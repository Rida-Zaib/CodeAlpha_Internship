"""Flask web app for the FAQ chatbot.

Run with: python app.py   (dev server, http://127.0.0.1:5000)
For a proper deployment use a real WSGI server instead of the Flask dev
server, e.g.:  waitress-serve --port=5000 app:app   (Windows-friendly)
           or: gunicorn -w 4 -b 0.0.0.0:5000 app:app   (Linux/Mac)
"""

import csv
import logging
import os
import re
import threading
import time
from datetime import datetime

from flask import Flask, jsonify, render_template, request

import ai_cache
from preprocess import nlp_status, nlp_summary
from matcher import FAQMatcher, FALLBACK_ANSWER
from llm_client import (
    LAST_ERRORS, clean_history, configured_providers, generate_answer, is_no_answer, warm_up_huggingface,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("faq-bot")

FAQ_FILE = "faqs.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_LOG = os.path.join(DATA_DIR, "feedback.csv")

app = Flask(__name__)

try:
    matcher = FAQMatcher(FAQ_FILE)
    log.info("Loaded %d FAQs from %s", len(matcher.questions), FAQ_FILE)
    _nlp_ok, _nlp_msg = nlp_summary()
    (log.info if _nlp_ok else log.warning)("Text preprocessing: %s", _nlp_msg)
except Exception:
    log.exception("Failed to load FAQ matcher at startup")
    raise

_warmed_models = []


def _warm_up_in_background():
    """Pings every configured Hugging Face model once at startup so they
    start loading now, instead of a real visitor eating that cold-start
    delay on their first Smart-mode question. Runs on a daemon thread so
    it never blocks or crashes app startup — if it's still running when
    the first real request comes in, that request just tries the models
    directly itself (see llm_client._call_huggingface), so nothing is
    lost by warm-up being slow or unconfigured."""
    global _warmed_models
    try:
        _warmed_models = warm_up_huggingface(log=log)
        if _warmed_models:
            log.info("Warm-up complete: %d Hugging Face model(s) ready", len(_warmed_models))
    except Exception:
        log.exception("Hugging Face warm-up thread failed (non-fatal)")


# With `python app.py` (debug mode) Flask runs this file twice: a watcher
# process and the real server. Only warm up in the real one, otherwise every
# startup message appears twice. Under gunicorn __name__ is "app", so it runs.
_is_watcher = __name__ == "__main__" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
if configured_providers() and not _is_watcher:
    threading.Thread(target=_warm_up_in_background, daemon=True).start()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight readiness check — useful for uptime monitors / load balancers."""
    return jsonify({
        "status": "ok",
        "faqs_loaded": len(matcher.questions),
        "ai_providers_configured": configured_providers(),
        "nlp": nlp_status(),
        "ai_errors": dict(LAST_ERRORS),   # e.g. {"groq": "HTTP 401"}: why a configured provider is failing
        "hf_models_warmed": _warmed_models,
        **ai_cache.stats(),
    })


MAX_MESSAGE_CHARS = 1000
MAX_FULL_KB = 40   # up to this many FAQs the AI reads them all; beyond it, only the closest 15

# Greetings / thanks / goodbyes: answered instantly and locally — they are
# not "unanswered questions" and shouldn't spend AI quota. Only triggers when
# the WHOLE message is the greeting ("hi, where is my order" goes to the AI).
_SMALLTALK = [
    (re.compile(r"^(hi+|hello+|hey+|hiya|heya|yo|salam|salaam|aoa|greetings|assalam[a-z ]{0,12}"
                r"|good (morning|afternoon|evening))( (there|team|bot|support|everyone|guys))?$"),
     "Hi! What can I help you with today?"),
    (re.compile(r"^((thanks|thank you|thank u|thx|ty|shukriya|jazakallah|jazak allah|appreciate it)"
                r"( (so|very) much| a lot)?|(great|ok|okay|cool) thanks)$"),
     "You're welcome! Let me know if there's anything else I can help with."),
    (re.compile(r"^(bye|goodbye|see you|see ya|take care|allah hafiz|khuda hafiz)$"),
     "Goodbye! Come back any time."),
]


def smalltalk_reply(message: str):
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", message.lower())).strip()
    for pattern, reply in _SMALLTALK:
        if pattern.match(t):
            return reply
    return None


def _normalise(message: str) -> str:
    return re.sub(r"\s+", " ", message.lower()).strip(" .!?,")


def _ask_ai(message: str, history: list):
    """Let the AI answer from the FAQ knowledge base. Returns
    {"text", "engine", "cached"}, {"no_answer": True}, or None when no AI
    provider could be reached (caller falls back to the local matcher)."""
    cache_key = "q:" + _normalise(message)
    use_cache = not history          # a follow-up depends on the chat so far, so only
    if use_cache:                    # stand-alone questions are safe to cache
        hit = ai_cache.get(cache_key)
        if hit:
            return {"text": hit["text"], "engine": hit["engine"], "cached": True}

    faqs = matcher.all_faqs()
    if len(faqs) > MAX_FULL_KB:      # a big knowledge base won't fit the prompt: send the closest ones
        faqs = [{"category": c["category"], "question": c["question"], "answer": c["answer"]}
                for c in matcher.get_top_k(message, k=15)]
        if not faqs:
            return {"no_answer": True}

    ai = generate_answer(message, faqs, history)
    if not ai:
        return None
    if is_no_answer(ai["text"]):
        return {"no_answer": True}
    if use_cache:
        ai_cache.set(cache_key, ai["text"], ai["engine"])
    return {"text": ai["text"], "engine": ai["engine"], "cached": False}


@app.route("/ask", methods=["POST"])
def ask():
    started = time.perf_counter()
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    mode = (data.get("mode") or "smart").strip().lower()   # "smart" (AI) or "fast" (local only)
    history = clean_history(data.get("history"))

    def reply(answer, answered, engine, **extra):
        return jsonify({
            "answer": answer, "answered": answered, "engine": engine,
            "suggestions": [], "matched_question": None,
            "response_ms": round((time.perf_counter() - started) * 1000, 1),
            **extra,
        })

    if not message:
        return reply("Please type a question.", False, "none"), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return reply(f"That message is a bit long for me — could you shorten it to under {MAX_MESSAGE_CHARS} characters?",
                     False, "none"), 400

    small = smalltalk_reply(message)
    if small:
        return reply(small, True, "smalltalk", smalltalk=True)

    try:
        local = matcher.get_answer(message, log_unanswered=False)
        nearest = local["matched_question"]

        ai = None
        if mode == "smart" and configured_providers():
            ai = _ask_ai(message, history)

        if ai and ai.get("text"):
            return reply(ai["text"], True, ai["engine"], matched_question=nearest, cached=ai["cached"])

        if ai and ai.get("no_answer"):
            matcher.log_unanswered(message, local["score"])
            tips = [c["question"] for c in matcher.get_top_k(message, k=3)]
            return reply(FALLBACK_ANSWER, False, "ai", suggestions=tips)

        # Fast mode, no AI key, or no AI provider reachable: use the local matcher.
        extra = {"ai_unavailable": True} if mode == "smart" and configured_providers() else {}
        if nearest:
            return reply(local["answer"], True, "local", matched_question=nearest, **extra)
        matcher.log_unanswered(message, local["score"])
        return reply(local["answer"], False, "local",
                     suggestions=[a["question"] for a in local["alternatives"]], **extra)
    except Exception:
        log.exception("Error while answering: %r", message)
        return reply("Something went wrong on my end. Please try again.", False, "error"), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """Logs a thumbs up/down on a given answer — this is how you find out
    which matches are actually wrong, not just which questions went unanswered."""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    matched_question = (data.get("matched_question") or "").strip()
    helpful = bool(data.get("helpful"))

    if not question:
        return jsonify({"ok": False, "error": "Missing question."}), 400

    os.makedirs(DATA_DIR, exist_ok=True)
    is_new = not os.path.exists(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "question", "matched_question", "helpful"])
        w.writerow([datetime.utcnow().isoformat(timespec="seconds"), question, matched_question, helpful])

    log.info("Feedback logged: helpful=%s question=%r", helpful, question)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
