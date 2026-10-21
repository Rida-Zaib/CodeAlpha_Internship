"""
Smart mode: lets a free LLM API read the customer's message and answer it
from the FAQ knowledge base — so it can cope with any wording, any length,
several questions at once, and follow-ups ("and for international?").

How it stays honest: the model is given the ENTIRE FAQ list (it's small)
plus the last few chat turns, and is told to answer only from that list.
If the list doesn't cover the question it must reply NO_ANSWER, which
app.py turns into a polite "I couldn't find that" and logs as a gap. This
is retrieval-augmented generation (RAG) with the whole knowledge base in
the prompt — not fine-tuning.

If no key is configured, or every provider fails (network down, rate
limited, bad key), generate_answer() returns None and app.py falls back to
the local matcher — the chatbot always works, with or without AI.

Four free-tier providers are supported; you only need ONE key (README.md
shows how to get each). They are raced: every configured provider is
called AT THE SAME TIME on its own thread (Hugging Face internally tries a
list of models), the first good answer wins and the rest are ignored.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError

import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("faq-bot")

TIMEOUT = 8          # seconds per Groq/Gemini/OpenRouter call
RACE_TIMEOUT = 20    # seconds to wait for the fastest racer overall before giving up
MAX_TOKENS = 1024     # newer free models spend part of this on hidden reasoning, so leave headroom
MAX_HISTORY = 6      # most recent chat messages sent along, so follow-ups make sense
NO_ANSWER = "NO_ANSWER"

SYSTEM_PROMPT = (
    "You are the customer support assistant for TechNest, an online electronics store. "
    "Answer the customer's message using ONLY the knowledge base below.\n\n"
    "How to answer:\n"
    "- Work out what the customer actually means. They may write casually, with typos, in "
    "Roman Urdu/Hindi, very briefly, or in a long rambling message. Never expect them to "
    "use the exact wording of an FAQ.\n"
    "- If they ask several things, answer every part the knowledge base covers.\n"
    "- Use the earlier messages to understand follow-ups such as \"and for international?\".\n"
    "- Sound like a helpful person: natural, friendly and concise (usually 1-4 sentences). "
    "Rephrase in your own words; do not paste the FAQ text and never mention a "
    "\"knowledge base\" or \"FAQ\".\n"
    "- Never invent policies, prices, timings, links or contact details that are not in the "
    "knowledge base. If only part of the question is covered, answer that part and say you "
    "are not sure about the rest.\n"
    "- For a plain greeting or thanks, reply briefly and warmly.\n"
    "- If the message has nothing to do with the knowledge base (or it does not cover it at "
    "all), reply with exactly: " + NO_ANSWER + "\n"
    "- Reply in the language the customer wrote in.\n\n"
    "KNOWLEDGE BASE:\n"
)


def build_messages(user_query: str, faqs: list, history=None) -> list:
    """OpenAI-style message list: system prompt (with every FAQ), the recent
    conversation, then the customer's new message."""
    kb = "\n\n".join(f"[{f['category']}] Q: {f['question']}\nA: {f['answer']}" for f in faqs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT + kb}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_query})
    return messages


def clean_history(raw) -> list:
    """Validate chat history coming from the browser (it is untrusted input):
    only user/assistant turns, text only, trimmed, strictly alternating,
    starting with a user turn and ending with an assistant turn (the new
    user message follows), at most MAX_HISTORY messages."""
    out = []
    for item in (raw if isinstance(raw, list) else [])[-(MAX_HISTORY * 2):]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        entry = {"role": role, "content": content.strip()[:600]}
        if out and out[-1]["role"] == role:
            out[-1] = entry              # two in a row: keep the latest
        else:
            out.append(entry)
    out = out[-MAX_HISTORY:]
    while out and out[0]["role"] != "user":
        out.pop(0)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out


def is_no_answer(text: str) -> bool:
    return text.strip().strip("\"'`.! ").upper().startswith(NO_ANSWER)


# ---------------------------------------------------------------------------
# Groq, Gemini and OpenRouter share one mechanism, because free-tier models are
# retired every few months (Gemini 2.0 in June 2026, Groq's llama-3.3-70b in
# August 2026) and a hard-coded model name silently kills the whole AI layer:
#   1. try the models in our list, newest first (or your *_MODEL in .env);
#   2. a model the provider says is gone is remembered and skipped;
#   3. if none are left, ask the provider which models it serves right now.
# ---------------------------------------------------------------------------

PREFERRED_MODELS = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "gemini": ["gemini-3.5-flash", "gemini-3.1-flash-lite"],
    "openrouter": ["openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free",
                   "meta-llama/llama-3.3-70b-instruct:free"],
}

_bad_models = {name: set() for name in PREFERRED_MODELS}   # retired / unavailable, until restart
_discovered = {}                                           # models found by asking the provider
LAST_ERRORS = {}   # provider -> short reason it last failed (shown on /health)

# Words that mark a model as not being a text chat model.
_NOT_CHAT = ("whisper", "guard", "safeguard", "tts", "orpheus", "embed", "playai",
             "image", "audio", "vision", "moderation", "live", "transcribe")


def _model_queue(name: str) -> list:
    """Models to try, in order: your *_MODEL from .env (comma-separated is
    fine), then our preferred list, then anything discovered, minus retired ones."""
    from_env = [m.strip() for m in os.getenv(f"{name.upper()}_MODEL", "").split(",") if m.strip()]
    out = []
    for m in from_env + PREFERRED_MODELS[name] + _discovered.get(name, []):
        if m not in out and m not in _bad_models[name]:
            out.append(m)
    return out


def _model_is_gone(resp) -> bool:
    """Does this error mean 'that model name no longer exists'?"""
    if resp.status_code == 404:
        return True
    if resp.status_code == 400:
        t = resp.text.lower()
        return "decommission" in t or ("model" in t and any(
            w in t for w in ("not found", "does not exist", "not supported", "invalid", "deprecated", "no longer")))
    return False


def _error_detail(resp) -> str:
    """The provider's own explanation of an error, shortened."""
    if resp is None:
        return ""
    try:
        err = resp.json().get("error")
        text = err.get("message") if isinstance(err, dict) else (err or "")
    except (ValueError, AttributeError):
        text = ""
    return (text or resp.text or "").strip().replace("\n", " ")[:220]


def _rank_chat_model(model_id: str):
    m = model_id.lower()
    for i, hint in enumerate(("gpt-oss-120b", "gpt-oss", "llama", "qwen", "gemma", "mistral", "kimi", "deepseek")):
        if hint in m:
            return (i, m)
    return (99, m)


def _get_json(url, headers=None, params=None):
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _discover_groq(key):
    data = _get_json("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
    ids = [m["id"] for m in data.get("data", []) if m.get("id") and not any(w in m["id"].lower() for w in _NOT_CHAT)]
    return sorted(ids, key=_rank_chat_model)[:3]


def _discover_gemini(key):
    data = _get_json("https://generativelanguage.googleapis.com/v1beta/models",
                     {"x-goog-api-key": key}, {"pageSize": 200})
    skip = _NOT_CHAT + ("native", "exp", "thinking", "robotics", "computer")
    names = []
    for m in data.get("models", []):
        n = m.get("name", "").replace("models/", "")
        if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in n \
                and not any(w in n for w in skip):
            names.append(n)
    names.sort(reverse=True)                      # newest-looking name first
    names.sort(key=lambda n: "preview" in n)      # but stable releases before previews
    return names[:3]


def _discover_openrouter(key):
    data = _get_json("https://openrouter.ai/api/v1/models", {"Authorization": f"Bearer {key}"})
    ids = [m["id"] for m in data.get("data", [])
           if m.get("id", "").endswith(":free") and not any(w in m["id"].lower() for w in _NOT_CHAT)]
    return sorted(ids, key=_rank_chat_model)[:4]


def _openai_style_request(url):
    def send(key, model, messages):
        return requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": MAX_TOKENS},
            timeout=TIMEOUT,
        )
    return send


def _gemini_request(key, model, messages):
    return requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [
                {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in messages[1:]
            ],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": MAX_TOKENS},
        },
        timeout=TIMEOUT,
    )


def _parse_openai_style(resp) -> str:
    return (resp.json()["choices"][0]["message"].get("content") or "").strip()


def _parse_gemini(resp) -> str:
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


_SPECS = {
    "groq": {"key_env": "GROQ_API_KEY", "send": _openai_style_request("https://api.groq.com/openai/v1/chat/completions"),
             "parse": _parse_openai_style, "discover": _discover_groq},
    "gemini": {"key_env": "GEMINI_API_KEY", "send": _gemini_request,
               "parse": _parse_gemini, "discover": _discover_gemini},
    "openrouter": {"key_env": "OPENROUTER_API_KEY", "send": _openai_style_request("https://openrouter.ai/api/v1/chat/completions"),
                   "parse": _parse_openai_style, "discover": _discover_openrouter},
}


def _run_provider(name: str, messages: list):
    """Ask one provider, walking through its models until one answers."""
    spec = _SPECS[name]
    key = os.getenv(spec["key_env"])
    if not key:
        return None

    tried, asked_provider = set(), False
    for _ in range(8):                                   # hard cap on attempts
        queue = [m for m in _model_queue(name) if m not in tried]
        if not queue:
            if asked_provider:
                break
            asked_provider = True                        # all our models are gone: what does it offer now?
            found = spec["discover"](key)
            if not found:
                break
            _discovered[name] = found
            log.info("AI provider %s: now trying the models it currently offers: %s", name, ", ".join(found))
            continue

        model = queue[0]
        tried.add(model)
        resp = spec["send"](key, model, messages)
        if _model_is_gone(resp):
            _bad_models[name].add(model)
            log.warning("AI provider %s: model %s is no longer available (HTTP %s: %s); trying another",
                        name, model, resp.status_code, _error_detail(resp))
            continue
        resp.raise_for_status()                          # bad key, rate limit, ... handled by _safe_call
        try:
            text = spec["parse"](resp)
        except (KeyError, IndexError, TypeError, ValueError):
            text = ""
        if text:
            LAST_ERRORS.pop(name, None)
            return {"text": text, "engine": f"{name}:{model}"}
        log.warning("AI provider %s: model %s returned no text; trying another", name, model)
    return None


def _call_groq(messages: list):
    return _run_provider("groq", messages)


def _call_gemini(messages: list):
    return _run_provider("gemini", messages)


def _call_openrouter(messages: list):
    return _run_provider("openrouter", messages)


# Six solid free chat models on Hugging Face's router, tried in order.
# Having several matters specifically for HF: only one needs to already
# be "warm" (recently used, so no cold-start delay) for Smart mode to
# stay fast — one slow/sleeping model is no longer a dead end.
# Override with HF_MODELS="model/one,model/two" in .env to use your own
# list instead. Browse more at huggingface.co/models.
DEFAULT_HF_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "HuggingFaceH4/zephyr-7b-beta",
]
HF_CALL_TIMEOUT = 6  # shorter than TIMEOUT — we may try up to 6 of these in one request
HF_WARMUP_TIMEOUT = 15  # generous, since warm-up runs in the background, not on the user's request


def _hf_models() -> list:
    raw = os.getenv("HF_MODELS")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    single = os.getenv("HF_MODEL")  # back-compat with the old single-model var
    return [single] if single else DEFAULT_HF_MODELS


def _hf_request(key: str, model: str, messages: list, timeout: int):
    return requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
        },
        timeout=timeout,
    )


HF_PAUSE_SECONDS = 30 * 60  # how long to leave Hugging Face alone after a "no credits" / "bad token" reply

# Remembered for the life of the process so permanent failures are not
# retried (and don't slow down) every single question.
_hf_pause_until = 0.0
_hf_bad_models = set()


def _hf_problem(status: int):
    """(scope, plain-English reason) for a Hugging Face error status.
    scope is 'account' (affects every model), 'model' (just that one), or
    None for temporary trouble such as a rate limit or a model still loading."""
    if status == 402:
        return "account", ("payment required: the free Inference Providers credits "
                           "on this Hugging Face account are used up, or not included")
    if status in (401, 403):
        return "account", "the token was rejected; check HF_TOKEN in .env"
    if status in (400, 404, 422):
        return "model", "this model is not available through Hugging Face's router for this account"
    return None, None


def _hf_note_failure(model: str, status: int, emit) -> bool:
    """Record a permanent failure. Returns True if Hugging Face as a whole
    should be abandoned (so the caller stops trying more models)."""
    global _hf_pause_until
    scope, why = _hf_problem(status)
    if scope == "account":
        _hf_pause_until = time.time() + HF_PAUSE_SECONDS
        emit("Hugging Face HTTP %s: %s. Skipping Hugging Face for %d minutes; other "
             "providers or the local matcher are used instead. (Remove HF_TOKEN from "
             ".env to stop this message.)", status, why, HF_PAUSE_SECONDS // 60)
        return True
    if scope == "model" and model not in _hf_bad_models:
        _hf_bad_models.add(model)
        emit("Hugging Face HTTP %s for %s: %s. Not trying it again.", status, model, why)
    return False


def _hf_usable() -> bool:
    """Has a token, isn't paused for lack of credits, and has at least one
    model that hasn't been rejected."""
    if not os.getenv("HF_TOKEN") or time.time() < _hf_pause_until:
        return False
    return any(m not in _hf_bad_models for m in _hf_models())


def _call_huggingface(messages: list):
    """Hugging Face's "Inference Providers" router (OpenAI-compatible). Tries
    each model in turn. Permanent failures (no credits, bad token, a model the
    router doesn't serve) are remembered, so they cost nothing on later
    questions instead of burning ~1s per model every time."""
    if not _hf_usable():
        return None
    key = os.getenv("HF_TOKEN")

    for model in _hf_models():
        if model in _hf_bad_models:
            continue
        try:
            resp = _hf_request(key, model, messages, HF_CALL_TIMEOUT)
        except requests.exceptions.RequestException:
            continue  # timeout / network blip: try the next model
        if resp.status_code != 200:
            if _hf_note_failure(model, resp.status_code, log.warning):
                return None
            continue
        try:
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError):
            continue
        if text:
            return {"text": text, "engine": f"huggingface:{model}"}

    log.warning("AI provider huggingface: none of its models answered")
    return None


def warm_up_huggingface(log=None):
    """Ping each Hugging Face model once at startup. Returns the models that
    answered. Also the first place a 'no credits' / 'bad token' problem shows
    up, and it reports that once, in plain words, instead of once per model."""
    emit_log = log or globals()["log"]
    if not _hf_usable():
        return []
    key = os.getenv("HF_TOKEN")

    ping = [{"role": "system", "content": "Reply with the single word: ok"},
            {"role": "user", "content": "ping"}]
    warmed = []
    for model in _hf_models():
        if model in _hf_bad_models:
            continue
        try:
            resp = _hf_request(key, model, ping, HF_WARMUP_TIMEOUT)
        except requests.exceptions.RequestException as e:
            emit_log.info("Hugging Face warm-up: %s did not respond (%s)", model, e.__class__.__name__)
            continue
        if resp.status_code == 200:
            warmed.append(model)
            emit_log.info("Hugging Face model warmed up: %s", model)
        elif _hf_note_failure(model, resp.status_code, emit_log.warning):
            break  # whole account problem: no point pinging the rest
        elif _hf_problem(resp.status_code)[0] is None:
            emit_log.info("Hugging Face warm-up: %s not ready (HTTP %s), may be loading or rate-limited",
                          model, resp.status_code)
    return warmed


PROVIDERS = [
    ("groq", _call_groq),
    ("gemini", _call_gemini),
    ("openrouter", _call_openrouter),
    ("huggingface", _call_huggingface),
]

# Each provider's env var name for its key — most follow {NAME}_API_KEY,
# but Hugging Face's own convention is HF_TOKEN, so it needs a manual
# mapping here (used by configured_providers() below).
_KEY_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HF_TOKEN",
}


def _safe_call(name: str, fn, messages: list):
    """Run one provider; log (never raise) why it failed, so a bad key or a
    retired model shows up in the terminal instead of failing silently."""
    try:
        return fn(messages)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        LAST_ERRORS[name] = f"HTTP {status}"
        hint = {401: " (the API key was rejected: check it in .env)",
                403: " (the API key was rejected: check it in .env)",
                429: " (rate limited: wait a minute)"}.get(status, "")
        log.warning("AI provider %s failed: HTTP %s %s%s", name, status, _error_detail(e.response), hint)
    except requests.exceptions.RequestException as e:
        LAST_ERRORS[name] = e.__class__.__name__
        log.warning("AI provider %s could not be reached: %s", name, e.__class__.__name__)
    except (KeyError, IndexError, ValueError) as e:
        LAST_ERRORS[name] = "unexpected response"
        log.warning("AI provider %s returned an unexpected response (%s)", name, e.__class__.__name__)
    return None


def generate_answer(user_query: str, faqs: list, history=None):
    """Race every configured provider and return {"text", "engine"} from the
    first that answers, or None if none are configured / none answer within
    RACE_TIMEOUT (the caller then falls back to the local matcher). The text
    may be the NO_ANSWER marker — see is_no_answer()."""
    configured = [(n, fn) for n, fn in PROVIDERS if _provider_usable(n)]
    if not configured:
        return None

    messages = build_messages(user_query, faqs, history)

    # One provider: nothing to race, so call it directly. No threads means no
    # dependence on the host's thread support (some shared hosts, e.g.
    # PythonAnywhere's free plan, are fussy about threads).
    if len(configured) == 1:
        name, fn = configured[0]
        result = _safe_call(name, fn, messages)
        return result if result and result.get("text") else None

    executor = ThreadPoolExecutor(max_workers=len(configured))
    futures = [executor.submit(_safe_call, n, fn, messages) for n, fn in configured]
    winner = None
    try:
        for future in as_completed(futures, timeout=RACE_TIMEOUT):
            result = future.result()
            if result and result.get("text"):
                winner = result
                break  # first good answer wins — stop waiting for the rest
    except FutureTimeoutError:
        log.warning("No AI provider answered within %ss", RACE_TIMEOUT)
    finally:
        # Don't block on the losers; any still-running HTTP calls finish in
        # the background and are discarded.
        executor.shutdown(wait=False, cancel_futures=True)
    return winner


def _provider_usable(name: str) -> bool:
    """Key is set and (for Hugging Face) it isn't currently written off."""
    if not os.getenv(_KEY_ENV_VARS[name]):
        return False
    return _hf_usable() if name == "huggingface" else True


def configured_providers() -> list:
    """Providers that have a key set and are actually usable right now. Shown
    on /health, and the page only offers the Smart/Fast switch when this is
    non-empty, so a dead Hugging Face token no longer makes it appear."""
    return [name for name, _ in PROVIDERS if _provider_usable(name)]