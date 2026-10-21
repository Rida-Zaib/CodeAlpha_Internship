"""
A small persistent cache for Smart-mode (AI-phrased) answers.

Keyed by the *matched FAQ question* (not the user's raw wording), because
the matcher already groups every paraphrase of "how long is shipping" /
"when will my order arrive" / "shiping time" onto the same FAQ entry — so
they should all reuse the same AI-phrased answer instead of each paying
the cold-start / latency cost of a fresh API call.

This directly addresses loading time: the very first time a given FAQ is
answered in Smart mode, it may take a second or two (or fall through
several Hugging Face models). Every time after that, for that FAQ, the
answer comes back instantly from this cache — no excuse left for it being
slow on repeat questions.

Backed by a plain JSON file so it survives app restarts. Thread-safe via
a simple lock (good enough for Flask's dev server / a handful of workers;
if you outgrow this, swap it for SQLite or Redis without changing the
public functions below).
"""

import json
import threading
from pathlib import Path

CACHE_FILE = Path("data/ai_cache.json")
_lock = threading.Lock()
_cache = {}
_loaded = False


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:  # re-check inside the lock (another thread may have just loaded it)
            return
        if CACHE_FILE.exists():
            try:
                _cache.update(json.loads(CACHE_FILE.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass  # corrupt or unreadable cache file — start fresh rather than crash
        _loaded = True


def _save():
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        CACHE_FILE.write_text(json.dumps(_cache, indent=2), encoding="utf-8")
    except OSError:
        pass  # caching is best-effort — never let a disk error break the chat


def get(matched_question: str):
    """Returns {"text", "engine"} if this FAQ's AI answer is cached, else None."""
    _ensure_loaded()
    return _cache.get(matched_question)


def set(matched_question: str, text: str, engine: str):
    _ensure_loaded()
    with _lock:
        _cache[matched_question] = {"text": text, "engine": engine}
        _save()


def stats() -> dict:
    _ensure_loaded()
    return {"cached_answers": len(_cache)}


def clear():
    """Wipe the cache — useful after editing faqs.csv wording or switching
    HF_MODELS, so stale phrasing doesn't linger."""
    global _cache
    _ensure_loaded()
    with _lock:
        _cache = {}
        _save()
