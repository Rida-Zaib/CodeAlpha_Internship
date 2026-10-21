"""Tests for how llm_client copes with free-tier models being retired, bad keys
and empty replies. No network is used: requests.post / requests.get are mocked.
Run with `pytest tests/` (or just read them: each one is a small scenario)."""

import json
import os
import sys
from unittest import mock

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llm_client as L  # noqa: E402

FAQS = [{"category": "Shipping", "question": "How long does shipping take?", "answer": "3 to 5 days."}]


def resp(status, body=None):
    r = requests.Response()
    r.status_code = status
    r.encoding = "utf-8"
    r._content = json.dumps(body if body is not None else {}).encode()
    return r


def chat(text):
    return {"choices": [{"message": {"content": text}}]}


def gone(model):
    return {"error": {"message": f"The model `{model}` does not exist or you do not have access to it."}}


def fresh(**env):
    """Reset module state and set only the keys given."""
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN",
              "GROQ_MODEL", "GEMINI_MODEL", "OPENROUTER_MODEL"):
        os.environ.pop(k, None)
    os.environ.update(env)
    for s in L._bad_models.values():
        s.clear()
    L._discovered.clear()
    L.LAST_ERRORS.clear()


def ask():
    return L._safe_call("groq", L._call_groq, L.build_messages("hi", FAQS))


def test_retired_model_is_skipped_and_remembered():
    fresh(GROQ_API_KEY="k", GROQ_MODEL="llama-3.3-70b-versatile")   # a model that no longer exists
    def post(url, **kw):
        m = kw["json"]["model"]
        return resp(404, gone(m)) if m == "llama-3.3-70b-versatile" else resp(200, chat("hello"))
    with mock.patch.object(L.requests, "post", side_effect=post) as p:
        assert ask()["engine"] == "groq:openai/gpt-oss-120b"
        assert p.call_count == 2                       # dead model once, then the working one
        ask()
        assert p.call_count == 3                       # dead model is never asked again


def test_all_known_models_gone_so_it_asks_the_provider_what_exists():
    fresh(GROQ_API_KEY="k")
    listing = {"data": [{"id": "whisper-large-v3"}, {"id": "meta-llama/llama-guard-4-12b"},
                        {"id": "qwen/qwen3.6-27b"}, {"id": "brand/new-model"}]}
    def post(url, **kw):
        m = kw["json"]["model"]
        return resp(200, chat("found one")) if m == "qwen/qwen3.6-27b" else resp(404, gone(m))
    with mock.patch.object(L.requests, "post", side_effect=post), \
         mock.patch.object(L.requests, "get", return_value=resp(200, listing)):
        assert ask()["engine"] == "groq:qwen/qwen3.6-27b"   # speech/guard models were ignored


def test_bad_key_is_reported_not_retried_forever():
    fresh(GROQ_API_KEY="bad")
    with mock.patch.object(L.requests, "post", return_value=resp(401, {"error": {"message": "Invalid API Key"}})) as p:
        assert ask() is None
        assert p.call_count == 1
    assert L.LAST_ERRORS["groq"] == "HTTP 401"


def test_empty_reply_moves_on_to_the_next_model():
    fresh(GROQ_API_KEY="k")
    def post(url, **kw):
        return resp(200, chat("")) if kw["json"]["model"] == "openai/gpt-oss-120b" else resp(200, chat("real answer"))
    with mock.patch.object(L.requests, "post", side_effect=post):
        assert ask()["text"] == "real answer"


def test_nothing_works_and_nothing_to_discover_gives_up_cleanly():
    fresh(GROQ_API_KEY="k")
    with mock.patch.object(L.requests, "post", side_effect=lambda u, **kw: resp(404, gone(kw["json"]["model"]))), \
         mock.patch.object(L.requests, "get", return_value=resp(200, {"data": []})):
        assert ask() is None


def test_gemini_skips_thinking_parts_and_retired_models():
    fresh(GEMINI_API_KEY="k")
    ok = {"candidates": [{"content": {"parts": [{"text": "internal", "thought": True}, {"text": "Hello there"}]}}]}
    def post(url, **kw):
        return resp(404, {"error": {"message": "models/gemini-3.5-flash is not found"}}) if "gemini-3.5-flash:" in url else resp(200, ok)
    with mock.patch.object(L.requests, "post", side_effect=post):
        out = L._safe_call("gemini", L._call_gemini, L.build_messages("hi", FAQS))
    assert out == {"text": "Hello there", "engine": "gemini:gemini-3.1-flash-lite"}


def test_model_is_gone_only_for_model_problems():
    assert L._model_is_gone(resp(404, gone("x")))
    assert L._model_is_gone(resp(400, {"error": {"message": "The model has been decommissioned", "code": "model_decommissioned"}}))
    assert not L._model_is_gone(resp(400, {"error": {"message": "messages must alternate"}}))
    assert not L._model_is_gone(resp(429, {"error": {"message": "rate limit"}}))
