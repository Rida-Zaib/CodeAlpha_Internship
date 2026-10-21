"""
Tests every AI key in your .env with one real request and tells you, in plain
words, what is wrong (bad key, retired model, rate limit, blocked network...).

    python check_ai.py

Run this whenever answers look like plain FAQ text instead of natural replies:
that is what the chatbot falls back to when no AI provider works.
"""

import logging
import os
import time

import requests

import llm_client as L
from matcher import FAQMatcher

logging.basicConfig(level=logging.INFO, format="    %(message)s")

QUESTION = "my parcel came cracked, how long until I get my money back?"

HINTS = {
    401: "The key was rejected. Re-copy it into .env (no quotes, no spaces) and restart.",
    403: "The key was rejected or this provider isn't available to your account/region.",
    402: "Payment required: this account's free credits are used up.",
    429: "Rate limited. Wait a minute and run this again.",
}


def main():
    faqs = FAQMatcher("faqs.csv").all_faqs()
    messages = L.build_messages(QUESTION, faqs)
    print(f"Test question: {QUESTION!r}\n")

    working = []
    for name, fn in L.PROVIDERS:
        if not L._provider_usable(name):
            why = "no key set in .env" if not os.getenv(L._KEY_ENV_VARS[name]) else "paused after earlier errors"
            print(f"- {name:<12} skipped ({why})")
            continue
        started = time.perf_counter()
        try:
            result = fn(messages)
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            print(f"- {name:<12} FAILED  HTTP {code}: {L._error_detail(e.response)}")
            print(f"{'':<15}{HINTS.get(code, '')}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"- {name:<12} FAILED  could not connect ({e.__class__.__name__}). "
                  "Check your internet, VPN or firewall.")
            continue
        took = time.perf_counter() - started
        if result:
            working.append(name)
            print(f"- {name:<12} OK  {result['engine']}  ({took:.1f}s)")
            print(f"{'':<15}reply: {result['text'][:300]}")
        else:
            print(f"- {name:<12} FAILED  no model answered (see the lines above for each model)")

    print()
    if working:
        print(f"Working: {', '.join(working)}. Smart mode will use the fastest of these.")
    else:
        print("No AI provider works right now, so the chatbot is using its basic FAQ matcher.")
        print("Fix the first FAILED line above, then run this again.")


if __name__ == "__main__":
    main()
