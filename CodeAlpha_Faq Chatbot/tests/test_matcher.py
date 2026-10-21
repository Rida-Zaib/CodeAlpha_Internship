"""
Basic sanity tests for the matcher. Run with: pytest
(requires: pip install pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from matcher import FAQMatcher

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "faqs.csv")


@pytest.fixture(scope="module")
def matcher():
    return FAQMatcher(CSV_PATH)


def test_exact_question_matches_itself(matcher):
    for question in matcher.questions[:5]:
        r = matcher.get_answer(question)
        assert r["matched_question"] == question
        assert r["score"] > 0.9


def test_paraphrase_matches(matcher):
    r = matcher.get_answer("how long will my package take to arrive")
    assert r["matched_question"] == "How long does shipping take?"


def test_synonym_bridges_vocabulary_gap(matcher):
    r = matcher.get_answer("can I get my money back")
    assert r["matched_question"] is not None
    assert "refund" in r["answer"].lower()


def test_typo_still_matches(matcher):
    r = matcher.get_answer("shiping time please")
    assert r["matched_question"] == "How long does shipping take?"


def test_unrelated_query_falls_back(matcher):
    r = matcher.get_answer("what is the weather today")
    assert r["matched_question"] is None
    assert r["confidence"] == "none"


def test_empty_query_handled_gracefully(matcher):
    r = matcher.get_answer("")
    assert r["matched_question"] is None
    assert r["score"] == 0.0


def test_gibberish_falls_back(matcher):
    r = matcher.get_answer("asdkjaslkdj qqzxcv random")
    assert r["matched_question"] is None
