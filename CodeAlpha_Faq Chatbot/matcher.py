"""
The local matching engine. Classic NLP, no external API, no GPU, answers in
a few milliseconds. It is the instant fallback when no AI key is set, and
the thing that suggests related questions when nothing matches.

How it copes with people asking things "in their own way":

1. Alternate phrasings   — every FAQ in faqs.csv can carry an `alternates`
                           column ("when will my parcel reach me | how soon
                           is delivery | ..."). Each phrasing is indexed, so
                           a customer doesn't have to guess the FAQ's exact
                           wording. This is the standard way FAQ bots are
                           made flexible; add more alternates over time.
2. Answer-text indexing  — the answer itself is searchable too ("hidden
                           charges" finds the tax FAQ even if the question
                           is worded differently).
3. Long questions        — a long message is also split into clauses and
                           each clause is scored, so filler ("I ordered a
                           laptop yesterday and I was wondering...") doesn't
                           drown out the actual question.
4. Synonyms + typos      — a small synonym map and difflib-based typo
                           correction against the FAQ vocabulary.
5. Honest fallback       — if nothing scores high enough, it says so and
                           offers the closest topics, and logs the question
                           to data/unanswered.csv (see gap_report.py) so you
                           know which alternates/FAQs to add next.
"""

import csv
import difflib
import os
import re
import time
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_tokens

DEFAULT_THRESHOLD = 0.30
TYPO_MATCH_CUTOFF = 0.82      # how close a word must be to a known word to be auto-corrected
HIGH_CONFIDENCE = 0.55
SECOND_INTENT_MIN = 0.40      # a clause must match a *different* FAQ this well to earn its own answer
SECOND_INTENT_MARGIN = 0.15   # ...and beat the main FAQ's score for that same clause by this much
MAX_ANSWERS = 3               # most answers combined into one reply for a multi-part question

# How the final score is built from four signals (they sum to 1.0).
W_BEST_PHRASING = 0.40   # best match against any single phrasing of the FAQ
W_PROFILE = 0.25         # match against all of the FAQ's phrasings combined
W_ANSWER = 0.10          # match against the answer text
W_COVERAGE = 0.25        # how much of the question's meaningful words the FAQ covers

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UNANSWERED_LOG = os.path.join(DATA_DIR, "unanswered.csv")

FALLBACK_ANSWER = (
    "Sorry, I couldn't find an answer to that. Try rephrasing it, or "
    "contact our support team and they'll help you out."
)

# Small synonym map: phrases customers use that the FAQ wording doesn't.
SYNONYMS = {
    "money back": "refund",
    "get my money": "refund",
    "reimburse": "refund",
    "cancel order": "cancel",
    "abroad": "international",
    "overseas": "international",
    "outside the country": "international",
    "other countries": "international",
    "parcel": "package",
    "shipment": "package",
    "guarantee": "warranty",
    "broken": "damaged",
    "defective": "damaged",
    "faulty": "damaged",
    "cracked": "damaged",
    "wholesale": "bulk",
    "cod": "cash on delivery",
    "sold out": "out of stock",
    "sign up": "create account",
    "signup": "create account",
    "register": "create account",
    "helpline": "customer support",
    "agent": "customer support",
    "gst": "tax",
    "vat": "tax",
    "present": "gift",
}


# Intent patterns: phrasings where the *shape* of the sentence carries the
# meaning, which plain keyword matching loses once "where/when" are dropped
# as stopwords. Each pattern appends the FAQ-style keyword for that intent.
INTENT_PATTERNS = [
    (re.compile(r"\bwhere\b.*\b(order|package|parcel|shipment|delivery|stuff|item)\b"), "track"),
    (re.compile(r"\b(status|whereabouts)\b"), "track"),
    (re.compile(r"\bwhen\b.*\b(show up|shows up|arrive|arrives|come|comes|reach|reaches|get here|get to me)\b"), "delivery time"),
]


def _expand_synonyms(raw_query: str) -> str:
    lowered = raw_query.lower()
    extra = [mapped for phrase, mapped in SYNONYMS.items() if re.search(r"\b" + re.escape(phrase) + r"\b", lowered)]
    extra += [mapped for pattern, mapped in INTENT_PATTERNS if pattern.search(lowered)]
    return raw_query + " " + " ".join(extra) if extra else raw_query


_CLAUSE_SPLIT = re.compile(r"[.?!;\n]+|\s+(?:and|but|also|plus)\s+|,\s+")


class FAQMatcher:
    def __init__(self, csv_path: str, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        faqs = self._load(csv_path)
        self.questions = [f["question"] for f in faqs]
        self.answers = [f["answer"] for f in faqs]
        self.categories = [f["category"] for f in faqs]
        self.alternates = [f["alternates"] for f in faqs]
        n = len(faqs)

        # One "document" per phrasing (the question itself + each alternate).
        doc_tokens, doc_owner = [], []
        for i, f in enumerate(faqs):
            for text in [f["question"]] + f["alternates"]:
                doc_tokens.append(clean_tokens(text))
                doc_owner.append(i)
        answer_tokens = [clean_tokens(a) for a in self.answers]
        self._doc_owner = np.array(doc_owner)

        self.vocab = sorted({t for toks in doc_tokens + answer_tokens for t in toks})

        corpus = [" ".join(t) for t in doc_tokens] + [" ".join(t) for t in answer_tokens]
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        self.vectorizer.fit(corpus)

        self._doc_matrix = self.vectorizer.transform([" ".join(t) for t in doc_tokens])
        self._answer_matrix = self.vectorizer.transform([" ".join(t) for t in answer_tokens])

        # "Profile" = every phrasing of a FAQ merged into one vector.
        profiles = []
        for i in range(n):
            rows = self._doc_matrix[self._doc_owner == i]
            vec = np.asarray(rows.sum(axis=0)).ravel()
            norm = np.linalg.norm(vec)
            profiles.append(vec / norm if norm else vec)
        self._profile_matrix = np.vstack(profiles)

        # Words each FAQ "covers" (phrasings + answer), and how rare each word
        # is, for the coverage signal.
        self._covered = [set() for _ in range(n)]
        for toks, owner in zip(doc_tokens, doc_owner):
            self._covered[owner].update(toks)
        for i, toks in enumerate(answer_tokens):
            self._covered[i].update(toks)
        self._idf = {
            term: float(self.vectorizer.idf_[idx])
            for term, idx in self.vectorizer.vocabulary_.items()
            if " " not in term
        }

        os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------ data
    @staticmethod
    def _load(csv_path: str) -> list:
        faqs = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = (row.get("question") or "").strip()
                a = (row.get("answer") or "").strip()
                if not (q and a):
                    continue
                alts = [x.strip() for x in (row.get("alternates") or "").split("|") if x.strip()]
                faqs.append({
                    "category": (row.get("category") or "General").strip(),
                    "question": q, "answer": a, "alternates": alts,
                })
        if not faqs:
            raise ValueError(f"No FAQs found in {csv_path}")
        return faqs

    def all_faqs(self) -> list:
        """Every FAQ as {category, question, answer} — the knowledge base the
        AI layer (llm_client.py) reads from."""
        return [
            {"category": c, "question": q, "answer": a}
            for c, q, a in zip(self.categories, self.questions, self.answers)
        ]

    # --------------------------------------------------------------- scoring
    def _correct_typos(self, tokens: list) -> list:
        vocab_set = set(self.vocab)
        corrected = []
        for tok in tokens:
            if tok in vocab_set or len(tok) < 4:
                corrected.append(tok)
                continue
            close = [
                w for w in difflib.get_close_matches(tok, self.vocab, n=3, cutoff=TYPO_MATCH_CUTOFF)
                if abs(len(w) - len(tok)) <= 1
            ]
            corrected.append(close[0] if close else tok)
        return corrected

    def _score_tokens(self, tokens: list) -> np.ndarray:
        """Score every FAQ for one bag of (cleaned) tokens."""
        n = len(self.questions)
        if not tokens:
            return np.zeros(n)

        q_vec = self.vectorizer.transform([" ".join(tokens)])

        doc_sims = cosine_similarity(q_vec, self._doc_matrix)[0]
        best_phrasing = np.zeros(n)
        np.maximum.at(best_phrasing, self._doc_owner, doc_sims)

        q_dense = q_vec.toarray().ravel()
        q_norm = np.linalg.norm(q_dense)
        profile = (self._profile_matrix @ q_dense) / q_norm if q_norm else np.zeros(n)

        answer = cosine_similarity(q_vec, self._answer_matrix)[0]

        known = [t for t in set(tokens) if t in self._idf]
        total = sum(self._idf[t] for t in known)
        coverage = np.zeros(n)
        if total:
            for i in range(n):
                coverage[i] = sum(self._idf[t] for t in known if t in self._covered[i]) / total

        return (W_BEST_PHRASING * best_phrasing + W_PROFILE * profile
                + W_ANSWER * answer + W_COVERAGE * coverage)

    def _scores_for_query(self, user_query: str, with_clauses: bool = False):
        """Scores for the whole message, and — for longer messages — for each
        clause separately; each FAQ keeps its best score. With
        with_clauses=True also returns [(clause_position, scores)] so a
        multi-part question can be answered part by part."""
        expanded = _expand_synonyms(user_query)
        whole = self._correct_typos(clean_tokens(expanded))
        scores = self._score_tokens(whole)

        if 0 < len(whole) <= 3:
            understood = sum(1 for t in whole if t in self._idf) / len(whole)
            scores = scores * (0.4 + 0.6 * understood)

        clauses = []
        parts = [c for c in _CLAUSE_SPLIT.split(expanded) if c and c.strip()]
        if len(parts) >= 2 or len(whole) >= 6:
            for pos, clause in enumerate(parts):
                toks = self._correct_typos(clean_tokens(clause))
                known = [t for t in toks if t in self._idf]
                if len(known) >= 2 and len(toks) < len(whole):
                    arr = self._score_tokens(toks)
                    if len(whole) >= 6:      # long messages: a clause can lift the main match
                        scores = np.maximum(scores, arr)
                    clauses.append((pos, arr))   # any message with 2+ clauses: each can earn its own answer
        return (scores, clauses) if with_clauses else scores

    def _answer_order(self, best_idx: int, clauses: list) -> list:
        """FAQ indices to answer, in the order the customer asked them. Always
        starts from the best overall match; adds at most MAX_ANSWERS-1 more,
        each only if one clause of the message clearly asks about it."""
        best_pos, extras = None, {}
        for pos, arr in clauses:
            j = int(np.argmax(arr))
            if j == best_idx:
                best_pos = pos if best_pos is None else best_pos
            elif (arr[j] >= SECOND_INTENT_MIN and arr[j] - arr[best_idx] >= SECOND_INTENT_MARGIN
                  and arr[j] > extras.get(j, (0, 0.0))[1]):
                extras[j] = (pos, float(arr[j]))
        strongest = sorted(extras.items(), key=lambda kv: -kv[1][1])[:MAX_ANSWERS - 1]
        picked = [(best_pos if best_pos is not None else 0, best_idx)] + [(pos, j) for j, (pos, _) in strongest]
        return [idx for _, idx in sorted(picked, key=lambda x: x[0])]

    # ----------------------------------------------------------------- public
    def log_unanswered(self, raw_query: str, best_score: float = 0.0):
        os.makedirs(DATA_DIR, exist_ok=True)
        is_new_file = not os.path.exists(UNANSWERED_LOG)
        with open(UNANSWERED_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new_file:
                w.writerow(["timestamp", "question", "best_score"])
            w.writerow([datetime.utcnow().isoformat(timespec="seconds"), raw_query, f"{best_score:.3f}"])

    def _suggestions(self, scores: np.ndarray, k: int = 3, floor: float = 0.08) -> list:
        ranked = np.argsort(-scores)[:k]
        return [
            {"question": self.questions[i], "score": round(float(scores[i]), 3)}
            for i in ranked if scores[i] >= floor
        ]

    def get_answer(self, user_query: str, log_unanswered: bool = True) -> dict:
        """Best local answer. log_unanswered=False lets the caller decide what
        counts as a gap (e.g. when an AI model might still answer it)."""
        start = time.perf_counter()

        def result(answer, idx=None, score=0.0, confidence="none", alternatives=None, also=None):
            return {
                "answer": answer,
                "matched_question": self.questions[idx] if idx is not None else None,
                "also_matched": also or [],
                "category": self.categories[idx] if idx is not None else None,
                "score": round(float(score), 3),
                "confidence": confidence,
                "alternatives": alternatives or [],
                "response_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        if not user_query or not user_query.strip():
            return result("Could you type a question?")

        scores, clauses = self._scores_for_query(user_query, with_clauses=True)
        best_idx = int(np.argmax(scores))
        best = float(scores[best_idx])

        if best < self.threshold:
            if log_unanswered:
                self.log_unanswered(user_query, best)
            return result(FALLBACK_ANSWER, score=best, alternatives=self._suggestions(scores))

        order = self._answer_order(best_idx, clauses)
        confidence = "high" if best >= HIGH_CONFIDENCE else "medium"
        combined = "\n\n".join(self.answers[i] for i in order)
        return result(combined, best_idx, best, confidence,
                      also=[self.questions[i] for i in order if i != best_idx])

    def get_top_k(self, user_query: str, k: int = 3) -> list:
        """The k closest FAQs regardless of the confidence threshold, as
        {question, answer, category, score}. Doesn't log anything."""
        if not user_query or not user_query.strip():
            return []
        scores = self._scores_for_query(user_query)
        ranked = np.argsort(-scores)[:k]
        return [
            {"question": self.questions[i], "answer": self.answers[i],
             "category": self.categories[i], "score": round(float(scores[i]), 3)}
            for i in ranked if scores[i] > 0.05
        ]
