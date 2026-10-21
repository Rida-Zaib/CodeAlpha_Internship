"""
Text preprocessing for the FAQ chatbot: lowercasing, punctuation removal,
tokenization, stopword removal, lemmatization (NLTK) and light stemming.

If the required NLTK data (punkt, stopwords, wordnet) has not been
downloaded yet, this module falls back to a simpler cleanup (lowercase +
whitespace split) instead of crashing, so the chatbot still runs — you just
get slightly less accurate matching until you run the downloads in
README.md.
"""

import os
import re

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True

    # On a hosting platform the data is downloaded into ./nltk_data by
    # setup_nltk.py (see README). Make NLTK look there too.
    _LOCAL_NLTK_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    if hasattr(nltk, "data") and _LOCAL_NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, _LOCAL_NLTK_DATA)
except ImportError:
    _NLTK_AVAILABLE = False

_STOPWORDS = set()
_lemmatizer = None
_stopwords_ok = False   # NLTK's own stopword list loaded
_tokenizer_ok = False   # NLTK's word_tokenize works (needs punkt / punkt_tab)

# Small built-in fallback so keyword-overlap and matching stay clean even if
# NLTK's own stopword corpus hasn't been downloaded yet (see README).
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "his", "her", "its", "our", "their", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "at",
    "for", "with", "about", "as", "by", "from", "up", "out", "can",
    "will", "would", "should", "could", "what", "which", "who", "whom",
    "how", "when", "where", "why", "there", "here", "please", "im", "id",
    # contraction fragments left over after stripping apostrophes ("haven't" -> "haven")
    "haven", "hasn", "hadn", "don", "doesn", "didn", "isn", "wasn", "aren",
    "weren", "couldn", "wouldn", "shouldn", "won", "ll", "ve", "re", "let",
    # filler / question words in Roman Urdu, so "kya aap ... hai" doesn't
    # count as meaningful overlap
    "kya", "hai", "hain", "hota", "hoti", "hote", "ho", "mein", "me", "ka",
    "ki", "ke", "ko", "se", "aap", "ap", "mera", "meri", "mere", "mujhe",
    "kar", "karna", "karun", "karen", "kaise", "kab", "kahan", "kitne",
    "kitni", "kitna", "to", "bhi", "ye", "yeh", "wo", "woh", "aur", "par",
    "nahi", "nahin", "koi", "kuch", "liye", "tak", "hum", "main",
}

# Words that appear in almost any message sent to this store and say nothing
# about *what* the customer wants ("I love your website", "hello technest").
# Always removed, whichever stopword list is in use.
_GENERIC_STOPWORDS = {
    "website", "site", "store", "shop", "company", "technest", "hello", "hi",
    "hey", "thanks", "thank", "sir", "madam", "kindly",
    "get", "gets", "getting", "got",
}

if _NLTK_AVAILABLE:
    try:
        _STOPWORDS = set(stopwords.words("english"))
        _stopwords_ok = True
    except LookupError:
        _STOPWORDS = _FALLBACK_STOPWORDS

    try:
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("testing")  # forces a check that wordnet data exists
    except LookupError:
        _lemmatizer = None

    try:
        word_tokenize("test")
        _tokenizer_ok = True
    except LookupError:
        pass
else:
    _STOPWORDS = _FALLBACK_STOPWORDS


def _tokenize(text: str):
    if _NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except LookupError:
            pass
    return text.split()


def stem(tok: str) -> str:
    """Very light suffix stripper applied on top of NLTK's lemmatizer, so
    "returning / returned / returns" and "shipping / ships" all collapse to
    one form whether or not the NLTK data is installed (WordNet's default
    lemmatizer only handles nouns, so verbs would otherwise differ between
    machines). Used identically on FAQ text and on user questions, so it
    only needs to be consistent, not linguistically perfect."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and not tok.endswith("ss") and len(tok) - len(suf) >= 3:
            tok = tok[: -len(suf)]
            if suf in ("ing", "ed") and len(tok) >= 4 and tok[-1] == tok[-2] and tok[-1] not in "lsz":
                tok = tok[:-1]  # shipping -> shipp -> ship
            break
    if tok.endswith("e") and len(tok) > 3:
        tok = tok[:-1]  # charge / charged / charges -> charg
    return tok


def clean_tokens(text: str) -> list:
    """Same cleanup as clean_text, but returns the token list instead of a
    joined string — used by the matcher for keyword-overlap scoring."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = _tokenize(text)

    cleaned = []
    for tok in tokens:
        if len(tok) <= 1 or tok in _STOPWORDS or tok in _GENERIC_STOPWORDS:
            continue
        if _lemmatizer is not None:
            try:
                tok = _lemmatizer.lemmatize(tok)
            except LookupError:
                pass
        cleaned.append(stem(tok))
    return cleaned


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers noise, tokenize, drop stopwords
    and short tokens, and lemmatize each remaining token."""
    return " ".join(clean_tokens(text))


def nlp_status() -> dict:
    """Which NLTK parts are really in use. Without the NLTK data the module
    still works (see the top of this file) but does not use NLTK for that part."""
    return {
        "nltk_installed": _NLTK_AVAILABLE,
        "tokenizer": _tokenizer_ok,
        "stopwords": _stopwords_ok,
        "lemmatizer": _lemmatizer is not None,
    }


def nlp_summary() -> tuple:
    """(everything_ok, one-line message) for the startup log."""
    st = nlp_status()
    if not st["nltk_installed"]:
        return False, "NLTK is not installed, using a basic fallback. Run: pip install -r requirements.txt"
    missing = [name for name in ("tokenizer", "stopwords", "lemmatizer") if not st[name]]
    if missing:
        return False, ("NLTK data missing for: " + ", ".join(missing) + ". Run: python -c \"import nltk; "
                       "[nltk.download(p) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]\"")
    return True, "NLTK active (word_tokenize, stopwords, WordNet lemmatizer)"