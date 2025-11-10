import re
import nltk
from nltk.corpus import stopwords

def _ensure_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

def clean_text(s: str) -> str:
    _ensure_nltk()
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [w for w in s.split() if w not in set(stopwords.words("english")) and len(w) > 2]
    return " ".join(tokens)
