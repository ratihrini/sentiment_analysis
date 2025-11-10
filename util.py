import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [w for w in s.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)
