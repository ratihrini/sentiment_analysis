import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
from utils import clean_text

df = pd.read_csv("data/processed/reviews_clean.csv")  # pastikan ada kolom text, label
X = df["text"].astype(str).apply(clean_text)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=2000, n_jobs=-1)
clf.fit(Xtr, y_train)

pred = clf.predict(Xte)
print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=4))

dump(tfidf, "models/tfidf.joblib")
dump(clf, "models/sentiment_lr.joblib")
