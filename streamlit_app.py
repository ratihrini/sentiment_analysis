import streamlit as st
from joblib import load
from util import clean_text

st.set_page_config(page_title="Sentiment Analysis", page_icon="🔎")

@st.cache_resource
def load_artifacts():
    tfidf = load("models/tfidf.joblib")
    model = load("models/sentiment_lr.joblib")
    return tfidf, model

st.title("Customer Review Sentiment")
st.write("Ketik atau tempelkan review, lalu lihat prediksinya.")

tfidf, model = load_artifacts()
user_text = st.text_area("Your review", height=150, placeholder="Type a review here")

if st.button("Predict") and user_text.strip():
    cleaned = clean_text(user_text)
    X = tfidf.transform([cleaned])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]

    label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    label = label_map.get(int(pred), str(pred))

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence")
    st.progress(float(max(proba)))
    st.json({label_map.get(i, str(i)): float(p) for i, p in enumerate(proba)})
