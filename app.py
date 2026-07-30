import joblib
import streamlit as st

from preprocessing import clean_text

# ----------------------------------------------------------------------------
# Language Identification App
# Backend model: TF-IDF (character n-grams 1-3) + Linear SVM (LinearSVC)
# Loads a PRETRAINED model - run train_model.py once beforehand to produce
# vectorizer.joblib and model.joblib. No training happens when this app runs.
# ----------------------------------------------------------------------------

VECTORIZER_PATH = "vectorizer.joblib"
MODEL_PATH = "model.joblib"

st.set_page_config(page_title="Language Identifier", page_icon="🌐", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def predict_language(text: str, vectorizer, model) -> str:
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    return model.predict(vector)[0]


# ---------------- UI ----------------

st.title("🌐 Language Identifier")
st.write(
    "This app detects the language of a piece of text. Paste a sentence or "
    "paragraph below and click **Predict Language** — the text will be run "
    "through a traditional Machine Learning model (TF-IDF character n-grams "
    "+ Linear SVM) trained on a multilingual dataset, and the app will tell "
    "you which language it is written in."
)

vectorizer, model = load_model()

text_input = st.text_area(
    "Paste your text here:",
    height=150,
    placeholder="Type or paste a sentence...",
)

if st.button("Predict Language", type="primary"):
    if not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        prediction = predict_language(text_input, vectorizer, model)
        st.markdown(f"### This text is written in **{prediction}** language.")

with st.expander("Supported languages"):
    st.write(sorted(model.classes_.tolist()))