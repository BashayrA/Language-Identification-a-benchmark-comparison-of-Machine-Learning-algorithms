import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from preprocessing import clean_text, contains_emoji, contains_url

# ----------------------------------------------------------------------------
# Run this script ONCE (locally, or in a one-off Colab/terminal session) to
# train the model and save it to disk:
#
#     python train_model.py
#
# It produces two files: vectorizer.joblib and model.joblib
# Commit those two files (NOT dataset.csv) alongside app.py and
# preprocessing.py to your deployment repo. The Streamlit app then just
# loads them - no training happens when the app starts.
# ----------------------------------------------------------------------------

DATASET_PATH = "dataset.csv"
VECTORIZER_PATH = "vectorizer.joblib"
MODEL_PATH = "model.joblib"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")

    # Drop duplicate text entries (same as notebook)
    df = df.drop_duplicates(subset="Text", keep="first").reset_index(drop=True)

    # Drop rows containing URLs or emojis; numbers/punctuation are kept since
    # they can help distinguish languages (same as notebook)
    has_url = df["Text"].apply(contains_url)
    has_emoji = df["Text"].apply(contains_emoji)
    df = df[~(has_url | has_emoji)].reset_index(drop=True)

    # Preprocessing: lowercase + strip whitespace
    df["clean_text"] = df["Text"].apply(clean_text)

    X = df["clean_text"]
    y = df["language"]

    print("Fitting TF-IDF vectorizer (character n-grams 1-3)...")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    X_tfidf = vectorizer.fit_transform(X)

    print("Training LinearSVC...")
    # Trained on the full cleaned dataset (rather than the notebook's 70/30
    # split used for benchmarking) to maximize prediction quality for the
    # deployed app.
    model = LinearSVC()
    model.fit(X_tfidf, y)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved {VECTORIZER_PATH} and {MODEL_PATH}")


if __name__ == "__main__":
    main()