"""
Small, self-contained resume screening demo.

This file contains a lightweight example that trains a RandomForest on a tiny in-memory
dataset and exposes a Gradio UI to upload a resume file (.pdf/.docx/.txt) and receive
a predicted category with confidence scores.

Notes:
- This demo uses spaCy if available, otherwise falls back to a simple tokenizer.
- For production use supply a larger labeled dataset and persist the trained model.
"""

import os
import re
import string
from typing import Optional

import pandas as pd
import docx2txt
import PyPDF2
import gradio as gr
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# --- NLP backend setup (try spaCy, otherwise simple fallback) ---
try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None


def ensure_nltk_resources():
    # Download stopwords if missing
    try:
        stopwords.words("english")
    except Exception:
        nltk.download("stopwords")


ensure_nltk_resources()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Tokenize + lemmatize (if spaCy available) and remove stopwords / non-alpha tokens."""
    text = (text or "").lower()
    if nlp:
        # use an explicit name to avoid confusion with other "doc" usages
        spacy_doc = nlp(text)
        tokens = [t.lemma_ for t in spacy_doc if t.is_alpha and t.lemma_ not in stop_words]
        return " ".join(tokens)

    # Fallback: basic regex tokenizer + stopword removal
    tokens = re.findall(r"[a-zA-Z]+", text)
    tokens = [t for t in (tok.lower() for tok in tokens) if t not in stop_words]
    return " ".join(tokens)


def build_and_train():
    # tiny demo dataset — replace with your real dataset for production
    data = {
        "resume_text": [
            "Java, Spring Boot, SQL, developed web applications",
            "Digital marketing, SEO, Google ads, social media",
            "Machine learning, Python, pandas, regression",
            "Sales strategy, client acquisition, B2B deals",
            "Graphic design, Photoshop, Illustrator, branding",
        ],
        "category": ["Software", "Marketing", "Data Science", "Sales", "Design"],
    }
    df = pd.DataFrame(data)
    df["cleaned"] = df["resume_text"].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (very small dataset; metrics not meaningful)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, vectorizer



def make_classifier(model, vectorizer):
    def classify_resume(file) -> str:
        if file is None:
            return "No file provided."

        # gradio provides a file object with a 'name' attribute (path on disk)
        path = getattr(file, "name", None) or (file.get("name") if isinstance(file, dict) else None)

        if not path or not os.path.exists(path):
            return "Uploaded file not found on disk."

        try:
            text = ""
            lower = path.lower()
            if lower.endswith(".pdf"):
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text
            elif lower.endswith(".docx"):
                # extract text from .docx files using docx2txt
                text = docx2txt.process(path)
            elif lower.endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                return "Unsupported file format. Please upload .pdf, .docx or .txt"

            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            probas = model.predict_proba(vec)[0]
            conf_score = max(probas)
            all_probs = dict(zip(model.classes_, probas))
            return (
                f"Predicted Category: {prediction}\nConfidence: {conf_score:.2f}\nAll Class Probabilities: {all_probs}"
            )
        except Exception as e:
            return f"Error while classifying: {e}"

    return classify_resume


def main():
    model, vectorizer = build_and_train()
    classifier = make_classifier(model, vectorizer)

    with gr.Blocks() as demo:
        gr.Markdown("## 📄 Advanced Resume Screening Classifier")
        gr.Markdown("Upload a resume file (.pdf, .docx, .txt) and get predicted job category and confidence scores.")
        file_input = gr.File(label="Upload Resume", file_types=[".pdf", ".docx", ".txt"])
        output = gr.Textbox(label="Prediction")
        file_input.change(classifier, inputs=file_input, outputs=output)

    demo.launch(share=False)


if __name__ == "__main__":
    main()