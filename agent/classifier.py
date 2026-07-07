"""
classifier.py
-------------
TF-IDF + Logistic Regression domain classifier.
Pre-baked at build time (no training at runtime).
~10 MB RAM, <1ms inference. Zero PyTorch dependency.
"""
import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DOMAINS = ["ner", "sentiment", "math", "summarization", "debugging", "codegen", "logic", "factual"]

# Domains where local model is almost always sufficient
LOCAL_SAFE_DOMAINS = {"sentiment", "factual", "math"}

# Serialized classifier path — baked at build time
CLASSIFIER_PKL = os.path.join(os.path.dirname(__file__), "classifier.pkl")
TRAINING_DATA  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "router_training.json")


class DomainClassifier:
    def __init__(self):
        self.pipeline: Pipeline | None = None
        self._load_or_train()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, text: str) -> tuple[str, float]:
        """Return (domain, confidence) for a given query text."""
        if self.pipeline is None:
            return "factual", 0.5  # safe fallback

        proba = self.pipeline.predict_proba([text])[0]
        idx   = int(np.argmax(proba))
        label = self.pipeline.classes_[idx]
        return label, float(proba[idx])

    def is_local_safe(self, domain: str) -> bool:
        return domain in LOCAL_SAFE_DOMAINS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_or_train(self):
        """Load pre-baked pickle if it exists, else train and save."""
        if os.path.exists(CLASSIFIER_PKL):
            with open(CLASSIFIER_PKL, "rb") as f:
                self.pipeline = pickle.load(f)
            return

        # Train from scratch
        self.pipeline = self._train(TRAINING_DATA)
        # Persist so the next call is instant
        with open(CLASSIFIER_PKL, "wb") as f:
            pickle.dump(self.pipeline, f)

    def _train(self, data_path: str) -> Pipeline:
        if os.path.exists(data_path):
            with open(data_path) as f:
                data = json.load(f)
        else:
            # Minimal inline fallback
            data = [
                {"text": "extract persons locations organizations dates from text", "domain": "ner"},
                {"text": "sentiment positive negative neutral classify", "domain": "sentiment"},
                {"text": "solve equation calculate sum area percent ratio", "domain": "math"},
                {"text": "summarize condense brief overview passage", "domain": "summarization"},
                {"text": "find bug debug fix error TypeError SyntaxError", "domain": "debugging"},
                {"text": "write function generate code python script class", "domain": "codegen"},
                {"text": "all some if then logical reasoning conclusion", "domain": "logic"},
                {"text": "who what when where capital year founded planet", "domain": "factual"},
            ]

        texts  = [d["text"] for d in data]
        labels = [d["domain"] for d in data]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=2000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=10.0,
                solver="lbfgs",
            )),
        ])
        pipeline.fit(texts, labels)
        return pipeline


# ------------------------------------------------------------------
# CLI: pre-bake the classifier pickle (run once before docker build)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Training and saving classifier pickle …")
    # Delete stale pickle so we always retrain fresh when run directly
    if os.path.exists(CLASSIFIER_PKL):
        os.remove(CLASSIFIER_PKL)
        print(f"Deleted old pickle: {CLASSIFIER_PKL}")
    clf = DomainClassifier()
    # Quick smoke test
    for sample in [
        "What is the capital of France?",
        "Fix this bug: def add(a, b): return a - b",
        "Summarize the passage in 3 sentences.",
        "Solve: 3x + 7 = 22",
        "Extract all named entities from the text below.",
        "Is this review positive or negative?",
        "Write a function to reverse a string in Python.",
        "If A > B and B > C, what can we say about A and C?",
    ]:
        domain, conf = clf.classify(sample)
        print(f"  [{conf:.2f}] {domain:15s} ← {sample[:55]}")
    print(f"\nClassifier saved to: {CLASSIFIER_PKL}")
