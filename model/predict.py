import json
import os
import re

import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "email_classifier.keras")
LABEL_PATH = os.path.join(BASE_DIR, "model", "labels.json")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
BINARY_MODEL_PATH = os.path.join(BASE_DIR, "model", "tf_spam_model.keras")


def load_labels():
    with open(LABEL_PATH, "r", encoding="utf-8") as label_file:
        return json.load(label_file)


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()
    MODEL_KIND = "multiclass"
elif os.path.exists(BINARY_MODEL_PATH):
    model = tf.keras.models.load_model(BINARY_MODEL_PATH)
    labels = ["ham", "spam"]
    MODEL_KIND = "binary"
else:
    raise FileNotFoundError(
        "No trained model found. Run `python model\\train.py` first."
    )

metrics = load_metrics()


def clean_text(value):
    text = str(value).lower()
    text = text.replace("\ufffd", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9@!$%:_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_email(email_text):
    cleaned_email = clean_text(email_text)
    sample = tf.constant([cleaned_email], dtype=tf.string)
    raw_prediction = model.predict(sample, verbose=0)[0]

    if MODEL_KIND == "binary":
        spam_score = float(raw_prediction[0])
        phishing_score = phishing_signal_score(email_text)

        if spam_score >= 0.7 or phishing_score >= 3:
            label = "phishing"
            result = "Phishing Email"
            category = "Spam"
        elif spam_score >= 0.5:
            label = "spam"
            result = "Spam Email"
            category = "Spam"
        else:
            label = "ham"
            result = "Ham / Safe Email"
            category = "Ham"

        return {
            "result": result,
            "label": label,
            "category": category,
            "confidence": round(max(spam_score, 1 - spam_score), 4),
            "spam_probability": round(spam_score, 4),
            "ham_probability": round(1 - spam_score, 4),
            "phishing_signal_score": phishing_score,
            "all_probabilities": {
                "ham": round(1 - spam_score, 4),
                "spam_or_phishing": round(spam_score, 4),
            },
            "model_accuracy": metrics.get("accuracy"),
            "model_precision": metrics.get("macro_precision", metrics.get("precision")),
            "model_recall": metrics.get("macro_recall", metrics.get("recall")),
            "model_f1_score": metrics.get("macro_f1_score", metrics.get("f1_score")),
            "model_kind": MODEL_KIND,
        }

    probabilities = raw_prediction
    predicted_index = int(np.argmax(probabilities))
    label = labels[predicted_index]
    confidence = float(probabilities[predicted_index])
    signal_score = phishing_signal_score(email_text)

    all_probabilities = {
        labels[index]: round(float(probability), 4)
        for index, probability in enumerate(probabilities)
    }

    if signal_score >= 3 and label != "ham":
        label = "phishing"
        confidence = max(confidence, all_probabilities.get("phishing", 0.0))

    spam_probability = all_probabilities.get("spam", 0.0) + all_probabilities.get("phishing", 0.0)
    ham_probability = all_probabilities.get("ham", 0.0)

    if label == "phishing":
        result = "Phishing Email"
        category = "Spam"
    elif label == "spam":
        result = "Spam Email"
        category = "Spam"
    else:
        result = "Ham / Safe Email"
        category = "Ham"

    return {
        "result": result,
        "label": label,
        "category": category,
        "confidence": round(confidence, 4),
        "spam_probability": round(float(spam_probability), 4),
        "ham_probability": round(float(ham_probability), 4),
        "phishing_signal_score": signal_score,
        "all_probabilities": all_probabilities,
        "model_accuracy": metrics.get("accuracy"),
        "model_precision": metrics.get("macro_precision", metrics.get("precision")),
        "model_recall": metrics.get("macro_recall", metrics.get("recall")),
        "model_f1_score": metrics.get("macro_f1_score", metrics.get("f1_score")),
        "model_kind": MODEL_KIND,
    }


def phishing_signal_score(text):
    normalized = " ".join(str(text).lower().split())
    score = 0

    finance_terms = [
        "binance",
        "bank",
        "wallet",
        "crypto",
        "withdrawal",
        "account",
        "payment",
        "paypal",
    ]
    account_action_terms = [
        "limited",
        "locked",
        "suspended",
        "restore",
        "verify",
        "update",
        "confirm",
        "login",
        "password",
        "employment information",
        "full access",
    ]
    urgency_terms = [
        "immediately",
        "urgent",
        "required",
        "within 24 hours",
        "deadline",
        "will be disabled",
    ]

    if any(term in normalized for term in finance_terms):
        score += 1
    if sum(term in normalized for term in account_action_terms) >= 2:
        score += 2
    if any(term in normalized for term in urgency_terms):
        score += 1
    if "http://" in normalized or "https://" in normalized or "www." in normalized:
        score += 1

    return score


if __name__ == "__main__":
    test_email = """
    Your Binance account is limited. Update your employment information
    immediately to restore full access.
    """

    print(json.dumps(predict_email(test_email), indent=2))
