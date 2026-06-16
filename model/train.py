import json
import os
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_CANDIDATES = [
    os.path.join(BASE_DIR, "spam_email"),
]
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "email_classifier.keras")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

MAX_TOKENS = 30000
SEQUENCE_LENGTH = 300
BATCH_SIZE = 64
EPOCHS = 15
RANDOM_STATE = 42
MIN_TEXT_LENGTH = 10


DATASET_SPECS = [
    {
        "filename": "emails.csv",
        "label_column": "spam",
        "positive_label": "spam",
        "required": True,
    },
    {
        "filename": "phishing_legit_dataset_KD_10000.csv",
        "label_column": "label",
        "positive_label": "phishing",
        "required": True,
    },
]


def find_dataset_path(filename, required=False):
    checked_paths = []

    for data_dir in DATA_DIR_CANDIDATES:
        path = os.path.join(data_dir, filename)
        checked_paths.append(path)
        if os.path.exists(path):
            return path

    if required:
        checked = "\n  ".join(checked_paths)
        raise FileNotFoundError(f"Required dataset not found: {filename}\nChecked:\n  {checked}")

    print(f"Skipping optional dataset: {filename}")
    return None


def clean_text(value):
    text = str(value).lower()
    text = text.replace("\ufffd", " ")
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"[^a-z0-9@!$%:_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_csv(path):
    for encoding in ("utf-8", "ISO-8859-1", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path, encoding="latin1", low_memory=False)


def get_text_column(df, filename):
    if "text" in df.columns:
        return df["text"].fillna("")
    if "text_combined" in df.columns:
        return df["text_combined"].fillna("")
    if "subject" in df.columns and "body" in df.columns:
        return df["subject"].fillna("") + " " + df["body"].fillna("")
    if "body" in df.columns:
        return df["body"].fillna("")

    raise ValueError(f"No usable text column found in {filename}")


def load_dataset(spec):
    filename = spec["filename"]
    path = find_dataset_path(filename, required=spec.get("required", False))

    if path is None:
        return None

    df = read_csv(path)
    text = get_text_column(df, filename)

    if "force_label" in spec:
        label = spec["force_label"]
        labels = np.full(len(df), label, dtype=object)
    else:
        label_column = spec.get("label_column", "label")

        if label_column not in df.columns:
            raise ValueError(f"No {label_column} column found in {filename}")

        raw_labels = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int)
        labels = np.where(raw_labels == 1, spec["positive_label"], "ham")

    data = pd.DataFrame({
        "text": text.map(clean_text),
        "label": labels,
    })
    data = data.dropna(subset=["text", "label"])
    data = data[data["text"].str.len() > MIN_TEXT_LENGTH]

    print(f"{filename}: {len(data)} rows from {os.path.dirname(path)}")
    return data


# def balance_dataset(data):
#     label_counts = data["label"].value_counts()
#     min_count = int(label_counts.min())
#     balanced_parts = []

#     for label in sorted(label_counts.index):
#         rows = data[data["label"] == label]
#         balanced_parts.append(rows.sample(n=min_count, random_state=RANDOM_STATE))

#     balanced = pd.concat(balanced_parts, ignore_index=True)
#     balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
#     return balanced


def build_dataset():

    datasets = []

    for spec in DATASET_SPECS:

        dataset = load_dataset(spec)

        if dataset is not None:
            datasets.append(dataset)

    if not datasets:
        raise RuntimeError("No datasets loaded.")

    data = pd.concat(
        datasets,
        ignore_index=True
    )

    # Remove duplicate emails
    data = data.drop_duplicates(subset=["text"])

    # Shuffle
    data = data.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("\n==============================")
    print("Combined Dataset")
    print("==============================")

    print(data["label"].value_counts())

    required_labels = {
        "ham",
        "spam",
        "phishing"
    }

    existing_labels = set(data["label"].unique())

    missing = required_labels - existing_labels

    if missing:
        raise RuntimeError(
            f"Missing labels: {missing}"
        )

    return data


# =========================================================
# BUILD MODEL
# =========================================================

def build_model(num_classes):

    text_input = tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="email_text"
    )

    # =====================================================
    # TEXT VECTORIZATION
    # =====================================================

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize=None,
        name="text_vectorizer",
    )

    x = vectorizer(text_input)

    # =====================================================
    # EMBEDDING
    # =====================================================

    x = tf.keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=128,
        name="embedding"
    )(x)

    x = tf.keras.layers.SpatialDropout1D(0.3)(x)

    # =====================================================
    # CNN
    # =====================================================

    x = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    )(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    # =====================================================
    # DENSE
    # =====================================================

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    # =====================================================
    # OUTPUT
    # =====================================================

    output = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="class_probability"
    )(x)

    model = tf.keras.Model(
        text_input,
        output
    )

    return model, vectorizer


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nStarting training...\n")

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # =====================================================
    # LOAD DATA
    # =====================================================

    data = build_dataset()

    # =====================================================
    # ENCODE LABELS
    # =====================================================

    encoder = LabelEncoder()

    y = encoder.fit_transform(data["label"])

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].to_numpy(),
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(
        enumerate(class_weights)
    )

    print("\nClass weights:")
    print(class_weights)

    # =====================================================
    # BUILD MODEL
    # =====================================================

    model, vectorizer = build_model(
        num_classes=len(encoder.classes_)
    )

    vectorizer.adapt(
        tf.constant(
            X_train,
            dtype=tf.string
        )
    )

    # =====================================================
    # COMPILE
    # =====================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =====================================================
    # CALLBACKS
    # =====================================================

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=0.00001
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        ),
    ]

    # =====================================================
    # MODEL SUMMARY
    # =====================================================

    model.summary()

    # =====================================================
    # TRAIN
    # =====================================================

    history = model.fit(

        tf.constant(
            X_train,
            dtype=tf.string
        ),

        y_train,

        validation_split=0.15,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=callbacks,

        class_weight=class_weights,

        verbose=1,
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    probabilities = model.predict(

        tf.constant(
            X_test,
            dtype=tf.string
        ),

        batch_size=BATCH_SIZE
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)

    # =====================================================
    # SAVE LABELS
    # =====================================================

    with open(
        LABEL_PATH,
        "w",
        encoding="utf-8"
    ) as label_file:

        json.dump(
            encoder.classes_.tolist(),
            label_file,
            indent=2
        )

    # =====================================================
    # SAVE METRICS
    # =====================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as metrics_file:

        json.dump({

            "accuracy": round(float(accuracy), 4),

            "macro_precision": round(
                float(report["macro avg"]["precision"]),
                4
            ),

            "macro_recall": round(
                float(report["macro avg"]["recall"]),
                4
            ),

            "macro_f1_score": round(
                float(report["macro avg"]["f1-score"]),
                4
            ),

            "test_rows": int(len(y_test)),

            "labels": encoder.classes_.tolist(),

            "max_tokens": MAX_TOKENS,

            "sequence_length": SEQUENCE_LENGTH,

            "epochs": EPOCHS,

        }, metrics_file, indent=2)

    # =====================================================
    # FINISH
    # =====================================================

    print("\n==============================")
    print("TRAINING COMPLETE")
    print("==============================")

    print(f"\nModel saved to:\n{MODEL_PATH}")

    print(f"\nLabels saved to:\n{LABEL_PATH}")

    print(f"\nMetrics saved to:\n{METRICS_PATH}")

    print(f"\nFinal Accuracy: {accuracy:.4f}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
