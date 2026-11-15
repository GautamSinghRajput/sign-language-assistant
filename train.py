"""Step 2: train a model and print accuracy.

    python train.py --mode static     # MLP on single photos (still letters)
    python train.py --mode dynamic    # LSTM on video sequences (moving letters)

Outputs: models/<mode>_model.keras, models/<mode>_labels.json, models/<mode>_confusion.png
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from common import normalize_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["static", "dynamic"], required=True)
mode = ap.parse_args().mode

data = np.load(f"data/{mode}.npz")
X, y_text = data["X"], data["y"]
if mode == "dynamic":
    X = np.array([normalize_sequence(s) for s in X])      # same step used in live mode
labels = sorted(set(y_text.tolist()))
y = np.array([labels.index(t) for t in y_text])
print("Samples:", X.shape, " Classes:", len(labels))

# 70% train, 15% validation, 15% test (test set is only used for the final report)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)


def augment(batch):
    out = batch + np.random.normal(0, 0.01, batch.shape).astype(np.float32)
    if mode == "static":                          # mirror = the other hand
        m = out.reshape(len(out), 21, 3).copy()
        m[:, :, 0] *= -1
        return np.concatenate([out, m.reshape(len(out), -1)]), None
    return out, None


if mode == "static":
    aug, _ = augment(X_tr)
    X_tr, y_tr = np.concatenate([X_tr, aug]), np.concatenate([y_tr, y_tr, y_tr])
    model = keras.Sequential([
        layers.Input(shape=X.shape[1:]),
        layers.Dense(128, activation="relu"), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"), layers.Dropout(0.2),
        layers.Dense(len(labels), activation="softmax"),
    ])
else:
    aug, _ = augment(X_tr)
    X_tr, y_tr = np.concatenate([X_tr, aug]), np.concatenate([y_tr, y_tr])
    model = keras.Sequential([
        layers.Input(shape=X.shape[1:]),
        layers.LSTM(64, return_sequences=True), layers.Dropout(0.3),
        layers.LSTM(128), layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(labels), activation="softmax"),
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=32,
          callbacks=[keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)])

pred = model.predict(X_te).argmax(axis=1)
print(classification_report(y_te, pred, labels=range(len(labels)), target_names=labels, zero_division=0))

os.makedirs("models", exist_ok=True)
model.save(f"models/{mode}_model.keras")
json.dump(labels, open(f"models/{mode}_labels.json", "w"))
size = max(6, len(labels) * 0.4)
fig, ax = plt.subplots(figsize=(size, size))
ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=range(len(labels))),
                       display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
plt.tight_layout()
plt.savefig(f"models/{mode}_confusion.png", dpi=150)
print(f"Saved models/{mode}_model.keras, labels and confusion matrix")
