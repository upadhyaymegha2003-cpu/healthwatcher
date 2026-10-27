#!/usr/bin/env python3
"""
Train the HealthWatcher *screening classifier* from healthwatcher_synthetic_vitals_demo.csv and export it
to TensorFlow Lite for the Android app.

Raw inputs (7, same order as ScreeningClassifier.java):
    age_years, is_male, heart_rate_bpm, systolic_bp_mmhg, diastolic_bp_mmhg, respiratory_rate_bpm, spo2_percent
Model input (17) = 7 raw values + 5 "distance outside the common range" + 5 "distance past the extreme cutoffs" features (see engineer()).
The same arithmetic is implemented in ScreeningClassifier.engineer() - keep the two in sync (a parity test checks it).
Output (3 softmax probabilities, order = CLASSES below):
    Within_demo_ranges, Needs_review, High_priority_review

Temperature and signal quality exist in the CSV but are NOT model inputs: the app has no thermometer and
the camera pipeline has no numeric quality score. Normalisation is baked into the model, so the app feeds raw values.

Usage:  pip install tensorflow pandas scikit-learn
        python train_screening.py --csv ../healthwatcher_synthetic_vitals_demo.csv
"""
import argparse, json, os
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

CLASSES = ["Within_demo_ranges", "Needs_review", "High_priority_review"]
FEATURES = ["age_years", "is_male", "heart_rate_bpm", "systolic_bp_mmhg",
            "diastolic_bp_mmhg", "respiratory_rate_bpm", "spo2_percent"]
SEED = 42


# Common adult resting ranges used by the demo labels (HR 60-100, SBP 90-139, DBP <=89, RR 12-20, SpO2 >=95).
def engineer(A):
    """A: [n,7] raw features -> [n,12] model inputs."""
    hr, sb, db, rr, sp = A[:, 2], A[:, 3], A[:, 4], A[:, 5], A[:, 6]
    ex = np.stack([
        np.maximum(0, 60 - hr) + np.maximum(0, hr - 100),
        np.maximum(0, 90 - sb) + np.maximum(0, sb - 139),
        np.maximum(0, db - 89),
        np.maximum(0, 12 - rr) + np.maximum(0, rr - 20),
        np.maximum(0, 95 - sp)], axis=1)
    # "extreme" depth: how far past the high-priority cutoffs (HR <40/>140, SBP <80/>=180, DBP >=120, RR <8/>30, SpO2 <90)
    xt = np.stack([
        np.maximum(0, 40 - hr) + np.maximum(0, hr - 140),
        np.maximum(0, 80 - sb) + np.maximum(0, sb - 179),
        np.maximum(0, db - 119),
        np.maximum(0, 8 - rr) + np.maximum(0, rr - 30),
        np.maximum(0, 90 - sp)], axis=1)
    return np.hstack([A, ex, xt]).astype("float32")


def load(csv):
    d = pd.read_csv(csv)
    d["is_male"] = (d["sex"] == "Male").astype(float)
    X = d[FEATURES].astype("float32").values
    y = d["screening_label"].map({c: i for i, c in enumerate(CLASSES)}).values
    assert not np.isnan(y.astype(float)).any(), "unknown label in CSV"
    return X, y


def build(norm_data):
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(norm_data)
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES) + 10,)),
        norm,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="../app/src/main/assets/models")
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(SEED)

    Xraw, y = load(a.csv)
    X = engineer(Xraw)
    Xtr, Xtmp, ytr, ytmp, _, Rtmp = train_test_split(X, y, Xraw, test_size=0.30, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte, _, Rte = train_test_split(Xtmp, ytmp, Rtmp, test_size=0.50, stratify=ytmp, random_state=SEED)
    print("train/val/test:", len(Xtr), len(Xva), len(Xte))

    # class weights: 'Within' and 'High priority' are rarer than 'Needs review'
    cnt = np.bincount(ytr)
    cw = {i: len(ytr) / (len(CLASSES) * cnt[i]) for i in range(len(CLASSES))}

    model = build(Xtr)
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=32,
              class_weight=cw, callbacks=[cb], verbose=0)

    p = model.predict(Xte, verbose=0)
    pred = p.argmax(1)
    acc = float((pred == yte).mean())
    maj = float((yte == np.bincount(ytr).argmax()).mean())
    f1 = float(f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred, labels=range(len(CLASSES))).tolist()
    print(classification_report(yte, pred, target_names=CLASSES, digits=3))
    print("confusion (rows=true):", cm)
    print(f"test accuracy {acc:.3f}  macro-F1 {f1:.3f}  majority-class baseline {maj:.3f}")

    # ---- TFLite export + parity check
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "health_screening.tflite")
    open(path, "wb").write(tfl)
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i0, o0 = it.get_input_details()[0], it.get_output_details()[0]
    worst = 0.0
    for row in Xte[:200]:
        it.set_tensor(i0["index"], row[None, :].astype("float32"))
        it.invoke()
        t = it.get_tensor(o0["index"])[0]
        k = model.predict(row[None, :], verbose=0)[0]
        worst = max(worst, float(np.abs(t - k).max()))
    print("Keras vs TFLite max abs diff:", worst)
    assert worst < 1e-4, "TFLite parity failed"

    # ---- parity vectors for the Android unit test
    pv = os.path.join("..", "app", "src", "test", "resources", "screening_parity.csv")
    os.makedirs(os.path.dirname(pv), exist_ok=True)
    idx = np.arange(0, len(Xte), max(1, len(Xte) // 25))[:25]
    pd.DataFrame(np.hstack([Rte[idx], model.predict(Xte[idx], verbose=0)]),
                 columns=FEATURES + ["p_" + c for c in CLASSES]).to_csv(pv, index=False, float_format="%.6f")

    props = {
        "model.version": "screening-v1",
        "model.file": "health_screening.tflite",
        "classes": ",".join(CLASSES),
        "features": ",".join(FEATURES),
        "engineered": "range_dev_hr,range_dev_sbp,range_dev_dbp,range_dev_rr,range_dev_spo2,extreme_dev_hr,extreme_dev_sbp,extreme_dev_dbp,extreme_dev_rr,extreme_dev_spo2",
        "training.data": "healthwatcher_synthetic_vitals_demo.csv (SYNTHETIC, rule-labelled)",
        "training.rows": str(len(X)),
        "input.size": str(X.shape[1]),
        "test.accuracy": f"{acc:.4f}",
        "test.macro_f1": f"{f1:.4f}",
        "test.majority_baseline": f"{maj:.4f}",
        "validation.status": "UNVALIDATED_SYNTHETIC",
    }
    with open(os.path.join(a.out, "health_screening.properties"), "w") as f:
        f.write("# Generated by ml_training/train_screening.py - do not edit by hand\n")
        for k, v in props.items():
            f.write(f"{k}={v}\n")
    os.makedirs("reports", exist_ok=True)
    json.dump({"test_accuracy": acc, "macro_f1": f1, "majority_baseline": maj, "confusion_matrix": cm,
               "classes": CLASSES, "features": FEATURES}, open("reports/evaluation_screening.json", "w"), indent=2)
    print("wrote", path, f"({len(tfl)} bytes)")


if __name__ == "__main__":
    main()
