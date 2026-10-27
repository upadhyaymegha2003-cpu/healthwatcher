#!/usr/bin/env python3
"""
Development utility: trains a small on-device model for ONE vital sign from REAL labelled recordings and
exports the files the Android app loads from app/src/main/assets/models/:

    <target>.tflite       float32 model, input [1, N] (normalised features), output [1, K]
    <target>.properties   manifest: feature order, EXACT normalisation parameters, output scaling,
                          plausibility limits, checksum, validation status, metrics

This script never fabricates data. Without a real dataset it only runs with --smoke-test, which uses
synthetic recordings to exercise the code path; the resulting model is stamped validation.status=SMOKE_TEST
(the Android app refuses to load such models) and the script refuses to write it into the app.

Usage:
  python train.py --data DATASET_DIR --target heart_rate [--baseline baseline.csv]
  python train.py --smoke-test --target heart_rate --out-dir /tmp/smoke
See README.md for the dataset layout.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

import ppg_features as pf

TARGETS = {
    # target: (label columns, output names, plausibility min, max, uses_demographics, baseline columns)
    "heart_rate": (["hr_bpm"], ["heart_rate_bpm"], [30.0], [220.0], False, ["hr_bpm"]),
    "respiratory_rate": (["rr_bpm"], ["respiratory_rate_brpm"], [4.0], [40.0], False, ["rr_bpm"]),
    "spo2": (["spo2"], ["spo2_percent"], [70.0], [100.0], False, ["spo2"]),
    "blood_pressure": (["sbp", "dbp"], ["systolic_mmhg", "diastolic_mmhg"], [70.0, 40.0], [250.0, 150.0], True,
                       ["sbp", "dbp"]),
}
MIN_SUBJECTS = 10
APP_ASSETS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "main", "assets"))


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


# ------------------------------------------------------------------------------------------ data
def read_recordings(data_dir):
    path = os.path.join(data_dir, "recordings.csv")
    if not os.path.exists(path):
        die("%s not found. See README.md for the required dataset layout." % path)
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def read_signal(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def build_matrix(rows, data_dir, target, baseline, require_quality_ok):
    label_cols, _, _, _, demo, base_cols = TARGETS[target]
    X, Y, G, B, rids = [], [], [], [], []
    skipped = {"no_label": 0, "bad_signal": 0, "bad_profile": 0, "quality": 0}
    for r in rows:
        y = [num(r.get(c)) for c in label_cols]
        if any(np.isnan(v) for v in y):
            skipped["no_label"] += 1
            continue
        if require_quality_ok:
            q = baseline.get(r["recording_id"], {}).get("quality_status")
            if q != "OK":
                skipped["quality"] += 1
                continue
        try:
            t, red, green, blue = read_signal(os.path.join(data_dir, r["file"]))
            f = pf.extract_features(t, red, green, blue)
        except (OSError, ValueError, pf.FeatureExtractionError, IndexError):
            skipped["bad_signal"] += 1
            continue
        if demo:
            d = pf.demographic_vector(num(r.get("age")), num(r.get("height_cm")), num(r.get("weight_kg")), num(r.get("gender")))
            if np.any(np.isnan(d)):
                skipped["bad_profile"] += 1
                continue
            f = np.concatenate([f, d])
        X.append(f)
        Y.append(y)
        G.append(r["subject_id"])
        rids.append(r["recording_id"])
        b = baseline.get(r["recording_id"], {})
        B.append([num(b.get(c)) for c in base_cols])
    if not X:
        die("no usable recordings (skipped: %s)" % skipped)
    names = list(pf.FEATURE_NAMES) + (pf.DEMOGRAPHIC_NAMES if demo else [])
    return np.array(X), np.array(Y), np.array(G), np.array(B), rids, names, skipped


def read_baseline(path):
    out = {}
    if path:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[r["recording_id"]] = r
    return out


def subject_split(groups, seed, frac=(0.70, 0.15, 0.15)):
    """Leakage-resistant split: every subject lands in exactly one of train / val / test."""
    subjects = sorted(set(groups))
    if len(subjects) < MIN_SUBJECTS:
        die("only %d subjects; at least %d are required for a subject-wise train/val/test split "
            "(far more are needed for a meaningful evaluation)." % (len(subjects), MIN_SUBJECTS))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_val = max(1, int(round(n * frac[1])))
    n_test = max(1, int(round(n * frac[2])))
    test = set(subjects[:n_test])
    val = set(subjects[n_test:n_test + n_val])
    idx = np.arange(len(groups))
    g = np.asarray(groups)
    return (idx[[s not in test and s not in val for s in g]],
            idx[[s in val for s in g]],
            idx[[s in test for s in g]])


def metrics(y, p):
    e = p - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "n": int(len(y))}


# ------------------------------------------------------------------------------------------ smoke data
def make_smoke_dataset(target, workdir, seed=0):
    """Synthetic recordings ONLY to test the tooling. Metrics from this are meaningless."""
    from make_golden_vectors import synth
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(workdir, "signals"), exist_ok=True)
    rows = []
    for s in range(24):
        age, h, w, gender = int(rng.integers(20, 70)), int(rng.integers(150, 195)), int(rng.integers(50, 100)), int(rng.integers(1, 3))
        for k in range(3):
            hr = float(rng.uniform(55, 110))
            rr = float(rng.uniform(10, 22))
            t, r, g, b = synth(int(rng.integers(0, 10 ** 6)), hr / 60.0, rr / 60.0, 30, 30.0, 0.05, 0.1, 0.2)
            rid = "s%02d_%d" % (s, k)
            fn = os.path.join("signals", rid + ".csv")
            np.savetxt(os.path.join(workdir, fn), np.column_stack([t, r, g, b]), delimiter=",",
                       header="t_sec,red,green,blue", comments="")
            rows.append({"recording_id": rid, "subject_id": "S%02d" % s, "file": fn, "hr_bpm": hr, "rr_bpm": rr,
                         "spo2": float(rng.uniform(94, 99)), "sbp": float(rng.uniform(100, 140)),
                         "dbp": float(rng.uniform(60, 90)), "age": age, "height_cm": h, "weight_kg": w, "gender": gender})
    with open(os.path.join(workdir, "recordings.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return workdir


# ------------------------------------------------------------------------------------------ model
def train_keras(Xtr, Ytr, Xva, Yva, seed, epochs):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    k = Ytr.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(k),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber())
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)]
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=epochs, batch_size=32, verbose=0, callbacks=cb)
    return model


def convert_tflite(model):
    import tensorflow as tf
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def run_tflite(tflite_bytes, X):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for row in X:
        it.set_tensor(i["index"], row.reshape(1, -1).astype(np.float32))
        it.invoke()
        out.append(it.get_tensor(o["index"])[0].copy())
    return np.array(out)


def fmt(arr):
    return ",".join(repr(float(v)) for v in arr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--data", help="dataset directory (recordings.csv + signal files)")
    ap.add_argument("--baseline", help="CSV produced by run_baseline.sh (original app algorithm) for comparison")
    ap.add_argument("--require-quality-ok", action="store_true",
                    help="train only on recordings the app's quality gate marks OK (needs --baseline)")
    ap.add_argument("--out-dir", default=os.path.join(APP_ASSETS, "models"))
    ap.add_argument("--reports-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true", help="synthetic pipeline test; output is NOT shippable")
    ap.add_argument("--i-understand-bp-needs-calibration", action="store_true")
    ap.add_argument("--description", default="", help="free text: dataset, reference devices, protocol")
    a = ap.parse_args()

    label_cols, out_names, omin, omax, demo, base_cols = TARGETS[a.target]
    if a.target == "blood_pressure" and not (a.i_understand_bp_needs_calibration or a.smoke_test):
        die("Cuffless blood pressure needs cuff reference labels, subject-independent validation (ISO 81060-2 style, "
            "~85+ subjects) and usually per-user calibration. Re-run with --i-understand-bp-needs-calibration.")

    if a.smoke_test:
        import tempfile
        work = tempfile.mkdtemp(prefix="hw_smoke_")
        data_dir = make_smoke_dataset(a.target, work, a.seed)
        status = "SMOKE_TEST"
        if os.path.commonpath([os.path.abspath(a.out_dir), APP_ASSETS]) == APP_ASSETS:
            a.out_dir = os.path.join(work, "models")
        print("*** SYNTHETIC SMOKE TEST: metrics are meaningless; the model will NOT load in the app. ***")
    else:
        if not a.data:
            die("--data is required (real labelled recordings). Use --smoke-test only to test the tooling.")
        data_dir, status = a.data, "UNVALIDATED"
    if a.require_quality_ok and not a.baseline:
        die("--require-quality-ok needs --baseline")

    rows = read_recordings(data_dir)
    baseline = read_baseline(a.baseline)
    X, Y, G, B, rids, names, skipped = build_matrix(rows, data_dir, a.target, baseline, a.require_quality_ok)
    print("recordings used: %d from %d subjects; skipped: %s" % (len(X), len(set(G)), skipped))
    tr, va, te = subject_split(G, a.seed)
    print("split (subjects): train %d / val %d / test %d" % (len(set(G[tr])), len(set(G[va])), len(set(G[te]))))
    assert not (set(G[tr]) & set(G[te])) and not (set(G[tr]) & set(G[va])) and not (set(G[va]) & set(G[te]))

    # normalisation parameters come from the TRAINING subjects only and are shipped in the manifest
    mean = X[tr].mean(axis=0)
    std = X[tr].std(axis=0)
    std[std < 1e-8] = 1.0
    ymean = Y[tr].mean(axis=0)
    ystd = Y[tr].std(axis=0)
    ystd[ystd < 1e-8] = 1.0
    Xn = (X - mean) / std
    Yn = (Y - ymean) / ystd

    model = train_keras(Xn[tr], Yn[tr], Xn[va], Yn[va], a.seed, a.epochs)
    tfl = convert_tflite(model)

    # development-side parity: converted model vs Keras model
    keras_pred = model.predict(Xn[te].astype(np.float32), verbose=0)
    tfl_pred = run_tflite(tfl, Xn[te])
    diff = float(np.max(np.abs(keras_pred - tfl_pred)))
    print("Keras vs TFLite max |diff| (normalised units): %.2e" % diff)
    if diff > 1e-4:
        die("TFLite conversion does not match the Keras model (%.2e)" % diff)

    pred = tfl_pred * ystd + ymean
    report = {"target": a.target, "status": status, "subjects": {"train": len(set(G[tr])), "val": len(set(G[va])),
              "test": len(set(G[te]))}, "recordings": {"train": len(tr), "val": len(va), "test": len(te)},
              "outputs": {}, "skipped": skipped}
    for j, nm in enumerate(out_names):
        yt = Y[te][:, j]
        entry = {"ml": metrics(yt, pred[:, j]),
                 "predict_train_mean": metrics(yt, np.full_like(yt, ymean[j]))}
        bj = B[te][:, j]
        ok = ~np.isnan(bj)
        if ok.any():
            entry["original_signal_processing"] = metrics(yt[ok], bj[ok])
            entry["original_signal_processing"]["coverage"] = float(ok.mean())
        report["outputs"][nm] = entry
        print(nm, json.dumps(entry))

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.reports_dir, exist_ok=True)
    tflite_path = os.path.join(a.out_dir, a.target + ".tflite")
    with open(tflite_path, "wb") as f:
        f.write(tfl)
    version = datetime.date.today().strftime("%Y%m%d") + "-" + hashlib.sha256(tfl).hexdigest()[:8]
    props = [
        "schema=1", "target=" + a.target, "model.name=" + a.target + "_mlp", "model.version=" + version,
        "model.sha256=" + hashlib.sha256(tfl).hexdigest(), "feature.version=" + pf.FEATURE_VERSION,
        "feature.names=" + ",".join(names), "norm.mean=" + fmt(mean), "norm.std=" + fmt(std),
        "output.names=" + ",".join(out_names), "output.mean=" + fmt(ymean), "output.std=" + fmt(ystd),
        "output.min=" + fmt(omin), "output.max=" + fmt(omax),
        # Stays UNVALIDATED until an independent validation has been completed by a human; only then
        # edit this line to VALIDATED (see README.md). The app only trusts models marked VALIDATED.
        "validation.status=" + status,
        "training.description=" + (a.description or ("synthetic smoke test" if a.smoke_test else "see README")).replace("\n", " "),
        "metrics.subjects_train=%d" % len(set(G[tr])), "metrics.subjects_test=%d" % len(set(G[te])),
    ]
    for nm, e in report["outputs"].items():
        props.append("metrics.test_mae_%s=%.4f" % (nm, e["ml"]["mae"]))
        props.append("metrics.test_rmse_%s=%.4f" % (nm, e["ml"]["rmse"]))
    with open(os.path.join(a.out_dir, a.target + ".properties"), "w") as f:
        f.write("\n".join(props) + "\n")
    with open(os.path.join(a.reports_dir, "evaluation_%s%s.json" % (a.target, "_SMOKE" if a.smoke_test else "")), "w") as f:
        json.dump(report, f, indent=2)

    # parity vectors for the Android instrumented test: RAW features -> expected outputs in physical units
    parity_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "test", "resources")
    if a.smoke_test:
        parity_dir = a.out_dir
    n = min(50, len(te))
    with open(os.path.join(parity_dir, "parity_%s.csv" % a.target), "w") as f:
        f.write(",".join(["in_%d" % i for i in range(X.shape[1])] + ["out_%d" % j for j in range(len(out_names))]) + "\n")
        for i in range(n):
            f.write(",".join(repr(float(v)) for v in list(X[te][i]) + list(pred[i])) + "\n")
    print("wrote", tflite_path, "(status=%s)" % status)
    if a.smoke_test:
        print("Smoke test finished. Nothing was written into the app.")


if __name__ == "__main__":
    main()
