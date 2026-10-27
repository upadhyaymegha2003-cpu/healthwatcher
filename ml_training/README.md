# HealthWatcher ML training utility (development only)

This folder is a **development tool**. The product is the Android app; it runs inference on the phone with
TensorFlow Lite and needs no Python, server or internet connection. Use this folder to turn *real labelled
recordings* into the two files the app loads (`<target>.tflite` + `<target>.properties`).

**The repository contains no dataset and no trained model**, so the app currently ships with none and uses the
original signal-processing algorithms. Nothing in here fabricates data or performance.

## Data you need

One folder, e.g. `my_dataset/`:

```
my_dataset/recordings.csv
my_dataset/signals/<recording>.csv        # columns: t_sec,red,green,blue   (per-frame mean 0..255 intensities)
```

`recordings.csv` columns (leave a label blank if you don't have it):

| column | meaning |
|---|---|
| `recording_id`, `subject_id`, `file` | unique id, **subject identifier (used for leakage-free splits)**, path of the signal CSV |
| `hr_bpm` | reference heart rate (ECG / chest strap), averaged over the same 30 s |
| `rr_bpm` | reference respiratory rate (capnography / respiration belt) |
| `spo2` | reference SpO2 from a clinically validated pulse oximeter (include low-saturation cases, otherwise the model cannot learn the range) |
| `sbp`, `dbp` | cuff / arterial-line reference, time-aligned |
| `age`, `height_cm`, `weight_kg`, `gender` | only for blood pressure (1 = male, 2 = female, as in the app). Collect with consent. |

Signal CSVs must be produced the way the app records them: same camera pipeline, flash on, finger covering lens
(the app's `PpgSignal.toCsv()` writes this format). Mixing phone models is fine and recommended, but keep the
phone model in your notes: camera pipelines differ and cuffless estimates are often device specific.

Minimum to run: 10 subjects (hard stop). To say anything meaningful: hundreds of subjects, several recordings
each, covering rest and varied conditions. Cuffless BP / SpO2 need clinical-grade protocols (ISO 81060-2 style BP
validation uses 85+ subjects; SpO2 validation needs a desaturation protocol) and usually per-user calibration.

## Steps

```bash
pip install -r requirements.txt

# 1. (optional but recommended) run the ORIGINAL app algorithms + quality gate on your data for comparison
./run_baseline.sh my_dataset baseline.csv            # needs a JDK

# 2. train + convert + evaluate + write into the app (app/src/main/assets/models)
python train.py --target heart_rate       --data my_dataset --baseline baseline.csv --require-quality-ok \
                --description "Dataset X, ECG reference, Pixel 7, N subjects"
python train.py --target respiratory_rate --data my_dataset --baseline baseline.csv --require-quality-ok
python train.py --target spo2             --data my_dataset --baseline baseline.csv --require-quality-ok
python train.py --target blood_pressure   --data my_dataset --baseline baseline.csv \
                --i-understand-bp-needs-calibration
```

What `train.py` does: feature extraction identical to the app (`ppg_features.py` mirrors `PPGFeatureExtractor.java`),
**subject-wise** train/val/test split (asserted), normalisation computed from training subjects only and written to
the manifest, a small MLP (26 features -> 32 -> 16 -> outputs), float32 TFLite export, a Keras-vs-TFLite parity
check (aborts if > 1e-4), MAE/RMSE on the held-out subjects next to the original algorithm and a
"predict the training mean" reference, `reports/evaluation_<target>.json`, and `parity_<target>.csv` in
`app/src/test/resources` for the on-device parity test.

### Validation status

Every model is written with `validation.status=UNVALIDATED`. In that state the app runs it but shows its number only
as an *experimental comparison* next to the signal-processing value. Edit the manifest to `VALIDATED` **only after**
an independent validation (separate subjects/devices, reference-standard comparison, agreed accuracy criteria) has
been completed and documented. Only then does the app display the model output as the main result.

### Keeping Java and Python identical

`PPGFeatureExtractor.java` / `PpgDsp.java` and `ppg_features.py` implement the same algorithm (`ppgfeat-1`). If you
change one: change the other, bump `FEATURE_VERSION` in both, run `python make_golden_vectors.py`, then
`./gradlew testDebugUnitTest` (the `PPGFeatureExtractorTest.javaMatchesPythonGoldenVectors` test must pass).
Models carry the feature version; the app refuses mismatching models.

### Smoke test (tooling check only)

`python train.py --smoke-test --target heart_rate --out-dir /tmp/smoke` uses synthetic signals to verify that
training/conversion/parity work on your machine. Its output is stamped `SMOKE_TEST`; the app refuses to load it, and
the script will not write it into the app. Its metrics mean nothing.

## Install into the app

`train.py` writes straight to `app/src/main/assets/models/`. By hand: copy `<target>.tflite` and
`<target>.properties` there together (both are required), rebuild, open *About* in the app to see the load status,
then run the instrumented parity test (`./gradlew connectedDebugAndroidTest`).
