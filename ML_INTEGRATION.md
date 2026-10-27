# HealthWatcher 2.0 - on-device ML integration

## 1. Audit summary (original project)

* Java, AppCompat, `android.hardware.Camera` (API 1), SQLite `UserDB`; AGP 3.6.2 / Gradle 5.6.4 / jcenter.
* **No datasets, no trained models, no ML code** in the repository (only images/launcher assets).
* Vital signs came from rule-based code copy-pasted across five activities: FFT peak (HR), FFT + Butterworth band-pass
  (respiration), a ratio-of-ratios formula (SpO2) and a literature formula from HR + age/height/weight/gender (BP).
* Defects found while auditing (fixed where noted):
  * Frame decoding, FFT and UI updates ran on the main thread (fixed: background decoding/analysis).
  * `onPause` dereferenced a possibly-null camera; no runtime CAMERA permission request (fixed).
  * Finger-off reset the counter but not the sample lists, so stale samples were analysed; the 30 s timer started at
    `onResume`, not at finger placement (fixed: recording starts on stable finger detection and restarts if lost).
  * Channel averages were truncated by integer division, quantising a 1-2 level pulse (fixed: double means).
  * The "average if plausible" checks used `||` and were always true (fixed to the documented intent).
  * `UserDB.onUpgrade` dropped the users table (fixed: non-destructive migration).
  * BP expression `3 / 2 * PP` is integer division, i.e. `PP` (kept, written explicitly and documented).
  * `(height, width)` are passed swapped to the YUV decoder (kept for identical colour behaviour; affects only chroma
    row stride, negligible with a fingertip over the lens).
  * Not changed: passwords are stored in plaintext in SQLite / "remember me" preferences, and `UserDB` builds SQL by
    string concatenation. Recommended follow-up: salted password hashing and parameterised queries.

## 2. Architecture

```
Camera+flash -> MeasurementActivity (frame thread: YUV -> mean R,G,B) -> PpgRecorder
   -> (30 s, finger-aware)  PpgSignal
   -> SignalQualityAnalyzer  --blocking issue--> "measure again" (no value shown)
   -> BaselineEstimator (original FFT / formulas)        \
   -> PPGFeatureExtractor (26 features, ppgfeat-1) -> MLModelManager (TFLite, manifest normalisation) 
   -> VitalSignsPredictor (source selection) -> VitalSignsReport -> ResultActivity + UserDB history
```

Package `com.example.yo7a.healthwatcher.ml` (pure Java, JVM-testable except `AssetModelSource`/`TfliteModelRunner`):
`PpgSignal`, `PpgRecorder`, `PpgDsp`, `PPGFeatureExtractor`, `SignalQualityAnalyzer`, `QualityReport`,
`BaselineEstimator`, `ModelManifest`, `MLModelManager`, `AssetModelSource`, `TfliteModelRunner`,
`VitalSignsPredictor`, `VitalEstimate`, `VitalSignsReport`, `UserProfile`, `VitalTarget`.

## 3. Which output uses what (current state of this repository)

| Vital | Shown value | ML status |
|---|---|---|
| Heart rate | FFT signal processing (original) | Interface ready, **no model installed** |
| Respiratory rate | FFT + band-pass (original) | Interface ready, **no model installed** |
| SpO2 | Original ratio-of-ratios formula - **experimental, uncalibrated, not validated** | Interface ready, **no model installed** |
| Blood pressure | Original HR/demographic formula - **experimental, uncalibrated, not validated** | Interface ready, **no model installed** |

A model's value becomes the headline number only if `<target>.properties` says `validation.status=VALIDATED`.
An installed but `UNVALIDATED` model is run and displayed only as "Experimental ML estimate ... NOT validated".
Models marked `SMOKE_TEST` are refused. There is no numeric confidence score anywhere: quality is a rule-based
gate (thresholds in `SignalQualityAnalyzer.Config`, calibrated on synthetic recordings only - retune on real data).

Quality checks: recording length (>= 25 s), sampling rate and frame gaps, invalid/NaN/out-of-range samples, finger
coverage, saturation, absent pulsation, motion bursts, cardiac-band spectral concentration, beat-rhythm regularity.

## 3b. System check screen (Home -> "System check")

`CapabilityActivity` runs real checks on the phone and shows PASS / ATTENTION / FAIL / INFO per row: rear camera,
flash, camera permission, PPG pipeline self-test (quality gate + features + estimators on a built-in 72 bpm test
waveform, never shown as patient data), TensorFlow Lite runtime self-test (loads `assets/selftest/runtime_selftest.tflite`,
a tiny NON-health linear function, and checks 3 outputs), each of the four ML models (not installed / unusable /
unvalidated / validated), overall validation status, the health-screening classifier (loads, then runs two built-in sample vectors),
and a local-database read/write probe (uses a separate probe database, never your data). The *About* screen also lists
model status.

## 4. Build and test

Requirements: Android Studio (current) with JDK 17+, Android SDK 34. Open the folder, let Gradle sync
(AGP 8.5.2, Gradle 8.7, compile/target SDK 34, minSdk 21; TensorFlow Lite 2.14 needs API 21+, so Android 4.4 is
no longer supported - the only compatibility reduction).

```
./gradlew assembleDebug
./gradlew testDebugUnitTest                 # 47 JVM tests (extractor parity with Python, quality gate, baseline,
                                            #   model loading/normalisation, predictor policy, recorder)
./gradlew connectedDebugAndroidTest         # DB history + v1->v2 migration; on-device model parity (skips if no model)
```

## 5. Modified / new / removed files

New: `app/src/main/java/.../ml/*` (16 classes), `CapabilityActivity`, `MeasurementActivity`, `MeasurementMode`, `MeasurementRecord`,
`ModelStatus`, `ResultActivity`, `HistoryActivity`, `TrendChartView`; layouts `activity_capability`, `item_capability`, `activity_measurement`,
`activity_result`, `item_result_row`, `activity_history`, `item_history`; drawables `bg_*`; unit tests
(`app/src/test/.../ml/*`, fixtures `app/src/test/resources/golden_*.csv`); instrumented tests
(`MeasurementDbInstrumentedTest`, `ModelParityInstrumentedTest`, `TfliteRuntimeInstrumentedTest`); `app/src/main/assets/models/README.md`, `app/src/main/assets/selftest/runtime_selftest.tflite`;
`ml_training/*` (train.py, ppg_features.py, make_golden_vectors.py, java/BaselineCli.java, run_baseline.sh, README).

Modified: `build.gradle`, `settings.gradle`, `gradle.properties`, `gradle/wrapper/gradle-wrapper.properties`,
`app/build.gradle`, `AndroidManifest.xml` (exported flags, flash not required, removed WAKE_LOCK/FLASHLIGHT, backup off),
`ImageProcessing`, `UserDB` (schema v2, history API, safe upgrade), `Primary`, `StartVitalSigns`, `AboutApp`,
the five `*Process` and five `*Result(s)` activities (now thin subclasses of the shared base classes),
layouts `activity_primary`, `activity_start_vital_signs`, `activity_about_app`, `colors.xml`, `styles.xml`.

Removed: the ten old process/result layouts (replaced by `activity_measurement` / `activity_result`; all screens and
features remain), and the two template `Example*Test` files (replaced by real tests).

## 6. What is missing for genuine ML inference / validation

1. A real dataset (format in `ml_training/README.md`) with reference devices per vital sign, many subjects, several
   phone models, and consent/ethics approval for health data.
2. SpO2: a controlled desaturation protocol with a validated oximeter; camera RGB alone is generally unreliable.
3. Blood pressure: cuff reference, 85+ subjects (ISO 81060-2 style), subject-independent evaluation, and probably
   per-user calibration (not implemented).
4. Independent validation reports before setting `validation.status=VALIDATED`; regulatory review before any
   clinical claim. The app states it is not a medical device.
5. Retuning of the quality-gate thresholds on real recordings.
6. Optional: a consent-gated research export of raw recordings (`PpgSignal.toCsv()` exists) to collect training data.

## 7. Verification status (honest)

Done in the authoring environment: all pure-Java logic compiled and **47 JUnit tests pass**, including exact
Java-vs-Python feature parity on golden vectors; the whole app source compiled against the Android API jar (androidx
and TFLite classes stubbed); resource references and XML well-formedness checked; the Python pipeline
(train -> TFLite -> parity) smoke-tested on synthetic data only.

**Not done (no Android SDK / Gradle / Maven access there):** a real Gradle build or APK, running the app on a device
or emulator, the instrumented tests, and loading a real `.tflite` with the TensorFlow Lite runtime. If Gradle
reports an error on first sync (e.g. a version/dependency resolution message), send me the text and I will fix it.

## 6. Health-screening classifier (good / needs review / high priority)

After a full "All vital signs" measurement, `ResultActivity` shows a colour-coded card under the vitals.

* Model: `assets/models/health_screening.tflite` + `.properties`, trained by `ml_training/train_screening.py`
  from `ml_training/healthwatcher_synthetic_vitals_demo.csv` (2000 synthetic rows, rule-based labels).
* Inputs: age, sex (from the profile) + HR, SBP, DBP, RR, SpO2 from the measurement. Temperature and
  signal quality are in the CSV but not used (the app cannot measure temperature).
* Output: `Within_demo_ranges` -> "Looks good", `Needs_review` -> "Needs review", `High_priority_review` -> "High-priority review",
  plus plain-language findings (e.g. "higher than the common range ... hypertension pattern").
* Safety net in `ScreeningClassifier.classify`: a verdict is never "Looks good" while a reading is outside the common
  range, and any reading past the extreme cutoffs forces "High-priority review".
* Held-out test (300 rows): accuracy 94.7 %, macro-F1 0.93 (majority-class baseline 67.7 %). Because the data are
  synthetic and labelled by threshold rules, this shows the model learned those rules, NOT clinical accuracy.
* Only adults (18+) are screened; children get "not available".
* Retrain: `cd ml_training && python train_screening.py --csv healthwatcher_synthetic_vitals_demo.csv --epochs 500`
