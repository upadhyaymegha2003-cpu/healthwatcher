package com.example.yo7a.healthwatcher.ml;

import java.io.IOException;
import java.util.Set;

/**
 * Orchestrates one measurement analysis:
 * quality gate -> baseline (original signal processing) -> feature extraction -> ML inference ->
 * source selection.
 *
 * Source policy: a trained model's number becomes THE result only if its manifest says
 * {@code validation.status=VALIDATED}. An installed but unvalidated model is run and shown only as an
 * "experimental" comparison next to the signal-processing value. Nothing is ever invented: with no
 * model installed, the ML side is reported as NOT_INSTALLED.
 */
public final class VitalSignsPredictor {
    private final MLModelManager models;          // may be null (ML disabled)
    private final SignalQualityAnalyzer analyzer;
    private final BaselineEstimator baseline = new BaselineEstimator();

    public VitalSignsPredictor(MLModelManager models) {
        this(models, new SignalQualityAnalyzer());
    }

    public VitalSignsPredictor(MLModelManager models, SignalQualityAnalyzer analyzer) {
        this.models = models;
        this.analyzer = analyzer;
    }

    /** Runs the full analysis. Blocking: call from a background thread. */
    public VitalSignsReport predict(PpgSignal signal, UserProfile profile, Set<VitalTarget> targets) {
        QualityReport quality = analyzer.analyze(signal);
        VitalSignsReport report = new VitalSignsReport(System.currentTimeMillis(),
                signal.durationSec(), signal.meanSamplingHz(), quality);

        // Quality gate: no value is produced from a recording that failed a blocking check.
        if (!quality.isUsable()) {
            for (VitalTarget t : targets) {
                VitalEstimate e = new VitalEstimate(t);
                e.available = false;
                e.mlDetail = "Recording rejected by quality checks";
                report.put(e);
            }
            return report;
        }

        BaselineEstimator.Result base = baseline.estimate(signal, profile);

        double[] features = null;
        String featureError = null;
        try {
            features = PPGFeatureExtractor.extract(signal);
        } catch (PPGFeatureExtractor.FeatureExtractionException ex) {
            featureError = ex.getMessage();
            quality.add(QualityReport.Issue.FEATURES_UNAVAILABLE, QualityReport.Severity.WARNING,
                    "Machine-learning features could not be computed: " + ex.getMessage());
        }

        for (VitalTarget t : targets) {
            VitalEstimate e = new VitalEstimate(t);
            switch (t) {
                case HEART_RATE:
                    e.baselineValue = base.heartRate;
                    break;
                case RESPIRATORY_RATE:
                    e.baselineValue = base.respiratoryRate;
                    break;
                case SPO2:
                    e.baselineValue = base.spo2;
                    break;
                case BLOOD_PRESSURE:
                    e.baselineValue = base.systolic;
                    e.baselineValue2 = base.diastolic;
                    break;
                default:
                    break;
            }
            runModel(e, features, featureError, profile);

            if (e.mlState == VitalEstimate.MlState.VALIDATED) {
                e.available = true;
                e.source = VitalEstimate.Source.ML_VALIDATED;
                e.value = e.mlValue;
                e.value2 = e.mlValue2;
            } else if (!Double.isNaN(e.baselineValue)) {
                e.available = true;
                e.source = VitalEstimate.Source.SIGNAL_PROCESSING;
                e.value = e.baselineValue;
                e.value2 = e.baselineValue2;
            } else {
                e.available = false;
            }
            report.put(e);
        }
        return report;
    }

    private void runModel(VitalEstimate e, double[] features, String featureError, UserProfile profile) {
        if (models == null) {
            e.mlState = VitalEstimate.MlState.NOT_INSTALLED;
            return;
        }
        MLModelManager.ModelInfo info = models.info(e.target);
        if (info.state == MLModelManager.State.NOT_INSTALLED) {
            e.mlState = VitalEstimate.MlState.NOT_INSTALLED;
            return;
        }
        if (!info.isReady()) {
            e.mlState = VitalEstimate.MlState.UNAVAILABLE;
            e.mlDetail = info.detail;
            return;
        }
        ModelManifest m = info.manifest;
        e.modelVersion = m.modelVersion;
        if (features == null) {
            e.mlState = VitalEstimate.MlState.UNAVAILABLE;
            e.mlDetail = featureError == null ? "no features" : featureError;
            return;
        }
        double[] input = new double[m.inputSize()];
        for (int i = 0; i < input.length; i++) {
            String name = m.featureNames[i];
            if (ModelManifest.isDemographic(name)) {
                if (profile == null || !profile.isPlausible()) {
                    e.mlState = VitalEstimate.MlState.UNAVAILABLE;
                    e.mlDetail = "a complete user profile is required by this model";
                    return;
                }
                input[i] = demographic(name, profile);
            } else {
                input[i] = features[PPGFeatureExtractor.indexOf(name)];
            }
        }
        double[] out;
        try {
            out = models.infer(e.target, input);
        } catch (IOException | RuntimeException ex) {
            e.mlState = VitalEstimate.MlState.UNAVAILABLE;
            e.mlDetail = ex.getMessage() == null ? ex.getClass().getSimpleName() : ex.getMessage();
            return;
        }
        for (int i = 0; i < out.length; i++) {
            if (Double.isNaN(out[i]) || Double.isInfinite(out[i])
                    || out[i] < m.outputMin[i] || out[i] > m.outputMax[i]) {
                e.mlState = VitalEstimate.MlState.REJECTED;
                e.mlDetail = "outside " + m.outputMin[i] + ".." + m.outputMax[i];
                e.mlValue = out[i];
                return;
            }
        }
        if (e.target == VitalTarget.BLOOD_PRESSURE && !(out[0] > out[1])) {
            e.mlState = VitalEstimate.MlState.REJECTED;
            e.mlDetail = "systolic not above diastolic";
            return;
        }
        e.mlValue = out[0];
        e.mlValue2 = out.length > 1 ? out[1] : Double.NaN;
        e.mlState = m.isValidated() ? VitalEstimate.MlState.VALIDATED : VitalEstimate.MlState.EXPERIMENTAL;
    }

    private static double demographic(String name, UserProfile p) {
        switch (name) {
            case "demo_age":
                return p.age;
            case "demo_height_cm":
                return p.heightCm;
            case "demo_weight_kg":
                return p.weightKg;
            default:
                return p.gender;
        }
    }
}
