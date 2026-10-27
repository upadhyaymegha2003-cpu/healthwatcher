package com.example.yo7a.healthwatcher.ml;

import java.util.EnumSet;
import java.util.Random;

/**
 * Runs the complete PPG pipeline (quality gate -> baseline -> features -> predictor) on a built-in
 * deterministic 72 bpm test waveform and checks the results. This verifies that the algorithms run correctly on
 * this device; it is NOT a measurement and the waveform is never shown as patient data.
 */
public final class PipelineSelfTest {
    private static final String TITLE = "PPG signal pipeline (quality gate, features, estimators)";

    private PipelineSelfTest() {
    }

    static PpgSignal testWaveform() {
        Random rnd = new Random(7);
        int n = 900;
        double fs = 30;
        double[] t = new double[n], r = new double[n], g = new double[n], b = new double[n];
        for (int i = 0; i < n; i++) {
            t[i] = i / fs;
            double ph = 2 * Math.PI * 1.2 * t[i];
            double pulse = Math.sin(ph) + 0.35 * Math.sin(2 * ph + 0.6);
            double resp = Math.sin(2 * Math.PI * 0.25 * t[i]);
            r[i] = 232 + 1.4 * pulse + 0.9 * resp + 0.05 * rnd.nextGaussian();
            g[i] = 118 + 1.0 * pulse + 0.6 * resp + 0.05 * rnd.nextGaussian();
            b[i] = 41 + 0.25 * pulse + 0.15 * resp + 0.03 * rnd.nextGaussian();
        }
        return new PpgSignal(t, r, g, b);
    }

    public static CapabilityItem run() {
        try {
            PpgSignal s = testWaveform();
            VitalSignsReport rep = new VitalSignsPredictor(null).predict(s, new UserProfile(30, 175, 70, 1),
                    EnumSet.allOf(VitalTarget.class));
            if (!rep.quality.isUsable()) {
                return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL,
                        "Quality gate rejected the clean test waveform: " + rep.quality.joinedMessages());
            }
            double[] f = PPGFeatureExtractor.extract(s);
            if (f.length != PPGFeatureExtractor.FEATURE_COUNT || Math.abs(f[0] - 1.2) > 0.03) {
                return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL, "Feature extraction returned unexpected values");
            }
            VitalEstimate hr = rep.get(VitalTarget.HEART_RATE);
            if (!hr.available || Math.abs(hr.value - 72) > 3) {
                return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL,
                        "Heart-rate estimator returned " + hr.displayValue() + " for a 72 bpm test waveform");
            }
            PpgSignal flat = new PpgSignal(s.t, s.red, new double[s.size()], s.blue);
            if (new SignalQualityAnalyzer().analyze(flat).isUsable()) {
                return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL, "Quality gate accepted a signal with no pulse");
            }
            return new CapabilityItem(TITLE, CapabilityItem.Status.PASS,
                    "Test waveform: quality gate accepted it, " + PPGFeatureExtractor.FEATURE_COUNT
                            + " features computed, heart rate " + hr.displayValue() + " bpm (expected ~72); a pulse-free signal was rejected.");
        } catch (Exception | LinkageError e) {
            return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL, "Pipeline self-test crashed: " + e);
        }
    }
}
