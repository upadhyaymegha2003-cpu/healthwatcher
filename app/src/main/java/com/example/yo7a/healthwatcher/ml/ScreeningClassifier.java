package com.example.yo7a.healthwatcher.ml;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * On-device health-screening classifier. Takes the four vital signs of a measurement plus age and sex and says
 * whether they look good, need review, or need high-priority review.
 *
 * Model: {@code assets/models/health_screening.tflite} (+ {@code .properties}), produced by
 * {@code ml_training/train_screening.py} from healthwatcher_synthetic_vitals_demo.csv. Input is 17 floats:
 * the 7 raw values (age, is_male, HR, SBP, DBP, RR, SpO2) + 5 "distance outside common range" + 5 "distance past extreme cutoff" features, computed
 * by {@link #engineer} (must match engineer() in train_screening.py). Normalisation is inside the model.
 *
 * No Android dependency: files come from an {@link MLModelManager.ModelSource}, inference from a
 * {@link MLModelManager.ModelRunner}.
 */
public final class ScreeningClassifier {
    public static final String MODEL_FILE = "health_screening.tflite";
    public static final String PROPS_FILE = "health_screening.properties";
    public static final int RAW_FEATURES = 7;
    public static final int INPUT_SIZE = 17;
    public static final int CLASS_COUNT = 3;
    public static final int MIN_AGE = 18;

    // Common adult resting ranges (same ones the training labels use).
    static final double HR_LOW = 60, HR_HIGH = 100;
    static final double SBP_LOW = 90, SBP_HIGH = 139;
    static final double DBP_HIGH = 89;
    static final double RR_LOW = 12, RR_HIGH = 20;
    static final double SPO2_LOW = 95, SPO2_VERY_LOW = 90;
    // High-priority cutoffs (derived from the training labels): HR <40 / >140, SBP <80 / >=180, DBP >=120, RR <8 / >30, SpO2 <90.
    static final double HR_X_LOW = 40, HR_X_HIGH = 140, SBP_X_LOW = 80, SBP_X_HIGH = 179, DBP_X_HIGH = 119;
    static final double RR_X_LOW = 8, RR_X_HIGH = 30;

    private final MLModelManager.ModelRunner runner;   // null when not usable
    private final String version;
    private final String status;
    private final String metrics;

    private ScreeningClassifier(MLModelManager.ModelRunner runner, String version, String status, String metrics) {
        this.runner = runner;
        this.version = version;
        this.status = status;
        this.metrics = metrics;
    }

    /** Loads the model; never throws. Check {@link #isReady()} / {@link #statusDetail()}. */
    public static ScreeningClassifier load(MLModelManager.ModelSource source, MLModelManager.RunnerFactory factory) {
        try {
            if (!source.exists(MODEL_FILE) || !source.exists(PROPS_FILE)) {
                return new ScreeningClassifier(null, "", "No screening model installed.", "");
            }
            Properties p = new Properties();
            try (InputStream in = source.open(PROPS_FILE)) {
                p.load(in);
            }
            String classes = p.getProperty("classes", "");
            if (!"Within_demo_ranges,Needs_review,High_priority_review".equals(classes)) {
                return new ScreeningClassifier(null, "", "Screening model manifest has unexpected classes.", "");
            }
            if (!String.valueOf(INPUT_SIZE).equals(p.getProperty("input.size"))) {
                return new ScreeningClassifier(null, "", "Screening model manifest has unexpected input size.", "");
            }
            byte[] bytes;
            try (InputStream in = source.open(MODEL_FILE)) {
                bytes = readAll(in);
            }
            ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
            buf.put(bytes);
            buf.rewind();
            MLModelManager.ModelRunner r = factory.create(buf);
            if (r.inputSize() != INPUT_SIZE || r.outputSize() != CLASS_COUNT) {
                String d = "Screening model shape " + r.inputSize() + "->" + r.outputSize() + " does not match the app ("
                        + INPUT_SIZE + "->" + CLASS_COUNT + ").";
                r.close();
                return new ScreeningClassifier(null, "", d, "");
            }
            String version = p.getProperty("model.version", "unknown");
            String metrics = "test accuracy " + p.getProperty("test.accuracy", "?") + " on "
                    + p.getProperty("training.data", "unknown data");
            return new ScreeningClassifier(r, version, "Model " + version + " loaded (" + p.getProperty("validation.status", "UNVALIDATED") + ").", metrics);
        } catch (Throwable t) {   // UnsatisfiedLinkError on unsupported ABIs, bad flatbuffers, IO errors ...
            return new ScreeningClassifier(null, "", "Screening model could not be loaded: " + t.getMessage(), "");
        }
    }

    public boolean isReady() {
        return runner != null;
    }

    public String statusDetail() {
        return status;
    }

    public String metricsDetail() {
        return metrics;
    }

    public String version() {
        return version;
    }

    public void close() {
        if (runner != null) {
            runner.close();
        }
    }

    /** Raw features -> the 17 model inputs. Mirrors train_screening.engineer(). */
    public static float[] engineer(double age, boolean male, double hr, double sbp, double dbp, double rr, double spo2) {
        float[] f = new float[INPUT_SIZE];
        f[0] = (float) age;
        f[1] = male ? 1f : 0f;
        f[2] = (float) hr;
        f[3] = (float) sbp;
        f[4] = (float) dbp;
        f[5] = (float) rr;
        f[6] = (float) spo2;
        f[7] = (float) (Math.max(0, HR_LOW - hr) + Math.max(0, hr - HR_HIGH));
        f[8] = (float) (Math.max(0, SBP_LOW - sbp) + Math.max(0, sbp - SBP_HIGH));
        f[9] = (float) Math.max(0, dbp - DBP_HIGH);
        f[10] = (float) (Math.max(0, RR_LOW - rr) + Math.max(0, rr - RR_HIGH));
        f[11] = (float) Math.max(0, SPO2_LOW - spo2);
        f[12] = (float) (Math.max(0, HR_X_LOW - hr) + Math.max(0, hr - HR_X_HIGH));
        f[13] = (float) (Math.max(0, SBP_X_LOW - sbp) + Math.max(0, sbp - SBP_X_HIGH));
        f[14] = (float) Math.max(0, dbp - DBP_X_HIGH);
        f[15] = (float) (Math.max(0, RR_X_LOW - rr) + Math.max(0, rr - RR_X_HIGH));
        f[16] = (float) Math.max(0, SPO2_VERY_LOW - spo2);
        return f;
    }

    /** Classifies a finished measurement. Needs all four vitals and an adult profile. Blocking but fast. */
    public ScreeningResult classify(VitalSignsReport report, UserProfile profile) {
        if (!isReady()) {
            return ScreeningResult.unavailable(status);
        }
        VitalEstimate hr = report.get(VitalTarget.HEART_RATE);
        VitalEstimate bp = report.get(VitalTarget.BLOOD_PRESSURE);
        VitalEstimate rr = report.get(VitalTarget.RESPIRATORY_RATE);
        VitalEstimate sp = report.get(VitalTarget.SPO2);
        if (!ok(hr) || !ok(bp) || !ok(rr) || !ok(sp)) {
            return ScreeningResult.unavailable("Screening needs all four measurements. Use \"All vital signs\" on the home screen.");
        }
        return classify(profile, hr.value, bp.value, bp.value2, rr.value, sp.value);
    }

    public ScreeningResult classify(UserProfile profile, double hr, double sbp, double dbp, double rr, double spo2) {
        if (!isReady()) {
            return ScreeningResult.unavailable(status);
        }
        if (profile == null || !profile.isPlausible()) {
            return ScreeningResult.unavailable("Screening needs your age and sex in your profile.");
        }
        if (profile.age < MIN_AGE) {
            return ScreeningResult.unavailable("The screening model was trained on adults (18+) only.");
        }
        float[] in = engineer(profile.age, profile.gender == 1, hr, sbp, dbp, rr, spo2);
        float[] out;
        try {
            out = runner.run(in);
        } catch (RuntimeException e) {
            return ScreeningResult.unavailable("Screening model failed to run: " + e.getMessage());
        }
        if (out == null || out.length != CLASS_COUNT) {
            return ScreeningResult.unavailable("Screening model returned an unexpected output.");
        }
        float sum = 0;
        for (float v : out) {
            if (Float.isNaN(v) || Float.isInfinite(v) || v < -1e-3f || v > 1 + 1e-3f) {
                return ScreeningResult.unavailable("Screening model returned an invalid output.");
            }
            sum += v;
        }
        if (Math.abs(sum - 1f) > 0.02f) {
            return ScreeningResult.unavailable("Screening model returned an invalid output.");
        }
        int best = 0;
        for (int i = 1; i < CLASS_COUNT; i++) {
            if (out[i] > out[best]) {
                best = i;
            }
        }
        ScreeningResult.Verdict verdict = ScreeningResult.Verdict.values()[best];
        List<String> findings = findings(hr, sbp, dbp, rr, spo2);

        // Safety net: never tell someone "looks good" while a reading is plainly outside the common range.
        boolean raised = false;
        if (verdict == ScreeningResult.Verdict.WITHIN_RANGES && !findings.isEmpty()) {
            verdict = ScreeningResult.Verdict.NEEDS_REVIEW;
            raised = true;
        }
        if (isExtreme(hr, sbp, dbp, rr, spo2) && verdict != ScreeningResult.Verdict.HIGH_PRIORITY) {
            verdict = ScreeningResult.Verdict.HIGH_PRIORITY;
            raised = true;
        }
        if (verdict != ScreeningResult.Verdict.WITHIN_RANGES && findings.isEmpty()) {
            findings.add("The combination of your readings looked unusual to the model. Measure again to confirm.");
        }
        return new ScreeningResult(true, "", verdict, out.clone(), version, raised, findings);
    }

    /** True when any reading is past the high-priority cutoffs. */
    static boolean isExtreme(double hr, double sbp, double dbp, double rr, double spo2) {
        return hr < HR_X_LOW || hr > HR_X_HIGH || sbp < SBP_X_LOW || sbp > SBP_X_HIGH || dbp > DBP_X_HIGH
                || rr < RR_X_LOW || rr > RR_X_HIGH || spo2 < SPO2_VERY_LOW;
    }

    /** Plain-language notes on which readings are outside the common adult resting range. */
    static List<String> findings(double hr, double sbp, double dbp, double rr, double spo2) {
        List<String> f = new ArrayList<>();
        long h = Math.round(hr), s = Math.round(sbp), d = Math.round(dbp), r = Math.round(rr), o = Math.round(spo2);
        if (hr < HR_LOW) {
            f.add("Heart rate " + h + " bpm is slow (below 60). This is called bradycardia and is normal in very fit people.");
        } else if (hr > HR_HIGH) {
            f.add("Heart rate " + h + " bpm is fast (above 100). This is called tachycardia and can follow exercise, stress, caffeine or fever.");
        }
        if (sbp > SBP_HIGH || dbp > DBP_HIGH) {
            f.add("Blood pressure " + s + "/" + d + " is higher than the common range (up to 139/89). A pattern like this is seen in hypertension.");
        } else if (sbp < SBP_LOW) {
            f.add("Blood pressure " + s + "/" + d + " is lower than the common range (systolic below 90). This is called hypotension.");
        }
        if (rr > RR_HIGH) {
            f.add("Breathing rate " + r + "/min is fast (above 20). This is called tachypnea.");
        } else if (rr < RR_LOW) {
            f.add("Breathing rate " + r + "/min is slow (below 12). This is called bradypnea.");
        }
        if (spo2 < SPO2_VERY_LOW) {
            f.add("Oxygen saturation " + o + "% is low (below 90). A value like this can mean too little oxygen in the blood (hypoxemia).");
        } else if (spo2 < SPO2_LOW) {
            f.add("Oxygen saturation " + o + "% is slightly reduced (below 95).");
        }
        return f;
    }

    /** Real check for the System-check screen: runs two built-in sample vectors (never shown as patient data). */
    public CapabilityItem selfTest() {
        String title = "Health screening classifier";
        if (!isReady()) {
            return new CapabilityItem(title, CapabilityItem.Status.INFO, status);
        }
        UserProfile adult = new UserProfile(45, 170, 70, 1);
        ScreeningResult healthy = classify(adult, 72, 118, 76, 16, 98);
        ScreeningResult unwell = classify(adult, 135, 190, 115, 30, 86);
        if (!healthy.available || !unwell.available) {
            return new CapabilityItem(title, CapabilityItem.Status.FAIL, "Model loaded but did not produce an output.");
        }
        boolean sane = healthy.verdict == ScreeningResult.Verdict.WITHIN_RANGES
                && unwell.verdict != ScreeningResult.Verdict.WITHIN_RANGES;
        if (!sane) {
            return new CapabilityItem(title, CapabilityItem.Status.FAIL,
                    "Sanity check failed: a normal sample was rated " + healthy.headline() + " and an abnormal sample "
                            + unwell.headline() + ".");
        }
        return new CapabilityItem(title, CapabilityItem.Status.WARN,
                status + " Sanity check passed (normal sample -> \"" + healthy.headline() + "\", abnormal sample -> \""
                        + unwell.headline() + "\"). Trained on synthetic data only (" + metrics + "); not clinically validated.");
    }

    private static boolean ok(VitalEstimate e) {
        return e != null && e.available && !Double.isNaN(e.value) && (e.target != VitalTarget.BLOOD_PRESSURE || !Double.isNaN(e.value2));
    }

    private static byte[] readAll(InputStream in) throws IOException {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        byte[] b = new byte[8192];
        int n;
        while ((n = in.read(b)) > 0) {
            bos.write(b, 0, n);
        }
        return bos.toByteArray();
    }
}
