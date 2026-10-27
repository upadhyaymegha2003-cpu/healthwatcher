package com.example.yo7a.healthwatcher.ml;

import java.io.Serializable;
import java.util.Locale;

/**
 * One vital-sign result with full provenance: which method produced the displayed value, what the
 * other method said, and the state of the ML model for this target.
 */
public final class VitalEstimate implements Serializable {
    private static final long serialVersionUID = 1L;

    public enum Source {
        /** Original FFT / formula based estimator. */
        SIGNAL_PROCESSING,
        /** Trained model whose manifest declares independent validation. */
        ML_VALIDATED
    }

    public enum MlState {
        NOT_INSTALLED,   // no model files in assets/models
        UNAVAILABLE,     // files present but could not be used (load failure, incompatible, bad input)
        EXPERIMENTAL,    // model ran, but is not declared validated: value shown only as a comparison
        VALIDATED,       // model ran and is declared validated: value is used as the result
        REJECTED         // model ran but produced an implausible output
    }

    public final VitalTarget target;
    public boolean available;
    public double value = Double.NaN;       // heart rate, respiratory rate, SpO2, or systolic
    public double value2 = Double.NaN;      // diastolic (blood pressure only)
    public Source source = Source.SIGNAL_PROCESSING;

    public double baselineValue = Double.NaN;
    public double baselineValue2 = Double.NaN;

    public MlState mlState = MlState.NOT_INSTALLED;
    public double mlValue = Double.NaN;
    public double mlValue2 = Double.NaN;
    public String modelVersion = "";
    public String mlDetail = "";

    VitalEstimate(VitalTarget target) {
        this.target = target;
    }

    public static String format(VitalTarget t, double v, double v2) {
        if (Double.isNaN(v)) {
            return "--";
        }
        if (t == VitalTarget.BLOOD_PRESSURE) {
            return String.format(Locale.US, "%d / %d", Math.round(v), Math.round(v2));
        }
        return String.format(Locale.US, "%d", Math.round(v));
    }

    public String displayValue() {
        return available ? format(target, value, value2) : "--";
    }

    public String baselineDisplay() {
        return format(target, baselineValue, baselineValue2);
    }

    public String mlDisplay() {
        return format(target, mlValue, mlValue2);
    }

    /** Short label for the badge next to the value. */
    public String sourceLabel() {
        if (!available) {
            return "Not available";
        }
        return source == Source.ML_VALIDATED ? "Validated ML model" : "Signal processing";
    }

    /** Honest caveat text that must accompany the value. */
    public String caution() {
        if (!available) {
            return "A reliable value could not be estimated from this recording. Please retry.";
        }
        if (source == Source.ML_VALIDATED) {
            return "Estimated on-device by a trained model that its manifest declares validated. "
                    + "Not a medical device reading.";
        }
        switch (target) {
            case BLOOD_PRESSURE:
                return "Experimental estimate from an uncalibrated formula using heart rate, age, height and weight. "
                        + "It is not a validated blood-pressure measurement. Use a cuff for any decision.";
            case SPO2:
                return "Experimental estimate from the camera's red/blue signal ratio with a generic formula. "
                        + "It is not a validated oxygen-saturation measurement. Use a pulse oximeter for any decision.";
            default:
                return "Estimated with frequency-domain signal processing of the camera signal. "
                        + "For information only, not a medical device reading.";
        }
    }

    /** Text describing the ML side-channel, or empty when there is nothing to say. */
    public String mlNote() {
        switch (mlState) {
            case EXPERIMENTAL:
                return "Experimental ML estimate (model " + modelVersion + ", NOT validated): " + mlDisplay() + " " + target.unit;
            case REJECTED:
                return "ML model output was rejected as implausible (" + mlDetail + ").";
            case UNAVAILABLE:
                return "ML model unavailable: " + mlDetail;
            case VALIDATED:
                return "Signal-processing comparison value: " + baselineDisplay() + " " + target.unit;
            case NOT_INSTALLED:
            default:
                return "";
        }
    }
}
