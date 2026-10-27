package com.example.yo7a.healthwatcher.ml;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Outcome of the on-device health-screening classifier for one measurement.
 *
 * This is a SCREENING hint, not a diagnosis: the three classes mirror the labels of the training data
 * (Within_demo_ranges / Needs_review / High_priority_review), and {@link #findings} lists in plain words
 * which readings are outside the common adult resting range.
 */
public final class ScreeningResult implements Serializable {
    private static final long serialVersionUID = 1L;

    public enum Verdict {WITHIN_RANGES, NEEDS_REVIEW, HIGH_PRIORITY}

    public final boolean available;
    /** Why no screening was produced (only when {@code available} is false). */
    public final String unavailableReason;
    public final Verdict verdict;
    /** Model output for [Within, Needs review, High priority]; null when unavailable. */
    public final float[] probabilities;
    public final String modelVersion;
    /** True when the simple range check raised the verdict above what the model said (safety net). */
    public final boolean raisedByRangeCheck;
    public final ArrayList<String> findings;

    ScreeningResult(boolean available, String unavailableReason, Verdict verdict, float[] probabilities,
                    String modelVersion, boolean raisedByRangeCheck, List<String> findings) {
        this.available = available;
        this.unavailableReason = unavailableReason == null ? "" : unavailableReason;
        this.verdict = verdict;
        this.probabilities = probabilities;
        this.modelVersion = modelVersion == null ? "" : modelVersion;
        this.raisedByRangeCheck = raisedByRangeCheck;
        this.findings = new ArrayList<>(findings);
    }

    static ScreeningResult unavailable(String reason) {
        return new ScreeningResult(false, reason, null, null, "", false, new ArrayList<String>());
    }

    /** Big headline, e.g. "Looks good". */
    public String headline() {
        if (!available) {
            return "Screening not available";
        }
        switch (verdict) {
            case WITHIN_RANGES:
                return "Looks good";
            case NEEDS_REVIEW:
                return "Needs review";
            default:
                return "High-priority review";
        }
    }

    public String summary() {
        if (!available) {
            return unavailableReason;
        }
        switch (verdict) {
            case WITHIN_RANGES:
                return "All your readings are within the common adult resting ranges. No signs of concern were found.";
            case NEEDS_REVIEW:
                return "One or more readings are outside the common resting range.";
            default:
                return "One or more readings are far outside the common resting range.";
        }
    }

    public String advice() {
        if (!available) {
            return "";
        }
        switch (verdict) {
            case WITHIN_RANGES:
                return "Keep up your healthy habits. Measure again if you ever feel unwell.";
            case NEEDS_REVIEW:
                return "Sit and rest for 5 minutes, then measure again. If it stays outside the range, or you have "
                        + "symptoms, please talk to a doctor. Confirm blood pressure with a cuff and oxygen with a pulse oximeter.";
            default:
                return "This may be a measurement error, so rest and measure again. If it is confirmed, or you have chest pain, "
                        + "trouble breathing, fainting or confusion, get medical help promptly.";
        }
    }

    /** One short line with the three model scores, e.g. "Model scores: good 3% / review 92% / high priority 5%". */
    public String scoreLine() {
        if (!available || probabilities == null) {
            return "";
        }
        return String.format(Locale.US, "Model scores: good %d%% / review %d%% / high priority %d%%",
                Math.round(probabilities[0] * 100), Math.round(probabilities[1] * 100), Math.round(probabilities[2] * 100));
    }

    public String caveat() {
        return "Screening hint from a small model trained on SYNTHETIC demo data, not on patients. It is not a diagnosis. "
                + "It also relies on camera-based blood-pressure and SpO\u2082 estimates, which are uncalibrated.";
    }

    /** Plain-text version for sharing. */
    public String shareText() {
        if (!available) {
            return "Health screening: not available (" + unavailableReason + ")";
        }
        StringBuilder sb = new StringBuilder("Health screening: ").append(headline()).append('\n');
        for (String f : findings) {
            sb.append(" - ").append(f).append('\n');
        }
        sb.append(advice());
        return sb.toString();
    }
}
