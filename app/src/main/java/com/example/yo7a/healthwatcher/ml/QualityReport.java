package com.example.yo7a.healthwatcher.ml;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Outcome of the pre-inference signal-quality checks. It deliberately contains NO numeric
 * "confidence" value: the checks are rule-based engineering thresholds, not a validated reliability model.
 */
public final class QualityReport implements Serializable {
    private static final long serialVersionUID = 1L;

    public enum Issue {
        TOO_SHORT, LOW_SAMPLING_RATE, SAMPLING_GAPS, INVALID_VALUES, NO_FINGER,
        SATURATED, NO_PULSATILE_SIGNAL, MOTION_ARTIFACT, LOW_CARDIAC_SNR, FEATURES_UNAVAILABLE
    }

    public enum Severity {BLOCKING, WARNING}

    public static final class Finding implements Serializable {
        private static final long serialVersionUID = 1L;
        public final Issue issue;
        public final Severity severity;
        public final String message;

        Finding(Issue issue, Severity severity, String message) {
            this.issue = issue;
            this.severity = severity;
            this.message = message;
        }
    }

    private final ArrayList<Finding> findings = new ArrayList<>();

    // Diagnostic metrics (informational only).
    public double durationSec;
    public double meanSamplingHz;
    public double maxGapSec;
    public double redMean;
    public double redSaturatedFraction;
    public double greenSaturatedFraction;
    public double cardiacPowerRatio = Double.NaN;

    void add(Issue issue, Severity severity, String message) {
        findings.add(new Finding(issue, severity, message));
    }

    public List<Finding> findings() {
        return Collections.unmodifiableList(findings);
    }

    /** True when no blocking issue was found: values may be shown (with any warnings). */
    public boolean isUsable() {
        for (Finding f : findings) {
            if (f.severity == Severity.BLOCKING) {
                return false;
            }
        }
        return true;
    }

    public boolean hasWarnings() {
        for (Finding f : findings) {
            if (f.severity == Severity.WARNING) {
                return true;
            }
        }
        return false;
    }

    public boolean has(Issue issue) {
        for (Finding f : findings) {
            if (f.issue == issue) {
                return true;
            }
        }
        return false;
    }

    public List<String> messages(Severity severity) {
        List<String> out = new ArrayList<>();
        for (Finding f : findings) {
            if (f.severity == severity) {
                out.add(f.message);
            }
        }
        return out;
    }

    /** One-line summary for storage: OK, WARNINGS or REJECTED. */
    public String statusLabel() {
        return !isUsable() ? "REJECTED" : (hasWarnings() ? "WARNINGS" : "OK");
    }

    /** Convenience for logs and test messages. */
    public String messagesDebug() {
        return statusLabel() + ": " + joinedMessages();
    }

    public String joinedMessages() {
        StringBuilder sb = new StringBuilder();
        for (Finding f : findings) {
            if (sb.length() > 0) {
                sb.append(" ");
            }
            sb.append(f.message);
        }
        return sb.toString();
    }
}
