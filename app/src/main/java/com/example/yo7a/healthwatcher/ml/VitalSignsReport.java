package com.example.yo7a.healthwatcher.ml;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/** Everything produced by one measurement: quality verdict plus one {@link VitalEstimate} per requested target. */
public final class VitalSignsReport implements Serializable {
    private static final long serialVersionUID = 1L;

    public final long timestampMillis;
    public final double durationSec;
    public final double samplingHz;
    public final QualityReport quality;
    public final String featureVersion;
    /** Health-screening verdict from the on-device classifier; null until the measurement flow sets it. */
    public ScreeningResult screening;
    private final EnumMap<VitalTarget, VitalEstimate> estimates = new EnumMap<>(VitalTarget.class);

    VitalSignsReport(long timestampMillis, double durationSec, double samplingHz, QualityReport quality) {
        this.timestampMillis = timestampMillis;
        this.durationSec = durationSec;
        this.samplingHz = samplingHz;
        this.quality = quality;
        this.featureVersion = PPGFeatureExtractor.FEATURE_VERSION;
    }

    void put(VitalEstimate e) {
        estimates.put(e.target, e);
    }

    /** Estimate for a target, or null if it was not requested. */
    public VitalEstimate get(VitalTarget t) {
        return estimates.get(t);
    }

    public Map<VitalTarget, VitalEstimate> all() {
        return estimates;
    }

    /** True when the signal passed the quality gate and at least one requested vital is available. */
    public boolean isSuccessful() {
        if (!quality.isUsable()) {
            return false;
        }
        for (VitalEstimate e : estimates.values()) {
            if (e.available) {
                return true;
            }
        }
        return false;
    }

    public List<VitalTarget> unavailableTargets() {
        List<VitalTarget> out = new ArrayList<>();
        for (VitalEstimate e : estimates.values()) {
            if (!e.available) {
                out.add(e.target);
            }
        }
        return out;
    }

    /** Message explaining why a retry is needed, or empty if the report is successful. */
    public String failureMessage() {
        if (!quality.isUsable()) {
            List<String> m = quality.messages(QualityReport.Severity.BLOCKING);
            StringBuilder sb = new StringBuilder();
            for (String s : m) {
                if (sb.length() > 0) {
                    sb.append("\n");
                }
                sb.append(s);
            }
            return sb.toString();
        }
        if (!isSuccessful()) {
            return "The signal could not be turned into a reliable value. Please retry and keep your finger still.";
        }
        return "";
    }
}
