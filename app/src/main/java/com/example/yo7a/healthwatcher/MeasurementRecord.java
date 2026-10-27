package com.example.yo7a.healthwatcher;

import com.example.yo7a.healthwatcher.ml.VitalEstimate;
import com.example.yo7a.healthwatcher.ml.VitalSignsReport;
import com.example.yo7a.healthwatcher.ml.VitalTarget;

/** One stored measurement (row of the {@code measurements} table). NaN = vital not measured. */
public final class MeasurementRecord {
    public long id;
    public String username;
    public long timestampMillis;
    public double heartRate = Double.NaN;
    public double systolic = Double.NaN;
    public double diastolic = Double.NaN;
    public double respiratoryRate = Double.NaN;
    public double spo2 = Double.NaN;
    /** Per-vital source, e.g. "hr=SIGNAL_PROCESSING;bp=SIGNAL_PROCESSING". */
    public String sources = "";
    /** Model versions that ran (experimental or validated), e.g. "heart_rate=v1". Empty if none. */
    public String modelVersions = "";
    public String qualityStatus = "";
    public String qualityNote = "";
    public double durationSec;
    public double samplingHz;

    public static MeasurementRecord fromReport(String user, VitalSignsReport r) {
        MeasurementRecord m = new MeasurementRecord();
        m.username = user;
        m.timestampMillis = r.timestampMillis;
        m.qualityStatus = r.quality.statusLabel();
        m.qualityNote = r.quality.joinedMessages();
        m.durationSec = r.durationSec;
        m.samplingHz = r.samplingHz;
        StringBuilder src = new StringBuilder();
        StringBuilder ver = new StringBuilder();
        for (VitalEstimate e : r.all().values()) {
            if (!e.modelVersion.isEmpty()) {
                ver.append(ver.length() > 0 ? ";" : "").append(e.target.assetName).append('=').append(e.modelVersion);
            }
            if (!e.available) {
                continue;
            }
            src.append(src.length() > 0 ? ";" : "").append(e.target.assetName).append('=').append(e.source.name());
            switch (e.target) {
                case HEART_RATE:
                    m.heartRate = e.value;
                    break;
                case RESPIRATORY_RATE:
                    m.respiratoryRate = e.value;
                    break;
                case SPO2:
                    m.spo2 = e.value;
                    break;
                case BLOOD_PRESSURE:
                    m.systolic = e.value;
                    m.diastolic = e.value2;
                    break;
                default:
                    break;
            }
        }
        m.sources = src.toString();
        m.modelVersions = ver.toString();
        return m;
    }

    /** True when at least one vital was stored. */
    public boolean hasValues() {
        return !Double.isNaN(heartRate) || !Double.isNaN(systolic)
                || !Double.isNaN(respiratoryRate) || !Double.isNaN(spo2);
    }

    /** "ML" or "SP" tag for a target, derived from {@link #sources}. */
    public String sourceTag(VitalTarget t) {
        for (String part : sources.split(";")) {
            if (part.startsWith(t.assetName + "=")) {
                return part.endsWith("ML_VALIDATED") ? "ML" : "SP";
            }
        }
        return "";
    }
}
