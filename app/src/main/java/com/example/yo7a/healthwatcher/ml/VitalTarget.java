package com.example.yo7a.healthwatcher.ml;

/** The four physiological quantities HealthWatcher estimates. */
public enum VitalTarget {
    HEART_RATE("heart_rate", "Heart rate", "bpm"),
    BLOOD_PRESSURE("blood_pressure", "Blood pressure", "mmHg"),
    RESPIRATORY_RATE("respiratory_rate", "Respiratory rate", "breaths/min"),
    SPO2("spo2", "Oxygen saturation (SpO\u2082)", "%");

    /** Base name of the model files in assets/models/ (e.g. heart_rate.tflite + heart_rate.properties). */
    public final String assetName;
    public final String displayName;
    public final String unit;

    VitalTarget(String assetName, String displayName, String unit) {
        this.assetName = assetName;
        this.displayName = displayName;
        this.unit = unit;
    }

    /** Number of values the target produces (blood pressure = systolic + diastolic). */
    public int outputCount() {
        return this == BLOOD_PRESSURE ? 2 : 1;
    }
}
