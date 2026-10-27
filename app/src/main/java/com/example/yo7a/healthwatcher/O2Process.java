package com.example.yo7a.healthwatcher;

/** Measurement screen for SPO2; all behaviour lives in {@link MeasurementActivity}. */
public class O2Process extends MeasurementActivity {
    @Override
    protected MeasurementMode mode() {
        return MeasurementMode.SPO2;
    }
}
