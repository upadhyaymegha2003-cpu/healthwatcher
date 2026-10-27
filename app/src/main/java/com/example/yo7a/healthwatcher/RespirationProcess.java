package com.example.yo7a.healthwatcher;

/** Measurement screen for RESPIRATION; all behaviour lives in {@link MeasurementActivity}. */
public class RespirationProcess extends MeasurementActivity {
    @Override
    protected MeasurementMode mode() {
        return MeasurementMode.RESPIRATION;
    }
}
