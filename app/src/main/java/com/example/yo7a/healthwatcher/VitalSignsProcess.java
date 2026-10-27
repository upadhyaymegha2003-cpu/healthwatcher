package com.example.yo7a.healthwatcher;

/** Measurement screen for ALL; all behaviour lives in {@link MeasurementActivity}. */
public class VitalSignsProcess extends MeasurementActivity {
    @Override
    protected MeasurementMode mode() {
        return MeasurementMode.ALL;
    }
}
