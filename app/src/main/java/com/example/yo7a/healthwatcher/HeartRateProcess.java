package com.example.yo7a.healthwatcher;

/** Measurement screen for HEART_RATE; all behaviour lives in {@link MeasurementActivity}. */
public class HeartRateProcess extends MeasurementActivity {
    @Override
    protected MeasurementMode mode() {
        return MeasurementMode.HEART_RATE;
    }
}
