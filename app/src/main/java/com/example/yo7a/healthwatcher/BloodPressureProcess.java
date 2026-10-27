package com.example.yo7a.healthwatcher;

/** Measurement screen for BLOOD_PRESSURE; all behaviour lives in {@link MeasurementActivity}. */
public class BloodPressureProcess extends MeasurementActivity {
    @Override
    protected MeasurementMode mode() {
        return MeasurementMode.BLOOD_PRESSURE;
    }
}
