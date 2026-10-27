package com.example.yo7a.healthwatcher;

import com.example.yo7a.healthwatcher.ml.VitalTarget;

import java.util.EnumSet;
import java.util.Set;

/** Which vital signs a measurement screen computes; {@code page} is the value Primary/StartVitalSigns already pass around. */
public enum MeasurementMode {
    HEART_RATE(1, EnumSet.of(VitalTarget.HEART_RATE), HeartRateResult.class),
    BLOOD_PRESSURE(2, EnumSet.of(VitalTarget.BLOOD_PRESSURE), BloodPressureResult.class),
    RESPIRATION(3, EnumSet.of(VitalTarget.RESPIRATORY_RATE), RespirationResult.class),
    SPO2(4, EnumSet.of(VitalTarget.SPO2), O2Result.class),
    ALL(5, EnumSet.allOf(VitalTarget.class), VitalSignsResults.class);

    public final int page;
    private final Set<VitalTarget> targets;
    public final Class<?> resultActivity;

    MeasurementMode(int page, Set<VitalTarget> targets, Class<?> resultActivity) {
        this.page = page;
        this.targets = targets;
        this.resultActivity = resultActivity;
    }

    public Set<VitalTarget> targets() {
        return EnumSet.copyOf(targets);
    }

    public static MeasurementMode fromPage(int page) {
        for (MeasurementMode m : values()) {
            if (m.page == page) {
                return m;
            }
        }
        return ALL;
    }
}
