package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class BaselineEstimatorTest {
    private final BaselineEstimator est = new BaselineEstimator();
    private final UserProfile profile = new UserProfile(30, 175, 70, 1);

    @Test
    public void originalFftHeartRateIsPreserved() {
        // The original FFT picks the bin and rounds up: a 1.2 Hz pulse (72 bpm) reads 72-74 bpm.
        BaselineEstimator.Result r = est.estimate(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile);
        assertEquals(73, r.heartRate, 2.5);
    }

    @Test
    public void respiratoryRateComesFromBandPassedFft() {
        BaselineEstimator.Result r = est.estimate(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile);
        assertEquals(15, r.respiratoryRate, 2.5);
    }

    @Test
    public void bloodPressureUsesOriginalFormulaAndIsPlausible() {
        BaselineEstimator.Result r = est.estimate(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile);
        assertTrue(r.systolic > r.diastolic);
        assertTrue(r.systolic >= BaselineEstimator.SBP_MIN && r.systolic <= BaselineEstimator.SBP_MAX);
        // Re-compute the original formula by hand for HR = r.heartRate (male, 70 kg, 175 cm, 30 y)
        double beats = r.heartRate;
        double bsa = 0.007184 * Math.pow(70, 0.425) * Math.pow(175, 0.725);
        double sv = -6.6 + 0.25 * ((364.5 - 1.23 * beats) - 35) - 0.62 * beats + 40.4 * bsa - 0.51 * 30;
        double pp = sv / ((0.013 * 70 - 0.007 * 30 - 0.004 * beats) + 1.307);
        double mpp = 5 * 18.5;
        assertEquals((int) (mpp + pp), r.systolic, 0.0);     // 3 / 2 * PP == PP in the original (integer division)
        assertEquals((int) (mpp - pp / 3), r.diastolic, 0.0);
    }

    @Test
    public void noBloodPressureWithoutProfile() {
        BaselineEstimator.Result r = est.estimate(TestSignals.clean(1.2, 0.25, 30, 30, 1), null);
        assertTrue(Double.isNaN(r.systolic) && Double.isNaN(r.diastolic));
        assertTrue(!Double.isNaN(r.heartRate));
    }

    @Test
    public void tooFewSamplesGivesNothing() {
        BaselineEstimator.Result r = est.estimate(TestSignals.clean(1.2, 0.25, 1, 30, 1), profile);
        assertTrue(Double.isNaN(r.heartRate) && Double.isNaN(r.respiratoryRate) && Double.isNaN(r.spo2));
    }
}
