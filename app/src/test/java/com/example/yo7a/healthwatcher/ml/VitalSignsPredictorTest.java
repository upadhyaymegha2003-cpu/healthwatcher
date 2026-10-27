package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.EnumSet;
import java.util.Set;

public class VitalSignsPredictorTest {
    private static final int N = PPGFeatureExtractor.FEATURE_COUNT;
    private static final byte[] MODEL = {1, 2, 3};
    private final UserProfile profile = new UserProfile(30, 175, 70, 1);
    private final Set<VitalTarget> all = EnumSet.allOf(VitalTarget.class);

    private static MLModelManager managerFor(String assetTarget, String status, int outputs, String mean, String std,
                                             String min, String max, float bias) {
        FakeEngine.MemorySource src = new FakeEngine.MemorySource()
                .put(assetTarget + ".tflite", MODEL)
                .put(assetTarget + ".properties", FakeEngine.manifest(assetTarget, status, outputs, mean, std, min, max,
                        PPGFeatureExtractor.FEATURE_VERSION, null));
        float[] bias2 = new float[outputs];
        java.util.Arrays.fill(bias2, bias);
        MLModelManager m = new MLModelManager(src, FakeEngine.factory(new FakeEngine.LinearRunner(N, outputs, 0f, bias2)));
        m.loadAll();
        return m;
    }

    @Test
    public void withoutAnyModelTheBaselineIsUsedAndMlIsNotInstalled() {
        VitalSignsReport rep = new VitalSignsPredictor(null).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile, all);
        assertTrue(rep.quality.isUsable());
        assertTrue(rep.isSuccessful());
        VitalEstimate hr = rep.get(VitalTarget.HEART_RATE);
        assertTrue(hr.available);
        assertEquals(VitalEstimate.Source.SIGNAL_PROCESSING, hr.source);
        assertEquals(VitalEstimate.MlState.NOT_INSTALLED, hr.mlState);
        assertEquals(73, hr.value, 2.5);
        assertEquals("Signal processing", hr.sourceLabel());
        assertTrue(rep.get(VitalTarget.BLOOD_PRESSURE).caution().contains("not a validated"));
        assertTrue(rep.get(VitalTarget.SPO2).caution().contains("not a validated"));
    }

    @Test
    public void emptyManagerBehavesLikeNoModel() {
        MLModelManager m = new MLModelManager(new FakeEngine.MemorySource(), FakeEngine.factory(null));
        m.loadAll();
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile, all);
        assertEquals(VitalEstimate.MlState.NOT_INSTALLED, rep.get(VitalTarget.RESPIRATORY_RATE).mlState);
    }

    @Test
    public void unvalidatedModelIsOnlyAnExperimentalComparison() {
        MLModelManager m = managerFor("heart_rate", "UNVALIDATED", 1, "70", "10", "30", "220", 1.0f);   // -> 80 bpm
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile,
                EnumSet.of(VitalTarget.HEART_RATE));
        VitalEstimate hr = rep.get(VitalTarget.HEART_RATE);
        assertEquals(VitalEstimate.MlState.EXPERIMENTAL, hr.mlState);
        assertEquals(80.0, hr.mlValue, 1e-3);
        assertEquals("displayed value must remain the signal-processing one", VitalEstimate.Source.SIGNAL_PROCESSING, hr.source);
        assertEquals(hr.baselineValue, hr.value, 0.0);
        assertTrue(hr.mlNote().contains("NOT validated"));
    }

    @Test
    public void validatedModelBecomesThePrimaryResult() {
        MLModelManager m = managerFor("heart_rate", "VALIDATED", 1, "70", "10", "30", "220", 1.0f);
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile,
                EnumSet.of(VitalTarget.HEART_RATE));
        VitalEstimate hr = rep.get(VitalTarget.HEART_RATE);
        assertEquals(VitalEstimate.MlState.VALIDATED, hr.mlState);
        assertEquals(VitalEstimate.Source.ML_VALIDATED, hr.source);
        assertEquals(80.0, hr.value, 1e-3);
        assertEquals("Validated ML model", hr.sourceLabel());
        assertTrue(hr.mlNote().startsWith("Signal-processing comparison"));
    }

    @Test
    public void implausibleModelOutputIsRejectedAndBaselineKept() {
        MLModelManager m = managerFor("heart_rate", "VALIDATED", 1, "70", "10", "30", "220", 50f);      // -> 570 bpm
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile,
                EnumSet.of(VitalTarget.HEART_RATE));
        VitalEstimate hr = rep.get(VitalTarget.HEART_RATE);
        assertEquals(VitalEstimate.MlState.REJECTED, hr.mlState);
        assertEquals(VitalEstimate.Source.SIGNAL_PROCESSING, hr.source);
        assertTrue(hr.available);
    }

    @Test
    public void bloodPressureModelNeedsSystolicAboveDiastolic() {
        // two equal outputs -> systolic == diastolic -> rejected
        MLModelManager m = managerFor("blood_pressure", "VALIDATED", 2, "100,100", "10,10", "50,30", "250,150", 1f);
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile,
                EnumSet.of(VitalTarget.BLOOD_PRESSURE));
        assertEquals(VitalEstimate.MlState.REJECTED, rep.get(VitalTarget.BLOOD_PRESSURE).mlState);
    }

    @Test
    public void poorSignalProducesNoValuesAndAsksForRetry() {
        PpgSignal bad = TestSignals.withRed(TestSignals.clean(1.2, 0.25, 30, 30, 1), 60);
        MLModelManager m = managerFor("heart_rate", "VALIDATED", 1, "70", "10", "30", "220", 1.0f);
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(bad, profile, all);
        assertFalse(rep.quality.isUsable());
        assertFalse(rep.isSuccessful());
        for (VitalEstimate e : rep.all().values()) {
            assertFalse("no value may be shown for a rejected recording", e.available);
            assertTrue(Double.isNaN(e.value));
            assertTrue(Double.isNaN(e.mlValue));
        }
        assertTrue(rep.failureMessage().length() > 10);
        assertEquals(all.size(), rep.unavailableTargets().size());
    }

    @Test
    public void unavailableModelStateIsSurfacedWithoutBreakingTheBaseline() {
        // manifest present but engine refuses the file
        FakeEngine.MemorySource src = new FakeEngine.MemorySource()
                .put("heart_rate.tflite", MODEL)
                .put("heart_rate.properties", FakeEngine.manifest("heart_rate", "VALIDATED", 1, "70", "10", "30", "220",
                        PPGFeatureExtractor.FEATURE_VERSION, null));
        MLModelManager m = new MLModelManager(src, new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(java.nio.ByteBuffer model) {
                throw new IllegalStateException("bad model");
            }
        });
        m.loadAll();
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile,
                EnumSet.of(VitalTarget.HEART_RATE));
        VitalEstimate hr = rep.get(VitalTarget.HEART_RATE);
        assertEquals(VitalEstimate.MlState.UNAVAILABLE, hr.mlState);
        assertTrue(hr.available);
        assertTrue(hr.mlNote().startsWith("ML model unavailable"));
    }

    @Test
    public void bloodPressureModelWithDemographicsGetsProfileValues() {
        String mf = FakeEngine.manifest("blood_pressure", "UNVALIDATED", 2, "110,70", "10,10", "50,30", "250,150",
                PPGFeatureExtractor.FEATURE_VERSION, null);
        mf = mf.replace("feature.names=", "feature.names=demo_age,demo_gender,").replace("norm.mean=", "norm.mean=0,0,")
                .replace("norm.std=", "norm.std=1,1,");
        FakeEngine.LinearRunner runner = new FakeEngine.LinearRunner(N + 2, 2, 0f, new float[]{1f, 0f});
        FakeEngine.MemorySource src = new FakeEngine.MemorySource().put("blood_pressure.tflite", MODEL).put("blood_pressure.properties", mf);
        MLModelManager m = new MLModelManager(src, FakeEngine.factory(runner));
        m.loadAll();
        assertTrue(m.info(VitalTarget.BLOOD_PRESSURE).detail, m.info(VitalTarget.BLOOD_PRESSURE).isReady());
        VitalSignsReport rep = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), profile,
                EnumSet.of(VitalTarget.BLOOD_PRESSURE));
        assertEquals(30f, runner.lastInput[0], 0f);    // demo_age
        assertEquals(1f, runner.lastInput[1], 0f);     // demo_gender
        VitalEstimate bp = rep.get(VitalTarget.BLOOD_PRESSURE);
        assertEquals(VitalEstimate.MlState.EXPERIMENTAL, bp.mlState);
        assertEquals(120.0, bp.mlValue, 1e-3);
        assertEquals(70.0, bp.mlValue2, 1e-3);

        // without a complete profile the model reports UNAVAILABLE instead of guessing
        VitalSignsReport rep2 = new VitalSignsPredictor(m).predict(TestSignals.clean(1.2, 0.25, 30, 30, 1), null,
                EnumSet.of(VitalTarget.BLOOD_PRESSURE));
        assertEquals(VitalEstimate.MlState.UNAVAILABLE, rep2.get(VitalTarget.BLOOD_PRESSURE).mlState);
    }

    @Test
    public void recorderProducesSortedSnapshotsAndRecentMeans() {
        PpgRecorder rec = new PpgRecorder();
        assertTrue(Double.isNaN(rec.recentMeans(1.0)[0]));
        for (int i = 0; i < 100; i++) {
            assertTrue(rec.add(5.0 + i * 0.1, 200 + i, 100, 50));
        }
        assertFalse("non-advancing timestamps are ignored", rec.add(5.0 + 99 * 0.1, 1, 1, 1));
        assertEquals(100, rec.size());
        PpgSignal s = rec.snapshot();
        assertEquals(0.0, s.t[0], 0.0);
        assertEquals(9.9, s.durationSec(), 1e-9);
        assertEquals(10.1, s.meanSamplingHz(), 0.1);
        assertEquals(298.0, rec.recentMeans(0.4)[0] + 0.0 - 0.0 + 0.0 * 0, 3.0);
        rec.clear();
        assertEquals(0, rec.size());
        assertTrue(s.toCsv().startsWith("t_sec,red,green,blue"));
    }
}
