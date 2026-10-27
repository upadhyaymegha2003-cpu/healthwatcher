package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;

/** Logic around the screening model (gating, safety net, feature engineering) with a scripted runner. */
public class ScreeningClassifierTest {
    private static final String PROPS = "classes=Within_demo_ranges,Needs_review,High_priority_review\n"
            + "input.size=17\nmodel.version=test\n";
    private static final UserProfile ADULT = new UserProfile(40, 170, 70, 1);

    private static ScreeningClassifier classifier(final float[] out) {
        return ScreeningClassifier.load(new MLModelManager.ModelSource() {
            @Override
            public boolean exists(String f) {
                return true;
            }

            @Override
            public InputStream open(String f) {
                return new ByteArrayInputStream(f.endsWith(".properties") ? PROPS.getBytes() : new byte[]{1});
            }
        }, new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(ByteBuffer model) {
                return new MLModelManager.ModelRunner() {
                    @Override
                    public int inputSize() {
                        return 17;
                    }

                    @Override
                    public int outputSize() {
                        return 3;
                    }

                    @Override
                    public float[] run(float[] input) {
                        return out;
                    }

                    @Override
                    public void close() {
                    }
                };
            }
        });
    }

    @Test
    public void engineeredFeaturesMatchPythonTrainingCode() {
        // expected values computed with train_screening.engineer() for the same raw inputs
        float[] f = ScreeningClassifier.engineer(50, true, 110, 150, 95, 22, 92);
        float[] expected = {50, 1, 110, 150, 95, 22, 92, 10, 11, 6, 2, 3, 0, 0, 0, 0, 0};
        assertEquals(17, f.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals("feature " + i, expected[i], f[i], 1e-6);
        }
        float[] x = ScreeningClassifier.engineer(70, false, 150, 200, 125, 35, 85);
        assertEquals(10, x[12], 1e-6);
        assertEquals(21, x[13], 1e-6);
        assertEquals(6, x[14], 1e-6);
        assertEquals(5, x[15], 1e-6);
        assertEquals(5, x[16], 1e-6);
    }

    @Test
    public void healthyReadingsLookGood() {
        ScreeningResult r = classifier(new float[]{0.95f, 0.04f, 0.01f}).classify(ADULT, 72, 118, 76, 16, 98);
        assertTrue(r.available);
        assertEquals(ScreeningResult.Verdict.WITHIN_RANGES, r.verdict);
        assertTrue(r.findings.isEmpty());
        assertFalse(r.raisedByRangeCheck);
    }

    @Test
    public void modelSaysGoodButReadingIsOutOfRange_verdictIsRaised() {
        ScreeningResult r = classifier(new float[]{0.9f, 0.09f, 0.01f}).classify(ADULT, 72, 150, 95, 16, 98);
        assertEquals(ScreeningResult.Verdict.NEEDS_REVIEW, r.verdict);
        assertTrue(r.raisedByRangeCheck);
        assertTrue(r.findings.get(0).contains("hypertension"));
    }

    @Test
    public void extremeReadingIsHighPriorityEvenIfModelIsLessSure() {
        ScreeningResult r = classifier(new float[]{0.0f, 0.7f, 0.3f}).classify(ADULT, 72, 118, 76, 16, 85);
        assertEquals(ScreeningResult.Verdict.HIGH_PRIORITY, r.verdict);
        assertTrue(r.raisedByRangeCheck);
    }

    @Test
    public void childrenAndMissingProfileAreNotScreened() {
        ScreeningClassifier c = classifier(new float[]{1f, 0f, 0f});
        assertFalse(c.classify(new UserProfile(12, 150, 40, 2), 72, 118, 76, 16, 98).available);
        assertFalse(c.classify(null, 72, 118, 76, 16, 98).available);
    }

    @Test
    public void invalidModelOutputIsRejected() {
        assertFalse(classifier(new float[]{0.2f, 0.2f, 0.2f}).classify(ADULT, 72, 118, 76, 16, 98).available);
        assertFalse(classifier(new float[]{Float.NaN, 0.5f, 0.5f}).classify(ADULT, 72, 118, 76, 16, 98).available);
    }

    @Test
    public void missingModelFilesGiveUnavailableNotACrash() {
        ScreeningClassifier c = ScreeningClassifier.load(new MLModelManager.ModelSource() {
            @Override
            public boolean exists(String f) {
                return false;
            }

            @Override
            public InputStream open(String f) {
                throw new IllegalStateException();
            }
        }, null);
        assertFalse(c.isReady());
        assertFalse(c.classify(ADULT, 72, 118, 76, 16, 98).available);
    }
}
