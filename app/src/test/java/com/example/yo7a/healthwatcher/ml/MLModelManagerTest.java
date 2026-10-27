package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class MLModelManagerTest {
    private static final int N = PPGFeatureExtractor.FEATURE_COUNT;
    private static final byte[] MODEL_BYTES = {1, 2, 3, 4, 5, 6, 7, 8};

    private static MLModelManager.ModelInfo loadHr(FakeEngine.MemorySource src, MLModelManager.RunnerFactory f) {
        MLModelManager m = new MLModelManager(src, f);
        m.loadAll();
        return m.info(VitalTarget.HEART_RATE);
    }

    private static String hrManifest(String status, String fv, String sha) {
        return FakeEngine.manifest("heart_rate", status, 1, "70", "10", "30", "220", fv, sha);
    }

    @Test
    public void missingModelsAreNotInstalledNotFaked() {
        MLModelManager m = new MLModelManager(new FakeEngine.MemorySource(), FakeEngine.factory(null));
        m.loadAll();
        for (VitalTarget t : VitalTarget.values()) {
            assertEquals(MLModelManager.State.NOT_INSTALLED, m.info(t).state);
        }
    }

    @Test
    public void infoBeforeLoadIsNotInstalled() {
        MLModelManager m = new MLModelManager(new FakeEngine.MemorySource(), FakeEngine.factory(null));
        assertEquals(MLModelManager.State.NOT_INSTALLED, m.info(VitalTarget.SPO2).state);
    }

    @Test
    public void normalizationAndOutputScalingUseManifestParameters() throws Exception {
        // norm: (x - mean)/std with mean=0,std=1 for all but feature 0 -> mean 5, std 2
        String mf = hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null)
                .replace("norm.mean=0,", "norm.mean=5,").replace("norm.std=1,", "norm.std=2,");
        FakeEngine.LinearRunner runner = new FakeEngine.LinearRunner(N, 1, 0f, new float[]{0.5f});
        FakeEngine.MemorySource src = new FakeEngine.MemorySource()
                .put("heart_rate.tflite", MODEL_BYTES).put("heart_rate.properties", mf);
        MLModelManager m = new MLModelManager(src, FakeEngine.factory(runner));
        m.loadAll();
        assertTrue(m.info(VitalTarget.HEART_RATE).detail, m.info(VitalTarget.HEART_RATE).isReady());

        double[] x = new double[N];
        x[0] = 9;
        x[1] = 3;
        double[] y = m.infer(VitalTarget.HEART_RATE, x);
        assertEquals((9 - 5) / 2.0f, runner.lastInput[0], 1e-6);   // normalised with manifest mean/std
        assertEquals(3.0f, runner.lastInput[1], 1e-6);
        assertEquals(0.5 * 10 + 70, y[0], 1e-4);                     // de-standardised: out*std+mean
    }

    @Test
    public void inferOnNotReadyOrWrongSizeThrows() throws Exception {
        MLModelManager m = new MLModelManager(new FakeEngine.MemorySource(), FakeEngine.factory(null));
        m.loadAll();
        try {
            m.infer(VitalTarget.HEART_RATE, new double[N]);
            fail();
        } catch (IOException expected) {
            // ok
        }
        FakeEngine.MemorySource src = new FakeEngine.MemorySource()
                .put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null));
        MLModelManager m2 = new MLModelManager(src, FakeEngine.factory(new FakeEngine.LinearRunner(N, 1, 0f, new float[1])));
        m2.loadAll();
        try {
            m2.infer(VitalTarget.HEART_RATE, new double[N - 1]);
            fail();
        } catch (IOException expected) {
            // ok
        }
    }

    @Test
    public void oneFileWithoutTheOtherFailsToLoad() {
        FakeEngine.MemorySource src = new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES);
        assertEquals(MLModelManager.State.LOAD_FAILED,
                loadHr(src, FakeEngine.factory(null)).state);
    }

    @Test
    public void incompatibleManifestsAreRefused() {
        FakeEngine.LinearRunner r = new FakeEngine.LinearRunner(N, 1, 0f, new float[1]);
        // wrong feature version
        MLModelManager.ModelInfo a = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", "ppgfeat-999", null)), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, a.state);
        // synthetic smoke-test models can never be loaded by the app
        MLModelManager.ModelInfo b = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("SMOKE_TEST", PPGFeatureExtractor.FEATURE_VERSION, null)), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, b.state);
        assertTrue(b.detail.toLowerCase().contains("smoke"));
        // unknown status
        MLModelManager.ModelInfo c = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("GREAT", PPGFeatureExtractor.FEATURE_VERSION, null)), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, c.state);
        // garbage manifest
        MLModelManager.ModelInfo d = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", "not a manifest"), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, d.state);
        // manifest for another target
        MLModelManager.ModelInfo e = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", FakeEngine.manifest("spo2", "UNVALIDATED", 1, "0", "1", "0", "100",
                        PPGFeatureExtractor.FEATURE_VERSION, null)), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, e.state);
        // demographic inputs are rejected for non-BP targets
        String demo = hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null)
                .replace("g_dom_freq_hz,", "demo_age,");
        MLModelManager.ModelInfo f = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", demo), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, f.state);
    }

    @Test
    public void inputSizeMismatchWithManifestIsIncompatibleAndReleased() {
        FakeEngine.LinearRunner r = new FakeEngine.LinearRunner(N + 3, 1, 0f, new float[1]);
        MLModelManager.ModelInfo i = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null)), FakeEngine.factory(r));
        assertEquals(MLModelManager.State.INCOMPATIBLE, i.state);
        assertTrue("runner must be closed when rejected", r.closed);
    }

    @Test
    public void checksumMismatchIsDetected() throws Exception {
        String good = MLModelManager.sha256(MODEL_BYTES);
        FakeEngine.LinearRunner r = new FakeEngine.LinearRunner(N, 1, 0f, new float[1]);
        assertTrue(loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, good)), FakeEngine.factory(r)).isReady());
        assertEquals(MLModelManager.State.LOAD_FAILED,
                loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", new byte[]{9, 9, 9})
                        .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, good)),
                        FakeEngine.factory(new FakeEngine.LinearRunner(N, 1, 0f, new float[1]))).state);
    }

    @Test
    public void engineFailureIsReportedNotThrown() {
        MLModelManager.RunnerFactory boom = new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(ByteBuffer model) {
                throw new IllegalArgumentException("ByteBuffer is not a valid flatbuffer model");
            }
        };
        MLModelManager.ModelInfo i = loadHr(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null)), boom);
        assertEquals(MLModelManager.State.LOAD_FAILED, i.state);
        assertTrue(i.detail.contains("IllegalArgumentException"));

        MLModelManager.RunnerFactory noNative = new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(ByteBuffer model) {
                throw new UnsatisfiedLinkError("no tensorflowlite_jni for this ABI");
            }
        };
        assertEquals(MLModelManager.State.LOAD_FAILED, loadHr(new FakeEngine.MemorySource()
                .put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null)), noNative).state);
    }

    @Test
    public void closeReleasesRunnersAndBlocksFurtherUse() throws Exception {
        FakeEngine.LinearRunner r = new FakeEngine.LinearRunner(N, 1, 0f, new float[1]);
        MLModelManager m = new MLModelManager(new FakeEngine.MemorySource().put("heart_rate.tflite", MODEL_BYTES)
                .put("heart_rate.properties", hrManifest("UNVALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null)),
                FakeEngine.factory(r));
        m.loadAll();
        m.close();
        assertTrue(r.closed);
        m.loadAll();   // no-op after close
        try {
            m.infer(VitalTarget.HEART_RATE, new double[N]);
            fail();
        } catch (IOException expected) {
            // ok
        }
    }

    @Test
    public void manifestParsingAndValidation() throws Exception {
        ModelManifest mf = ModelManifest.parse(new ByteArrayInputStream(
                hrManifest("VALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null).getBytes("UTF-8")));
        assertTrue(mf.isValidated());
        assertEquals(N, mf.inputSize());
        assertFalse(mf.usesDemographics());
        assertNotNull(mf.metrics);
        assertArrayEquals(new double[]{70}, mf.outputMean, 0);
        try {
            ModelManifest.parse(new ByteArrayInputStream(
                    hrManifest("VALIDATED", PPGFeatureExtractor.FEATURE_VERSION, null).replace("output.std=10", "output.std=0").getBytes("UTF-8")));
            fail("zero std must be rejected");
        } catch (ModelManifest.FormatException expected) {
            // ok
        }
    }
}
