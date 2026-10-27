package com.example.yo7a.healthwatcher.ml;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Proves that the inference engine loads a model and returns correct numbers on THIS device.
 * Uses assets/selftest/runtime_selftest.tflite, a tiny non-health linear function
 * y = 0.5*x0 - x1 + 2*x2 + 0.25*x3 + 0.1. It can never act as a vital-sign estimator.
 */
public final class RuntimeSelfTest {
    public static final String ASSET = "selftest/runtime_selftest.tflite";
    private static final float[] W = {0.5f, -1f, 2f, 0.25f};
    private static final float BIAS = 0.1f;
    private static final float[][] INPUTS = {{1, 2, 3, 4}, {0, 0, 0, 0}, {-1, 0.5f, 2, 8}};
    private static final String TITLE = "On-device inference (TensorFlow Lite runtime)";

    private RuntimeSelfTest() {
    }

    public static CapabilityItem run(InputStream model, MLModelManager.RunnerFactory factory) {
        byte[] bytes;
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            int r;
            while ((r = model.read(chunk)) != -1) {
                bos.write(chunk, 0, r);
            }
            bytes = bos.toByteArray();
        } catch (IOException e) {
            return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL, "Self-test model could not be read: " + e.getMessage());
        }
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
        buf.put(bytes);
        buf.rewind();
        MLModelManager.ModelRunner runner = null;
        try {
            runner = factory.create(buf);
            if (runner.inputSize() != 4 || runner.outputSize() != 1) {
                return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL,
                        "Self-test model has unexpected tensor sizes " + runner.inputSize() + "->" + runner.outputSize());
            }
            for (float[] in : INPUTS) {
                float want = BIAS;
                for (int i = 0; i < 4; i++) {
                    want += W[i] * in[i];
                }
                float got = runner.run(in.clone())[0];
                if (Math.abs(got - want) > 1e-3f) {
                    return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL,
                            "Runtime returned " + got + " but " + want + " was expected");
                }
            }
            return new CapabilityItem(TITLE, CapabilityItem.Status.PASS,
                    "TensorFlow Lite loaded a test model and produced correct outputs on this device (3/3 vectors).");
        } catch (Throwable t) {
            return new CapabilityItem(TITLE, CapabilityItem.Status.FAIL,
                    "TensorFlow Lite could not run on this device: " + t.getClass().getSimpleName()
                            + (t.getMessage() != null ? " - " + t.getMessage() : ""));
        } finally {
            if (runner != null) {
                try {
                    runner.close();
                } catch (RuntimeException ignored) {
                    // ignore
                }
            }
        }
    }
}
