package com.example.yo7a.healthwatcher.ml;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * Test doubles for the model plumbing. {@code LinearRunner} is a tiny deterministic linear
 * function used ONLY in unit tests to verify normalization / de-normalization / selection logic;
 * it is not shipped and is never used as an estimator.
 */
final class FakeEngine {
    private FakeEngine() {
    }

    static final class MemorySource implements MLModelManager.ModelSource {
        final Map<String, byte[]> files = new HashMap<>();

        MemorySource put(String name, byte[] data) {
            files.put(name, data);
            return this;
        }

        MemorySource put(String name, String text) {
            return put(name, text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }

        @Override
        public boolean exists(String fileName) {
            return files.containsKey(fileName);
        }

        @Override
        public InputStream open(String fileName) throws IOException {
            byte[] b = files.get(fileName);
            if (b == null) {
                throw new IOException("missing " + fileName);
            }
            return new ByteArrayInputStream(b);
        }
    }

    /** y_j = bias_j + sum_i w[i] * x[i]; records the last input it saw. */
    static class LinearRunner implements MLModelManager.ModelRunner {
        final int in, out;
        final float[] w;
        final float[] bias;
        float[] lastInput;
        boolean closed;

        LinearRunner(int in, int out, float w, float[] bias) {
            this.in = in;
            this.out = out;
            this.w = new float[in];
            java.util.Arrays.fill(this.w, w);
            this.bias = bias;
        }

        @Override
        public int inputSize() {
            return in;
        }

        @Override
        public int outputSize() {
            return out;
        }

        @Override
        public float[] run(float[] input) {
            lastInput = input.clone();
            float s = 0;
            for (int i = 0; i < in; i++) {
                s += w[i] * input[i];
            }
            float[] y = new float[out];
            for (int j = 0; j < out; j++) {
                y[j] = s + bias[j];
            }
            return y;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    static MLModelManager.RunnerFactory factory(final MLModelManager.ModelRunner runner) {
        return new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(ByteBuffer model) {
                return runner;
            }
        };
    }

    static String manifest(String target, String status, int outputs, String outMean, String outStd,
                           String outMin, String outMax, String featureVersion, String sha) {
        StringBuilder names = new StringBuilder();
        StringBuilder mean = new StringBuilder();
        StringBuilder std = new StringBuilder();
        for (int i = 0; i < PPGFeatureExtractor.FEATURE_COUNT; i++) {
            if (i > 0) {
                names.append(',');
                mean.append(',');
                std.append(',');
            }
            names.append(PPGFeatureExtractor.FEATURE_NAMES[i]);
            mean.append("0");
            std.append("1");
        }
        StringBuilder on = new StringBuilder();
        for (int j = 0; j < outputs; j++) {
            on.append(j > 0 ? "," : "").append("out").append(j);
        }
        return "schema=1\ntarget=" + target + "\nmodel.name=test_model\nmodel.version=v-test\n"
                + (sha == null ? "" : "model.sha256=" + sha + "\n")
                + "feature.version=" + featureVersion + "\nfeature.names=" + names + "\n"
                + "norm.mean=" + mean + "\nnorm.std=" + std + "\noutput.names=" + on + "\n"
                + "output.mean=" + outMean + "\noutput.std=" + outStd + "\n"
                + "output.min=" + outMin + "\noutput.max=" + outMax + "\n"
                + "validation.status=" + status + "\ntraining.description=unit test\n";
    }
}
