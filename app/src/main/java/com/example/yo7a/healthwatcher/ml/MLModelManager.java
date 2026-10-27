package com.example.yo7a.healthwatcher.ml;

import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.EnumMap;
import java.util.Map;

/**
 * Loads, validates, runs and releases the on-device models (one per {@link VitalTarget}).
 *
 * Each target needs two files in {@code assets/models/}: {@code <target>.tflite} and
 * {@code <target>.properties} (see {@link ModelManifest}). If they are absent the target is simply
 * {@link State#NOT_INSTALLED} and the app keeps using the signal-processing baseline -- nothing is faked.
 *
 * The class has no Android dependency: files come from a {@link ModelSource} and the inference
 * engine from a {@link RunnerFactory}, so the production wiring (assets + TensorFlow Lite) lives in
 * {@link AssetModelSource} / {@link TfliteModelRunner} and the logic here is unit-testable on a JVM.
 */
public final class MLModelManager implements Closeable {

    public enum State {NOT_INSTALLED, LOAD_FAILED, INCOMPATIBLE, READY}

    /** Access to model files. */
    public interface ModelSource {
        boolean exists(String fileName);

        InputStream open(String fileName) throws IOException;
    }

    /** One loaded inference engine for a single model file. */
    public interface ModelRunner extends Closeable {
        int inputSize();

        int outputSize();

        /** Runs the model on already-normalized features and returns the raw model outputs. */
        float[] run(float[] input);

        @Override
        void close();
    }

    public interface RunnerFactory {
        ModelRunner create(ByteBuffer model) throws Exception;
    }

    /** Immutable view of a model slot. */
    public static final class ModelInfo {
        public final VitalTarget target;
        public final State state;
        public final String detail;
        public final ModelManifest manifest;     // null unless the manifest could be parsed
        private final ModelRunner runner;

        ModelInfo(VitalTarget target, State state, String detail, ModelManifest manifest, ModelRunner runner) {
            this.target = target;
            this.state = state;
            this.detail = detail;
            this.manifest = manifest;
            this.runner = runner;
        }

        public boolean isReady() {
            return state == State.READY;
        }
    }

    private final ModelSource source;
    private final RunnerFactory factory;
    private final Map<VitalTarget, ModelInfo> models = new EnumMap<>(VitalTarget.class);
    private boolean loaded = false;
    private boolean closed = false;

    public MLModelManager(ModelSource source, RunnerFactory factory) {
        this.source = source;
        this.factory = factory;
    }

    /** Loads every model that is present. Safe to call repeatedly; never throws. */
    public synchronized void loadAll() {
        if (loaded || closed) {
            return;
        }
        for (VitalTarget t : VitalTarget.values()) {
            models.put(t, load(t));
        }
        loaded = true;
    }

    public synchronized ModelInfo info(VitalTarget t) {
        ModelInfo mi = models.get(t);
        if (mi == null) {
            return new ModelInfo(t, State.NOT_INSTALLED, "Models have not been loaded", null, null);
        }
        return mi;
    }

    /**
     * Runs the model for {@code t}. {@code features} must already be arranged in the manifest's
     * feature order; normalization (mean/std from the manifest) is applied here, and the raw outputs
     * are de-standardised with the manifest's output mean/std.
     */
    public synchronized double[] infer(VitalTarget t, double[] features) throws IOException {
        ModelInfo mi = models.get(t);
        if (closed || mi == null || !mi.isReady()) {
            throw new IOException("Model for " + t + " is not ready");
        }
        ModelManifest m = mi.manifest;
        if (features.length != m.inputSize()) {
            throw new IOException("Expected " + m.inputSize() + " features but got " + features.length);
        }
        float[] x = new float[features.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) ((features[i] - m.normMean[i]) / m.normStd[i]);
        }
        float[] y = mi.runner.run(x);
        if (y == null || y.length != m.outputNames.length) {
            throw new IOException("Unexpected model output size");
        }
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i] * m.outputStd[i] + m.outputMean[i];
        }
        return out;
    }

    @Override
    public synchronized void close() {
        closed = true;
        for (ModelInfo mi : models.values()) {
            if (mi.runner != null) {
                try {
                    mi.runner.close();
                } catch (RuntimeException ignored) {
                    // releasing native resources must never crash the UI
                }
            }
        }
        models.clear();
    }

    // ------------------------------------------------------------------------------------------

    private ModelInfo load(VitalTarget t) {
        String modelFile = t.assetName + ".tflite";
        String manifestFile = t.assetName + ".properties";
        boolean hasModel;
        boolean hasManifest;
        try {
            hasModel = source.exists(modelFile);
            hasManifest = source.exists(manifestFile);
        } catch (RuntimeException e) {
            return new ModelInfo(t, State.LOAD_FAILED, "Cannot read model folder: " + e.getMessage(), null, null);
        }
        if (!hasModel && !hasManifest) {
            return new ModelInfo(t, State.NOT_INSTALLED, "No trained model installed", null, null);
        }
        if (hasModel != hasManifest) {
            return new ModelInfo(t, State.LOAD_FAILED,
                    "Both " + modelFile + " and " + manifestFile + " are required", null, null);
        }

        ModelManifest manifest;
        try (InputStream in = source.open(manifestFile)) {
            manifest = ModelManifest.parse(in);
        } catch (IOException | ModelManifest.FormatException | RuntimeException e) {
            return new ModelInfo(t, State.INCOMPATIBLE, "Invalid manifest: " + e.getMessage(), null, null);
        }
        if (!manifest.target.equals(t.assetName)) {
            return new ModelInfo(t, State.INCOMPATIBLE,
                    "Manifest target '" + manifest.target + "' does not match " + t.assetName, manifest, null);
        }
        if (ModelManifest.STATUS_SMOKE_TEST.equals(manifest.validationStatus)) {
            return new ModelInfo(t, State.INCOMPATIBLE,
                    "Refused: model was produced by the synthetic pipeline smoke test", manifest, null);
        }
        if (!ModelManifest.STATUS_UNVALIDATED.equals(manifest.validationStatus)
                && !manifest.isValidated()) {
            return new ModelInfo(t, State.INCOMPATIBLE,
                    "Unknown validation status: " + manifest.validationStatus, manifest, null);
        }
        try {
            manifest.checkCompatibleWithApp(t == VitalTarget.BLOOD_PRESSURE);
        } catch (ModelManifest.FormatException e) {
            return new ModelInfo(t, State.INCOMPATIBLE, e.getMessage(), manifest, null);
        }
        if (manifest.outputNames.length != t.outputCount()) {
            return new ModelInfo(t, State.INCOMPATIBLE,
                    "Expected " + t.outputCount() + " output(s) for " + t, manifest, null);
        }

        byte[] bytes;
        try (InputStream in = source.open(modelFile)) {
            bytes = readAll(in);
        } catch (IOException | RuntimeException e) {
            return new ModelInfo(t, State.LOAD_FAILED, "Cannot read model file: " + e.getMessage(), manifest, null);
        }
        if (!manifest.sha256.isEmpty()) {
            String actual = sha256(bytes);
            if (!manifest.sha256.equals(actual)) {
                return new ModelInfo(t, State.LOAD_FAILED,
                        "Model file does not match the checksum in its manifest", manifest, null);
            }
        }
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
        buf.put(bytes);
        buf.rewind();

        ModelRunner runner = null;
        try {
            runner = factory.create(buf);
            if (runner.inputSize() != manifest.inputSize()) {
                String d = "Model input size " + runner.inputSize() + " != manifest " + manifest.inputSize();
                runner.close();
                return new ModelInfo(t, State.INCOMPATIBLE, d, manifest, null);
            }
            if (runner.outputSize() != manifest.outputNames.length) {
                String d = "Model output size " + runner.outputSize() + " != manifest " + manifest.outputNames.length;
                runner.close();
                return new ModelInfo(t, State.INCOMPATIBLE, d, manifest, null);
            }
        } catch (Throwable e) {   // UnsatisfiedLinkError on unsupported ABIs, IllegalArgumentException on bad flatbuffers...
            if (runner != null) {
                try {
                    runner.close();
                } catch (RuntimeException ignored) {
                    // ignore
                }
            }
            return new ModelInfo(t, State.LOAD_FAILED,
                    "Inference engine could not load the model: " + e.getClass().getSimpleName()
                            + (e.getMessage() != null ? " - " + e.getMessage() : ""), manifest, null);
        }
        return new ModelInfo(t, State.READY, "Loaded " + manifest.modelName + " " + manifest.modelVersion,
                manifest, runner);
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream(64 * 1024);
        byte[] chunk = new byte[16 * 1024];
        int r;
        while ((r = in.read(chunk)) != -1) {
            bos.write(chunk, 0, r);
        }
        return bos.toByteArray();
    }

    static String sha256(byte[] data) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(data);
            StringBuilder sb = new StringBuilder();
            for (byte b : d) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
