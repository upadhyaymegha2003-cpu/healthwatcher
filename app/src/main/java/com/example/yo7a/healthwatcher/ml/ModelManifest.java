package com.example.yo7a.healthwatcher.ml;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Properties;
import java.util.TreeMap;
import java.util.Map;

/**
 * Metadata that ships next to every model ({@code <target>.properties}). It carries the exact
 * normalization parameters used in training, the feature layout, plausibility limits and the
 * validation status. Parsed with plain {@link Properties} so it also works in JVM unit tests.
 */
public final class ModelManifest {
    public static final String STATUS_UNVALIDATED = "UNVALIDATED";
    public static final String STATUS_VALIDATED = "VALIDATED";
    public static final String STATUS_SMOKE_TEST = "SMOKE_TEST";

    public static final class FormatException extends Exception {
        private static final long serialVersionUID = 1L;

        public FormatException(String message) {
            super(message);
        }
    }

    public final String target;
    public final String modelName;
    public final String modelVersion;
    public final String sha256;                // optional, hex of the .tflite file
    public final String featureVersion;
    public final String[] featureNames;
    public final double[] normMean;
    public final double[] normStd;
    public final String[] outputNames;
    public final double[] outputMean;
    public final double[] outputStd;
    public final double[] outputMin;
    public final double[] outputMax;
    public final String validationStatus;
    public final String trainingDescription;
    public final Map<String, String> metrics = new TreeMap<>();

    private ModelManifest(Properties p) throws FormatException {
        target = req(p, "target");
        modelName = req(p, "model.name");
        modelVersion = req(p, "model.version");
        sha256 = p.getProperty("model.sha256", "").trim().toLowerCase();
        featureVersion = req(p, "feature.version");
        featureNames = csv(req(p, "feature.names"));
        normMean = nums(req(p, "norm.mean"), "norm.mean");
        normStd = nums(req(p, "norm.std"), "norm.std");
        outputNames = csv(req(p, "output.names"));
        int k = outputNames.length;
        outputMean = p.containsKey("output.mean") ? nums(p.getProperty("output.mean"), "output.mean") : new double[k];
        double[] os = new double[k];
        Arrays.fill(os, 1.0);
        outputStd = p.containsKey("output.std") ? nums(p.getProperty("output.std"), "output.std") : os;
        outputMin = nums(req(p, "output.min"), "output.min");
        outputMax = nums(req(p, "output.max"), "output.max");
        validationStatus = req(p, "validation.status").trim().toUpperCase();
        trainingDescription = p.getProperty("training.description", "");

        if (featureNames.length == 0 || normMean.length != featureNames.length || normStd.length != featureNames.length) {
            throw new FormatException("feature.names, norm.mean and norm.std must have the same non-zero length");
        }
        for (double s : normStd) {
            if (!(s > 0)) {
                throw new FormatException("norm.std values must be > 0");
            }
        }
        for (double s : outputStd) {
            if (!(s > 0)) {
                throw new FormatException("output.std values must be > 0");
            }
        }
        if (outputMean.length != k || outputStd.length != k || outputMin.length != k || outputMax.length != k) {
            throw new FormatException("output.* arrays must match output.names length");
        }
        for (String key : p.stringPropertyNames()) {
            if (key.startsWith("metrics.")) {
                metrics.put(key.substring("metrics.".length()), p.getProperty(key));
            }
        }
    }

    public static ModelManifest parse(InputStream in) throws IOException, FormatException {
        Properties p = new Properties();
        p.load(in);
        return new ModelManifest(p);
    }

    public int inputSize() {
        return featureNames.length;
    }

    public boolean isValidated() {
        return STATUS_VALIDATED.equals(validationStatus);
    }

    /** Verifies feature names/version against what this build of the app can produce. */
    public void checkCompatibleWithApp(boolean demographicsAllowed) throws FormatException {
        if (!PPGFeatureExtractor.FEATURE_VERSION.equals(featureVersion)) {
            throw new FormatException("Model expects feature set '" + featureVersion
                    + "' but the app provides '" + PPGFeatureExtractor.FEATURE_VERSION + "'");
        }
        for (String name : featureNames) {
            if (PPGFeatureExtractor.indexOf(name) < 0 && !isDemographic(name)) {
                throw new FormatException("Unknown input feature: " + name);
            }
            if (isDemographic(name) && !demographicsAllowed) {
                throw new FormatException("Demographic input '" + name + "' is not allowed for this target");
            }
        }
    }

    public static boolean isDemographic(String name) {
        return "demo_age".equals(name) || "demo_height_cm".equals(name)
                || "demo_weight_kg".equals(name) || "demo_gender".equals(name);
    }

    public boolean usesDemographics() {
        for (String n : featureNames) {
            if (isDemographic(n)) {
                return true;
            }
        }
        return false;
    }

    private static String req(Properties p, String key) throws FormatException {
        String v = p.getProperty(key);
        if (v == null || v.trim().isEmpty()) {
            throw new FormatException("Missing manifest key: " + key);
        }
        return v.trim();
    }

    private static String[] csv(String s) {
        String[] parts = s.split(",");
        for (int i = 0; i < parts.length; i++) {
            parts[i] = parts[i].trim();
        }
        return parts;
    }

    private static double[] nums(String s, String key) throws FormatException {
        String[] parts = csv(s);
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                out[i] = Double.parseDouble(parts[i]);
            } catch (NumberFormatException e) {
                throw new FormatException("Bad number in " + key + ": " + parts[i]);
            }
            if (Double.isNaN(out[i]) || Double.isInfinite(out[i])) {
                throw new FormatException("Non-finite number in " + key);
            }
        }
        return out;
    }
}
