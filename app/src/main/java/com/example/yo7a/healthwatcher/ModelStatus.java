package com.example.yo7a.healthwatcher;

import android.content.Context;

import com.example.yo7a.healthwatcher.ml.AssetModelSource;
import com.example.yo7a.healthwatcher.ml.MLModelManager;
import com.example.yo7a.healthwatcher.ml.ScreeningClassifier;
import com.example.yo7a.healthwatcher.ml.TfliteModelRunner;
import com.example.yo7a.healthwatcher.ml.VitalTarget;

/** Builds the model manager used by the app and a human-readable summary of installed models. */
final class ModelStatus {
    private ModelStatus() {
    }

    static MLModelManager newManager(Context ctx) {
        return new MLModelManager(new AssetModelSource(ctx.getApplicationContext().getAssets()),
                new MLModelManager.RunnerFactory() {
                    @Override
                    public MLModelManager.ModelRunner create(java.nio.ByteBuffer model) {
                        return new TfliteModelRunner(model);
                    }
                });
    }

    /** Loads the health-screening classifier (never null; check isReady()). Blocking: call off the main thread. */
    static ScreeningClassifier newScreening(Context ctx) {
        return ScreeningClassifier.load(new AssetModelSource(ctx.getApplicationContext().getAssets()),
                new MLModelManager.RunnerFactory() {
                    @Override
                    public MLModelManager.ModelRunner create(java.nio.ByteBuffer model) {
                        return new TfliteModelRunner(model);
                    }
                });
    }

    /** Loads (and releases) the models to describe them. Blocking: call off the main thread. */
    static String describe(Context ctx) {
        MLModelManager m = newManager(ctx);
        try {
            m.loadAll();
            StringBuilder sb = new StringBuilder();
            for (VitalTarget t : VitalTarget.values()) {
                MLModelManager.ModelInfo i = m.info(t);
                sb.append(t.displayName).append(": ");
                switch (i.state) {
                    case READY:
                        sb.append(i.manifest.isValidated() ? "validated ML model " : "ML model (NOT validated, comparison only) ")
                                .append(i.manifest.modelVersion);
                        break;
                    case NOT_INSTALLED:
                        sb.append("no trained model installed - signal processing used");
                        break;
                    default:
                        sb.append("model unusable (").append(i.detail).append(") - signal processing used");
                        break;
                }
                sb.append('\n');
            }
            ScreeningClassifier sc = newScreening(ctx);
            try {
                sb.append("Health screening: ").append(sc.isReady() ? "ML classifier " + sc.version() + " (synthetic-data demo, NOT validated)"
                        : sc.statusDetail());
            } finally {
                sc.close();
            }
            return sb.toString().trim();
        } finally {
            m.close();
        }
    }
}
