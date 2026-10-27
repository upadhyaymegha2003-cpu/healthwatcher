package com.example.yo7a.healthwatcher;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import android.content.Context;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.yo7a.healthwatcher.ml.AssetModelSource;
import com.example.yo7a.healthwatcher.ml.MLModelManager;
import com.example.yo7a.healthwatcher.ml.TfliteModelRunner;
import com.example.yo7a.healthwatcher.ml.VitalTarget;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

/**
 * Proves that the model running in the app (TensorFlow Lite on Android) reproduces the development
 * (Python) model. It reads parity_<target>.csv (written by ml_training/export_parity.py: raw features and the
 * expected outputs in physical units) from the TEST assets and compares them with real on-device
 * inference of the model installed in assets/models/. A target with no installed model is skipped.
 */
public class ModelParityInstrumentedTest {

    private void check(VitalTarget t) throws Exception {
        Context app = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context test = InstrumentationRegistry.getInstrumentation().getContext();
        MLModelManager m = new MLModelManager(new AssetModelSource(app.getAssets()),
                new MLModelManager.RunnerFactory() {
                    @Override
                    public MLModelManager.ModelRunner create(ByteBuffer model) {
                        return new TfliteModelRunner(model);
                    }
                });
        try {
            m.loadAll();
            MLModelManager.ModelInfo info = m.info(t);
            assumeTrue("no model installed for " + t, info.state != MLModelManager.State.NOT_INSTALLED);
            assertEquals(info.detail, MLModelManager.State.READY, info.state);

            InputStream in;
            try {
                in = test.getAssets().open("parity_" + t.assetName + ".csv");
            } catch (java.io.IOException e) {
                assumeTrue("no parity vectors for " + t + " (run ml_training/export_parity.py)", false);
                return;
            }
            int n = info.manifest.inputSize();
            int k = info.manifest.outputNames.length;
            try (BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
                br.readLine();   // header
                String line;
                int rows = 0;
                while ((line = br.readLine()) != null) {
                    String[] p = line.split(",");
                    double[] x = new double[n];
                    for (int i = 0; i < n; i++) {
                        x[i] = Double.parseDouble(p[i]);
                    }
                    double[] y = m.infer(t, x);
                    for (int j = 0; j < k; j++) {
                        double want = Double.parseDouble(p[n + j]);
                        assertEquals("row " + rows + " output " + j, want, y[j], 1e-3 * Math.max(1.0, Math.abs(want)));
                    }
                    rows++;
                }
                assertEquals("parity file must contain vectors", true, rows > 0);
            }
        } finally {
            m.close();
        }
    }

    @Test
    public void heartRateParity() throws Exception {
        check(VitalTarget.HEART_RATE);
    }

    @Test
    public void respiratoryRateParity() throws Exception {
        check(VitalTarget.RESPIRATORY_RATE);
    }

    @Test
    public void bloodPressureParity() throws Exception {
        check(VitalTarget.BLOOD_PRESSURE);
    }

    @Test
    public void spo2Parity() throws Exception {
        check(VitalTarget.SPO2);
    }
}
