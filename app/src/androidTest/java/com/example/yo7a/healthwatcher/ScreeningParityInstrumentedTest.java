package com.example.yo7a.healthwatcher;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.yo7a.healthwatcher.ml.AssetModelSource;
import com.example.yo7a.healthwatcher.ml.MLModelManager;
import com.example.yo7a.healthwatcher.ml.ScreeningClassifier;
import com.example.yo7a.healthwatcher.ml.ScreeningResult;
import com.example.yo7a.healthwatcher.ml.TfliteModelRunner;
import com.example.yo7a.healthwatcher.ml.UserProfile;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

/** On-device TFLite screening model must reproduce the Python model (screening_parity.csv from train_screening.py). */
public class ScreeningParityInstrumentedTest {
    @Test
    public void onDeviceModelMatchesPython() throws Exception {
        Context app = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context test = InstrumentationRegistry.getInstrumentation().getContext();
        ScreeningClassifier c = ScreeningClassifier.load(new AssetModelSource(app.getAssets()),
                new MLModelManager.RunnerFactory() {
                    @Override
                    public MLModelManager.ModelRunner create(ByteBuffer model) {
                        return new TfliteModelRunner(model);
                    }
                });
        try {
            assertTrue(c.statusDetail(), c.isReady());
            int rows = 0;
            try (InputStream in = test.getAssets().open("screening_parity.csv");
                 BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
                br.readLine();
                String line;
                while ((line = br.readLine()) != null) {
                    String[] p = line.split(",");
                    double[] v = new double[p.length];
                    for (int i = 0; i < p.length; i++) {
                        v[i] = Double.parseDouble(p[i]);
                    }
                    UserProfile prof = new UserProfile((int) v[0], 170, 70, v[1] == 1 ? 1 : 2);
                    ScreeningResult r = c.classify(prof, v[2], v[3], v[4], v[5], v[6]);
                    assertTrue(r.available);
                    for (int k = 0; k < 3; k++) {
                        assertEquals("row " + rows + " class " + k, v[7 + k], r.probabilities[k], 1e-3);
                    }
                    rows++;
                }
            }
            assertTrue(rows > 10);
        } finally {
            c.close();
        }
    }
}
