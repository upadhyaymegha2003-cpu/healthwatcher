package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class PPGFeatureExtractorTest {

    private static BufferedReader open(String name) {
        InputStream in = PPGFeatureExtractorTest.class.getClassLoader().getResourceAsStream(name);
        if (in == null) {
            throw new IllegalStateException("missing test resource " + name);
        }
        return new BufferedReader(new InputStreamReader(in));
    }

    /** Java feature extraction must reproduce the Python (training-side) implementation. */
    @Test
    public void javaMatchesPythonGoldenVectors() throws Exception {
        Map<String, List<double[]>> signals = new LinkedHashMap<>();
        try (BufferedReader br = open("golden_signals.csv")) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                List<double[]> rows = signals.get(p[0]);
                if (rows == null) {
                    rows = new ArrayList<>();
                    signals.put(p[0], rows);
                }
                rows.add(new double[]{Double.parseDouble(p[1]), Double.parseDouble(p[2]),
                        Double.parseDouble(p[3]), Double.parseDouble(p[4])});
            }
        }
        Map<String, double[]> expected = new LinkedHashMap<>();
        try (BufferedReader br = open("golden_features.csv")) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                double[] v = expected.get(p[0]);
                if (v == null) {
                    v = new double[PPGFeatureExtractor.FEATURE_COUNT];
                    expected.put(p[0], v);
                }
                assertEquals("feature order/name mismatch", PPGFeatureExtractor.FEATURE_NAMES[Integer.parseInt(p[1])], p[2]);
                v[Integer.parseInt(p[1])] = Double.parseDouble(p[3]);
            }
        }
        assertTrue(signals.size() >= 3);
        for (Map.Entry<String, List<double[]>> e : signals.entrySet()) {
            List<double[]> rows = e.getValue();
            int n = rows.size();
            double[] t = new double[n], r = new double[n], g = new double[n], b = new double[n];
            for (int i = 0; i < n; i++) {
                t[i] = rows.get(i)[0];
                r[i] = rows.get(i)[1];
                g[i] = rows.get(i)[2];
                b[i] = rows.get(i)[3];
            }
            double[] actual = PPGFeatureExtractor.extract(new PpgSignal(t, r, g, b));
            double[] want = expected.get(e.getKey());
            for (int i = 0; i < want.length; i++) {
                double tol = 1e-6 * Math.max(1.0, Math.abs(want[i]));
                assertEquals(e.getKey() + ":" + PPGFeatureExtractor.FEATURE_NAMES[i], want[i], actual[i], tol);
            }
        }
    }

    @Test
    public void dominantFrequencyTracksHeartRate() throws Exception {
        double[] f = PPGFeatureExtractor.extract(TestSignals.clean(1.5, 0.25, 30, 30, 5));
        assertEquals(1.5, f[0], 0.02);
        assertEquals(90.0, f[11], 2.0);
        assertEquals(0.25, f[18], 0.03);
        assertTrue("clean signal should concentrate power at the peak", f[1] > 0.4);
    }

    @Test
    public void allFeaturesFinite() throws Exception {
        double[] f = PPGFeatureExtractor.extract(TestSignals.clean(1.1, 0.3, 30, 29.7, 9));
        assertEquals(PPGFeatureExtractor.FEATURE_COUNT, f.length);
        for (double v : f) {
            assertTrue(!Double.isNaN(v) && !Double.isInfinite(v));
        }
    }

    @Test
    public void featureIndexLookup() {
        assertEquals(0, PPGFeatureExtractor.indexOf("g_dom_freq_hz"));
        assertEquals(-1, PPGFeatureExtractor.indexOf("nope"));
    }

    @Test
    public void rejectsTooShortNonFiniteAndNonMonotonic() {
        expectFailure(TestSignals.clean(1.2, 0.25, 6, 30, 1));
        PpgSignal ok = TestSignals.clean(1.2, 0.25, 20, 30, 1);
        double[] g = ok.green.clone();
        g[100] = Double.NaN;
        expectFailure(TestSignals.withGreen(ok, g));
        double[] t = ok.t.clone();
        t[50] = t[49];
        expectFailure(new PpgSignal(t, ok.red, ok.green, ok.blue));
    }

    private static void expectFailure(PpgSignal s) {
        try {
            PPGFeatureExtractor.extract(s);
            fail("expected FeatureExtractionException");
        } catch (PPGFeatureExtractor.FeatureExtractionException expected) {
            // ok
        }
    }
}
