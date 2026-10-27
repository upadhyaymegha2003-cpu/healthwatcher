package com.example.yo7a.healthwatcher.ml;

import java.util.Random;

/** Deterministic synthetic recordings used as unit-test inputs (not training data, not performance claims). */
final class TestSignals {
    private TestSignals() {
    }

    /** Camera-like recording: finger on lens, pulse at hrHz, respiration at respHz. */
    static PpgSignal clean(double hrHz, double respHz, double seconds, double fs, long seed) {
        Random rnd = new Random(seed);
        int n = (int) (seconds * fs);
        double[] t = new double[n], r = new double[n], g = new double[n], b = new double[n];
        for (int i = 0; i < n; i++) {
            t[i] = i / fs;
            double ph = 2 * Math.PI * hrHz * t[i];
            double pulse = Math.sin(ph) + 0.35 * Math.sin(2 * ph + 0.6);
            double resp = Math.sin(2 * Math.PI * respHz * t[i]);
            r[i] = 232 + 1.4 * pulse + 0.9 * resp + 0.05 * rnd.nextGaussian();
            g[i] = 118 + 1.0 * pulse + 0.6 * resp + 0.05 * rnd.nextGaussian();
            b[i] = 41 + 0.25 * pulse + 0.15 * resp + 0.03 * rnd.nextGaussian();
        }
        return new PpgSignal(t, r, g, b);
    }

    static PpgSignal withRed(PpgSignal s, double value) {
        double[] r = new double[s.size()];
        java.util.Arrays.fill(r, value);
        return new PpgSignal(s.t, r, s.green, s.blue);
    }

    static PpgSignal withGreen(PpgSignal s, double[] g) {
        return new PpgSignal(s.t, s.red, g, s.blue);
    }
}
