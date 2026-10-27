package com.example.yo7a.healthwatcher.ml;

import java.util.Arrays;

/** Thread-safe, growable buffer that collects frame samples and produces an immutable {@link PpgSignal}. */
public final class PpgRecorder {
    private double[] t = new double[1024];
    private double[] r = new double[1024];
    private double[] g = new double[1024];
    private double[] b = new double[1024];
    private int n = 0;

    /** Adds a sample. Samples whose timestamp does not advance are ignored. Returns true if stored. */
    public synchronized boolean add(double tSec, double red, double green, double blue) {
        if (n > 0 && !(tSec > t[n - 1])) {
            return false;
        }
        if (n == t.length) {
            int cap = n * 2;
            t = Arrays.copyOf(t, cap);
            r = Arrays.copyOf(r, cap);
            g = Arrays.copyOf(g, cap);
            b = Arrays.copyOf(b, cap);
        }
        t[n] = tSec;
        r[n] = red;
        g[n] = green;
        b[n] = blue;
        n++;
        return true;
    }

    public synchronized int size() {
        return n;
    }

    public synchronized void clear() {
        n = 0;
    }

    /** Seconds covered by the recording so far. */
    public synchronized double durationSec() {
        return n < 2 ? 0.0 : t[n - 1] - t[0];
    }

    /** Immutable snapshot with timestamps shifted so that the first sample is at t = 0. */
    public synchronized PpgSignal snapshot() {
        double[] tt = new double[n];
        double t0 = n > 0 ? t[0] : 0.0;
        for (int i = 0; i < n; i++) {
            tt[i] = t[i] - t0;
        }
        return new PpgSignal(tt, Arrays.copyOf(r, n), Arrays.copyOf(g, n), Arrays.copyOf(b, n));
    }

    /** Mean red/green values of the samples recorded during the last {@code seconds} (NaN if empty). */
    public synchronized double[] recentMeans(double seconds) {
        if (n == 0) {
            return new double[]{Double.NaN, Double.NaN};
        }
        double cutoff = t[n - 1] - seconds;
        double sr = 0, sg = 0;
        int c = 0;
        for (int i = n - 1; i >= 0 && t[i] >= cutoff; i--) {
            sr += r[i];
            sg += g[i];
            c++;
        }
        return new double[]{sr / c, sg / c};
    }
}
