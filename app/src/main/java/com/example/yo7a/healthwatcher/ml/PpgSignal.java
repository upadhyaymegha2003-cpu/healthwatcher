package com.example.yo7a.healthwatcher.ml;

import java.util.Locale;

/**
 * Immutable raw camera-PPG recording: per-frame mean red/green/blue intensity (0..255) with
 * frame timestamps in seconds (relative to the first frame).
 */
public final class PpgSignal {
    public final double[] t;
    public final double[] red;
    public final double[] green;
    public final double[] blue;

    public PpgSignal(double[] t, double[] red, double[] green, double[] blue) {
        if (t == null || red == null || green == null || blue == null) {
            throw new IllegalArgumentException("Signal arrays must not be null");
        }
        if (t.length != red.length || t.length != green.length || t.length != blue.length) {
            throw new IllegalArgumentException("Signal arrays must have equal length");
        }
        this.t = t.clone();
        this.red = red.clone();
        this.green = green.clone();
        this.blue = blue.clone();
    }

    public int size() {
        return t.length;
    }

    public double durationSec() {
        return t.length < 2 ? 0.0 : t[t.length - 1] - t[0];
    }

    /** Mean sampling rate in Hz computed from the first and last timestamps. */
    public double meanSamplingHz() {
        double d = durationSec();
        return d <= 0 ? 0.0 : (t.length - 1) / d;
    }

    /** CSV (t_sec,red,green,blue) -- the format read by the Python training utility. */
    public String toCsv() {
        StringBuilder sb = new StringBuilder("t_sec,red,green,blue\n");
        for (int i = 0; i < t.length; i++) {
            sb.append(String.format(Locale.US, "%.6f,%.6f,%.6f,%.6f%n", t[i], red[i], green[i], blue[i]));
        }
        return sb.toString();
    }
}
