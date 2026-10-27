package com.example.yo7a.healthwatcher.ml;

/**
 * Small, deterministic DSP primitives shared by {@link PPGFeatureExtractor} and
 * {@link SignalQualityAnalyzer}. Every routine here has a line-for-line numpy twin in
 * {@code ml_training/ppg_features.py}; if you change one, change the other and regenerate the
 * golden test vectors (see ml_training/README.md).
 */
final class PpgDsp {
    private PpgDsp() {
    }

    /** Linear resampling onto the uniform grid k / fs, k = 0..floor(T*fs). t must be strictly increasing. */
    static double[] resampleLinear(double[] t, double[] x, double fs) {
        int n = t.length;
        double t0 = t[0];
        double total = t[n - 1] - t0;
        int m = (int) Math.floor(total * fs + 1e-9) + 1;
        double[] out = new double[m];
        int j = 0;
        for (int k = 0; k < m; k++) {
            double tk = k / fs;
            while (j < n - 2 && (t[j + 1] - t0) <= tk) {
                j++;
            }
            double x0 = t[j] - t0;
            double x1 = t[j + 1] - t0;
            double slope = (x[j + 1] - x[j]) / (x1 - x0);
            double v = slope * (tk - x0) + x[j];
            if (tk >= x1) {
                v = x[j + 1];
            }
            out[k] = v;
        }
        return out;
    }

    /** Centered moving average with a window of {@code w} (odd) samples; the window shrinks at the edges. */
    static double[] movingAverage(double[] x, int w) {
        int n = x.length;
        int h = w / 2;
        double[] prefix = new double[n + 1];
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + x[i];
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            int lo = Math.max(0, i - h);
            int hi = Math.min(n - 1, i + h);
            out[i] = (prefix[hi + 1] - prefix[lo]) / (hi - lo + 1);
        }
        return out;
    }

    /** x minus its least-squares straight line over the sample index. */
    static double[] linearDetrend(double[] x) {
        int n = x.length;
        double mi = (n - 1) / 2.0;
        double mx = mean(x);
        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++) {
            sxy += (i - mi) * (x[i] - mx);
            sxx += (i - mi) * (i - mi);
        }
        double slope = sxx == 0 ? 0 : sxy / sxx;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[i] - (mx + slope * (i - mi));
        }
        return out;
    }

    /** Cardiac-band conditioning: remove a 45-sample moving average (~1.5 s) then smooth over 5 samples. */
    static double[] cardiacBand(double[] x) {
        double[] base = movingAverage(x, 45);
        double[] hp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            hp[i] = x[i] - base[i];
        }
        return movingAverage(hp, 5);
    }

    /** Hann-windowed copy of x. */
    static double[] hann(double[] x) {
        int n = x.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double w = n > 1 ? 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / (n - 1)) : 1.0;
            out[i] = x[i] * w;
        }
        return out;
    }

    /** Squared DFT magnitude of an (already windowed) signal at frequency f (Hz). */
    static double powerAt(double[] s, double fs, double f) {
        double re = 0, im = 0;
        double w = 2.0 * Math.PI * f / fs;
        for (int i = 0; i < s.length; i++) {
            double a = w * i;
            re += s[i] * Math.cos(a);
            im -= s[i] * Math.sin(a);
        }
        return re * re + im * im;
    }

    /** Power on the grid f0 + j*step, j = 0..count-1, of the Hann-windowed signal. */
    static double[] spectrum(double[] x, double fs, double f0, double step, int count) {
        double[] s = hann(x);
        double[] p = new double[count];
        for (int j = 0; j < count; j++) {
            p[j] = powerAt(s, fs, f0 + j * step);
        }
        return p;
    }

    static int argmax(double[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best;
    }

    static double sum(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s;
    }

    static double mean(double[] x) {
        return x.length == 0 ? 0.0 : sum(x) / x.length;
    }

    /** Population standard deviation. */
    static double std(double[] x) {
        if (x.length == 0) {
            return 0.0;
        }
        double m = mean(x);
        double s = 0;
        for (double v : x) {
            s += (v - m) * (v - m);
        }
        return Math.sqrt(s / x.length);
    }
}
