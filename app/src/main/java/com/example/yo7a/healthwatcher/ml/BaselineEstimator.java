package com.example.yo7a.healthwatcher.ml;

import com.example.yo7a.healthwatcher.Math.Fft;
import com.example.yo7a.healthwatcher.Math.Fft2;

/**
 * The ORIGINAL HealthWatcher signal-processing estimators, extracted from the five measurement
 * activities into one reusable, unit-testable class. The FFT / band-pass code ({@link Fft},
 * {@link Fft2}) and all formulas are unchanged. Differences from the old in-activity code:
 * <ul>
 *   <li>The red/green fallback now implements the documented intent (use the average when both
 *       channels give a plausible value, otherwise the plausible one). The old conditions used
 *       {@code ||} and were always true.</li>
 *   <li>The blood-pressure expression {@code 3 / 2 * PP} is kept as the integer-division value it
 *       always evaluated to ({@code PP}); it is written explicitly so the behaviour is visible.</li>
 *   <li>Implausible results are reported as unavailable instead of being displayed.</li>
 * </ul>
 * Blood pressure and SpO2 here come from uncalibrated literature formulas and are NOT validated measurements.
 */
public final class BaselineEstimator {
    public static final double HR_MIN = 45, HR_MAX = 200;
    public static final double RR_MIN = 10, RR_MAX = 24;
    public static final double SPO2_MIN = 80, SPO2_MAX = 99;
    public static final double SBP_MIN = 70, SBP_MAX = 250, DBP_MIN = 40, DBP_MAX = 150;

    /** Estimates; NaN means "not available / implausible". */
    public static final class Result {
        public double heartRate = Double.NaN;
        public double respiratoryRate = Double.NaN;
        public double spo2 = Double.NaN;
        public double systolic = Double.NaN;
        public double diastolic = Double.NaN;
    }

    public Result estimate(PpgSignal s, UserProfile profile) {
        Result res = new Result();
        int n = s.size();
        if (n < 40 || s.durationSec() <= 0) {
            return res;
        }
        double fs = s.meanSamplingHz();
        Double[] green = box(s.green);
        Double[] red = box(s.red);

        // Heart rate: FFT peak of green and red channels
        double bpmG = Math.ceil(Fft.FFT(green, n, fs) * 60);
        double bpmR = Math.ceil(Fft.FFT(red, n, fs) * 60);
        double hr = combine(bpmG, bpmR, HR_MIN, HR_MAX);
        if (!Double.isNaN(hr)) {
            res.heartRate = (int) hr;
        }

        // Respiratory rate: FFT with Butterworth band-pass on green and red
        double rrG = Math.ceil(Fft2.FFT(green, n, fs) * 60);
        double rrR = Math.ceil(Fft2.FFT(red, n, fs) * 60);
        double rr = combine(rrG, rrR, RR_MIN, RR_MAX);
        if (!Double.isNaN(rr)) {
            res.respiratoryRate = (int) rr;
        }

        // SpO2: ratio of ratios of red and blue (Kanva et al.), uncalibrated
        double meanR = PpgDsp.mean(s.red);
        double meanB = PpgDsp.mean(s.blue);
        double sdR = sampleStd(s.red, meanR);
        double sdB = sampleStd(s.blue, meanB);
        if (meanR > 0 && meanB > 0 && sdB > 0) {
            double ratio = (sdR / meanR) / (sdB / meanB);
            int o2 = (int) (100 - 5 * ratio);
            if (o2 >= SPO2_MIN && o2 <= SPO2_MAX) {
                res.spo2 = o2;
            }
        }

        // Blood pressure from heart rate and demographics (Chandrasekaran et al.), uncalibrated
        if (!Double.isNaN(res.heartRate) && profile != null && profile.isPlausible()) {
            double beats = res.heartRate;
            double q = profile.gender == 1 ? 5 : 4.5;
            double rob = 18.5;
            double et = 364.5 - 1.23 * beats;
            double bsa = 0.007184 * Math.pow(profile.weightKg, 0.425) * Math.pow(profile.heightCm, 0.725);
            double sv = -6.6 + 0.25 * (et - 35) - 0.62 * beats + 40.4 * bsa - 0.51 * profile.age;
            double pp = sv / ((0.013 * profile.weightKg - 0.007 * profile.age - 0.004 * beats) + 1.307);
            double mpp = q * rob;
            // Original source: SP = (int) (MPP + 3 / 2 * PP). In Java 3 / 2 == 1, so this is MPP + PP.
            int sp = (int) (mpp + 1 * pp);
            int dp = (int) (mpp - pp / 3);
            if (sp >= SBP_MIN && sp <= SBP_MAX && dp >= DBP_MIN && dp <= DBP_MAX && sp > dp) {
                res.systolic = sp;
                res.diastolic = dp;
            }
        }
        return res;
    }

    private static double combine(double a, double b, double lo, double hi) {
        boolean okA = a >= lo && a <= hi;
        boolean okB = b >= lo && b <= hi;
        if (okA && okB) {
            return (a + b) / 2;
        }
        if (okA) {
            return a;
        }
        if (okB) {
            return b;
        }
        return Double.NaN;
    }

    private static double sampleStd(double[] x, double mean) {
        if (x.length < 2) {
            return 0;
        }
        double s = 0;
        for (double v : x) {
            s += (v - mean) * (v - mean);
        }
        return Math.sqrt(s / (x.length - 1));
    }

    private static Double[] box(double[] x) {
        Double[] out = new Double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i];
        }
        return out;
    }
}
