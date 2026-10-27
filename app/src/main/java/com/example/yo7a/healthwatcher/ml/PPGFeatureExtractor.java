package com.example.yo7a.healthwatcher.ml;

/**
 * Converts a raw camera PPG recording into the fixed-length feature vector consumed by the
 * on-device models. The algorithm is versioned ({@link #FEATURE_VERSION}) and is mirrored exactly
 * by {@code ml_training/ppg_features.py}; model manifests record the version they were trained with
 * and the app refuses models whose version differs.
 *
 * Pipeline: linear resampling to 30 Hz -> per-channel cardiac conditioning (moving-average high-pass
 * + 5-sample smoothing) -> Hann-windowed DFT power on a 0.01 Hz grid (cardiac 0.6-3.6 Hz,
 * respiratory 0.08-0.60 Hz) -> peak detection / inter-beat intervals -> statistics.
 */
public final class PPGFeatureExtractor {
    public static final String FEATURE_VERSION = "ppgfeat-1";
    public static final double TARGET_FS = 30.0;
    /** Minimum recording length (seconds) the extractor accepts. The quality analyzer is stricter. */
    public static final double MIN_DURATION_SEC = 10.0;

    static final double CARDIAC_F0 = 0.60;
    static final double CARDIAC_STEP = 0.01;
    static final int CARDIAC_BINS = 301;      // 0.60 .. 3.60 Hz
    static final double RESP_F0 = 0.08;
    static final double RESP_STEP = 0.01;
    static final int RESP_BINS = 53;          // 0.08 .. 0.60 Hz
    static final int PEAK_HALF_WINDOW = 7;    // +-7 samples at 30 Hz (~0.23 s)

    public static final String[] FEATURE_NAMES = {
            "g_dom_freq_hz",
            "g_peak_power_ratio",
            "g_second_peak_ratio",
            "g_spectral_entropy",
            "r_dom_freq_hz",
            "r_peak_power_ratio",
            "g_harmonic_ratio",
            "g_ibi_mean_s",
            "g_ibi_std_s",
            "g_rmssd_s",
            "g_peak_count",
            "g_hr_from_peaks_bpm",
            "g_ac_std",
            "g_ac_dc_ratio",
            "g_skewness",
            "g_excess_kurtosis",
            "g_zero_cross_rate_hz",
            "g_peak_amp_mean",
            "resp_g_dom_freq_hz",
            "resp_g_peak_power_ratio",
            "resp_r_dom_freq_hz",
            "red_ac_dc_ratio",
            "blue_ac_dc_ratio",
            "ratio_of_ratios",
            "red_dc_norm",
            "duration_s"
    };

    public static final int FEATURE_COUNT = FEATURE_NAMES.length;

    /** Index of a feature by name, or -1. */
    public static int indexOf(String name) {
        for (int i = 0; i < FEATURE_NAMES.length; i++) {
            if (FEATURE_NAMES[i].equals(name)) {
                return i;
            }
        }
        return -1;
    }

    /** Thrown when a recording cannot be converted into valid features. */
    public static final class FeatureExtractionException extends Exception {
        private static final long serialVersionUID = 1L;

        public FeatureExtractionException(String message) {
            super(message);
        }
    }

    /** Pre-conditioned channels, reused by the quality analyzer. */
    static final class Conditioned {
        double[] red, green, blue;           // resampled to TARGET_FS
        double[] cardiacRed, cardiacGreen, cardiacBlue;
        double durationSec;
    }

    static Conditioned condition(PpgSignal s) throws FeatureExtractionException {
        int n = s.size();
        if (n < 2) {
            throw new FeatureExtractionException("Recording has fewer than two samples");
        }
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(s.t[i]) || Double.isInfinite(s.t[i])
                    || Double.isNaN(s.red[i]) || Double.isInfinite(s.red[i])
                    || Double.isNaN(s.green[i]) || Double.isInfinite(s.green[i])
                    || Double.isNaN(s.blue[i]) || Double.isInfinite(s.blue[i])) {
                throw new FeatureExtractionException("Recording contains non-finite values");
            }
            if (i > 0 && !(s.t[i] > s.t[i - 1])) {
                throw new FeatureExtractionException("Timestamps are not strictly increasing");
            }
        }
        if (s.durationSec() < MIN_DURATION_SEC) {
            throw new FeatureExtractionException("Recording shorter than " + (int) MIN_DURATION_SEC + " s");
        }
        Conditioned c = new Conditioned();
        c.durationSec = s.durationSec();
        c.red = PpgDsp.resampleLinear(s.t, s.red, TARGET_FS);
        c.green = PpgDsp.resampleLinear(s.t, s.green, TARGET_FS);
        c.blue = PpgDsp.resampleLinear(s.t, s.blue, TARGET_FS);
        c.cardiacRed = PpgDsp.cardiacBand(c.red);
        c.cardiacGreen = PpgDsp.cardiacBand(c.green);
        c.cardiacBlue = PpgDsp.cardiacBand(c.blue);
        return c;
    }

    /** Dominant-frequency summary of a conditioned cardiac signal. */
    static final class CardiacSpectrum {
        double[] power;
        int peakIndex;
        double peakFreqHz;
        double total;
        double peakRatio;   // power within +-0.10 Hz of the peak / total band power
    }

    static CardiacSpectrum cardiacSpectrum(double[] cardiac) {
        CardiacSpectrum cs = new CardiacSpectrum();
        cs.power = PpgDsp.spectrum(cardiac, TARGET_FS, CARDIAC_F0, CARDIAC_STEP, CARDIAC_BINS);
        cs.peakIndex = PpgDsp.argmax(cs.power);
        cs.peakFreqHz = CARDIAC_F0 + cs.peakIndex * CARDIAC_STEP;
        cs.total = PpgDsp.sum(cs.power);
        double near = 0;
        for (int j = Math.max(0, cs.peakIndex - 10); j <= Math.min(CARDIAC_BINS - 1, cs.peakIndex + 10); j++) {
            near += cs.power[j];
        }
        cs.peakRatio = cs.total > 0 ? near / cs.total : 0.0;
        return cs;
    }

    /** Local maxima above zero that dominate +-PEAK_HALF_WINDOW samples (min spacing ~0.27 s). */
    static int[] detectPeaks(double[] cg) {
        int m = cg.length;
        int[] tmp = new int[m];
        int np = 0;
        for (int i = PEAK_HALF_WINDOW; i <= m - 1 - PEAK_HALF_WINDOW; i++) {
            if (!(cg[i] > 0)) {
                continue;
            }
            boolean ok = true;
            for (int k = 1; k <= PEAK_HALF_WINDOW && ok; k++) {
                if (!(cg[i] > cg[i - k]) || !(cg[i] >= cg[i + k])) {
                    ok = false;
                }
            }
            if (ok) {
                tmp[np++] = i;
            }
        }
        return java.util.Arrays.copyOf(tmp, np);
    }

    /** Extracts the {@link #FEATURE_COUNT} features. All returned values are finite. */
    public static double[] extract(PpgSignal signal) throws FeatureExtractionException {
        Conditioned c = condition(signal);
        double fs = TARGET_FS;
        double[] cg = c.cardiacGreen;
        int m = cg.length;
        double[] f = new double[FEATURE_COUNT];

        // --- cardiac spectrum, green + red ---
        CardiacSpectrum sg = cardiacSpectrum(cg);
        CardiacSpectrum sr = cardiacSpectrum(c.cardiacRed);
        f[0] = sg.peakFreqHz;
        f[1] = sg.peakRatio;

        double second = 0;
        for (int j = 0; j < CARDIAC_BINS; j++) {
            if (Math.abs(j - sg.peakIndex) > 15 && sg.power[j] > second) {
                second = sg.power[j];
            }
        }
        double peakPower = sg.power[sg.peakIndex];
        f[2] = peakPower > 0 ? second / peakPower : 0.0;

        double ent = 0;
        if (sg.total > 0) {
            for (int j = 0; j < CARDIAC_BINS; j++) {
                double p = sg.power[j] / sg.total;
                if (p > 0) {
                    ent -= p * Math.log(p);
                }
            }
            ent /= Math.log(CARDIAC_BINS);
        }
        f[3] = ent;
        f[4] = sr.peakFreqHz;
        f[5] = sr.peakRatio;

        double[] windowed = PpgDsp.hann(cg);
        double harm = PpgDsp.powerAt(windowed, fs, 2.0 * sg.peakFreqHz);
        f[6] = peakPower > 0 ? harm / peakPower : 0.0;

        // --- peaks and inter-beat intervals on the green cardiac signal ---
        int[] peaks = detectPeaks(cg);
        int np = peaks.length;
        int nibi = Math.max(0, np - 1);
        double[] ibi = new double[nibi];
        for (int k = 0; k < nibi; k++) {
            ibi[k] = (peaks[k + 1] - peaks[k]) / fs;
        }
        double ibiMean = PpgDsp.mean(ibi);
        f[7] = ibiMean;
        f[8] = PpgDsp.std(ibi);
        double rmssd = 0;
        if (nibi >= 2) {
            double s = 0;
            for (int k = 0; k < nibi - 1; k++) {
                double d = ibi[k + 1] - ibi[k];
                s += d * d;
            }
            rmssd = Math.sqrt(s / (nibi - 1));
        }
        f[9] = rmssd;
        f[10] = np;
        f[11] = ibiMean > 0 ? 60.0 / ibiMean : 0.0;

        // --- amplitude / shape statistics of the green cardiac signal ---
        double acStd = PpgDsp.std(cg);
        double dcGreen = PpgDsp.mean(c.green);
        f[12] = acStd;
        f[13] = dcGreen > 0 ? acStd / dcGreen : 0.0;
        double mu = PpgDsp.mean(cg);
        double m2 = 0, m3 = 0, m4 = 0;
        for (double v : cg) {
            double d = v - mu;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
        }
        m2 /= m;
        m3 /= m;
        m4 /= m;
        f[14] = m2 > 1e-12 ? m3 / Math.pow(m2, 1.5) : 0.0;
        f[15] = m2 > 1e-12 ? m4 / (m2 * m2) - 3.0 : 0.0;
        int zc = 0;
        for (int i = 1; i < m; i++) {
            if (cg[i - 1] * cg[i] < 0) {
                zc++;
            }
        }
        f[16] = zc / ((m - 1) / fs);
        double pa = 0;
        for (int k = 0; k < np; k++) {
            pa += cg[peaks[k]];
        }
        f[17] = np > 0 ? pa / np : 0.0;

        // --- respiratory-band spectrum ---
        double[] respG = PpgDsp.movingAverage(PpgDsp.linearDetrend(c.green), 15);
        double[] respR = PpgDsp.movingAverage(PpgDsp.linearDetrend(c.red), 15);
        double[] pg = PpgDsp.spectrum(respG, fs, RESP_F0, RESP_STEP, RESP_BINS);
        double[] pr = PpgDsp.spectrum(respR, fs, RESP_F0, RESP_STEP, RESP_BINS);
        int ig = PpgDsp.argmax(pg);
        int ir = PpgDsp.argmax(pr);
        f[18] = RESP_F0 + ig * RESP_STEP;
        double tot = PpgDsp.sum(pg);
        double near = 0;
        for (int j = Math.max(0, ig - 5); j <= Math.min(RESP_BINS - 1, ig + 5); j++) {
            near += pg[j];
        }
        f[19] = tot > 0 ? near / tot : 0.0;
        f[20] = RESP_F0 + ir * RESP_STEP;

        // --- optical ratios (AC/DC) ---
        double dcRed = PpgDsp.mean(c.red);
        double dcBlue = PpgDsp.mean(c.blue);
        double redAcDc = dcRed > 0 ? PpgDsp.std(c.cardiacRed) / dcRed : 0.0;
        double blueAcDc = dcBlue > 0 ? PpgDsp.std(c.cardiacBlue) / dcBlue : 0.0;
        f[21] = redAcDc;
        f[22] = blueAcDc;
        f[23] = blueAcDc > 1e-12 ? redAcDc / blueAcDc : 0.0;
        f[24] = dcRed / 255.0;
        f[25] = c.durationSec;

        for (int i = 0; i < f.length; i++) {
            if (Double.isNaN(f[i]) || Double.isInfinite(f[i])) {
                throw new FeatureExtractionException("Non-finite feature: " + FEATURE_NAMES[i]);
            }
        }
        return f;
    }
}
