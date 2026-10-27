package com.example.yo7a.healthwatcher.ml;

import java.util.Arrays;

/**
 * Rule-based quality gate that runs BEFORE any estimate is shown. All thresholds are engineering
 * defaults (see {@link Config}); they are not clinically validated and should be tuned on real data.
 */
public final class SignalQualityAnalyzer {

    /** Tunable thresholds. */
    public static final class Config {
        public double minDurationSec = 25.0;          // recording target is 30 s
        public double minSamplingHz = 10.0;
        public double blockingGapSec = 1.0;
        public double warningGapSec = 0.5;
        public double fingerRedMin = 200.0;           // same threshold the original app used (0..255 scale)
        public double maxLowRedFraction = 0.05;
        public double saturationLevel = 254.5;
        public double maxSaturatedFraction = 0.25;
        public double minGreenAcStd = 0.01;           // intensity units; below this there is no pulse signal
        public double motionWindowSec = 2.0;
        public double motionStdFactor = 3.0;          // window std > factor * median window std
        public double motionWarnFraction = 0.15;
        public double motionBlockFraction = 0.30;
        // Cardiac peak power (+-0.1 Hz) / total 0.6-3.6 Hz power. Calibrated on synthetic recordings:
        // white noise reaches 0.27 (95th pct) / 0.34 (max); a pulse with moderate noise stays >= 0.4.
        // Heuristic engineering values -- re-tune on real recordings before relying on them.
        public double snrWarnRatio = 0.45;
        public double snrBlockRatio = 0.30;
        // Beat-interval coefficient of variation. Noise/drift gives >= 0.33; a pulse with moderate noise <= 0.23.
        public double rhythmWarnCv = 0.20;
        public double rhythmBlockCv = 0.33;
        public double minPulseHz = 40.0 / 60.0;       // no plausible rhythm below 40 bpm ...
        public double maxPulseHz = 210.0 / 60.0;      // ... or above 210 bpm
    }

    /** Instantaneous feedback shown while recording. */
    public enum LiveState {NO_FINGER, SATURATED, OK}

    private final Config cfg;

    public SignalQualityAnalyzer() {
        this(new Config());
    }

    public SignalQualityAnalyzer(Config cfg) {
        this.cfg = cfg;
    }

    /** Cheap per-second check on the most recent mean red/green values. */
    public LiveState assessLive(double recentRedMean, double recentGreenMean) {
        if (Double.isNaN(recentRedMean) || recentRedMean < cfg.fingerRedMin) {
            return LiveState.NO_FINGER;
        }
        if (recentRedMean >= cfg.saturationLevel && recentGreenMean >= cfg.saturationLevel) {
            return LiveState.SATURATED;
        }
        return LiveState.OK;
    }

    public QualityReport analyze(PpgSignal s) {
        QualityReport q = new QualityReport();
        int n = s.size();
        q.durationSec = s.durationSec();
        q.meanSamplingHz = s.meanSamplingHz();

        if (n < 2) {
            q.add(QualityReport.Issue.TOO_SHORT, QualityReport.Severity.BLOCKING,
                    "No camera frames were recorded. Please try again.");
            return q;
        }

        // 1. Invalid values / timestamps
        boolean invalid = false;
        double maxGap = 0;
        for (int i = 0; i < n; i++) {
            double[] v = {s.t[i], s.red[i], s.green[i], s.blue[i]};
            for (double x : v) {
                if (Double.isNaN(x) || Double.isInfinite(x)) {
                    invalid = true;
                }
            }
            if (s.red[i] < 0 || s.red[i] > 255.5 || s.green[i] < 0 || s.green[i] > 255.5
                    || s.blue[i] < 0 || s.blue[i] > 255.5) {
                invalid = true;
            }
            if (i > 0) {
                double dt = s.t[i] - s.t[i - 1];
                if (!(dt > 0)) {
                    invalid = true;
                } else if (dt > maxGap) {
                    maxGap = dt;
                }
            }
        }
        q.maxGapSec = maxGap;
        if (invalid) {
            q.add(QualityReport.Issue.INVALID_VALUES, QualityReport.Severity.BLOCKING,
                    "The camera returned invalid data. Please restart the measurement.");
            return q;
        }

        // 2. Duration and sampling consistency
        if (q.durationSec < cfg.minDurationSec) {
            q.add(QualityReport.Issue.TOO_SHORT, QualityReport.Severity.BLOCKING,
                    "The recording was too short. Keep your finger still for the full 30 seconds.");
        }
        if (q.meanSamplingHz < cfg.minSamplingHz) {
            q.add(QualityReport.Issue.LOW_SAMPLING_RATE, QualityReport.Severity.BLOCKING,
                    "The camera delivered too few frames per second on this device.");
        }
        if (maxGap > cfg.blockingGapSec) {
            q.add(QualityReport.Issue.SAMPLING_GAPS, QualityReport.Severity.BLOCKING,
                    "Camera frames were interrupted during the recording. Close other apps and retry.");
        } else if (maxGap > cfg.warningGapSec) {
            q.add(QualityReport.Issue.SAMPLING_GAPS, QualityReport.Severity.WARNING,
                    "Some camera frames were delayed during the recording.");
        }

        // 3. Finger placement and saturation
        int lowRed = 0, satRed = 0, satGreen = 0;
        double redSum = 0;
        for (int i = 0; i < n; i++) {
            redSum += s.red[i];
            if (s.red[i] < cfg.fingerRedMin) {
                lowRed++;
            }
            if (s.red[i] >= cfg.saturationLevel) {
                satRed++;
            }
            if (s.green[i] >= cfg.saturationLevel) {
                satGreen++;
            }
        }
        q.redMean = redSum / n;
        q.redSaturatedFraction = satRed / (double) n;
        q.greenSaturatedFraction = satGreen / (double) n;
        if (lowRed / (double) n > cfg.maxLowRedFraction) {
            q.add(QualityReport.Issue.NO_FINGER, QualityReport.Severity.BLOCKING,
                    "Your finger was not covering the camera and flash the whole time. Cover both completely.");
        }
        if (q.greenSaturatedFraction > cfg.maxSaturatedFraction) {
            q.add(QualityReport.Issue.SATURATED, QualityReport.Severity.BLOCKING,
                    "The camera image is over-exposed. Press more lightly and avoid bright surroundings.");
        } else if (q.redSaturatedFraction > cfg.maxSaturatedFraction) {
            q.add(QualityReport.Issue.SATURATED, QualityReport.Severity.WARNING,
                    "The red channel is saturated; oxygen-saturation estimates are especially unreliable.");
        }

        // The remaining checks need a usable, long-enough recording.
        if (!q.isUsable()) {
            return q;
        }

        PPGFeatureExtractor.Conditioned c;
        try {
            c = PPGFeatureExtractor.condition(s);
        } catch (PPGFeatureExtractor.FeatureExtractionException e) {
            q.add(QualityReport.Issue.FEATURES_UNAVAILABLE, QualityReport.Severity.BLOCKING,
                    "The recording could not be analysed: " + e.getMessage());
            return q;
        }

        // 4. Pulsatile amplitude
        double acStd = PpgDsp.std(c.cardiacGreen);
        if (acStd < cfg.minGreenAcStd) {
            q.add(QualityReport.Issue.NO_PULSATILE_SIGNAL, QualityReport.Severity.BLOCKING,
                    "No pulse signal was detected. Make sure your fingertip fully covers the lens and flash.");
            return q;
        }

        // 5. Motion artifacts: windows whose amplitude is far above the typical window amplitude
        int w = (int) Math.round(cfg.motionWindowSec * PPGFeatureExtractor.TARGET_FS);
        int windows = c.cardiacGreen.length / w;
        if (windows >= 3) {
            double[] ws = new double[windows];
            for (int k = 0; k < windows; k++) {
                ws[k] = PpgDsp.std(Arrays.copyOfRange(c.cardiacGreen, k * w, (k + 1) * w));
            }
            double[] sorted = ws.clone();
            Arrays.sort(sorted);
            double median = sorted[windows / 2];
            int bad = 0;
            for (double v : ws) {
                if (v > cfg.motionStdFactor * Math.max(median, 1e-9)) {
                    bad++;
                }
            }
            double frac = bad / (double) windows;
            if (frac > cfg.motionBlockFraction) {
                q.add(QualityReport.Issue.MOTION_ARTIFACT, QualityReport.Severity.BLOCKING,
                        "Too much movement was detected. Rest your hand on a table and keep still.");
            } else if (frac > cfg.motionWarnFraction) {
                q.add(QualityReport.Issue.MOTION_ARTIFACT, QualityReport.Severity.WARNING,
                        "Some movement was detected during the recording.");
            }
        }

        // 6. Cardiac-band spectral concentration
        PPGFeatureExtractor.CardiacSpectrum cs = PPGFeatureExtractor.cardiacSpectrum(c.cardiacGreen);
        q.cardiacPowerRatio = cs.peakRatio;
        if (cs.peakFreqHz < cfg.minPulseHz || cs.peakFreqHz > cfg.maxPulseHz) {
            q.add(QualityReport.Issue.LOW_CARDIAC_SNR, QualityReport.Severity.BLOCKING,
                    "No clear pulse rhythm was found in the signal. Please retry with your finger still.");
        } else if (rhythmCv(c.cardiacGreen) > cfg.rhythmBlockCv) {
            q.add(QualityReport.Issue.LOW_CARDIAC_SNR, QualityReport.Severity.BLOCKING,
                    "The heartbeat rhythm could not be tracked reliably. Please retry with your finger still.");
        } else if (cs.peakRatio < cfg.snrBlockRatio) {
            q.add(QualityReport.Issue.LOW_CARDIAC_SNR, QualityReport.Severity.BLOCKING,
                    "The pulse signal is too noisy to analyse. Please retry in a calm, shaded place.");
        } else if (cs.peakRatio < cfg.snrWarnRatio || rhythmCv(c.cardiacGreen) > cfg.rhythmWarnCv) {
            q.add(QualityReport.Issue.LOW_CARDIAC_SNR, QualityReport.Severity.WARNING,
                    "The pulse signal is weak; results may be less reliable.");
        }
        return q;
    }

    /** Coefficient of variation of detected beat intervals (1.0 when too few beats to judge). */
    private static double rhythmCv(double[] cardiac) {
        int[] peaks = PPGFeatureExtractor.detectPeaks(cardiac);
        if (peaks.length < 6) {
            return 1.0;
        }
        double[] ibi = new double[peaks.length - 1];
        for (int i = 0; i < ibi.length; i++) {
            ibi[i] = (peaks[i + 1] - peaks[i]) / PPGFeatureExtractor.TARGET_FS;
        }
        double mean = PpgDsp.mean(ibi);
        return mean > 0 ? PpgDsp.std(ibi) / mean : 1.0;
    }
}
