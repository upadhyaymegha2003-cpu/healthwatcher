package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Random;

public class SignalQualityAnalyzerTest {
    private final SignalQualityAnalyzer analyzer = new SignalQualityAnalyzer();

    @Test
    public void goodRecordingPasses() {
        QualityReport q = analyzer.analyze(TestSignals.clean(1.2, 0.25, 30, 30, 1));
        assertTrue(q.joinedMessages(), q.isUsable());
        assertEquals("OK", q.statusLabel());
        assertTrue(q.cardiacPowerRatio > 0.3);
    }

    @Test
    public void tooShortIsBlocked() {
        QualityReport q = analyzer.analyze(TestSignals.clean(1.2, 0.25, 12, 30, 1));
        assertFalse(q.isUsable());
        assertTrue(q.has(QualityReport.Issue.TOO_SHORT));
        assertEquals("REJECTED", q.statusLabel());
    }

    @Test
    public void lowSamplingRateIsBlocked() {
        QualityReport q = analyzer.analyze(TestSignals.clean(0.8, 0.2, 30, 6, 1));
        assertTrue(q.has(QualityReport.Issue.LOW_SAMPLING_RATE));
        assertFalse(q.isUsable());
    }

    @Test
    public void missingFingerIsBlocked() {
        QualityReport q = analyzer.analyze(TestSignals.withRed(TestSignals.clean(1.2, 0.25, 30, 30, 1), 90));
        assertTrue(q.has(QualityReport.Issue.NO_FINGER));
        assertFalse(q.isUsable());
    }

    @Test
    public void flatSignalHasNoPulse() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        double[] g = new double[base.size()];
        java.util.Arrays.fill(g, 118.0);
        QualityReport q = analyzer.analyze(TestSignals.withGreen(base, g));
        assertTrue(q.has(QualityReport.Issue.NO_PULSATILE_SIGNAL));
        assertFalse(q.isUsable());
    }

    @Test
    public void saturatedGreenIsBlockedAndSaturatedRedWarns() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        double[] g = new double[base.size()];
        java.util.Arrays.fill(g, 255.0);
        QualityReport q1 = analyzer.analyze(TestSignals.withGreen(base, g));
        assertTrue(q1.has(QualityReport.Issue.SATURATED));
        assertFalse(q1.isUsable());

        QualityReport q2 = analyzer.analyze(TestSignals.withRed(base, 255.0));
        assertTrue(q2.has(QualityReport.Issue.SATURATED));
        assertTrue("red saturation alone is only a warning", q2.isUsable());
        assertTrue(q2.hasWarnings());
    }

    @Test
    public void pureNoiseIsRejected() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        Random rnd = new Random(3);
        double[] g = new double[base.size()];
        for (int i = 0; i < g.length; i++) {
            g[i] = 118 + 1.5 * rnd.nextGaussian();
        }
        QualityReport q = analyzer.analyze(TestSignals.withGreen(base, g));
        assertFalse(q.messagesDebug(), q.isUsable());
        assertTrue(q.has(QualityReport.Issue.LOW_CARDIAC_SNR) || q.has(QualityReport.Issue.MOTION_ARTIFACT));
    }

    @Test
    public void slowDriftWithoutPulseIsRejected() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        Random rnd = new Random(11);
        double[] g = new double[base.size()];
        double v = 118;
        for (int i = 0; i < g.length; i++) {
            v += 0.05 * rnd.nextGaussian();     // random-walk brightness drift, no heartbeat
            g[i] = v;
        }
        QualityReport q = analyzer.analyze(TestSignals.withGreen(base, g));
        assertFalse(q.messagesDebug(), q.isUsable());
    }

    @Test
    public void bigMotionBurstsAreFlagged() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        double[] g = base.green.clone();
        Random rnd = new Random(8);
        for (int i = 0; i < g.length; i++) {
            if ((i / 60) % 3 == 1) {            // every third 2-second window thrashes
                g[i] += 25 * rnd.nextGaussian();
            }
        }
        QualityReport q = analyzer.analyze(TestSignals.withGreen(base, g));
        assertTrue(q.has(QualityReport.Issue.MOTION_ARTIFACT) || q.has(QualityReport.Issue.LOW_CARDIAC_SNR));
        assertFalse(q.statusLabel(), "OK".equals(q.statusLabel()));
    }

    @Test
    public void nanAndNonMonotonicTimestampsAreInvalid() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        double[] r = base.red.clone();
        r[10] = Double.NaN;
        QualityReport q = analyzer.analyze(new PpgSignal(base.t, r, base.green, base.blue));
        assertTrue(q.has(QualityReport.Issue.INVALID_VALUES));
        double[] t = base.t.clone();
        t[20] = t[19];
        QualityReport q2 = analyzer.analyze(new PpgSignal(t, base.red, base.green, base.blue));
        assertTrue(q2.has(QualityReport.Issue.INVALID_VALUES));
        double[] g = base.green.clone();
        g[5] = 400;
        assertTrue(analyzer.analyze(TestSignals.withGreen(base, g)).has(QualityReport.Issue.INVALID_VALUES));
    }

    @Test
    public void longFrameGapIsBlocked() {
        PpgSignal base = TestSignals.clean(1.2, 0.25, 30, 30, 1);
        double[] t = base.t.clone();
        for (int i = 400; i < t.length; i++) {
            t[i] += 1.5;                         // 1.5 s stall in the middle
        }
        QualityReport q = analyzer.analyze(new PpgSignal(t, base.red, base.green, base.blue));
        assertTrue(q.has(QualityReport.Issue.SAMPLING_GAPS));
        assertFalse(q.isUsable());
    }

    @Test
    public void emptyRecordingIsHandled() {
        QualityReport q = analyzer.analyze(new PpgSignal(new double[0], new double[0], new double[0], new double[0]));
        assertFalse(q.isUsable());
    }

    @Test
    public void liveStateReflectsFingerAndSaturation() {
        assertEquals(SignalQualityAnalyzer.LiveState.NO_FINGER, analyzer.assessLive(80, 60));
        assertEquals(SignalQualityAnalyzer.LiveState.NO_FINGER, analyzer.assessLive(Double.NaN, 0));
        assertEquals(SignalQualityAnalyzer.LiveState.SATURATED, analyzer.assessLive(255, 255));
        assertEquals(SignalQualityAnalyzer.LiveState.OK, analyzer.assessLive(235, 120));
    }
}
