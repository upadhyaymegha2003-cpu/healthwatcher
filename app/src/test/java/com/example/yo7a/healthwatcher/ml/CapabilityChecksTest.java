package com.example.yo7a.healthwatcher.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class CapabilityChecksTest {
    private static final InputStream MODEL = new ByteArrayInputStream(new byte[]{1, 2, 3});

    private static MLModelManager.RunnerFactory linear(final float scale, final int in, final int out) {
        return new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(ByteBuffer model) {
                return new FakeEngine.LinearRunner(in, out, 0f, new float[out]) {
                    @Override
                    public float[] run(float[] x) {
                        float y = 0.1f + 0.5f * x[0] - x[1] + 2 * x[2] + 0.25f * x[3];
                        return new float[]{y * scale};
                    }
                };
            }
        };
    }

    private static InputStream stream() {
        return new ByteArrayInputStream(new byte[]{1, 2, 3});
    }

    @Test
    public void runtimeSelfTestPassesWhenEngineIsCorrect() {
        CapabilityItem i = RuntimeSelfTest.run(stream(), linear(1f, 4, 1));
        assertEquals(i.detail, CapabilityItem.Status.PASS, i.status);
    }

    @Test
    public void runtimeSelfTestFailsOnWrongNumbersWrongShapeOrEngineErrors() {
        assertEquals(CapabilityItem.Status.FAIL, RuntimeSelfTest.run(stream(), linear(1.5f, 4, 1)).status);
        assertEquals(CapabilityItem.Status.FAIL, RuntimeSelfTest.run(stream(), linear(1f, 5, 1)).status);
        CapabilityItem boom = RuntimeSelfTest.run(stream(), new MLModelManager.RunnerFactory() {
            @Override
            public MLModelManager.ModelRunner create(ByteBuffer model) {
                throw new UnsatisfiedLinkError("no native library for this ABI");
            }
        });
        assertEquals(CapabilityItem.Status.FAIL, boom.status);
        assertTrue(boom.detail.contains("UnsatisfiedLinkError"));
        InputStream broken = new InputStream() {
            @Override
            public int read() throws IOException {
                throw new IOException("disk");
            }
        };
        assertEquals(CapabilityItem.Status.FAIL, RuntimeSelfTest.run(broken, linear(1f, 4, 1)).status);
    }

    @Test
    public void pipelineSelfTestPassesOnTheBuiltInWaveform() {
        CapabilityItem i = PipelineSelfTest.run();
        assertEquals(i.detail, CapabilityItem.Status.PASS, i.status);
    }
}
