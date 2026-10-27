package com.example.yo7a.healthwatcher;

import static org.junit.Assert.assertEquals;

import androidx.test.platform.app.InstrumentationRegistry;

import com.example.yo7a.healthwatcher.ml.CapabilityItem;
import com.example.yo7a.healthwatcher.ml.MLModelManager;
import com.example.yo7a.healthwatcher.ml.RuntimeSelfTest;
import com.example.yo7a.healthwatcher.ml.TfliteModelRunner;

import org.junit.Test;

import java.io.InputStream;
import java.nio.ByteBuffer;

/** Runs the REAL TensorFlow Lite runtime on the device against the bundled non-health self-test model. */
public class TfliteRuntimeInstrumentedTest {
    @Test
    public void tensorFlowLiteRunsOnThisDevice() throws Exception {
        try (InputStream in = InstrumentationRegistry.getInstrumentation().getTargetContext()
                .getAssets().open(RuntimeSelfTest.ASSET)) {
            CapabilityItem item = RuntimeSelfTest.run(in, new MLModelManager.RunnerFactory() {
                @Override
                public MLModelManager.ModelRunner create(ByteBuffer model) {
                    return new TfliteModelRunner(model);
                }
            });
            assertEquals(item.detail, CapabilityItem.Status.PASS, item.status);
        }
    }
}
