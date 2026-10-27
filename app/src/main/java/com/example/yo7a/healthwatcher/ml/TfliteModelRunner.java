package com.example.yo7a.healthwatcher.ml;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.nio.ByteBuffer;

/** TensorFlow Lite implementation of {@link MLModelManager.ModelRunner} (float32 [1,N] -> [1,M] models). */
public final class TfliteModelRunner implements MLModelManager.ModelRunner {
    private final Interpreter interpreter;
    private final int inputSize;
    private final int outputSize;

    public TfliteModelRunner(ByteBuffer model) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        Interpreter it = new Interpreter(model, options);
        try {
            int[] in = it.getInputTensor(0).shape();
            int[] out = it.getOutputTensor(0).shape();
            if (in.length != 2 || in[0] != 1 || out.length != 2 || out[0] != 1) {
                throw new IllegalArgumentException("Model must have input [1,N] and output [1,M]");
            }
            if (it.getInputTensor(0).dataType() != DataType.FLOAT32
                    || it.getOutputTensor(0).dataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException("Model must use float32 input and output");
            }
            inputSize = in[1];
            outputSize = out[1];
        } catch (RuntimeException e) {
            it.close();
            throw e;
        }
        interpreter = it;
    }

    @Override
    public int inputSize() {
        return inputSize;
    }

    @Override
    public int outputSize() {
        return outputSize;
    }

    @Override
    public float[] run(float[] input) {
        float[][] in = new float[1][];
        in[0] = input;
        float[][] out = new float[1][outputSize];
        interpreter.run(in, out);
        return out[0];
    }

    @Override
    public void close() {
        interpreter.close();
    }
}
