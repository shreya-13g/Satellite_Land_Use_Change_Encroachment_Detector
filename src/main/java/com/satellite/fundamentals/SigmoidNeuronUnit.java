package com.satellite.fundamentals;

import java.util.Arrays;

/**
 * Continuous Sigmoid Neuron:
 * sigma(x) = 1 / (1 + exp(-x))
 * Analytical derivative:
 * sigma'(x) = sigma(x) * (1 - sigma(x))
 */
public class SigmoidNeuronUnit {

    private final double[] weights;
    private final double bias;

    public SigmoidNeuronUnit(double[] weights, double bias) {
        this.weights = Arrays.copyOf(weights, weights.length);
        this.bias = bias;
    }

    /**
     * Sigmoid mathematical activation function.
     */
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Analytical derivative of sigmoid.
     */
    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    public double forward(double... inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Input dimension mismatch");
        }
        double z = bias;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * inputs[i];
        }
        return sigmoid(z);
    }
}

