package com.satellite.neurons;

import java.util.Arrays;

/**
 * Fundamental McCulloch-Pitts artificial neuron (1943).
 * Operates on binary or weighted inputs with a hard threshold activation:
 * y = 1 if sum(w_i * x_i) >= theta, else 0.
 */
public class McCullochPittsNeuron {

    private final double[] weights;
    private final double threshold;

    public McCullochPittsNeuron(double[] weights, double threshold) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights array must be non-empty");
        }
        this.weights = Arrays.copyOf(weights, weights.length);
        this.threshold = threshold;
    }

    public int activate(double... inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(String.format("Expected %d inputs, got %d", weights.length, inputs.length));
        }
        double sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return (sum >= threshold) ? 1 : 0;
    }

    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double getThreshold() {
        return threshold;
    }

    // Factory methods for classic logic gates
    public static McCullochPittsNeuron createAndGate() {
        return new McCullochPittsNeuron(new double[]{1.0, 1.0}, 2.0);
    }

    public static McCullochPittsNeuron createOrGate() {
        return new McCullochPittsNeuron(new double[]{1.0, 1.0}, 1.0);
    }

    public static McCullochPittsNeuron createNotGate() {
        return new McCullochPittsNeuron(new double[]{-1.0}, 0.0);
    }
}
