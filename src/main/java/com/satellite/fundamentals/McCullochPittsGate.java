package com.satellite.fundamentals;

import java.util.Arrays;

/**
 * McCulloch-Pitts Artificial Neuron (1943).
 * Computes:
 * 1. Weighted sum: z = sum(w_i * x_i)
 * 2. Threshold: y = 1 if z >= threshold, else 0
 */
public class McCullochPittsGate {

    private final double[] weights;
    private final double threshold;

    public McCullochPittsGate(double[] weights, double threshold) {
        this.weights = Arrays.copyOf(weights, weights.length);
        this.threshold = threshold;
    }

    public int activate(double... inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Expected " + weights.length + " inputs, got " + inputs.length);
        }
        double weightedSum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            weightedSum += weights[i] * inputs[i];
        }
        return (weightedSum >= threshold) ? 1 : 0;
    }

    /**
     * Classic AND gate:
     * Weights = [1.0, 1.0], Threshold = 2.0
     * Only activates (1) when both inputs are 1.
     */
    public static McCullochPittsGate createAndGate() {
        return new McCullochPittsGate(new double[]{1.0, 1.0}, 2.0);
    }

    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double getThreshold() {
        return threshold;
    }
}

