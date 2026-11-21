package com.satellite.learning;

import java.util.Arrays;

/**
 * Classical Hebbian Learning rule (Donald Hebb, 1949).
 * "Neurons that fire together, wire together."
 * Weight update: Delta w_ij = eta * x_i * y_j
 */
public class HebbianLearning {

    public static class Result {
        private final double[][] initialWeights;
        private final double[][] finalWeights;
        private final double[][] inputPatterns;
        private final double learningRate;

        public Result(double[][] initialWeights, double[][] finalWeights, double[][] inputPatterns, double learningRate) {
            this.initialWeights = initialWeights;
            this.finalWeights = finalWeights;
            this.inputPatterns = inputPatterns;
            this.learningRate = learningRate;
        }

        public double[][] getInitialWeights() { return initialWeights; }
        public double[][] getFinalWeights() { return finalWeights; }
        public double[][] getInputPatterns() { return inputPatterns; }
        public double getLearningRate() { return learningRate; }
    }

    public static Result runDemo(double[][] patterns, double learningRate, int epochs) {
        int dim = patterns[0].length;
        double[][] weights = new double[dim][dim];
        double[][] initial = new double[dim][dim];

        // Zero or small initialization
        for (int i = 0; i < dim; i++) {
            System.arraycopy(weights[i], 0, initial[i], 0, dim);
        }

        for (int ep = 0; ep < epochs; ep++) {
            for (double[] x : patterns) {
                // For an auto-associative Hebbian rule, output y = x
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        if (i != j) {
                            weights[i][j] += learningRate * x[i] * x[j];
                        }
                    }
                }
            }
        }

        return new Result(initial, weights, patterns, learningRate);
    }
}
