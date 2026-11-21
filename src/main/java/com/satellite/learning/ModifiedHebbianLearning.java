package com.satellite.learning;

import java.util.Arrays;

/**
 * Modified Hebbian Learning using Oja's Rule (Erkki Oja, 1982).
 * Normalizes Hebbian weight growth and extracts the principal component:
 * Delta w = eta * y * (x - y * w), where y = w^T * x.
 */
public class ModifiedHebbianLearning {

    public static class Result {
        private final double[] initialWeights;
        private final double[] finalWeights;
        private final double[][] inputData;
        private final double finalWeightNorm;

        public Result(double[] initialWeights, double[] finalWeights, double[][] inputData, double finalWeightNorm) {
            this.initialWeights = initialWeights;
            this.finalWeights = finalWeights;
            this.inputData = inputData;
            this.finalWeightNorm = finalWeightNorm;
        }

        public double[] getInitialWeights() { return initialWeights; }
        public double[] getFinalWeights() { return finalWeights; }
        public double[][] getInputData() { return inputData; }
        public double getFinalWeightNorm() { return finalWeightNorm; }
    }

    public static Result runDemo(double[][] data, double[] initialWeights, double learningRate, int epochs) {
        int dim = initialWeights.length;
        double[] w = Arrays.copyOf(initialWeights, dim);

        for (int ep = 0; ep < epochs; ep++) {
            for (double[] x : data) {
                // y = w^T * x
                double y = 0.0;
                for (int i = 0; i < dim; i++) {
                    y += w[i] * x[i];
                }

                // Delta w_i = eta * y * (x_i - y * w_i)
                for (int i = 0; i < dim; i++) {
                    w[i] += learningRate * y * (x[i] - y * w[i]);
                }
            }
        }

        double normSq = 0.0;
        for (double v : w) normSq += v * v;
        double norm = Math.sqrt(normSq);

        return new Result(initialWeights, w, data, norm);
    }
}
