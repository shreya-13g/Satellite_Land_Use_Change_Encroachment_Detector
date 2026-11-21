package com.satellite.learning;

import java.util.Arrays;

/**
 * Error-Correction Learning (Widrow-Hoff / Delta Rule, 1960).
 * Adjusts weights proportional to the difference between desired target and actual output:
 * error = target - y
 * Delta w = eta * error * x
 * Delta b = eta * error
 */
public class ErrorCorrectionLearning {

    public static class Result {
        private final double[] initialWeights;
        private final double[] finalWeights;
        private final double initialBias;
        private final double finalBias;
        private final double initialMse;
        private final double finalMse;

        public Result(double[] initialWeights, double[] finalWeights, double initialBias, double finalBias, double initialMse, double finalMse) {
            this.initialWeights = initialWeights;
            this.finalWeights = finalWeights;
            this.initialBias = initialBias;
            this.finalBias = finalBias;
            this.initialMse = initialMse;
            this.finalMse = finalMse;
        }

        public double[] getInitialWeights() { return initialWeights; }
        public double[] getFinalWeights() { return finalWeights; }
        public double getInitialBias() { return initialBias; }
        public double getFinalBias() { return finalBias; }
        public double getInitialMse() { return initialMse; }
        public double getFinalMse() { return finalMse; }
    }

    public static Result runDemo(double[][] X, double[] Y, double[] initialW, double initialB, double learningRate, int epochs) {
        int dim = initialW.length;
        double[] w = Arrays.copyOf(initialW, dim);
        double b = initialB;

        double initialMse = computeMse(X, Y, w, b);

        for (int ep = 0; ep < epochs; ep++) {
            for (int s = 0; s < X.length; s++) {
                double[] x = X[s];
                double target = Y[s];

                // Linear output
                double y = b;
                for (int i = 0; i < dim; i++) {
                    y += w[i] * x[i];
                }

                double error = target - y;
                for (int i = 0; i < dim; i++) {
                    w[i] += learningRate * error * x[i];
                }
                b += learningRate * error;
            }
        }

        double finalMse = computeMse(X, Y, w, b);
        return new Result(initialW, w, initialB, b, initialMse, finalMse);
    }

    private static double computeMse(double[][] X, double[] Y, double[] w, double b) {
        double sum = 0.0;
        for (int s = 0; s < X.length; s++) {
            double y = b;
            for (int i = 0; i < w.length; i++) {
                y += w[i] * X[s][i];
            }
            double diff = Y[s] - y;
            sum += diff * diff;
        }
        return sum / X.length;
    }
}
