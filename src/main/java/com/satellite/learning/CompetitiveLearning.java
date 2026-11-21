package com.satellite.learning;

import java.util.Arrays;

/**
 * Competitive Learning / Winner-Take-All (WTA) unsupervised clustering.
 * Identifies the winning neuron closest to the input vector and updates only that winner:
 * w_winner = w_winner + eta * (x - w_winner).
 */
public class CompetitiveLearning {

    public static class Result {
        private final double[][] initialPrototypes;
        private final double[][] finalPrototypes;
        private final double[][] data;
        private final int[] clusterAssignments;

        public Result(double[][] initialPrototypes, double[][] finalPrototypes, double[][] data, int[] clusterAssignments) {
            this.initialPrototypes = initialPrototypes;
            this.finalPrototypes = finalPrototypes;
            this.data = data;
            this.clusterAssignments = clusterAssignments;
        }

        public double[][] getInitialPrototypes() { return initialPrototypes; }
        public double[][] getFinalPrototypes() { return finalPrototypes; }
        public double[][] getData() { return data; }
        public int[] getClusterAssignments() { return clusterAssignments; }
    }

    public static Result runDemo(double[][] data, double[][] initialPrototypes, double learningRate, int epochs) {
        int k = initialPrototypes.length;
        int dim = initialPrototypes[0].length;

        double[][] prototypes = new double[k][dim];
        for (int i = 0; i < k; i++) {
            System.arraycopy(initialPrototypes[i], 0, prototypes[i], 0, dim);
        }

        for (int ep = 0; ep < epochs; ep++) {
            for (double[] x : data) {
                // Find nearest prototype
                int winner = 0;
                double minDistSq = Double.MAX_VALUE;
                for (int i = 0; i < k; i++) {
                    double distSq = 0.0;
                    for (int d = 0; d < dim; d++) {
                        double diff = x[d] - prototypes[i][d];
                        distSq += diff * diff;
                    }
                    if (distSq < minDistSq) {
                        minDistSq = distSq;
                        winner = i;
                    }
                }

                // Update only the winning prototype
                for (int d = 0; d < dim; d++) {
                    prototypes[winner][d] += learningRate * (x[d] - prototypes[winner][d]);
                }
            }
        }

        // Final cluster assignment
        int[] assignments = new int[data.length];
        for (int s = 0; s < data.length; s++) {
            double[] x = data[s];
            int winner = 0;
            double minDistSq = Double.MAX_VALUE;
            for (int i = 0; i < k; i++) {
                double distSq = 0.0;
                for (int d = 0; d < dim; d++) {
                    double diff = x[d] - prototypes[i][d];
                    distSq += diff * diff;
                }
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    winner = i;
                }
            }
            assignments[s] = winner;
        }

        return new Result(initialPrototypes, prototypes, data, assignments);
    }
}
