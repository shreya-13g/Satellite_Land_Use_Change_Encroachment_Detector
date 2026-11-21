package com.satellite.mlp;

import com.satellite.tensor.Tensor;

/**
 * Standard loss functions and analytical loss gradients for neural network training.
 */
public enum LossFunction {

    /**
     * Mean Squared Error: L = 1/(2N) * sum((y_pred - y_true)^2)
     * Gradient dL/dy_pred = (y_pred - y_true) / N
     */
    MSE {
        @Override
        public double compute(Tensor yPred, Tensor yTrue) {
            Tensor diff = yPred.subtract(yTrue);
            return 0.5 * diff.multiply(diff).mean();
        }

        @Override
        public Tensor gradient(Tensor yPred, Tensor yTrue) {
            int n = yPred.getRows();
            return yPred.subtract(yTrue).divide(n);
        }
    },

    /**
     * Binary Cross Entropy:
     * L = -1/N * sum(y*ln(p + eps) + (1-y)*ln(1-p + eps))
     */
    BINARY_CROSS_ENTROPY {
        private static final double EPS = 1e-12;

        @Override
        public double compute(Tensor yPred, Tensor yTrue) {
            double[] p = yPred.getData();
            double[] y = yTrue.getData();
            int n = p.length;
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                double pi = Math.max(EPS, Math.min(1.0 - EPS, p[i]));
                double yi = y[i];
                sum += yi * Math.log(pi) + (1.0 - yi) * Math.log(1.0 - pi);
            }
            int samples = yPred.getRows();
            return -sum / samples;
        }

        @Override
        public Tensor gradient(Tensor yPred, Tensor yTrue) {
            int rows = yPred.getRows();
            int cols = yPred.getCols();
            double[] p = yPred.getData();
            double[] y = yTrue.getData();
            double[] grad = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                double pi = Math.max(EPS, Math.min(1.0 - EPS, p[i]));
                double yi = y[i];
                // dL/dp = (p - y) / (p*(1-p) * rows)
                grad[i] = ((pi - yi) / (pi * (1.0 - pi))) / rows;
            }
            return new Tensor(grad, rows, cols);
        }
    };

    public abstract double compute(Tensor yPred, Tensor yTrue);

    public abstract Tensor gradient(Tensor yPred, Tensor yTrue);
}
