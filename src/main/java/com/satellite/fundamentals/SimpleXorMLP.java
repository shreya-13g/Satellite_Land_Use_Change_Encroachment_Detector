package com.satellite.fundamentals;

import java.util.Random;

/**
 * Educational Multilayer Perceptron (MLP) for XOR:
 * 2 Inputs -> Small Hidden Layer (4 neurons) -> 1 Output.
 *
 * Implements in plain Java with visible mathematics:
 * 1. Forward propagation: z1 = X * W1 + b1, a1 = sigmoid(z1), z2 = a1 * W2 + b2, a2 = sigmoid(z2)
 * 2. Mean Squared Error (MSE) loss: L = 0.5 * mean((y_pred - y_true)^2)
 * 3. Backpropagation: delta2 = (a2 - y) * a2 * (1 - a2), delta1 = (delta2 * W2^T) * a1 * (1 - a1)
 * 4. Batch parameter updates: W = W - eta * grad
 */
public class SimpleXorMLP {

    private final int inputDim = 2;
    private final int hiddenDim = 4;
    private final int outputDim = 1;

    private final double[][] w1; // [2][4]
    private final double[] b1;   // [4]
    private final double[][] w2; // [4][1]
    private final double[] b2;   // [1]

    public SimpleXorMLP(long seed) {
        Random rng = new Random(seed);
        w1 = new double[inputDim][hiddenDim];
        b1 = new double[hiddenDim];
        w2 = new double[hiddenDim][outputDim];
        b2 = new double[outputDim];

        // Standard Gaussian random initialization
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < hiddenDim; j++) {
                w1[i][j] = rng.nextGaussian();
            }
        }
        for (int j = 0; j < hiddenDim; j++) {
            w2[j][0] = rng.nextGaussian();
        }
    }

    public SimpleXorMLP() {
        this(42L);
    }

    private static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double sigmoidDeriv(double a) {
        return a * (1.0 - a);
    }

    public double predict(double[] x) {
        double[] a1 = new double[hiddenDim];
        for (int j = 0; j < hiddenDim; j++) {
            double z1 = b1[j] + x[0] * w1[0][j] + x[1] * w1[1][j];
            a1[j] = sigmoid(z1);
        }
        double z2 = b2[0];
        for (int j = 0; j < hiddenDim; j++) {
            z2 += a1[j] * w2[j][0];
        }
        return sigmoid(z2);
    }

    /**
     * Trains the MLP on the XOR dataset using batch gradient descent.
     */
    public double train(double[][] X, double[] Y, int epochs, double learningRate) {
        int n = X.length;
        double finalLoss = 0.0;

        for (int ep = 0; ep < epochs; ep++) {
            // Forward pass for all samples
            double[][] a1 = new double[n][hiddenDim];
            double[] a2 = new double[n];
            double epochLoss = 0.0;

            for (int s = 0; s < n; s++) {
                for (int j = 0; j < hiddenDim; j++) {
                    double z1 = b1[j] + X[s][0] * w1[0][j] + X[s][1] * w1[1][j];
                    a1[s][j] = sigmoid(z1);
                }
                double z2 = b2[0];
                for (int j = 0; j < hiddenDim; j++) {
                    z2 += a1[s][j] * w2[j][0];
                }
                a2[s] = sigmoid(z2);

                double err = a2[s] - Y[s];
                epochLoss += 0.5 * err * err;
            }

            finalLoss = epochLoss / n;

            // Backward pass (Batch Gradient Descent)
            double[] delta2 = new double[n];
            for (int s = 0; s < n; s++) {
                delta2[s] = (a2[s] - Y[s]) * sigmoidDeriv(a2[s]);
            }

            // Gradients for layer 2
            double[][] dW2 = new double[hiddenDim][1];
            double db2 = 0.0;
            for (int s = 0; s < n; s++) {
                db2 += delta2[s];
                for (int j = 0; j < hiddenDim; j++) {
                    dW2[j][0] += a1[s][j] * delta2[s];
                }
            }

            // Gradients for layer 1
            double[][] delta1 = new double[n][hiddenDim];
            double[][] dW1 = new double[inputDim][hiddenDim];
            double[] db1 = new double[hiddenDim];

            for (int s = 0; s < n; s++) {
                for (int j = 0; j < hiddenDim; j++) {
                    delta1[s][j] = delta2[s] * w2[j][0] * sigmoidDeriv(a1[s][j]);
                    db1[j] += delta1[s][j];
                    dW1[0][j] += X[s][0] * delta1[s][j];
                    dW1[1][j] += X[s][1] * delta1[s][j];
                }
            }

            // Parameter updates
            for (int j = 0; j < hiddenDim; j++) {
                w2[j][0] -= learningRate * dW2[j][0];
                b1[j] -= learningRate * db1[j];
                w1[0][j] -= learningRate * dW1[0][j];
                w1[1][j] -= learningRate * dW1[1][j];
            }
            b2[0] -= learningRate * db2;
        }

        return finalLoss;
    }
}
