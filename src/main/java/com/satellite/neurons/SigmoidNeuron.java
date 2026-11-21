package com.satellite.neurons;

import com.satellite.tensor.Tensor;

import java.util.Arrays;

/**
 * Continuous Sigmoid artificial neuron with weighted sum, bias,
 * numerically stable activation, derivative computation, and binary thresholding.
 */
public class SigmoidNeuron {

    private final double[] weights;
    private double bias;
    private double threshold;

    public SigmoidNeuron(double[] weights, double bias, double threshold) {
        if (weights == null || weights.length == 0) {
            throw new IllegalArgumentException("Weights array must be non-empty");
        }
        this.weights = Arrays.copyOf(weights, weights.length);
        this.bias = bias;
        this.threshold = threshold;
    }

    public SigmoidNeuron(double[] weights, double bias) {
        this(weights, bias, 0.5);
    }

    /**
     * Numerically stable sigmoid activation:
     * sigma(z) = 1 / (1 + e^(-z)) for z >= 0
     * sigma(z) = e^z / (1 + e^z) for z < 0
     */
    public static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    /**
     * Analytical derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z)).
     */
    public static double sigmoidDerivative(double z) {
        double s = sigmoid(z);
        return s * (1.0 - s);
    }

    public static double sigmoidDerivativeFromOutput(double output) {
        return output * (1.0 - output);
    }

    public double computeWeightedSum(double... inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(String.format("Expected %d inputs, got %d", weights.length, inputs.length));
        }
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    public double forward(double... inputs) {
        return sigmoid(computeWeightedSum(inputs));
    }

    public int predict(double... inputs) {
        return forward(inputs) >= threshold ? 1 : 0;
    }

    public double[] getWeights() {
        return Arrays.copyOf(weights, weights.length);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}
