package com.satellite.mlp;

import com.satellite.optimizers.BatchGradientDescent;
import com.satellite.optimizers.Optimizer;
import com.satellite.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Educational Multilayer Perceptron (MLP) with one hidden layer,
 * configurable activations, loss functions, weight initializations,
 * forward propagation, backpropagation, and pluggable optimizers.
 */
public class MultilayerPerceptron {

    private final int inputDim;
    private final int hiddenDim;
    private final int outputDim;
    private final Activation hiddenActivation;
    private final Activation outputActivation;
    private final LossFunction lossFunction;

    // Trainable parameters
    private Tensor w1; // [inputDim, hiddenDim]
    private Tensor b1; // [1, hiddenDim]
    private Tensor w2; // [hiddenDim, outputDim]
    private Tensor b2; // [1, outputDim]

    public MultilayerPerceptron(int inputDim, int hiddenDim, int outputDim,
                                Activation hiddenActivation, Activation outputActivation,
                                LossFunction lossFunction, WeightInitializer initializer,
                                long seed) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;
        this.hiddenActivation = hiddenActivation;
        this.outputActivation = outputActivation;
        this.lossFunction = lossFunction;

        Random rng = new Random(seed);
        this.w1 = initializer.initialize(inputDim, hiddenDim, rng);
        this.b1 = Tensor.zeros(1, hiddenDim);
        this.w2 = initializer.initialize(hiddenDim, outputDim, rng);
        this.b2 = Tensor.zeros(1, outputDim);
    }

    public MultilayerPerceptron(int inputDim, int hiddenDim, int outputDim) {
        this(inputDim, hiddenDim, outputDim, Activation.SIGMOID, Activation.SIGMOID,
                LossFunction.MSE, WeightInitializer.XAVIER, 42L);
    }

    public static class ForwardCache {
        public final Tensor x;
        public final Tensor z1;
        public final Tensor a1;
        public final Tensor z2;
        public final Tensor a2;

        public ForwardCache(Tensor x, Tensor z1, Tensor a1, Tensor z2, Tensor a2) {
            this.x = x;
            this.z1 = z1;
            this.a1 = a1;
            this.z2 = z2;
            this.a2 = a2;
        }
    }

    public static class Gradients {
        public final Tensor dw1;
        public final Tensor db1;
        public final Tensor dw2;
        public final Tensor db2;

        public Gradients(Tensor dw1, Tensor db1, Tensor dw2, Tensor db2) {
            this.dw1 = dw1;
            this.db1 = db1;
            this.dw2 = dw2;
            this.db2 = db2;
        }

        public List<Tensor> toList() {
            return Arrays.asList(dw1, db1, dw2, db2);
        }
    }

    // ------------------------------------------------------------------------
    // Forward Pass
    // ------------------------------------------------------------------------

    public ForwardCache forward(Tensor x) {
        if (x.getRank() != 2 || x.getCols() != inputDim) {
            throw new IllegalArgumentException(String.format("Expected input shape [N, %d], got %s",
                    inputDim, Arrays.toString(x.getShape())));
        }

        // Hidden layer: z1 = X * W1 + b1, a1 = g1(z1)
        Tensor z1 = x.matmul(w1).add(b1);
        Tensor a1 = hiddenActivation.activate(z1);

        // Output layer: z2 = a1 * W2 + b2, a2 = g2(z2)
        Tensor z2 = a1.matmul(w2).add(b2);
        Tensor a2 = outputActivation.activate(z2);

        return new ForwardCache(x, z1, a1, z2, a2);
    }

    public Tensor predict(Tensor x) {
        return forward(x).a2;
    }

    public Tensor predictClass(Tensor x, double threshold) {
        Tensor prob = predict(x);
        return prob.apply(v -> v >= threshold ? 1.0 : 0.0);
    }

    // ------------------------------------------------------------------------
    // Backward Pass (Backpropagation)
    // ------------------------------------------------------------------------

    public Gradients backward(ForwardCache cache, Tensor yTrue, double l2Lambda) {
        int n = cache.x.getRows();

        Tensor delta2;
        // Optimization: When output is Sigmoid + BCE, gradient simplifies cleanly to (a2 - y) / n
        if (outputActivation == Activation.SIGMOID && lossFunction == LossFunction.BINARY_CROSS_ENTROPY) {
            delta2 = cache.a2.subtract(yTrue).divide(n);
        } else {
            Tensor dL_da2 = lossFunction.gradient(cache.a2, yTrue);
            Tensor da2_dz2 = outputActivation.derivative(cache.z2, cache.a2);
            delta2 = dL_da2.multiply(da2_dz2);
        }

        // Gradients for layer 2
        Tensor dw2 = cache.a1.transpose().matmul(delta2);
        if (l2Lambda > 0.0) {
            dw2 = dw2.add(w2.multiply(l2Lambda));
        }
        Tensor db2 = delta2.sum(0);

        // Backpropagate to hidden layer
        Tensor delta1 = delta2.matmul(w2.transpose()).multiply(hiddenActivation.derivative(cache.z1, cache.a1));

        // Gradients for layer 1
        Tensor dw1 = cache.x.transpose().matmul(delta1);
        if (l2Lambda > 0.0) {
            dw1 = dw1.add(w1.multiply(l2Lambda));
        }
        Tensor db1 = delta1.sum(0);

        return new Gradients(dw1, db1, dw2, db2);
    }

    // ------------------------------------------------------------------------
    // Parameter Access
    // ------------------------------------------------------------------------

    public List<Tensor> getParameters() {
        return Arrays.asList(w1, b1, w2, b2);
    }

    public Tensor getW1() { return w1; }
    public Tensor getB1() { return b1; }
    public Tensor getW2() { return w2; }
    public Tensor getB2() { return b2; }

    public void setParameters(Tensor w1, Tensor b1, Tensor w2, Tensor b2) {
        this.w1 = w1;
        this.b1 = b1;
        this.w2 = w2;
        this.b2 = b2;
    }

    public LossFunction getLossFunction() { return lossFunction; }
    public Activation getHiddenActivation() { return hiddenActivation; }
    public Activation getOutputActivation() { return outputActivation; }

    // ------------------------------------------------------------------------
    // Training Loop
    // ------------------------------------------------------------------------

    public static class TrainingHistory {
        private final List<Integer> epochs = new ArrayList<>();
        private final List<Double> losses = new ArrayList<>();
        private final List<Double> accuracies = new ArrayList<>();

        public void add(int epoch, double loss, double accuracy) {
            epochs.add(epoch);
            losses.add(loss);
            accuracies.add(accuracy);
        }

        public List<Integer> getEpochs() { return epochs; }
        public List<Double> getLosses() { return losses; }
        public List<Double> getAccuracies() { return accuracies; }
        public double getFinalLoss() { return losses.isEmpty() ? 0.0 : losses.get(losses.size() - 1); }
        public double getFinalAccuracy() { return accuracies.isEmpty() ? 0.0 : accuracies.get(accuracies.size() - 1); }
    }

    public TrainingHistory fit(Tensor x, Tensor y, int epochs, double learningRate,
                               Optimizer optimizer, double l2Lambda, int logInterval) {
        if (optimizer == null) {
            optimizer = new BatchGradientDescent();
        }
        optimizer.initialize(getParameters());
        TrainingHistory history = new TrainingHistory();

        for (int ep = 1; ep <= epochs; ep++) {
            ForwardCache cache = forward(x);
            double loss = lossFunction.compute(cache.a2, y);

            // Compute accuracy for binary classification
            double acc = computeBinaryAccuracy(cache.a2, y);
            history.add(ep, loss, acc);

            Gradients grads = backward(cache, y, l2Lambda);
            optimizer.update(getParameters(), grads.toList(), learningRate);
        }

        return history;
    }

    public TrainingHistory fit(Tensor x, Tensor y, int epochs, double learningRate) {
        return fit(x, y, epochs, learningRate, new BatchGradientDescent(), 0.0, 100);
    }

    private double computeBinaryAccuracy(Tensor pred, Tensor target) {
        double[] p = pred.getData();
        double[] t = target.getData();
        int correct = 0;
        for (int i = 0; i < p.length; i++) {
            int predClass = p[i] >= 0.5 ? 1 : 0;
            int targetClass = t[i] >= 0.5 ? 1 : 0;
            if (predClass == targetClass) correct++;
        }
        return (double) correct / p.length;
    }
}
