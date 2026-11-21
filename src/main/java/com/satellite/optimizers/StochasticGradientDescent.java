package com.satellite.optimizers;

import com.satellite.tensor.Tensor;

import java.util.List;

/**
 * Stochastic Gradient Descent (SGD) with optional learning rate decay.
 */
public class StochasticGradientDescent implements Optimizer {

    private final double decayRate;
    private int stepCount = 0;

    public StochasticGradientDescent(double decayRate) {
        this.decayRate = decayRate;
    }

    public StochasticGradientDescent() {
        this(0.0);
    }

    @Override
    public String getName() {
        return "Stochastic Gradient Descent (SGD)";
    }

    @Override
    public void initialize(List<Tensor> parameters) {
        stepCount = 0;
    }

    @Override
    public void update(List<Tensor> parameters, List<Tensor> gradients, double learningRate) {
        stepCount++;
        double effectiveLr = learningRate / (1.0 + decayRate * stepCount);
        for (int i = 0; i < parameters.size(); i++) {
            Tensor p = parameters.get(i);
            Tensor g = gradients.get(i);
            double[] pData = p.getData();
            double[] gData = g.getData();
            for (int j = 0; j < pData.length; j++) {
                pData[j] -= effectiveLr * gData[j];
            }
        }
    }
}
