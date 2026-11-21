package com.satellite.optimizers;

import com.satellite.tensor.Tensor;

import java.util.List;

/**
 * Standard Batch Gradient Descent (BGD).
 * Updates parameters: theta = theta - eta * grad.
 */
public class BatchGradientDescent implements Optimizer {

    @Override
    public String getName() {
        return "Batch Gradient Descent (BGD)";
    }

    @Override
    public void initialize(List<Tensor> parameters) {
        // Stateless optimizer
    }

    @Override
    public void update(List<Tensor> parameters, List<Tensor> gradients, double learningRate) {
        for (int i = 0; i < parameters.size(); i++) {
            Tensor p = parameters.get(i);
            Tensor g = gradients.get(i);
            double[] pData = p.getData();
            double[] gData = g.getData();
            for (int j = 0; j < pData.length; j++) {
                pData[j] -= learningRate * gData[j];
            }
        }
    }
}
