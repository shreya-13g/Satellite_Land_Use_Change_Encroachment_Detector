package com.satellite.optimizers;

import com.satellite.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Classical Momentum Optimizer (Polyak, 1964).
 * Maintains a velocity buffer:
 * v = beta * v + eta * grad
 * theta = theta - v
 */
public class MomentumOptimizer implements Optimizer {

    private final double beta;
    private final List<Tensor> velocities = new ArrayList<>();

    public MomentumOptimizer(double beta) {
        this.beta = beta;
    }

    public MomentumOptimizer() {
        this(0.9);
    }

    @Override
    public String getName() {
        return "Momentum (beta=" + beta + ")";
    }

    @Override
    public void initialize(List<Tensor> parameters) {
        velocities.clear();
        for (Tensor p : parameters) {
            velocities.add(Tensor.zeros(p.getShape()));
        }
    }

    @Override
    public void update(List<Tensor> parameters, List<Tensor> gradients, double learningRate) {
        if (velocities.size() != parameters.size()) {
            initialize(parameters);
        }
        for (int i = 0; i < parameters.size(); i++) {
            Tensor p = parameters.get(i);
            Tensor g = gradients.get(i);
            Tensor v = velocities.get(i);

            double[] pData = p.getData();
            double[] gData = g.getData();
            double[] vData = v.getData();

            for (int j = 0; j < pData.length; j++) {
                vData[j] = beta * vData[j] + learningRate * gData[j];
                pData[j] -= vData[j];
            }
        }
    }

    public double getBeta() {
        return beta;
    }
}
