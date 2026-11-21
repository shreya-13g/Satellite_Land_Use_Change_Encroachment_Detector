package com.satellite.optimizers;

import com.satellite.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Nesterov Accelerated Gradient (NAG) optimizer.
 * Computes lookahead momentum step:
 * v_next = beta * v + eta * grad
 * theta = theta - (beta * v_next + eta * grad)
 * v = v_next
 */
public class NesterovOptimizer implements Optimizer {

    private final double beta;
    private final List<Tensor> velocities = new ArrayList<>();

    public NesterovOptimizer(double beta) {
        this.beta = beta;
    }

    public NesterovOptimizer() {
        this(0.9);
    }

    @Override
    public String getName() {
        return "Nesterov Momentum (beta=" + beta + ")";
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
                double vNext = beta * vData[j] + learningRate * gData[j];
                pData[j] -= (beta * vNext + learningRate * gData[j]);
                vData[j] = vNext;
            }
        }
    }

    public double getBeta() {
        return beta;
    }
}
