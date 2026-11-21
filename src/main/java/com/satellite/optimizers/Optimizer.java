package com.satellite.optimizers;

import com.satellite.tensor.Tensor;

import java.util.List;

/**
 * Common contract for parameter optimization algorithms.
 */
public interface Optimizer {

    String getName();

    void initialize(List<Tensor> parameters);

    void update(List<Tensor> parameters, List<Tensor> gradients, double learningRate);
}
