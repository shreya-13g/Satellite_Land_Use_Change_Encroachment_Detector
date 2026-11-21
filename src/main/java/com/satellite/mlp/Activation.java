package com.satellite.mlp;

import com.satellite.neurons.SigmoidNeuron;
import com.satellite.tensor.Tensor;

/**
 * Common activation functions and their analytical derivatives for neural network layers.
 */
public enum Activation {

    SIGMOID {
        @Override
        public Tensor activate(Tensor z) {
            return z.apply(SigmoidNeuron::sigmoid);
        }

        @Override
        public Tensor derivative(Tensor z, Tensor activated) {
            // sigma'(z) = a * (1 - a)
            return activated.apply(SigmoidNeuron::sigmoidDerivativeFromOutput);
        }
    },

    TANH {
        @Override
        public Tensor activate(Tensor z) {
            return z.apply(Math::tanh);
        }

        @Override
        public Tensor derivative(Tensor z, Tensor activated) {
            // tanh'(z) = 1 - tanh^2(z)
            return activated.apply(a -> 1.0 - a * a);
        }
    },

    RELU {
        @Override
        public Tensor activate(Tensor z) {
            return z.apply(v -> Math.max(0.0, v));
        }

        @Override
        public Tensor derivative(Tensor z, Tensor activated) {
            // relu'(z) = z > 0 ? 1 : 0
            return z.apply(v -> v > 0.0 ? 1.0 : 0.0);
        }
    },

    LINEAR {
        @Override
        public Tensor activate(Tensor z) {
            return z.copy();
        }

        @Override
        public Tensor derivative(Tensor z, Tensor activated) {
            return Tensor.ones(z.getShape());
        }
    };

    public abstract Tensor activate(Tensor z);

    public abstract Tensor derivative(Tensor z, Tensor activated);
}
