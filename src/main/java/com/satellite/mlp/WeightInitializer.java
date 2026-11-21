package com.satellite.mlp;

import com.satellite.tensor.Tensor;

import java.util.Random;

/**
 * Standard neural network weight initialization strategies.
 */
public enum WeightInitializer {

    ZERO {
        @Override
        public Tensor initialize(int fanIn, int fanOut, Random rng) {
            return Tensor.zeros(fanIn, fanOut);
        }
    },

    RANDOM_NORMAL {
        @Override
        public Tensor initialize(int fanIn, int fanOut, Random rng) {
            return Tensor.randn(new int[]{fanIn, fanOut}, 0.0, 0.1, rng);
        }
    },

    XAVIER {
        @Override
        public Tensor initialize(int fanIn, int fanOut, Random rng) {
            // Glorot uniform: limit = sqrt(6 / (fanIn + fanOut))
            double limit = Math.sqrt(6.0 / (fanIn + fanOut));
            return Tensor.random(new int[]{fanIn, fanOut}, -limit, limit, rng);
        }
    },

    HE {
        @Override
        public Tensor initialize(int fanIn, int fanOut, Random rng) {
            // He normal: std = sqrt(2 / fanIn)
            double std = Math.sqrt(2.0 / fanIn);
            return Tensor.randn(new int[]{fanIn, fanOut}, 0.0, std, rng);
        }
    };

    public abstract Tensor initialize(int fanIn, int fanOut, Random rng);
}
