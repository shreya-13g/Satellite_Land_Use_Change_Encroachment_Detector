package com.satellite.fundamentals;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 test suite verifying pure Java deep-learning fundamentals:
 * - Tensor reshape, matmul, inner product, transpose, addition, scalar mult
 * - Sigmoid activation & derivative
 * - McCulloch-Pitts AND logic gate
 * - Numerical gradient checking vs analytical gradient
 * - SVD reconstruction
 * - Simple MLP forward pass and training on XOR
 */
class FundamentalsTest {

    @Test
    @DisplayName("Verify Tensor Creation, Reshape, and Flatten")
    void testTensorReshapeAndFlatten() {
        SimpleTensor t = SimpleTensor.of(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assertArrayEquals(new int[]{6}, t.getShape());

        SimpleTensor reshaped = t.reshape(2, 3);
        assertArrayEquals(new int[]{2, 3}, reshaped.getShape());
        assertEquals(1.0, reshaped.get(0, 0));
        assertEquals(6.0, reshaped.get(1, 2));

        SimpleTensor flattened = reshaped.flatten();
        assertArrayEquals(new int[]{6}, flattened.getShape());
        assertEquals(4.0, flattened.getData()[3]);
    }

    @Test
    @DisplayName("Verify Tensor Transpose and Matrix Multiplication")
    void testTensorMatmulAndTranspose() {
        // A = [[1, 2], [3, 4]] (2x2)
        SimpleTensor a = SimpleTensor.of2D(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        // B = [[2, 0], [1, 2]] (2x2)
        SimpleTensor b = SimpleTensor.of2D(new double[][]{{2.0, 0.0}, {1.0, 2.0}});

        // C = A * B
        // C[0][0] = 1*2 + 2*1 = 4.0
        // C[0][1] = 1*0 + 2*2 = 4.0
        // C[1][0] = 3*2 + 4*1 = 10.0
        // C[1][1] = 3*0 + 4*2 = 8.0
        SimpleTensor c = a.matmul(b);
        assertEquals(4.0, c.get(0, 0), 1e-6);
        assertEquals(4.0, c.get(0, 1), 1e-6);
        assertEquals(10.0, c.get(1, 0), 1e-6);
        assertEquals(8.0, c.get(1, 1), 1e-6);

        // A^T = [[1, 3], [2, 4]]
        SimpleTensor aT = a.transpose();
        assertEquals(3.0, aT.get(0, 1), 1e-6);
        assertEquals(2.0, aT.get(1, 0), 1e-6);
    }

    @Test
    @DisplayName("Verify Vector Inner Product (Dot Product)")
    void testInnerProduct() {
        SimpleTensor v1 = SimpleTensor.of(1.0, 3.0, -5.0);
        SimpleTensor v2 = SimpleTensor.of(4.0, -2.0, -1.0);

        // 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3.0
        double dot = v1.innerProduct(v2);
        assertEquals(3.0, dot, 1e-6);
    }

    @Test
    @DisplayName("Verify Sigmoid Neuron Activation and Derivative")
    void testSigmoidNeuron() {
        assertEquals(0.5, SigmoidNeuronUnit.sigmoid(0.0), 1e-6, "sigmoid(0) must be 0.5");
        assertTrue(SigmoidNeuronUnit.sigmoid(5.0) > 0.99);
        assertTrue(SigmoidNeuronUnit.sigmoid(-5.0) < 0.01);

        // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
        assertEquals(0.25, SigmoidNeuronUnit.sigmoidDerivative(0.0), 1e-6);

        SigmoidNeuronUnit unit = new SigmoidNeuronUnit(new double[]{1.0, -1.0}, 0.0);
        assertEquals(0.5, unit.forward(2.0, 2.0), 1e-6);
    }

    @Test
    @DisplayName("Verify McCulloch-Pitts Neuron AND Gate")
    void testMcCullochPittsAndGate() {
        McCullochPittsGate andGate = McCullochPittsGate.createAndGate();
        assertEquals(0, andGate.activate(0.0, 0.0));
        assertEquals(0, andGate.activate(0.0, 1.0));
        assertEquals(0, andGate.activate(1.0, 0.0));
        assertEquals(1, andGate.activate(1.0, 1.0));
    }

    @Test
    @DisplayName("Verify Numerical Gradient Checking (Central Differences)")
    void testNumericalGradientCheck() {
        // f(x) = x^2, f'(x) = 2x
        NumericalGradientChecker.CheckResult res = NumericalGradientChecker.check(
                x -> x * x,
                x -> 2.0 * x,
                4.0,
                1e-5,
                1e-4
        );
        assertTrue(res.passed, "Numerical gradient check for x^2 must pass");
        assertEquals(8.0, res.analyticalGradient, 1e-6);
        assertEquals(8.0, res.numericalGradient, 1e-4);
        assertTrue(res.relativeError < 1e-5);
    }

    @Test
    @DisplayName("Verify SVD Decomposition and Matrix Reconstruction")
    void testSvdDecomposition() {
        double[][] a = {
                {3.0, 2.0},
                {2.0, 3.0}
        };
        SvdDemo.SvdResult res = SvdDemo.decompose(a);
        assertNotNull(res.singularValues);
        assertEquals(2, res.singularValues.length);
        assertTrue(res.singularValues[0] >= res.singularValues[1]);
        assertTrue(res.reconstructionFrobeniusError < 1e-9, "Reconstruction error must be virtually zero");
    }

    @Test
    @DisplayName("Verify Simple MLP Forward Pass and XOR Convergence")
    void testSimpleXorMLP() {
        double[][] x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] y = {0, 1, 1, 0};

        SimpleXorMLP mlp = new SimpleXorMLP(42L);

        // Verify initial forward pass produces valid probabilities
        for (double[] sample : x) {
            double pred = mlp.predict(sample);
            assertTrue(pred >= 0.0 && pred <= 1.0);
        }

        // Train network on XOR
        double finalLoss = mlp.train(x, y, 6000, 1.5);
        assertTrue(finalLoss < 0.05, "MLP should converge on XOR with MSE < 0.05");

        // Verify accurate classification after training
        assertTrue(mlp.predict(x[0]) < 0.25, "(0,0) -> 0");
        assertTrue(mlp.predict(x[1]) > 0.75, "(0,1) -> 1");
        assertTrue(mlp.predict(x[2]) > 0.75, "(1,0) -> 1");
        assertTrue(mlp.predict(x[3]) < 0.25, "(1,1) -> 0");
    }
}

