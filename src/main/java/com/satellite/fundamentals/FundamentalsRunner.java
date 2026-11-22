package com.satellite.fundamentals;

/**
 * Console demonstration runner for Course Deep-Learning Fundamentals.
 * Completely separate from production satellite detection pipeline.
 */
public class FundamentalsRunner {

    public static void runAll() {
        System.out.println("================================================================================");
        System.out.println("  DEEP LEARNING COURSE FUNDAMENTALS (PURE JAVA)");
        System.out.println("================================================================================");

        // 1. Tensor linear algebra
        System.out.println("\n[1/6] Tensor Operations:");
        SimpleTensor t = SimpleTensor.of2D(new double[][]{{1, 2}, {3, 4}});
        SimpleTensor tT = t.transpose();
        SimpleTensor tProd = t.matmul(tT);
        System.out.printf("  Input (2x2) Matmul Transpose -> top-left: %.1f, top-right: %.1f%n",
                tProd.get(0, 0), tProd.get(0, 1));

        // 2. SVD Demonstration
        System.out.println("\n[2/6] Singular Value Decomposition (A = U * S * V^T):");
        double[][] sampleMatrix = {
                {4.0, 0.0, 0.0},
                {0.0, 3.0, 0.0},
                {0.0, 0.0, 2.0}
        };
        SvdDemo.SvdResult svdRes = SvdDemo.decompose(sampleMatrix);
        System.out.printf("  Singular values: [%.2f, %.2f, %.2f], Reconstruction Error: %.2e%n",
                svdRes.singularValues[0], svdRes.singularValues[1], svdRes.singularValues[2],
                svdRes.reconstructionFrobeniusError);

        // 3. Numerical Gradient Checking
        System.out.println("\n[3/6] Numerical Gradient Checking (Central Finite Differences):");
        NumericalGradientChecker.CheckResult gradCheck = NumericalGradientChecker.check(
                x -> x * x, // f(x) = x^2
                x -> 2.0 * x, // f'(x) = 2x
                3.0, 1e-5, 1e-5
        );
        System.out.printf("  f(x)=x^2 at x=3.0 -> Numerical: %.6f, Analytical: %.6f, Relative Error: %.2e (Passed: %b)%n",
                gradCheck.numericalGradient, gradCheck.analyticalGradient, gradCheck.relativeError, gradCheck.passed);

        // 4. McCulloch-Pitts Neuron
        System.out.println("\n[4/6] McCulloch-Pitts Neuron (AND Logic Gate):");
        McCullochPittsGate andGate = McCullochPittsGate.createAndGate();
        System.out.printf("  AND(0,0)=%d, AND(0,1)=%d, AND(1,0)=%d, AND(1,1)=%d%n",
                andGate.activate(0, 0), andGate.activate(0, 1), andGate.activate(1, 0), andGate.activate(1, 1));

        // 5. Sigmoid Neuron
        System.out.println("\n[5/6] Sigmoid Continuous Neuron:");
        SigmoidNeuronUnit sigmoidUnit = new SigmoidNeuronUnit(new double[]{0.5, 0.5}, -0.5);
        double sigOut = sigmoidUnit.forward(1.0, 1.0);
        System.out.printf("  Sigmoid forward([1.0, 1.0], w=[0.5, 0.5], b=-0.5) -> %.4f (deriv=%.4f)%n",
                sigOut, SigmoidNeuronUnit.sigmoidDerivative(sigOut));

        // 6. Simple XOR MLP
        System.out.println("\n[6/6] Simple MLP Learning XOR Problem (2-3-1):");
        double[][] xorX = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[] xorY = {0, 1, 1, 0};
        SimpleXorMLP mlp = new SimpleXorMLP(42L);
        double finalLoss = mlp.train(xorX, xorY, 5000, 1.5);
        System.out.printf("  Training complete. Final MSE Loss: %.4f%n", finalLoss);
        System.out.printf("  Predictions: (0,0)->%.3f, (0,1)->%.3f, (1,0)->%.3f, (1,1)->%.3f%n",
                mlp.predict(xorX[0]), mlp.predict(xorX[1]), mlp.predict(xorX[2]), mlp.predict(xorX[3]));
        System.out.println("================================================================================");
    }
}

