package com.satellite.learning;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * Demonstration runner for educational learning rules:
 * - Hebbian Learning
 * - Modified Hebbian (Oja's rule)
 * - Competitive Learning (Winner-Take-All)
 * - Error-Correction Learning (Widrow-Hoff delta rule)
 * - Hopfield Attractor Energy minimization
 */
public class LearningRulesDemo {

    public static void runAll() {
        System.out.println("================================================================================");
        System.out.println("  EDUCATIONAL NEURAL LEARNING RULES DEMONSTRATION");
        System.out.println("================================================================================");

        // 1. Classical Hebbian
        System.out.println("\n[1/5] Classical Hebbian Learning (\"Neurons that fire together wire together\"):");
        double[][] patterns = {
                {1.0, 1.0, -1.0},
                {-1.0, -1.0, 1.0}
        };
        HebbianLearning.Result hebRes = HebbianLearning.runDemo(patterns, 0.1, 5);
        System.out.printf("  Learned correlation weight W[0][1]: %.3f, W[0][2]: %.3f%n",
                hebRes.getFinalWeights()[0][1], hebRes.getFinalWeights()[0][2]);

        // 2. Modified Hebbian (Oja's Rule)
        System.out.println("\n[2/5] Modified Hebbian (Oja's Rule - Normalizes weight vector):");
        double[][] ojaData = {
                {1.0, 2.0},
                {2.0, 4.0},
                {-1.0, -2.0}
        };
        double[] initW = {0.5, 0.5};
        ModifiedHebbianLearning.Result ojaRes = ModifiedHebbianLearning.runDemo(ojaData, initW, 0.05, 100);
        System.out.printf("  Final weight vector: [%.3f, %.3f] (L2 Norm: %.4f - bounded)%n",
                ojaRes.getFinalWeights()[0], ojaRes.getFinalWeights()[1], ojaRes.getFinalWeightNorm());

        // 3. Competitive Learning (Winner-Take-All)
        System.out.println("\n[3/5] Competitive Learning (WTA Prototype Clustering):");
        double[][] clusterData = {
                {0.1, 0.2}, {0.2, 0.1},
                {0.9, 0.8}, {0.8, 0.9}
        };
        double[][] initPrototypes = {
                {0.0, 0.0},
                {1.0, 1.0}
        };
        CompetitiveLearning.Result compRes = CompetitiveLearning.runDemo(clusterData, initPrototypes, 0.1, 50);
        System.out.printf("  Cluster prototype 0: [%.3f, %.3f], Prototype 1: [%.3f, %.3f]%n",
                compRes.getFinalPrototypes()[0][0], compRes.getFinalPrototypes()[0][1],
                compRes.getFinalPrototypes()[1][0], compRes.getFinalPrototypes()[1][1]);

        // 4. Error-Correction Learning (Widrow-Hoff)
        System.out.println("\n[4/5] Error-Correction Learning (Delta Rule):");
        double[][] deltaX = {{1.0}, {2.0}, {3.0}, {4.0}};
        double[] deltaY = {2.0, 4.0, 6.0, 8.0}; // target: y = 2*x
        double[] initDeltaW = {0.1};
        ErrorCorrectionLearning.Result deltaRes = ErrorCorrectionLearning.runDemo(deltaX, deltaY, initDeltaW, 0.0, 0.05, 200);
        System.out.printf("  Initial MSE: %.4f -> Final MSE: %.4f, Learned weight: %.4f (target=2.000)%n",
                deltaRes.getInitialMse(), deltaRes.getFinalMse(), deltaRes.getFinalWeights()[0]);

        // 5. Hopfield Energy Minimization
        System.out.println("\n[5/5] Hopfield Network Energy Minimization (Attractor Dynamics):");
        int[] targetPattern = {1, 1, -1, -1, 1};
        int[] corrupted = {0, 1}; // corrupt bits at index 0 and 1
        HopfieldEnergyDemo.EnergyResult hopRes = HopfieldEnergyDemo.run(targetPattern, corrupted, 10);
        HopfieldEnergyDemo.StepRecord initStep = hopRes.getTrajectory().get(0);
        HopfieldEnergyDemo.StepRecord finalStep = hopRes.getTrajectory().get(hopRes.getTrajectory().size() - 1);
        System.out.printf("  Initial Energy: %.2f -> Converged Energy: %.2f (Stable: %b, Target Recovered: %b)%n",
                initStep.getEnergy(), finalStep.getEnergy(), finalStep.isStable(), finalStep.matchesTarget());

        try {
            HopfieldEnergyDemo.exportResults(hopRes, Paths.get("experiments", "hopfield_energy_trajectory.csv"));
            System.out.println("  Saved Hopfield energy trajectory to experiments/hopfield_energy_trajectory.csv");
        } catch (IOException e) {
            System.err.println("  Warning: could not save Hopfield trajectory: " + e.getMessage());
        }

        System.out.println("================================================================================");
    }

    public static void main(String[] args) {
        runAll();
    }
}

