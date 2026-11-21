package com.satellite.learning;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Educational Energy-Based Model demonstration using a discrete Hopfield Network / Boltzmann Energy model.
 * Demonstrates associative memory, Lyapunov energy minimization, and deterministic pattern recovery:
 * E(s) = -0.5 * sum_{i != j} W_{ij} * s_i * s_j
 */
public class HopfieldEnergyDemo {

    public static class StepRecord {
        private final int step;
        private final int[] state;
        private final double energy;
        private final boolean isStable;
        private final boolean matchesTarget;

        public StepRecord(int step, int[] state, double energy, boolean isStable, boolean matchesTarget) {
            this.step = step;
            this.state = Arrays.copyOf(state, state.length);
            this.energy = energy;
            this.isStable = isStable;
            this.matchesTarget = matchesTarget;
        }

        public int getStep() { return step; }
        public int[] getState() { return state; }
        public double getEnergy() { return energy; }
        public boolean isStable() { return isStable; }
        public boolean isMatchesTarget() { return matchesTarget; }
    }

    public static class EnergyResult {
        private final int[] targetPattern;
        private final int[] initialCorrupted;
        private final double[][] weights;
        private final List<StepRecord> trajectory;

        public EnergyResult(int[] targetPattern, int[] initialCorrupted, double[][] weights, List<StepRecord> trajectory) {
            this.targetPattern = targetPattern;
            this.initialCorrupted = initialCorrupted;
            this.weights = weights;
            this.trajectory = trajectory;
        }

        public int[] getTargetPattern() { return targetPattern; }
        public int[] getInitialCorrupted() { return initialCorrupted; }
        public double[][] getWeights() { return weights; }
        public List<StepRecord> getTrajectory() { return trajectory; }
    }

    public static EnergyResult run(int[] target, int[] corruptedIndices, int maxIterations) {
        int n = target.length;
        double[][] w = new double[n][n];

        // Hebbian weight calculation: W_ij = (1/n) * xi_i * xi_j with zero diagonal
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    w[i][j] = (double) (target[i] * target[j]) / n;
                }
            }
        }

        // Initialize state with corrupted pattern
        int[] s = Arrays.copyOf(target, n);
        for (int idx : corruptedIndices) {
            if (idx >= 0 && idx < n) {
                s[idx] = -s[idx]; // bit flip
            }
        }
        int[] initialCorrupted = Arrays.copyOf(s, n);

        List<StepRecord> trajectory = new ArrayList<>();
        double currentEnergy = computeEnergy(s, w);
        trajectory.add(new StepRecord(0, s, currentEnergy, false, Arrays.equals(s, target)));

        for (int step = 1; step <= maxIterations; step++) {
            boolean changed = false;

            // Deterministic asynchronous update across all units
            for (int i = 0; i < n; i++) {
                double field = 0.0;
                for (int j = 0; j < n; j++) {
                    field += w[i][j] * s[j];
                }
                int newState = field >= 0.0 ? 1 : -1;
                if (newState != s[i]) {
                    s[i] = newState;
                    changed = true;
                }
            }

            currentEnergy = computeEnergy(s, w);
            boolean isTarget = Arrays.equals(s, target);
            trajectory.add(new StepRecord(step, s, currentEnergy, !changed, isTarget));

            if (!changed) {
                break; // Converged to stable attractor state
            }
        }

        return new EnergyResult(target, initialCorrupted, w, trajectory);
    }

    public static double computeEnergy(int[] s, double[][] w) {
        int n = s.length;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += w[i][j] * s[i] * s[j];
            }
        }
        return -0.5 * sum;
    }

    public static void exportResults(EnergyResult result, Path csvPath) throws IOException {
        Files.createDirectories(csvPath.getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(csvPath)) {
            writer.write("step,state,energy,is_stable,matches_target\n");
            for (StepRecord rec : result.getTrajectory()) {
                StringBuilder stateStr = new StringBuilder();
                for (int bit : rec.getState()) {
                    stateStr.append(bit > 0 ? "+" : "-");
                }
                writer.write(String.format("%d,%s,%.4f,%b,%b\n",
                        rec.getStep(), stateStr, rec.getEnergy(), rec.isStable(), rec.matchesTarget()));
            }
        }
    }
}
