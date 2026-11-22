package com.satellite.optimizers;

import com.satellite.mlp.*;
import com.satellite.tensor.Tensor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Fair comparison of 4 parameter optimizers on the exact same XOR learning problem:
 * 1. Batch Gradient Descent (BGD)
 * 2. Stochastic Gradient Descent (SGD)
 * 3. Momentum
 * 4. Nesterov Accelerated Gradient (NAG)
 *
 * Saves genuinely computed loss histories to experiments/optimizer_comparison.csv
 * and summary to experiments/optimizer_summary.csv.
 */
public class OptimizerComparison {

    public static class OptimizerRecord {
        public final String optimizerName;
        public final int epoch;
        public final double loss;

        public OptimizerRecord(String name, int epoch, double loss) {
            this.optimizerName = name;
            this.epoch = epoch;
            this.loss = loss;
        }
    }

    public static class OptimizerSummary {
        public final String optimizerName;
        public final double finalLoss;

        public OptimizerSummary(String name, double finalLoss) {
            this.optimizerName = name;
            this.finalLoss = finalLoss;
        }
    }

    public static void runComparison(Path outputDir) throws IOException {
        Files.createDirectories(outputDir);

        // Standard XOR problem
        Tensor x = Tensor.of(new double[][]{
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        });
        Tensor y = Tensor.of(new double[][]{
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        });

        int epochs = 1000;
        double learningRate = 1.0;
        long fixedSeed = 100L;

        Optimizer[] optimizers = {
                new BatchGradientDescent(),
                new StochasticGradientDescent(0.001),
                new MomentumOptimizer(0.85),
                new NesterovOptimizer(0.85)
        };

        List<OptimizerRecord> epochRecords = new ArrayList<>();
        List<OptimizerSummary> summaryRecords = new ArrayList<>();

        for (Optimizer opt : optimizers) {
            MultilayerPerceptron mlp = new MultilayerPerceptron(
                    2, 4, 1,
                    Activation.SIGMOID, Activation.SIGMOID,
                    LossFunction.MSE, WeightInitializer.XAVIER,
                    fixedSeed
            );

            // Record epoch-by-epoch loss
            opt.initialize(mlp.getParameters());
            for (int ep = 1; ep <= epochs; ep++) {
                MultilayerPerceptron.ForwardCache cache = mlp.forward(x);
                double loss = mlp.computeLoss(cache.a2, y);

                MultilayerPerceptron.Gradients grads = mlp.backward(cache, y);
                opt.update(mlp.getParameters(), grads.toList(), learningRate);

                // Record every 50 epochs and final epoch
                if (ep % 50 == 0 || ep == 1 || ep == epochs) {
                    epochRecords.add(new OptimizerRecord(opt.getName(), ep, loss));
                }

                if (ep == epochs) {
                    MultilayerPerceptron.ForwardCache finalCache = mlp.forward(x);
                    double finalLoss = mlp.computeLoss(finalCache.a2, y);
                    summaryRecords.add(new OptimizerSummary(opt.getName(), finalLoss));
                }
            }
        }

        // Export epoch-level CSV
        Path compCsv = outputDir.resolve("optimizer_comparison.csv");
        try (BufferedWriter bw = Files.newBufferedWriter(compCsv)) {
            bw.write("Optimizer,Epoch,Loss\n");
            for (OptimizerRecord r : epochRecords) {
                bw.write(String.format("%s,%d,%.6f\n", r.optimizerName, r.epoch, r.loss));
            }
        }

        // Export summary CSV
        Path sumCsv = outputDir.resolve("optimizer_summary.csv");
        try (BufferedWriter bw = Files.newBufferedWriter(sumCsv)) {
            bw.write("Optimizer,FinalLoss\n");
            for (OptimizerSummary s : summaryRecords) {
                bw.write(String.format("%s,%.6f\n", s.optimizerName, s.finalLoss));
            }
        }

        System.out.println("Optimizer comparison successfully executed:");
        for (OptimizerSummary s : summaryRecords) {
            System.out.printf("  %s -> Final Loss: %.6f%n", s.optimizerName, s.finalLoss);
        }
        System.out.printf("  Saved results to %s and %s%n", compCsv, sumCsv);
    }

    public static void main(String[] args) throws IOException {
        runComparison(Paths.get("experiments"));
    }
}

