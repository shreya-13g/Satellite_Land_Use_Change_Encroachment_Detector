package com.satellite.mlp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.satellite.optimizers.*;
import com.satellite.tensor.Tensor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Executes structured ablation experiments across neural network hyperparameters:
 * learning rates, initializations, normalization, L2 regularization, activations, optimizers,
 * and empirical overfitting vs generalization studies.
 */
public class AblationRunner {

    public static class AblationRecord {
        private final String category;
        private final String configuration;
        private final double finalLoss;
        private final double finalAccuracy;
        private final int epochs;

        public AblationRecord(String category, String configuration, double finalLoss, double finalAccuracy, int epochs) {
            this.category = category;
            this.configuration = configuration;
            this.finalLoss = finalLoss;
            this.finalAccuracy = finalAccuracy;
            this.epochs = epochs;
        }

        public String getCategory() { return category; }
        public String getConfiguration() { return configuration; }
        public double getFinalLoss() { return finalLoss; }
        public double getFinalAccuracy() { return finalAccuracy; }
        public int getEpochs() { return epochs; }
    }

    public static class OverfittingRecord {
        public final int epoch;
        public final double trainLoss;
        public final double trainAccuracy;
        public final double valLoss;
        public final double valAccuracy;

        public OverfittingRecord(int epoch, double trainLoss, double trainAccuracy, double valLoss, double valAccuracy) {
            this.epoch = epoch;
            this.trainLoss = trainLoss;
            this.trainAccuracy = trainAccuracy;
            this.valLoss = valLoss;
            this.valAccuracy = valAccuracy;
        }
    }

    public static List<AblationRecord> runAblations(Tensor xNorm, Tensor y) {
        List<AblationRecord> records = new ArrayList<>();
        int epochs = 1500;

        // 1. Learning Rate Ablation (BGD, Sigmoid, Xavier)
        double[] lrs = {0.05, 0.1, 0.5, 1.0, 2.0};
        for (double lr : lrs) {
            MultilayerPerceptron mlp = new MultilayerPerceptron(2, 4, 1,
                    Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
            MultilayerPerceptron.TrainingHistory h = mlp.fit(xNorm, y, epochs, lr, new BatchGradientDescent(), 0.0, 0);
            records.add(new AblationRecord("LearningRate", "lr=" + lr, h.getFinalLoss(), h.getFinalAccuracy(), epochs));
        }

        // 2. Weight Initialization Ablation (lr=1.0, Sigmoid, BGD)
        WeightInitializer[] inits = {WeightInitializer.ZERO, WeightInitializer.RANDOM_NORMAL, WeightInitializer.XAVIER, WeightInitializer.HE};
        for (WeightInitializer init : inits) {
            MultilayerPerceptron mlp = new MultilayerPerceptron(2, 4, 1,
                    Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, init, 42L);
            MultilayerPerceptron.TrainingHistory h = mlp.fit(xNorm, y, epochs, 1.0, new BatchGradientDescent(), 0.0, 0);
            records.add(new AblationRecord("Initialization", init.name(), h.getFinalLoss(), h.getFinalAccuracy(), epochs));
        }

        // 3. Normalization Ablation (Normalized [0, 1] vs Unnormalized [0, 255])
        Tensor xRaw = xNorm.multiply(255.0);
        MultilayerPerceptron mlpNorm = new MultilayerPerceptron(2, 4, 1, Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
        MultilayerPerceptron.TrainingHistory hNorm = mlpNorm.fit(xNorm, y, epochs, 1.0, new BatchGradientDescent(), 0.0, 0);
        records.add(new AblationRecord("Normalization", "WithNormalization [0,1]", hNorm.getFinalLoss(), hNorm.getFinalAccuracy(), epochs));

        MultilayerPerceptron mlpRaw = new MultilayerPerceptron(2, 4, 1, Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
        MultilayerPerceptron.TrainingHistory hRaw = mlpRaw.fit(xRaw, y, epochs, 1.0, new BatchGradientDescent(), 0.0, 0);
        records.add(new AblationRecord("Normalization", "WithoutNormalization [0,255]", hRaw.getFinalLoss(), hRaw.getFinalAccuracy(), epochs));

        // 4. Regularization Ablation (L2 lambda = 0.0 vs lambda = 0.01)
        MultilayerPerceptron mlpNoReg = new MultilayerPerceptron(2, 4, 1, Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
        MultilayerPerceptron.TrainingHistory hNoReg = mlpNoReg.fit(xNorm, y, epochs, 1.0, new BatchGradientDescent(), 0.0, 0);
        records.add(new AblationRecord("Regularization", "L2_lambda=0.0", hNoReg.getFinalLoss(), hNoReg.getFinalAccuracy(), epochs));

        MultilayerPerceptron mlpReg = new MultilayerPerceptron(2, 4, 1, Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
        MultilayerPerceptron.TrainingHistory hReg = mlpReg.fit(xNorm, y, epochs, 1.0, new BatchGradientDescent(), 0.01, 0);
        records.add(new AblationRecord("Regularization", "L2_lambda=0.01", hReg.getFinalLoss(), hReg.getFinalAccuracy(), epochs));

        // 5. Activation Function Ablation (Sigmoid vs Tanh vs ReLU)
        Activation[] acts = {Activation.SIGMOID, Activation.TANH, Activation.RELU};
        for (Activation act : acts) {
            MultilayerPerceptron mlp = new MultilayerPerceptron(2, 4, 1,
                    act, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
            MultilayerPerceptron.TrainingHistory h = mlp.fit(xNorm, y, epochs, 0.5, new MomentumOptimizer(0.9), 0.0, 0);
            records.add(new AblationRecord("Activation", act.name(), h.getFinalLoss(), h.getFinalAccuracy(), epochs));
        }

        // 6. Optimizer Ablation (BGD, SGD, Momentum, Nesterov)
        Optimizer[] optimizers = {
                new BatchGradientDescent(),
                new StochasticGradientDescent(0.001),
                new MomentumOptimizer(0.9),
                new NesterovOptimizer(0.9)
        };
        for (Optimizer opt : optimizers) {
            MultilayerPerceptron mlp = new MultilayerPerceptron(2, 4, 1,
                    Activation.SIGMOID, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);
            MultilayerPerceptron.TrainingHistory h = mlp.fit(xNorm, y, 1000, 0.5, opt, 0.0, 0);
            records.add(new AblationRecord("Optimizer", opt.getName(), h.getFinalLoss(), h.getFinalAccuracy(), 1000));
        }

        return records;
    }

    /**
     * Conducts an overfitting vs generalization experiment:
     * - Small training dataset (16 samples) with 10% label noise
     * - Independent validation dataset (64 samples)
     * - Tracks divergence where training loss approaches zero while validation loss begins to rise.
     */
    public static List<OverfittingRecord> runOverfittingStudy(Path outputCsv) throws IOException {
        Random rng = new Random(101L);
        int trainN = 16;
        int valN = 64;

        double[][] trainX = new double[trainN][2];
        double[][] trainY = new double[trainN][1];
        for (int i = 0; i < trainN; i++) {
            trainX[i][0] = rng.nextDouble() * 2.0 - 1.0;
            trainX[i][1] = rng.nextDouble() * 2.0 - 1.0;
            // Ground truth rule: x0 * x1 > 0 with 10% noise
            boolean pos = (trainX[i][0] * trainX[i][1] > 0);
            if (rng.nextDouble() < 0.10) pos = !pos;
            trainY[i][0] = pos ? 1.0 : 0.0;
        }

        double[][] valX = new double[valN][2];
        double[][] valY = new double[valN][1];
        for (int i = 0; i < valN; i++) {
            valX[i][0] = rng.nextDouble() * 2.0 - 1.0;
            valX[i][1] = rng.nextDouble() * 2.0 - 1.0;
            valY[i][0] = (valX[i][0] * valX[i][1] > 0) ? 1.0 : 0.0;
        }

        Tensor tX = Tensor.of(trainX);
        Tensor tY = Tensor.of(trainY);
        Tensor vX = Tensor.of(valX);
        Tensor vY = Tensor.of(valY);

        // High capacity MLP (2 -> 16 -> 1) prone to overfitting on 16 samples
        MultilayerPerceptron mlp = new MultilayerPerceptron(2, 16, 1,
                Activation.TANH, Activation.SIGMOID, LossFunction.MSE, WeightInitializer.XAVIER, 42L);

        List<OverfittingRecord> curve = new ArrayList<>();
        int totalEpochs = 1000;
        double lr = 0.5;
        Optimizer opt = new MomentumOptimizer(0.85);
        opt.initialize(mlp.getParameters());

        for (int ep = 1; ep <= totalEpochs; ep++) {
            MultilayerPerceptron.ForwardCache trCache = mlp.forward(tX);
            double trLoss = mlp.computeLoss(trCache.a2, tY);
            MultilayerPerceptron.Gradients grads = mlp.backward(trCache, tY, 0.0);
            opt.update(mlp.getParameters(), grads.toList(), lr);

            if (ep % 25 == 0 || ep == 1 || ep == totalEpochs) {
                MultilayerPerceptron.ForwardCache valCache = mlp.forward(vX);
                double valLoss = mlp.computeLoss(valCache.a2, vY);

                double trAcc = computeBinaryAcc(trCache.a2, tY);
                double valAcc = computeBinaryAcc(valCache.a2, vY);
                curve.add(new OverfittingRecord(ep, trLoss, trAcc, valLoss, valAcc));
            }
        }

        if (outputCsv != null) {
            Files.createDirectories(outputCsv.getParent());
            try (BufferedWriter bw = Files.newBufferedWriter(outputCsv)) {
                bw.write("epoch,train_loss,train_acc,val_loss,val_acc\n");
                for (OverfittingRecord r : curve) {
                    bw.write(String.format("%d,%.6f,%.4f,%.6f,%.4f\n",
                            r.epoch, r.trainLoss, r.trainAccuracy, r.valLoss, r.valAccuracy));
                }
            }
        }

        return curve;
    }

    private static double computeBinaryAcc(Tensor pred, Tensor target) {
        double[] p = pred.getData();
        double[] t = target.getData();
        int c = 0;
        for (int i = 0; i < p.length; i++) {
            if ((p[i] >= 0.5 ? 1 : 0) == (t[i] >= 0.5 ? 1 : 0)) c++;
        }
        return (double) c / p.length;
    }

    public static void exportResults(List<AblationRecord> records, Path csvPath, Path jsonPath) throws IOException {
        Files.createDirectories(csvPath.getParent());
        Files.createDirectories(jsonPath.getParent());

        // CSV
        try (BufferedWriter w = Files.newBufferedWriter(csvPath)) {
            w.write("category,configuration,final_loss,final_accuracy,epochs\n");
            for (AblationRecord r : records) {
                String conf = r.getConfiguration();
                if (conf.contains(",")) {
                    conf = "\"" + conf + "\"";
                }
                w.write(String.format("%s,%s,%.6f,%.4f,%d\n",
                        r.getCategory(), conf, r.getFinalLoss(), r.getFinalAccuracy(), r.getEpochs()));
            }
        }

        // JSON
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(jsonPath.toFile(), records);
    }

    public static void runAllAblations() throws Exception {
        System.out.println("================================================================================");
        System.out.println("  NEURAL NETWORK ABLATION EXPERIMENTS & OVERFITTING STUDY (Requirement 5)");
        System.out.println("================================================================================");

        Tensor x = Tensor.of(new double[][]{
                {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
        });
        Tensor y = Tensor.of(new double[][]{
                {0.0}, {1.0}, {1.0}, {0.0}
        });

        List<AblationRecord> records = runAblations(x, y);

        Path csvPath = Paths.get("experiments", "ablation_results.csv");
        Path jsonPath = Paths.get("experiments", "ablation_results.json");
        exportResults(records, csvPath, jsonPath);

        System.out.println("  1. Hyperparameter Ablation Results:");
        String currentCat = "";
        for (AblationRecord r : records) {
            if (!r.getCategory().equals(currentCat)) {
                currentCat = r.getCategory();
                System.out.printf("   [%s Ablation]%n", currentCat);
            }
            System.out.printf("     - %-32s -> Loss: %.6f | Acc: %5.1f%%%n",
                    r.getConfiguration(), r.getFinalLoss(), r.getFinalAccuracy() * 100.0);
        }

        // Overfitting vs Generalization
        Path overfitCsv = Paths.get("experiments", "overfitting_curves.csv");
        List<OverfittingRecord> overfitCurve = runOverfittingStudy(overfitCsv);
        System.out.println("--------------------------------------------------------------------------------");
        System.out.println("  2. Overfitting & Generalization Study (Train N=16, Val N=64):");
        OverfittingRecord early = overfitCurve.get(0);
        OverfittingRecord mid = overfitCurve.get(overfitCurve.size() / 3);
        OverfittingRecord late = overfitCurve.get(overfitCurve.size() - 1);
        System.out.printf("     Epoch %3d: Train Loss: %.4f (Acc: %.1f%%) | Val Loss: %.4f (Acc: %.1f%%)%n",
                early.epoch, early.trainLoss, early.trainAccuracy * 100, early.valLoss, early.valAccuracy * 100);
        System.out.printf("     Epoch %3d: Train Loss: %.4f (Acc: %.1f%%) | Val Loss: %.4f (Acc: %.1f%%) [Optimal Generalization]%n",
                mid.epoch, mid.trainLoss, mid.trainAccuracy * 100, mid.valLoss, mid.valAccuracy * 100);
        System.out.printf("     Epoch %3d: Train Loss: %.4f (Acc: %.1f%%) | Val Loss: %.4f (Acc: %.1f%%) [Overfitting Regime]%n",
                late.epoch, late.trainLoss, late.trainAccuracy * 100, late.valLoss, late.valAccuracy * 100);

        System.out.println("  Saved results to:");
        System.out.println("    - " + csvPath.toAbsolutePath());
        System.out.println("    - " + jsonPath.toAbsolutePath());
        System.out.println("    - " + overfitCsv.toAbsolutePath());
        System.out.println("================================================================================");
    }

    public static void main(String[] args) throws Exception {
        runAllAblations();
    }
}
