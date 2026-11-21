package com.satellite.mlp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.satellite.optimizers.*;
import com.satellite.tensor.Tensor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Executes structured ablation experiments across neural network hyperparameters:
 * learning rates, initializations, normalization, L2 regularization, activations, and optimizers.
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

    public static List<AblationRecord> runAblations(Tensor xNorm, Tensor y) {
        List<AblationRecord> records = new ArrayList<>();
        int epochs = 2000;

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

    public static void exportResults(List<AblationRecord> records, Path csvPath, Path jsonPath) throws IOException {
        Files.createDirectories(csvPath.getParent());
        Files.createDirectories(jsonPath.getParent());

        // CSV
        try (BufferedWriter w = Files.newBufferedWriter(csvPath)) {
            w.write("category,configuration,final_loss,final_accuracy,epochs\n");
            for (AblationRecord r : records) {
                w.write(String.format("%s,%s,%.6f,%.4f,%d\n",
                        r.getCategory(), r.getConfiguration(), r.getFinalLoss(), r.getFinalAccuracy(), r.getEpochs()));
            }
        }

        // JSON
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(jsonPath.toFile(), records);
    }
}
