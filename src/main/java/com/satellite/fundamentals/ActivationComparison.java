package com.satellite.fundamentals;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Educational comparison of primary neural activation functions:
 * 1. Step: y = 1 if x >= 0, else 0
 * 2. Sigmoid: y = 1 / (1 + exp(-x))
 * 3. Tanh: y = tanh(x)
 * 4. ReLU: y = max(0, x)
 *
 * Exports comparison values across a domain sweep to experiments/activation_comparison.csv.
 */
public class ActivationComparison {

    public static double step(double x) {
        return x >= 0.0 ? 1.0 : 0.0;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double relu(double x) {
        return Math.max(0.0, x);
    }

    public static void exportCsv(Path outputPath) throws IOException {
        Files.createDirectories(outputPath.getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(outputPath)) {
            bw.write("x,step,sigmoid,tanh,relu\n");
            // Sweep x from -3.0 to +3.0 with step 0.5
            for (double x = -3.0; x <= 3.001; x += 0.5) {
                bw.write(String.format("%.1f,%.4f,%.4f,%.4f,%.4f\n",
                        x, step(x), sigmoid(x), tanh(x), relu(x)));
            }
        }
        System.out.println("Activation function comparison exported to: " + outputPath.toAbsolutePath());
    }

    public static void main(String[] args) throws IOException {
        exportCsv(Paths.get("experiments", "activation_comparison.csv"));
    }
}

