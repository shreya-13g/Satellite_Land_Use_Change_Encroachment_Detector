package com.satellite.optimizers;

import com.satellite.mlp.*;
import com.satellite.tensor.Tensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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

        // Render Convergence Chart PNG
        Path chartPath = outputDir.resolve("optimizer_convergence_chart.png");
        renderConvergenceChart(epochRecords, chartPath);

        System.out.println("Optimizer comparison successfully executed:");
        for (OptimizerSummary s : summaryRecords) {
            System.out.printf("  %s -> Final Loss: %.6f%n", s.optimizerName, s.finalLoss);
        }
        System.out.printf("  Saved results to %s, %s, and %s%n", compCsv, sumCsv, chartPath);
    }

    /**
     * Renders loss convergence curves across all 4 optimizers using Java AWT Graphics2D.
     */
    public static void renderConvergenceChart(List<OptimizerRecord> records, Path outputPath) throws IOException {
        int width = 720;
        int height = 450;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Background
        g.setColor(new Color(248, 249, 250));
        g.fillRect(0, 0, width, height);

        // Header
        g.setColor(new Color(33, 37, 41));
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.drawString("Optimizer Convergence Comparison (XOR Baseline)", 40, 35);

        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g.setColor(new Color(108, 117, 125));
        g.drawString("Empirical Loss Curves over 1,000 Epochs (BGD, SGD, Momentum, Nesterov)", 40, 56);

        // Chart bounds
        int plotX = 80;
        int plotY = 80;
        int plotW = width - 140;
        int plotH = height - 160;

        // Axes
        g.setColor(new Color(200, 205, 210));
        g.drawRect(plotX, plotY, plotW, plotH);

        // Grid lines & Y-axis labels (Loss from 0.0 to 0.35)
        double maxLoss = 0.35;
        int yTicks = 5;
        g.setFont(new Font("SansSerif", Font.PLAIN, 10));
        g.setColor(new Color(140, 150, 160));
        for (int i = 0; i <= yTicks; i++) {
            double val = (maxLoss / yTicks) * i;
            int yPos = plotY + plotH - (int) Math.round((val / maxLoss) * plotH);
            g.drawString(String.format("%.2f", val), plotX - 35, yPos + 4);
            g.setColor(new Color(230, 233, 238));
            g.drawLine(plotX, yPos, plotX + plotW, yPos);
            g.setColor(new Color(140, 150, 160));
        }

        // X-axis labels
        int[] epochTicks = {0, 200, 400, 600, 800, 1000};
        for (int ep : epochTicks) {
            int xPos = plotX + (int) Math.round(((double) ep / 1000.0) * plotW);
            g.drawString(String.valueOf(ep), xPos - 10, plotY + plotH + 18);
        }
        g.drawString("Training Epochs", plotX + plotW / 2 - 40, plotY + plotH + 34);

        // Group records by optimizer
        Map<String, List<OptimizerRecord>> grouped = new LinkedHashMap<>();
        for (OptimizerRecord r : records) {
            grouped.computeIfAbsent(r.optimizerName, k -> new ArrayList<>()).add(r);
        }

        Map<String, Color> colors = new LinkedHashMap<>();
        colors.put("Batch Gradient Descent", new Color(0, 123, 255));      // Blue
        colors.put("Stochastic Gradient Descent", new Color(220, 53, 69)); // Red
        colors.put("Momentum Optimizer (beta=0.85)", new Color(40, 167, 69)); // Green
        colors.put("Nesterov Accelerated Gradient (beta=0.85)", new Color(255, 136, 0)); // Orange

        // Plot lines
        g.setStroke(new BasicStroke(2.2f));
        int legX = plotX + 20;
        int legY = height - 30;

        for (Map.Entry<String, List<OptimizerRecord>> entry : grouped.entrySet()) {
            String name = entry.getKey();
            List<OptimizerRecord> pts = entry.getValue();
            Color c = colors.getOrDefault(name, Color.DARK_GRAY);
            g.setColor(c);

            for (int i = 0; i < pts.size() - 1; i++) {
                OptimizerRecord p1 = pts.get(i);
                OptimizerRecord p2 = pts.get(i + 1);

                int x1 = plotX + (int) Math.round(((double) p1.epoch / 1000.0) * plotW);
                int y1 = plotY + plotH - (int) Math.round((Math.min(maxLoss, p1.loss) / maxLoss) * plotH);
                int x2 = plotX + (int) Math.round(((double) p2.epoch / 1000.0) * plotW);
                int y2 = plotY + plotH - (int) Math.round((Math.min(maxLoss, p2.loss) / maxLoss) * plotH);

                g.drawLine(x1, y1, x2, y2);
            }

            // Legend item
            g.fillRect(legX, legY, 14, 10);
            g.setColor(new Color(33, 37, 41));
            g.setFont(new Font("SansSerif", Font.PLAIN, 10));
            String shortName = name.replace(" Optimizer (beta=0.85)", "").replace(" (beta=0.85)", "");
            g.drawString(shortName, legX + 18, legY + 9);
            legX += g.getFontMetrics().stringWidth(shortName) + 32;
        }

        g.dispose();
        Files.createDirectories(outputPath.getParent());
        ImageIO.write(img, "PNG", outputPath.toFile());
    }

    public static void main(String[] args) throws IOException {
        runComparison(Paths.get("experiments"));
    }
}

