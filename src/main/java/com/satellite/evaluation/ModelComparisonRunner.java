package com.satellite.evaluation;

import com.satellite.change.ChangeDetector;
import com.satellite.classification.OnnxResNetChangeDetector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Executes a fair, head-to-head empirical comparison between:
 * 1. Classical Pixel-Difference Baseline
 * 2. Deep-Learning Component (Siamese Pretrained ResNet-18)
 *
 * Saves genuinely computed metrics to results/model_comparison.csv
 * and renders a multi-metric diagnostic comparison chart.
 */
public class ModelComparisonRunner {

    private static final Logger logger = LoggerFactory.getLogger(ModelComparisonRunner.class);

    public static class ModelBenchmarkResult {
        public final String method;
        public final double precision;
        public final double recall;
        public final double f1;
        public final double iou;
        public final long runtimeMs;

        public ModelBenchmarkResult(String method, double precision, double recall, double f1, double iou, long runtimeMs) {
            this.method = method;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
            this.iou = iou;
            this.runtimeMs = runtimeMs;
        }
    }

    public static List<ModelBenchmarkResult> compareModels(
            List<Path> filesA, List<Path> filesB, List<Path> filesL,
            Path resultsDir, int sampleLimit) throws Exception {

        Files.createDirectories(resultsDir);
        List<ModelBenchmarkResult> comparison = new ArrayList<>();

        // 1. Evaluate Classical Baseline
        logger.info("Evaluating Classical Change-Detection Baseline...");
        ChangeDetector baseline = new ChangeDetector(35.0, 30);
        Path baselineOut = resultsDir.resolve("baseline");
        Path baselineViz = Paths.get("visualizations", "change-detection");
        ChangeDetector.BatchEvaluationResult baseRes = baseline.evaluateBatch(
                filesA, filesB, filesL, baselineOut, baselineViz, sampleLimit);

        comparison.add(new ModelBenchmarkResult(
                "Classical Baseline (Pixel Difference)",
                baseRes.getMeanPrecision(),
                baseRes.getMeanRecall(),
                baseRes.getMeanF1(),
                baseRes.getMeanIou(),
                baseRes.getTotalRuntimeMs()
        ));

        // 2. Evaluate Deep Learning Component (Siamese ResNet-18)
        logger.info("Evaluating Deep-Learning Model (Siamese ResNet-18)...");
        Path onnxPath = Paths.get("models", "onnx", "resnet18_change_detector.onnx");
        if (Files.exists(onnxPath)) {
            try (OnnxResNetChangeDetector dlDetector = new OnnxResNetChangeDetector(onnxPath, 30)) {
                Path dlOut = resultsDir.resolve("deep-learning");
                Path dlViz = Paths.get("visualizations", "deep-learning");
                ChangeDetector.BatchEvaluationResult dlRes = dlDetector.evaluateBatch(
                        filesA, filesB, filesL, dlOut, dlViz, sampleLimit);

                comparison.add(new ModelBenchmarkResult(
                        "Deep Learning (Siamese ResNet-18)",
                        dlRes.getMeanPrecision(),
                        dlRes.getMeanRecall(),
                        dlRes.getMeanF1(),
                        dlRes.getMeanIou(),
                        dlRes.getTotalRuntimeMs()
                ));
            }
        } else {
            logger.warn("ONNX model file not found at {}. Skipping DL comparison.", onnxPath);
        }

        // 3. Export results/model_comparison.csv
        Path csvPath = resultsDir.resolve("model_comparison.csv");
        try (BufferedWriter bw = Files.newBufferedWriter(csvPath)) {
            bw.write("Method,Precision,Recall,F1,IoU,RuntimeMs\n");
            for (ModelBenchmarkResult r : comparison) {
                bw.write(String.format("%s,%.4f,%.4f,%.4f,%.4f,%d\n",
                        r.method, r.precision, r.recall, r.f1, r.iou, r.runtimeMs));
            }
        }
        logger.info("Saved model comparison CSV to {}", csvPath.toAbsolutePath());

        // 4. Render visual comparison chart
        renderComparisonChart(comparison, resultsDir.resolve("model_comparison_chart.png"));

        return comparison;
    }

    /**
     * Renders a clean 2D comparison bar chart using standard Java AWT Graphics.
     */
    public static void renderComparisonChart(List<ModelBenchmarkResult> results, Path outputPath) throws IOException {
        if (results.isEmpty()) return;

        int width = 700;
        int height = 400;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Background
        g.setColor(new Color(248, 249, 250));
        g.fillRect(0, 0, width, height);

        // Title
        g.setColor(new Color(33, 37, 41));
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.drawString("Benchmark Comparison: Baseline vs Deep Learning", 40, 40);

        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g.setColor(new Color(108, 117, 125));
        g.drawString("Dataset: LEVIR-CD (256x256 Tiles) | Metrics Computed on Ground Truth", 40, 62);

        // Metrics to plot: Precision, Recall, F1, IoU
        String[] metricNames = {"Precision", "Recall", "F1-Score", "IoU"};
        int groupCount = metricNames.length;
        int groupWidth = 120;
        int startX = 80;
        int baseY = 320;
        int maxBarHeight = 200;

        // Draw axes
        g.setColor(new Color(200, 205, 210));
        g.drawLine(StartXAxis(startX), baseY, startX + groupCount * groupWidth, baseY);

        Color colorBaseline = new Color(70, 130, 180);  // Steel Blue
        Color colorDL = new Color(40, 167, 69);         // Forest Green

        for (int m = 0; m < groupCount; m++) {
            int gx = startX + m * groupWidth;

            // Baseline bar
            double valBase = getMetricValue(results.get(0), m);
            int hBase = (int) Math.round(valBase * maxBarHeight);
            g.setColor(colorBaseline);
            g.fillRect(gx, baseY - hBase, 30, hBase);
            g.setColor(Color.DARK_GRAY);
            g.setFont(new Font("SansSerif", Font.PLAIN, 10));
            g.drawString(String.format("%.2f", valBase), gx + 2, baseY - hBase - 5);

            // DL bar
            if (results.size() > 1) {
                double valDL = getMetricValue(results.get(1), m);
                int hDL = (int) Math.round(valDL * maxBarHeight);
                g.setColor(colorDL);
                g.fillRect(gx + 35, baseY - hDL, 30, hDL);
                g.setColor(Color.DARK_GRAY);
                g.drawString(String.format("%.2f", valDL), gx + 37, baseY - hDL - 5);
            }

            // Metric label
            g.setColor(new Color(33, 37, 41));
            g.setFont(new Font("SansSerif", Font.BOLD, 12));
            g.drawString(metricNames[m], gx + 8, baseY + 20);
        }

        // Legend
        int legX = 400;
        int legY = 360;
        g.setColor(colorBaseline);
        g.fillRect(legX, legY, 15, 12);
        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.PLAIN, 11));
        g.drawString("Classical Baseline", legX + 22, legY + 10);

        if (results.size() > 1) {
            g.setColor(colorDL);
            g.fillRect(legX + 150, legY, 15, 12);
            g.setColor(Color.BLACK);
            g.drawString("Siamese ResNet-18", legX + 172, legY + 10);
        }

        g.dispose();
        ImageIO.write(img, "PNG", outputPath.toFile());
        logger.info("Saved comparison chart to {}", outputPath.toAbsolutePath());
    }

    private static int StartXAxis(int startX) {
        return startX - 10;
    }

    private static double getMetricValue(ModelBenchmarkResult r, int metricIdx) {
        switch (metricIdx) {
            case 0: return r.precision;
            case 1: return r.recall;
            case 2: return r.f1;
            case 3: return r.iou;
            default: return 0.0;
        }
    }
}

