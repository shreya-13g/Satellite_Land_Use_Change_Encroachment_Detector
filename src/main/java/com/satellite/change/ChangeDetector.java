package com.satellite.change;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.satellite.detection.EncroachmentDetector;
import com.satellite.detection.EncroachmentResult;
import com.satellite.evaluation.ChangeMetrics;
import com.satellite.visualization.ChangeVisualizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates classical bi-temporal change detection, noise removal,
 * connected component extraction, encroachment screening, evaluation metrics, and visualization.
 */
public class ChangeDetector {

    private static final Logger logger = LoggerFactory.getLogger(ChangeDetector.class);

    private final DifferenceCalculator diffCalculator;
    private final ChangeMaskGenerator maskGenerator;
    private final NoiseRemover noiseRemover;
    private final ConnectedComponentAnalyzer componentAnalyzer;
    private final EncroachmentDetector encroachmentDetector;
    private final ChangeVisualizer visualizer;
    private boolean useOtsu = false;

    public ChangeDetector(double threshold, int minRegionArea, DifferenceCalculator.Mode diffMode) {
        this.diffCalculator = new DifferenceCalculator(diffMode);
        this.maskGenerator = new ChangeMaskGenerator(threshold);
        this.noiseRemover = new NoiseRemover(minRegionArea, true);
        this.componentAnalyzer = new ConnectedComponentAnalyzer(minRegionArea);
        this.encroachmentDetector = new EncroachmentDetector();
        this.visualizer = new ChangeVisualizer();
        this.useOtsu = false;
    }

    public ChangeDetector(double threshold, int minRegionArea, boolean useOtsu) {
        this(threshold, minRegionArea, DifferenceCalculator.Mode.EUCLIDEAN_RGB);
        this.useOtsu = useOtsu;
    }

    public ChangeDetector(double threshold, int minRegionArea) {
        this(threshold, minRegionArea, DifferenceCalculator.Mode.EUCLIDEAN_RGB);
    }

    public ChangeDetector() {
        this(ChangeMaskGenerator.DEFAULT_THRESHOLD, NoiseRemover.DEFAULT_MIN_REGION_AREA);
    }

    public boolean isUseOtsu() {
        return useOtsu;
    }

    public void setUseOtsu(boolean useOtsu) {
        this.useOtsu = useOtsu;
    }

    /**
     * Executes change detection on an aligned pair of images (with optional ground-truth mask).
     */
    public DetectionOutput processPair(BufferedImage imgA, BufferedImage imgB, BufferedImage gtMask, String sampleId) {
        long startTime = System.currentTimeMillis();

        // 1. Difference map
        float[][] diffMap = diffCalculator.calculateDifference(imgA, imgB);

        // 2. Binary mask generation (static threshold or Otsu adaptive)
        byte[][] rawMask = maskGenerator.generateMask(diffMap, useOtsu);

        // 3. Noise removal & morphological cleanup
        byte[][] cleanMask = noiseRemover.cleanMask(rawMask);

        // 4. Connected component analysis
        List<ChangeRegion> regions = componentAnalyzer.findRegions(cleanMask);

        // 5. Total changed pixels
        int changedPixels = 0;
        int totalPixels = cleanMask.length * cleanMask[0].length;
        for (int y = 0; y < cleanMask.length; y++) {
            for (int x = 0; x < cleanMask[0].length; x++) {
                if (cleanMask[y][x] == 1) changedPixels++;
            }
        }

        // 6. Encroachment screening classification
        EncroachmentResult result = encroachmentDetector.evaluate(sampleId, changedPixels, totalPixels, regions);

        // 7. Metrics evaluation if ground truth is supplied
        byte[][] gtArray = null;
        if (gtMask != null) {
            gtArray = extractBinaryArray(gtMask);
            long runtimeMs = System.currentTimeMillis() - startTime;
            ChangeMetrics metrics = ChangeMetrics.compute(cleanMask, gtArray, sampleId, runtimeMs);
            result.setMetrics(metrics);
        }

        return new DetectionOutput(result, cleanMask, gtArray, diffMap);
    }

    /**
     * Saves full diagnostic visualization assets for a processed pair.
     */
    public void saveVisualizations(BufferedImage imgA, BufferedImage imgB, BufferedImage gtMask,
                                   DetectionOutput output, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);
        String id = output.getResult().getSampleId();

        BufferedImage predMaskImg = maskGenerator.toMaskImage(output.getCleanMask());
        BufferedImage bboxImg = visualizer.renderBoundingBoxes(imgB, output.getResult().getRegions(), output.getResult().getStatus());
        BufferedImage confusionImg = visualizer.renderConfusionOverlay(imgB, output.getCleanMask(), output.getGroundTruthArray());

        // Individual visual files
        ImageIO.write(imgA, "PNG", outputDir.resolve(id + "_before.png").toFile());
        ImageIO.write(imgB, "PNG", outputDir.resolve(id + "_after.png").toFile());
        ImageIO.write(predMaskImg, "PNG", outputDir.resolve(id + "_predicted_mask.png").toFile());
        ImageIO.write(bboxImg, "PNG", outputDir.resolve(id + "_bounding_boxes.png").toFile());
        ImageIO.write(confusionImg, "PNG", outputDir.resolve(id + "_highlighted_changes.png").toFile());

        BufferedImage gtImg = (gtMask != null) ? gtMask : new BufferedImage(imgA.getWidth(), imgA.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        if (gtMask != null) {
            ImageIO.write(gtMask, "PNG", outputDir.resolve(id + "_ground_truth.png").toFile());
        }

        // Composite 5-panel board
        BufferedImage board = visualizer.renderSideBySideBoard(imgA, imgB, gtImg, predMaskImg, bboxImg, output.getResult());
        Path boardPath = outputDir.resolve(id + "_comparison_board.png");
        ImageIO.write(board, "PNG", boardPath.toFile());

        logger.info("Saved visualizations for {} to {}", id, outputDir.toAbsolutePath());
    }

    /**
     * Evaluates a collection of pairs and exports summary reports, CSVs, and JSON.
     */
    public BatchEvaluationResult evaluateBatch(List<Path> filesA, List<Path> filesB, List<Path> filesLabel,
                                              Path resultsDir, Path vizDir, int maxSamplesToVisualize) throws IOException {
        Files.createDirectories(resultsDir);
        if (vizDir != null) Files.createDirectories(vizDir);

        List<EncroachmentResult> results = new ArrayList<>();
        List<ChangeMetrics> metricsList = new ArrayList<>();

        long totalTp = 0;
        long totalTn = 0;
        long totalFp = 0;
        long totalFn = 0;

        int n = filesA.size();
        logger.info("Starting batch evaluation on {} tile pairs...", n);

        for (int i = 0; i < n; i++) {
            Path pA = filesA.get(i);
            Path pB = filesB.get(i);
            Path pL = (filesLabel != null && i < filesLabel.size()) ? filesLabel.get(i) : null;

            String sampleId = stripExtension(pA.getFileName().toString()).replace("_before", "").replace("_tile", "");

            BufferedImage imgA = ImageIO.read(pA.toFile());
            BufferedImage imgB = ImageIO.read(pB.toFile());
            BufferedImage imgL = (pL != null && Files.exists(pL)) ? ImageIO.read(pL.toFile()) : null;

            DetectionOutput output = processPair(imgA, imgB, imgL, sampleId);
            results.add(output.getResult());

            if (output.getResult().getMetrics() != null) {
                ChangeMetrics m = output.getResult().getMetrics();
                metricsList.add(m);
                totalTp += m.getTruePositives();
                totalTn += m.getTrueNegatives();
                totalFp += m.getFalsePositives();
                totalFn += m.getFalseNegatives();
            }

            if (vizDir != null && i < maxSamplesToVisualize) {
                saveVisualizations(imgA, imgB, imgL, output, vizDir);
            }
        }

        // Aggregate statistics
        double meanIou = metricsList.stream().mapToDouble(ChangeMetrics::getIou).average().orElse(0.0);
        double meanF1 = metricsList.stream().mapToDouble(ChangeMetrics::getF1Score).average().orElse(0.0);
        double meanPrecision = metricsList.stream().mapToDouble(ChangeMetrics::getPrecision).average().orElse(0.0);
        double meanRecall = metricsList.stream().mapToDouble(ChangeMetrics::getRecall).average().orElse(0.0);
        double meanAreaError = metricsList.stream().mapToDouble(ChangeMetrics::getAreaErrorPixels).average().orElse(0.0);
        long totalRuntime = metricsList.stream().mapToLong(ChangeMetrics::getRuntimeMs).sum();

        BatchEvaluationResult batchResult = new BatchEvaluationResult(
                results, metricsList, meanIou, meanF1, meanPrecision, meanRecall, meanAreaError,
                totalTp, totalTn, totalFp, totalFn, totalRuntime
        );

        // 1. change-results.json
        Path jsonPath = resultsDir.resolve("change-results.json");
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(jsonPath.toFile(), results);

        // 2. change-metrics.csv
        Path csvPath = resultsDir.resolve("change-metrics.csv");
        try (BufferedWriter w = Files.newBufferedWriter(csvPath)) {
            w.write("sample_id,status,iou,f1_score,precision,recall,changed_area_px,ground_truth_px,area_error_px,runtime_ms\n");
            for (EncroachmentResult r : results) {
                ChangeMetrics m = r.getMetrics();
                if (m != null) {
                    w.write(String.format("%s,%s,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n",
                            r.getSampleId(), r.getStatus(),
                            m.getIou(), m.getF1Score(), m.getPrecision(), m.getRecall(),
                            r.getTotalChangedPixels(), m.getGroundTruthChangedPixels(),
                            m.getAreaErrorPixels(), m.getRuntimeMs()));
                } else {
                    w.write(String.format("%s,%s,,,,,%d,,,\n", r.getSampleId(), r.getStatus(), r.getTotalChangedPixels()));
                }
            }
        }

        // 3. confusion-matrix.csv
        Path cmPath = resultsDir.resolve("confusion-matrix.csv");
        try (BufferedWriter w = Files.newBufferedWriter(cmPath)) {
            w.write("metric,pixel_count\n");
            w.write(String.format("true_positives,%d\n", totalTp));
            w.write(String.format("true_negatives,%d\n", totalTn));
            w.write(String.format("false_positives,%d\n", totalFp));
            w.write(String.format("false_negatives,%d\n", totalFn));
            w.write(String.format("mean_iou,%.4f\n", meanIou));
            w.write(String.format("mean_f1,%.4f\n", meanF1));
            w.write(String.format("mean_precision,%.4f\n", meanPrecision));
            w.write(String.format("mean_recall,%.4f\n", meanRecall));
            w.write(String.format("mean_area_error_pixels,%.1f\n", meanAreaError));
        }

        // 4. summary-report.md
        Path reportPath = resultsDir.resolve("summary-report.md");
        writeSummaryReport(batchResult, reportPath);

        logger.info("Batch evaluation complete across {} samples. Mean IoU: {:.4f}, Mean F1: {:.4f}", n, meanIou, meanF1);
        return batchResult;
    }

    private void writeSummaryReport(BatchEvaluationResult res, Path reportPath) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("# Stage 3 Change Detection & Encroachment Screening Report\n\n");
        sb.append("**Evaluation Scope:** LEVIR-CD Processed Test Dataset (256x256 Tiles)\n");
        sb.append("**Method:** Classical Multitemporal Radiometric Differencing, Morphological Noise Removal & Connected Component Screening\n\n");

        sb.append("## 1. Quantitative Benchmark Metrics\n\n");
        sb.append("| Metric | Evaluated Value |\n");
        sb.append("|:---|:---:|\n");
        sb.append(String.format("| **Evaluated Samples** | %d tiles |\n", res.getResults().size()));
        sb.append(String.format("| **Mean IoU (Intersection over Union)** | **%.4f** |\n", res.getMeanIou()));
        sb.append(String.format("| **Mean F1-Score** | **%.4f** |\n", res.getMeanF1()));
        sb.append(String.format("| **Mean Precision** | **%.4f** |\n", res.getMeanPrecision()));
        sb.append(String.format("| **Mean Change Recall** | **%.4f** |\n", res.getMeanRecall()));
        sb.append(String.format("| **Mean Changed-Area Error** | **%.1f pixels** |\n", res.getMeanAreaError()));
        sb.append(String.format("| **Total Runtime** | **%d ms** |\n\n", res.getTotalRuntimeMs()));

        sb.append("## 2. Global Confusion Matrix (Pixel Aggregation)\n\n");
        sb.append("| Classification | Ground Truth Change | Ground Truth Unchanged |\n");
        sb.append("|:---|:---:|:---:|\n");
        sb.append(String.format("| **Predicted Change** | TP: %,d | FP: %,d |\n", res.getTotalTp(), res.getTotalFp()));
        sb.append(String.format("| **Predicted Unchanged** | FN: %,d | TN: %,d |\n\n", res.getTotalFn(), res.getTotalTn()));

        sb.append("## 3. Encroachment Screening Breakdown\n\n");
        long countEncroachment = res.getResults().stream().filter(r -> r.getStatus() == EncroachmentResult.Status.POSSIBLE_ENCROACHMENT).count();
        long countChange = res.getResults().stream().filter(r -> r.getStatus() == EncroachmentResult.Status.POSSIBLE_CHANGE).count();
        long countNoChange = res.getResults().stream().filter(r -> r.getStatus() == EncroachmentResult.Status.NO_SIGNIFICANT_CHANGE).count();

        sb.append(String.format("- **Possible Encroachments:** %d\n", countEncroachment));
        sb.append(String.format("- **Possible Changes:** %d\n", countChange));
        sb.append(String.format("- **No Significant Change:** %d\n\n", countNoChange));

        sb.append("## 4. Operational Disclaimer\n\n");
        sb.append("> [!NOTE]\n");
        sb.append("> This report represents automated satellite screening outputs. Findings indicate priority zones for field verification and cadastral inspection and do not constitute a legal or definitive land ownership adjudication.\n");

        Files.writeString(reportPath, sb.toString());
    }

    public static byte[][] extractBinaryArray(BufferedImage mask) {
        int w = mask.getWidth();
        int h = mask.getHeight();
        byte[][] arr = new byte[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int r = (mask.getRGB(x, y) >> 16) & 0xFF;
                arr[y][x] = (r > 127 || r == 1) ? (byte) 1 : (byte) 0;
            }
        }
        return arr;
    }

    private String stripExtension(String name) {
        int idx = name.lastIndexOf('.');
        return idx > 0 ? name.substring(0, idx) : name;
    }

    public static class DetectionOutput {
        private final EncroachmentResult result;
        private final byte[][] cleanMask;
        private final byte[][] groundTruthArray;
        private final float[][] diffMap;

        public DetectionOutput(EncroachmentResult result, byte[][] cleanMask, byte[][] groundTruthArray, float[][] diffMap) {
            this.result = result;
            this.cleanMask = cleanMask;
            this.groundTruthArray = groundTruthArray;
            this.diffMap = diffMap;
        }

        public EncroachmentResult getResult() {
            return result;
        }

        public byte[][] getCleanMask() {
            return cleanMask;
        }

        public byte[][] getGroundTruthArray() {
            return groundTruthArray;
        }

        public byte[][] getGroundTruthMask() {
            return groundTruthArray;
        }

        public float[][] getDiffMap() {
            return diffMap;
        }
    }

    public static class BatchEvaluationResult {
        private final List<EncroachmentResult> results;
        private final List<ChangeMetrics> metrics;
        private final double meanIou;
        private final double meanF1;
        private final double meanPrecision;
        private final double meanRecall;
        private final double meanAreaError;
        private final long totalTp;
        private final long totalTn;
        private final long totalFp;
        private final long totalFn;
        private final long totalRuntimeMs;

        public BatchEvaluationResult(List<EncroachmentResult> results, List<ChangeMetrics> metrics,
                                     double meanIou, double meanF1, double meanPrecision, double meanRecall, double meanAreaError,
                                     long totalTp, long totalTn, long totalFp, long totalFn, long totalRuntimeMs) {
            this.results = results;
            this.metrics = metrics;
            this.meanIou = meanIou;
            this.meanF1 = meanF1;
            this.meanPrecision = meanPrecision;
            this.meanRecall = meanRecall;
            this.meanAreaError = meanAreaError;
            this.totalTp = totalTp;
            this.totalTn = totalTn;
            this.totalFp = totalFp;
            this.totalFn = totalFn;
            this.totalRuntimeMs = totalRuntimeMs;
        }

        public List<EncroachmentResult> getResults() {
            return results;
        }

        public List<ChangeMetrics> getMetrics() {
            return metrics;
        }

        public double getMeanIou() {
            return meanIou;
        }

        public double getMeanF1() {
            return meanF1;
        }

        public double getMeanPrecision() {
            return meanPrecision;
        }

        public double getMeanRecall() {
            return meanRecall;
        }

        public double getMeanAreaError() {
            return meanAreaError;
        }

        public long getTotalTp() {
            return totalTp;
        }

        public long getTotalTn() {
            return totalTn;
        }

        public long getTotalFp() {
            return totalFp;
        }

        public long getTotalFn() {
            return totalFn;
        }

        public long getTotalRuntimeMs() {
            return totalRuntimeMs;
        }
    }
}
