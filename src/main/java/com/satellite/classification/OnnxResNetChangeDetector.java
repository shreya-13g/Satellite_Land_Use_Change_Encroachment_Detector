package com.satellite.classification;

import ai.onnxruntime.*;
import com.satellite.change.*;
import com.satellite.detection.EncroachmentDetector;
import com.satellite.detection.EncroachmentResult;
import com.satellite.evaluation.ChangeMetrics;
import com.satellite.visualization.ChangeVisualizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Deep Learning Change Detector utilizing a Siamese Pretrained ResNet-18 Backbone.
 *
 * Architecture:
 *   Input: Before Image [1, 3, 256, 256] + After Image [1, 3, 256, 256]
 *   Backbone: Pretrained ResNet-18 (ImageNet weights)
 *   Representation: 512-channel deep convolutional feature maps (layer4)
 *   Differencing: Spatial Euclidean L2 distance across 512 channels
 *   Output: Continuous change heatmap [1, 1, 256, 256]
 *   Inference Engine: ONNX Runtime Java (CPU)
 *   Postprocessing: Adaptive thresholding, morphological noise removal, 8-connectivity BFS
 */
public class OnnxResNetChangeDetector implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(OnnxResNetChangeDetector.class);

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD  = {0.229f, 0.224f, 0.225f};

    private final OrtEnvironment env;
    private final OrtSession session;
    private final NoiseRemover noiseRemover;
    private final ConnectedComponentAnalyzer componentAnalyzer;
    private final EncroachmentDetector encroachmentDetector;
    private final ChangeVisualizer visualizer;

    public OnnxResNetChangeDetector(Path modelPath, int minRegionArea) throws OrtException {
        if (!Files.exists(modelPath)) {
            throw new IllegalArgumentException("ONNX model file not found: " + modelPath.toAbsolutePath());
        }
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
        this.session = env.createSession(modelPath.toString(), opts);

        this.noiseRemover = new NoiseRemover(minRegionArea, true);
        this.componentAnalyzer = new ConnectedComponentAnalyzer(minRegionArea);
        this.encroachmentDetector = new EncroachmentDetector();
        this.visualizer = new ChangeVisualizer();
        logger.info("Initialized OnnxResNetChangeDetector with model: {}", modelPath);
    }

    public OnnxResNetChangeDetector() throws OrtException {
        this(Paths.get("models", "onnx", "resnet18_change_detector.onnx"), 30);
    }

    /**
     * Executes Siamese ResNet-18 inference on a bi-temporal image pair.
     */
    public ChangeDetector.DetectionOutput processPair(BufferedImage imgA, BufferedImage imgB,
                                                      BufferedImage gtMask, String sampleId) throws OrtException {
        long startTime = System.currentTimeMillis();
        int width = imgA.getWidth();
        int height = imgA.getHeight();

        // 1. Prepare normalized NCHW float buffers [1, 3, 256, 256]
        FloatBuffer bufA = imageToNormalizedBuffer(imgA);
        FloatBuffer bufB = imageToNormalizedBuffer(imgB);

        long[] shape = new long[]{1, 3, height, width};
        try (OnnxTensor tensorA = OnnxTensor.createTensor(env, bufA, shape);
             OnnxTensor tensorB = OnnxTensor.createTensor(env, bufB, shape)) {

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("before", tensorA);
            inputs.put("after", tensorB);

            // 2. Execute deep neural network inference
            try (OrtSession.Result ortResult = session.run(inputs)) {
                float[][][][] outputMap = (float[][][][]) ortResult.get(0).getValue();
                float[][] diffMap = outputMap[0][0]; // [256, 256]

                // 3. Adaptive thresholding on deep feature distance map
                byte[][] rawMask = thresholdDeepDistanceMap(diffMap);

                // 4. Morphological noise removal & small region cleanup
                byte[][] cleanMask = noiseRemover.cleanMask(rawMask);

                // 5. Connected component analysis (8-neighbour BFS)
                List<ChangeRegion> regions = componentAnalyzer.findRegions(cleanMask);

                // 6. Changed pixel count
                int changedPixels = 0;
                int totalPixels = height * width;
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        if (cleanMask[y][x] == 1) changedPixels++;
                    }
                }

                // 7. Encroachment screening classification
                EncroachmentResult result = encroachmentDetector.evaluate(sampleId, changedPixels, totalPixels, regions);

                // 8. Quantitative evaluation against ground truth if provided
                byte[][] gtArray = null;
                if (gtMask != null) {
                    gtArray = ChangeDetector.extractBinaryArray(gtMask);
                    long runtimeMs = System.currentTimeMillis() - startTime;
                    ChangeMetrics metrics = ChangeMetrics.compute(cleanMask, gtArray, sampleId, runtimeMs);
                    result.setMetrics(metrics);
                }

                return new ChangeDetector.DetectionOutput(result, cleanMask, gtArray, diffMap);
            }
        }
    }

    /**
     * Converts RGB BufferedImage into normalized NCHW FloatBuffer for ResNet input.
     */
    private FloatBuffer imageToNormalizedBuffer(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        FloatBuffer buffer = FloatBuffer.allocate(3 * h * w);

        float[] rPlane = new float[h * w];
        float[] gPlane = new float[h * w];
        float[] bPlane = new float[h * w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                int idx = y * w + x;
                rPlane[idx] = ((r / 255.0f) - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                gPlane[idx] = ((g / 255.0f) - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                bPlane[idx] = ((b / 255.0f) - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        buffer.put(rPlane);
        buffer.put(gPlane);
        buffer.put(bPlane);
        buffer.flip();
        return buffer;
    }

    /**
     * Applies adaptive thresholding to the deep feature distance map.
     * Uses Otsu-style or statistical threshold: T = mean + 1.2 * std.
     */
    private byte[][] thresholdDeepDistanceMap(float[][] diffMap) {
        int h = diffMap.length;
        int w = diffMap[0].length;
        double sum = 0.0;
        float minVal = Float.MAX_VALUE;
        float maxVal = Float.MIN_VALUE;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float v = diffMap[y][x];
                sum += v;
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }

        double mean = sum / (h * w);
        double varSum = 0.0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double diff = diffMap[y][x] - mean;
                varSum += diff * diff;
            }
        }
        double std = Math.sqrt(varSum / (h * w));

        // Threshold separates high-distance change clusters from background noise
        double threshold = mean + 1.25 * std;

        byte[][] mask = new byte[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                mask[y][x] = (diffMap[y][x] >= threshold) ? (byte) 1 : (byte) 0;
            }
        }
        return mask;
    }

    /**
     * Evaluates a batch of test tile pairs and returns summary metrics.
     */
    public ChangeDetector.BatchEvaluationResult evaluateBatch(
            List<Path> filesA, List<Path> filesB, List<Path> filesL,
            Path outputDir, Path vizDir, int limit) throws Exception {

        Files.createDirectories(outputDir);
        Files.createDirectories(vizDir);

        int n = Math.min(limit, filesA.size());
        List<EncroachmentResult> results = new ArrayList<>();
        long batchStartTime = System.currentTimeMillis();

        for (int i = 0; i < n; i++) {
            Path pA = filesA.get(i);
            Path pB = filesB.get(i);
            Path pL = (filesL != null && i < filesL.size()) ? filesL.get(i) : null;

            BufferedImage imgA = ImageIO.read(pA.toFile());
            BufferedImage imgB = ImageIO.read(pB.toFile());
            BufferedImage imgL = (pL != null && Files.exists(pL)) ? ImageIO.read(pL.toFile()) : null;

            String sampleId = pA.getFileName().toString().replaceFirst("\\.[^.]+$", "");
            ChangeDetector.DetectionOutput out = processPair(imgA, imgB, imgL, sampleId);
            results.add(out.getResult());

            // Save visual overlay for first few tiles
            if (i < 6) {
                saveVisualizations(imgA, imgB, imgL, out, vizDir);
            }
        }

        long totalTime = System.currentTimeMillis() - batchStartTime;
        long totalTp = 0, totalTn = 0, totalFp = 0, totalFn = 0;
        double sumIou = 0, sumF1 = 0, sumPrec = 0, sumRec = 0, sumAreaErr = 0;
        int countMetrics = 0;
        List<ChangeMetrics> metricsList = new ArrayList<>();
        for (EncroachmentResult r : results) {
            ChangeMetrics m = r.getMetrics();
            if (m != null) {
                metricsList.add(m);
                sumIou += m.getIou();
                sumF1 += m.getF1Score();
                sumPrec += m.getPrecision();
                sumRec += m.getRecall();
                sumAreaErr += m.getAreaErrorPixels();
                totalTp += m.getTruePositives();
                totalTn += m.getTrueNegatives();
                totalFp += m.getFalsePositives();
                totalFn += m.getFalseNegatives();
                countMetrics++;
            }
        }
        double meanIou = countMetrics > 0 ? sumIou / countMetrics : 0.0;
        double meanF1 = countMetrics > 0 ? sumF1 / countMetrics : 0.0;
        double meanPrec = countMetrics > 0 ? sumPrec / countMetrics : 0.0;
        double meanRec = countMetrics > 0 ? sumRec / countMetrics : 0.0;
        double meanAreaErr = countMetrics > 0 ? sumAreaErr / countMetrics : 0.0;

        return new ChangeDetector.BatchEvaluationResult(
                results, metricsList, meanIou, meanF1, meanPrec, meanRec, meanAreaErr,
                totalTp, totalTn, totalFp, totalFn, totalTime
        );
    }

    public void saveVisualizations(BufferedImage imgA, BufferedImage imgB, BufferedImage gtMask,
                                   ChangeDetector.DetectionOutput output, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);
        EncroachmentResult res = output.getResult();
        String sampleId = res.getSampleId();
        byte[][] cleanMask = output.getCleanMask();

        BufferedImage maskImg = visualizer.createMaskImage(cleanMask);
        ImageIO.write(maskImg, "PNG", outputDir.resolve(sampleId + "_dl_predicted_mask.png").toFile());

        BufferedImage bboxImg = visualizer.renderBoundingBoxes(imgB, res.getRegions(), res.getStatus());
        ImageIO.write(bboxImg, "PNG", outputDir.resolve(sampleId + "_dl_bounding_boxes.png").toFile());

        BufferedImage gtImg = (gtMask != null) ? gtMask : maskImg;
        BufferedImage board = visualizer.renderSideBySideBoard(
                imgA, imgB, gtImg, maskImg, bboxImg, res
        );
        ImageIO.write(board, "PNG", outputDir.resolve(sampleId + "_dl_comparison_board.png").toFile());
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (OrtException e) {
            logger.warn("Error closing ONNX resources: {}", e.getMessage());
        }
    }
}
