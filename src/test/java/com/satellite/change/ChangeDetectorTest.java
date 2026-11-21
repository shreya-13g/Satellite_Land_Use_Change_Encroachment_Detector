package com.satellite.change;

import com.satellite.app.Main;
import com.satellite.detection.EncroachmentDetector;
import com.satellite.detection.EncroachmentResult;
import com.satellite.evaluation.ChangeMetrics;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive JUnit 5 test suite for Stage 3 Change Detection & Encroachment Screening.
 */
class ChangeDetectorTest {

    @Test
    @DisplayName("Verify Pixel-Difference Calculation (RGB & Grayscale)")
    void testPixelDifferenceCalculation() {
        BufferedImage imgA = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        BufferedImage imgB = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);

        // Pixel (0,0): Black in A (0,0,0), Pure White in B (255,255,255)
        imgA.setRGB(0, 0, 0);
        imgB.setRGB(0, 0, (255 << 16) | (255 << 8) | 255);

        // Pixel (1,1): Identical Red in both
        imgA.setRGB(1, 1, 255 << 16);
        imgB.setRGB(1, 1, 255 << 16);

        DifferenceCalculator calcRgb = new DifferenceCalculator(DifferenceCalculator.Mode.EUCLIDEAN_RGB);
        float[][] diff = calcRgb.calculateDifference(imgA, imgB);

        assertEquals(255.0f, diff[0][0], 1e-3, "White vs Black should have 255.0 difference");
        assertEquals(0.0f, diff[1][1], 1e-3, "Identical pixel should have 0.0 difference");

        DifferenceCalculator calcGray = new DifferenceCalculator(DifferenceCalculator.Mode.GRAYSCALE_ABSOLUTE);
        float[][] diffGray = calcGray.calculateDifference(imgA, imgB);
        assertEquals(255.0f, diffGray[0][0], 1e-3);
        assertEquals(0.0f, diffGray[1][1], 1e-3);
    }

    @Test
    @DisplayName("Verify Threshold-Based Mask Generation (Static & Otsu)")
    void testThresholdBasedMaskGeneration() {
        float[][] diffMap = {
                {10.0f, 40.0f},
                {20.0f, 60.0f}
        };

        ChangeMaskGenerator generator = new ChangeMaskGenerator(30.0);
        byte[][] mask = generator.generateMask(diffMap);

        assertEquals((byte) 0, mask[0][0], "10.0 < 30.0 -> 0");
        assertEquals((byte) 1, mask[0][1], "40.0 >= 30.0 -> 1");
        assertEquals((byte) 0, mask[1][0], "20.0 < 30.0 -> 0");
        assertEquals((byte) 1, mask[1][1], "60.0 >= 30.0 -> 1");

        // Otsu threshold computation should execute deterministically
        double otsuT = ChangeMaskGenerator.computeOtsuThreshold(diffMap);
        assertTrue(otsuT >= 0 && otsuT <= 255, "Otsu threshold must be valid byte value");
    }

    @Test
    @DisplayName("Verify Noise Removal & Morphological Cleanup")
    void testNoiseRemoval() {
        // Create 20x20 mask with one isolated speck (at 2,2) and one large block (10x10)
        byte[][] mask = new byte[20][20];
        mask[2][2] = 1; // isolated noise speck

        // Large block from (8,8) to (17,17) (100 pixels)
        for (int y = 8; y < 18; y++) {
            for (int x = 8; x < 18; x++) {
                mask[y][x] = 1;
            }
        }

        NoiseRemover remover = new NoiseRemover(25, true);
        byte[][] cleaned = remover.cleanMask(mask);

        // Isolated speck should be removed
        assertEquals((byte) 0, cleaned[2][2], "Isolated single-pixel speck must be eliminated");
        // Interior of large block must be preserved
        assertEquals((byte) 1, cleaned[12][12], "Interior of large block must be preserved");
    }

    @Test
    @DisplayName("Verify Connected-Component Detection & Bounding Box Calculation")
    void testConnectedComponentAndBoundingBox() {
        byte[][] mask = new byte[50][50];

        // Component 1: 5x10 rectangle from (x=10, y=10) to (x=19, y=14) -> 50 pixels
        for (int y = 10; y <= 14; y++) {
            for (int x = 10; x <= 19; x++) {
                mask[y][x] = 1;
            }
        }

        ConnectedComponentAnalyzer analyzer = new ConnectedComponentAnalyzer(20);
        List<ChangeRegion> regions = analyzer.findRegions(mask);

        assertEquals(1, regions.size(), "Should identify exactly one connected component");
        ChangeRegion r = regions.get(0);

        assertEquals(10, r.getMinX());
        assertEquals(10, r.getMinY());
        assertEquals(19, r.getMaxX());
        assertEquals(14, r.getMaxY());
        assertEquals(10, r.getWidth());
        assertEquals(5, r.getHeight());
        assertEquals(50, r.getAreaPixels());
        assertEquals(14.5, r.getCentroidX(), 1e-4);
        assertEquals(12.0, r.getCentroidY(), 1e-4);
        assertEquals(2.0, r.getPercentageOfImage(), 1e-4); // 50 / 2500 = 2.0%
    }

    @Test
    @DisplayName("Verify Quantitative Metrics: IoU, Precision, Recall, F1-Score")
    void testQuantitativeMetrics() {
        // 10x10 images: total 100 pixels
        byte[][] pred = new byte[10][10];
        byte[][] gt = new byte[10][10];

        // Overlapping ground truth and prediction:
        // Pixels (0-4, 0) = 5 pixels (TP = 5)
        for (int x = 0; x < 5; x++) {
            pred[0][x] = 1;
            gt[0][x] = 1;
        }
        // Prediction false positive at (0, 5) -> FP = 1
        pred[0][5] = 1;
        // Ground truth false negative at (0, 6) -> FN = 1
        gt[0][6] = 1;

        ChangeMetrics m = ChangeMetrics.compute(pred, gt, "test_pair", 15L);

        assertEquals(5, m.getTruePositives());
        assertEquals(1, m.getFalsePositives());
        assertEquals(1, m.getFalseNegatives());
        assertEquals(93, m.getTrueNegatives());

        // Precision = TP / (TP + FP) = 5 / 6 = 0.8333
        assertEquals(5.0 / 6.0, m.getPrecision(), 1e-4);

        // Recall = TP / (TP + FN) = 5 / 6 = 0.8333
        assertEquals(5.0 / 6.0, m.getRecall(), 1e-4);

        // F1 = 5 / 6 = 0.8333
        assertEquals(5.0 / 6.0, m.getF1Score(), 1e-4);

        // IoU = TP / (TP + FP + FN) = 5 / 7 = 0.7143
        assertEquals(5.0 / 7.0, m.getIou(), 1e-4);

        // Area error = |6 - 6| = 0
        assertEquals(0, m.getAreaErrorPixels());
    }

    @Test
    @DisplayName("Verify Encroachment Screening Classification Rules")
    void testEncroachmentClassification() {
        EncroachmentDetector detector = new EncroachmentDetector();
        int totalImagePixels = 65536; // 256x256

        // Case 1: Very minor noise (10 pixels, largest cluster 5)
        ChangeRegion minor = new ChangeRegion(1, 0, 0, 2, 2, 5, 0.01, 1, 1);
        EncroachmentResult res1 = detector.evaluate("sample_1", 10, totalImagePixels, Collections.singletonList(minor));
        assertEquals(EncroachmentResult.Status.NO_SIGNIFICANT_CHANGE, res1.getStatus());

        // Case 2: Intermediate change (70 pixels, largest cluster 70)
        ChangeRegion mod = new ChangeRegion(1, 0, 0, 9, 7, 70, 0.11, 4, 3);
        EncroachmentResult res2 = detector.evaluate("sample_2", 70, totalImagePixels, Collections.singletonList(mod));
        assertEquals(EncroachmentResult.Status.POSSIBLE_CHANGE, res2.getStatus());

        // Case 3: Substantial structural encroachment (350 pixels, cluster >= 150)
        ChangeRegion large = new ChangeRegion(1, 10, 10, 30, 25, 300, 0.46, 20, 17);
        EncroachmentResult res3 = detector.evaluate("sample_3", 350, totalImagePixels, Collections.singletonList(large));
        assertEquals(EncroachmentResult.Status.POSSIBLE_ENCROACHMENT, res3.getStatus());
        assertTrue(res3.getConfidence() >= 0.65);
    }

    @Test
    @DisplayName("Verify Invalid Command-Line Arguments Rejection")
    void testInvalidCommandLineArguments() {
        // Unknown command flag
        assertThrows(IllegalArgumentException.class, () ->
                Main.handleCommandLine(new String[]{"--invalid-flag"}));

        // --detect-change missing required --after
        assertThrows(IllegalArgumentException.class, () ->
                Main.handleCommandLine(new String[]{"--detect-change", "--before", "some/path.png"}));

        // Unexpected bare argument
        assertThrows(IllegalArgumentException.class, () ->
                Main.handleCommandLine(new String[]{"bareArgument"}));
    }
}
