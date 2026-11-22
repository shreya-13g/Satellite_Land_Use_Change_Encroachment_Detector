package com.satellite.classification;

import com.satellite.change.ChangeDetector;
import com.satellite.temporal.TemporalLandUseDemo;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 test suite verifying deep learning ONNX runtime integration and temporal models.
 */
class OnnxResNetChangeDetectorTest {

    @Test
    @DisplayName("Verify ONNX Siamese ResNet-18 Model Inference")
    void testOnnxResNetInference() throws Exception {
        Path modelPath = Paths.get("models", "onnx", "resnet18_change_detector.onnx");
        if (!Files.exists(modelPath)) {
            System.out.println("ONNX model file not found, skipping live ONNX inference test.");
            return;
        }

        try (OnnxResNetChangeDetector detector = new OnnxResNetChangeDetector(modelPath, 30)) {
            // Synthetic 256x256 images
            BufferedImage imgA = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
            BufferedImage imgB = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);

            // Add change patch in imgB
            for (int y = 50; y < 100; y++) {
                for (int x = 50; x < 100; x++) {
                    imgB.setRGB(x, y, 0xFFFFFF); // White patch
                }
            }

            ChangeDetector.DetectionOutput out = detector.processPair(imgA, imgB, null, "unit_test_sample");
            assertNotNull(out);
            assertNotNull(out.getCleanMask());
            assertEquals(256, out.getCleanMask().length);
            assertEquals(256, out.getCleanMask()[0].length);
            assertNotNull(out.getResult());
            assertNotNull(out.getDiffMap());
        }
    }

    @Test
    @DisplayName("Verify Temporal Sequence Land-Use Forecast")
    void testTemporalForecast() {
        int[] years = {2018, 2020, 2022, 2024, 2026};
        double[] builtUp = {10.0, 13.0, 17.0, 23.0, 31.0};

        TemporalLandUseDemo.ForecastResult res = TemporalLandUseDemo.runTemporalDemo(years, builtUp);
        assertNotNull(res);
        assertEquals(2028, res.forecastYear);
        assertTrue(res.forecastValue > 31.0, "Forecast for 2028 should predict continued expansion");
        assertTrue(res.meanAnnualGrowthRate > 0.0);
    }
}
