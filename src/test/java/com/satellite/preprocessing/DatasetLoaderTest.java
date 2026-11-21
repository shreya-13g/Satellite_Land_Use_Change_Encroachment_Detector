package com.satellite.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 test suite for Stage 2 dataset handling and preprocessing.
 */
class DatasetLoaderTest {

    private DatasetConfig config;
    private ImageValidator validator;
    private ImagePreprocessor preprocessor;

    @BeforeEach
    void setUp() {
        config = DatasetConfig.defaultConfig();
        validator = new ImageValidator(config.getSupportedExtensions());
        preprocessor = new ImagePreprocessor(256, 256, 256);
    }

    @Test
    @DisplayName("Verify Dataset Path & Archive Detection")
    void testDatasetPathDetection() {
        DatasetLoader loader = new DatasetLoader(config);
        DatasetMetadata meta = loader.inspectDataset();

        assertNotNull(meta, "Dataset metadata should not be null");
        Path root = config.getDatasetRoot();
        if (Files.exists(root)) {
            // Verify archive tracking
            assertTrue(meta.getZipArchivesDetected().containsKey("train.zip"));
            assertTrue(meta.getZipArchivesDetected().containsKey("val.zip"));
            assertTrue(meta.getZipArchivesDetected().containsKey("test.zip"));

            // Verify splits found
            assertTrue(meta.getTrainPairs().size() > 0, "Train split should contain pairs");
            assertTrue(meta.getValPairs().size() > 0, "Val split should contain pairs");
            assertTrue(meta.getTestPairs().size() > 0, "Test split should contain pairs");
        }
    }

    @Test
    @DisplayName("Verify A/B/Label Matching Across Stems")
    void testABLabelMatching(@TempDir Path tempDir) throws IOException {
        Path splitDir = tempDir.resolve("sample_split");
        Path dirA = splitDir.resolve("A");
        Path dirB = splitDir.resolve("B");
        Path dirLabel = splitDir.resolve("label");
        Files.createDirectories(dirA);
        Files.createDirectories(dirB);
        Files.createDirectories(dirLabel);

        // Create dummy matching files for pair "sample_001"
        Files.write(dirA.resolve("sample_001.png"), new byte[]{1, 2, 3});
        Files.write(dirB.resolve("sample_001.png"), new byte[]{1, 2, 3});
        Files.write(dirLabel.resolve("sample_001.png"), new byte[]{1, 2, 3});

        DatasetConfig customConfig = new DatasetConfig()
                .setDatasetRoot(tempDir)
                .setSplits(Collections.singletonList("sample_split"));

        DatasetLoader loader = new DatasetLoader(customConfig);
        DatasetMetadata meta = loader.inspectDataset();

        // One pair should be detected and grouped
        assertNotNull(meta);
        assertEquals(1, meta.getAllPairs().size());
        ImagePair pair = meta.getAllPairs().get(0);
        assertEquals("sample_001", pair.getId());
        assertEquals("sample_split", pair.getSplit());
        assertNotNull(pair.getPathA());
        assertNotNull(pair.getPathB());
        assertNotNull(pair.getPathLabel());
    }

    @Test
    @DisplayName("Verify Missing-File Detection in ImagePair")
    void testMissingFileDetection(@TempDir Path tempDir) {
        Path existingA = tempDir.resolve("existingA.png");
        Path nonExistentB = tempDir.resolve("missingB.png");
        Path nonExistentLabel = tempDir.resolve("missingLabel.png");

        try {
            Files.write(existingA, new byte[]{0});
        } catch (IOException ignored) {}

        ImagePair brokenPair = new ImagePair("broken_1", "train", existingA, nonExistentB, nonExistentLabel);
        boolean isValid = validator.validate(brokenPair);

        assertFalse(isValid, "Pair with missing B and label must be invalid");
        assertFalse(brokenPair.isValid());
        assertTrue(brokenPair.getValidationMessage().contains("Missing later image"));
    }

    @Test
    @DisplayName("Verify Image Readability and Normalization to [0.0, 1.0]")
    void testImageReadabilityAndNormalization() {
        BufferedImage img = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(255, 128, 0));
        g.fillRect(0, 0, 10, 10);
        g.dispose();

        float[][][] norm = preprocessor.normalizeRGB(img);
        assertNotNull(norm);
        assertEquals(10, norm.length);
        assertEquals(10, norm[0].length);
        assertEquals(3, norm[0][0].length);

        // Red = 255/255 = 1.0, Green = 128/255 ~ 0.502, Blue = 0/255 = 0.0
        assertEquals(1.0f, norm[0][0][0], 1e-4);
        assertEquals(128.0f / 255.0f, norm[0][0][1], 1e-4);
        assertEquals(0.0f, norm[0][0][2], 1e-4);
    }

    @Test
    @DisplayName("Verify Dimension Mismatch Detection")
    void testDimensionValidation(@TempDir Path tempDir) throws IOException {
        Path fileA = tempDir.resolve("mismatchA.png");
        Path fileB = tempDir.resolve("mismatchB.png");
        Path fileLabel = tempDir.resolve("mismatchLabel.png");

        // A is 100x100
        BufferedImage imgA = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(imgA, "PNG", fileA.toFile());

        // B is 200x200 (mismatched)
        BufferedImage imgB = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(imgB, "PNG", fileB.toFile());

        // Label is 100x100
        BufferedImage imgLabel = new BufferedImage(100, 100, BufferedImage.TYPE_BYTE_GRAY);
        ImageIO.write(imgLabel, "PNG", fileLabel.toFile());

        ImagePair pair = new ImagePair("mismatch_1", "train", fileA, fileB, fileLabel);
        boolean ok = validator.validate(pair);

        assertFalse(ok, "Validator must reject mismatched dimensions");
        assertTrue(pair.getValidationMessage().contains("Dimension mismatch"));
    }

    @Test
    @DisplayName("Verify Binary-Mask Validation and Pixel Counting")
    void testBinaryMaskValidation() {
        BufferedImage mask = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        // Half black (0), half white (255)
        for (int y = 0; y < 20; y++) {
            for (int x = 0; x < 20; x++) {
                int val = x < 10 ? 0 : 255;
                int rgb = (val << 16) | (val << 8) | val;
                mask.setRGB(x, y, rgb);
            }
        }

        ImageValidator.ValidationMaskResult res = validator.validateAndCountMask(mask);
        assertTrue(res.isValidBinary());
        assertEquals(200, res.getUnchangedPixels(), "Left half should be 200 unchanged pixels");
        assertEquals(200, res.getChangedPixels(), "Right half should be 200 changed pixels");

        byte[][] binaryArr = preprocessor.extractBinaryMask(mask);
        assertEquals(20, binaryArr.length);
        assertEquals((byte) 0, binaryArr[0][0]);
        assertEquals((byte) 1, binaryArr[0][15]);
    }

    @Test
    @DisplayName("Verify 256x256 Tile Processing")
    void testTileProcessing256x256(@TempDir Path tempDir) throws IOException {
        Path fileA = tempDir.resolve("tileA.png");
        Path fileB = tempDir.resolve("tileB.png");
        Path fileLabel = tempDir.resolve("tileLabel.png");

        // Create 512x512 images (should yield exactly 4 tiles of 256x256)
        BufferedImage img512 = new BufferedImage(512, 512, BufferedImage.TYPE_INT_RGB);
        ImageIO.write(img512, "PNG", fileA.toFile());
        ImageIO.write(img512, "PNG", fileB.toFile());
        ImageIO.write(img512, "PNG", fileLabel.toFile());

        ImagePair pair = new ImagePair("tile_test", "val", fileA, fileB, fileLabel);
        Path processedOut = tempDir.resolve("processed");

        int tiles = preprocessor.processAndSaveTiles(pair, processedOut);
        assertEquals(4, tiles, "512x512 image split by 256x256 stride 256 must yield 4 tiles");

        Path tileA0 = processedOut.resolve("val").resolve("A").resolve("tile_test_tile_y0000_x0000.png");
        Path tileB0 = processedOut.resolve("val").resolve("B").resolve("tile_test_tile_y0000_x0000.png");
        Path tileL0 = processedOut.resolve("val").resolve("label").resolve("tile_test_tile_y0000_x0000.png");

        assertTrue(Files.exists(tileA0));
        assertTrue(Files.exists(tileB0));
        assertTrue(Files.exists(tileL0));

        BufferedImage loadedTile = ImageIO.read(tileA0.toFile());
        assertEquals(256, loadedTile.getWidth());
        assertEquals(256, loadedTile.getHeight());
    }

    @Test
    @DisplayName("Verify DatasetSummary Calculation & File Serialization")
    void testDatasetSummaryCalculation(@TempDir Path tempDir) throws IOException {
        DatasetSummary summary = new DatasetSummary();
        summary.setTrainPairsCount(445);
        summary.setValPairsCount(64);
        summary.setTestPairsCount(128);
        summary.setValidPairsCount(637);
        summary.setInvalidPairsCount(0);
        summary.setMissingFilesCount(0);
        summary.setImageDimensions("1024x1024");
        summary.setMaskDimensions("1024x1024");
        summary.setChangedPixelCount(10_000_000L);
        summary.setUnchangedPixelCount(90_000_000L);
        summary.setTotalProcessedTiles(160);
        summary.setProcessingTimeMs(1500L);

        assertEquals(637, summary.getTotalPairsCount());
        assertEquals(0.10, summary.getChangeRatio(), 1e-4);

        Path jsonFile = tempDir.resolve("summary.json");
        Path csvFile = tempDir.resolve("summary.csv");

        summary.toJsonFile(jsonFile);
        summary.toCsvFile(csvFile);

        assertTrue(Files.exists(jsonFile));
        assertTrue(Files.exists(csvFile));

        String jsonContent = Files.readString(jsonFile);
        assertTrue(jsonContent.contains("\"trainPairs\" : 445"));
        assertTrue(jsonContent.contains("\"changedPixelCount\" : 10000000"));

        String csvContent = Files.readString(csvFile);
        assertTrue(csvContent.contains("number_of_train_pairs,445"));
        assertTrue(csvContent.contains("changed_pixel_count,10000000"));
    }
}
