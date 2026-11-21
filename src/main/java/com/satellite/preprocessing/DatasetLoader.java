package com.satellite.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Primary loader and preprocessor for the multitemporal LEVIR-CD dataset.
 * Discovers, matches, validates, tiles, and visualizes satellite image pairs.
 */
public class DatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(DatasetLoader.class);

    private final DatasetConfig config;
    private final ImageValidator validator;
    private final ImagePreprocessor preprocessor;

    public DatasetLoader(DatasetConfig config) {
        this.config = config != null ? config : DatasetConfig.defaultConfig();
        this.validator = new ImageValidator(this.config.getSupportedExtensions());
        this.preprocessor = new ImagePreprocessor(
                this.config.getTileWidth(),
                this.config.getTileHeight(),
                this.config.getTileStride()
        );
    }

    public DatasetLoader() {
        this(DatasetConfig.defaultConfig());
    }

    /**
     * Inspects the dataset root directory, reports ZIP archives and extraction state.
     */
    public DatasetMetadata inspectDataset() {
        Path root = config.getDatasetRoot();
        DatasetMetadata metadata = new DatasetMetadata(root);

        logger.info("Scanning dataset root directory: {}", root.toAbsolutePath());
        if (!Files.exists(root) || !Files.isDirectory(root)) {
            logger.error("Dataset root does not exist: {}", root);
            return metadata;
        }

        // 1. Check for ZIP archives
        String[] archives = {"train.zip", "val.zip", "test.zip"};
        for (String arc : archives) {
            Path arcPath = root.resolve(arc);
            String folderName = arc.replace(".zip", "");
            Path folderPath = root.resolve(folderName);

            boolean arcExists = Files.exists(arcPath) && Files.isRegularFile(arcPath);
            boolean folderExtracted = Files.exists(folderPath) && Files.isDirectory(folderPath);

            metadata.recordArchive(arc, arcExists, folderExtracted);

            if (arcExists) {
                logger.info("Detected ZIP archive: {} (Extracted: {})", arc, folderExtracted);
            }
            if (arcExists && !folderExtracted) {
                logger.warn("ARCHIVE NOT EXTRACTED: {} is present but folder {} is missing!", arc, folderName);
            }
        }

        // 2. Discover image pairs in extracted folders
        for (String split : config.getSplits()) {
            Path splitDir = root.resolve(split);
            if (!Files.exists(splitDir) || !Files.isDirectory(splitDir)) {
                logger.warn("Split directory not found: {}", splitDir);
                continue;
            }

            Path dirA = splitDir.resolve(config.getSubfolderA());
            Path dirB = splitDir.resolve(config.getSubfolderB());
            Path dirLabel = splitDir.resolve(config.getSubfolderLabel());

            if (!Files.exists(dirA)) {
                metadata.addMissingFileError("Missing folder A in split: " + split);
            }
            if (!Files.exists(dirB)) {
                metadata.addMissingFileError("Missing folder B in split: " + split);
            }
            if (!Files.exists(dirLabel)) {
                metadata.addMissingFileError("Missing folder label in split: " + split);
            }

            // Find all files in A, B, and label and match by filename
            Map<String, Path> filesA = listFilesByStem(dirA);
            Map<String, Path> filesB = listFilesByStem(dirB);
            Map<String, Path> filesLabel = listFilesByStem(dirLabel);

            // Set of all distinct stems across all 3 folders
            Set<String> allStems = new TreeSet<>();
            allStems.addAll(filesA.keySet());
            allStems.addAll(filesB.keySet());
            allStems.addAll(filesLabel.keySet());

            for (String stem : allStems) {
                Path pA = filesA.get(stem);
                Path pB = filesB.get(stem);
                Path pL = filesLabel.get(stem);

                ImagePair pair = new ImagePair(stem, split, pA, pB, pL);

                if (pA == null) {
                    pair.setValid(false);
                    pair.setValidationMessage("Missing image A for stem: " + stem);
                    metadata.addMissingFileError(pair.getValidationMessage());
                } else if (pB == null) {
                    pair.setValid(false);
                    pair.setValidationMessage("Missing image B for stem: " + stem);
                    metadata.addMissingFileError(pair.getValidationMessage());
                } else if (pL == null) {
                    pair.setValid(false);
                    pair.setValidationMessage("Missing label for stem: " + stem);
                    metadata.addMissingFileError(pair.getValidationMessage());
                }

                metadata.addPair(pair);
            }
            logger.info("Discovered {} pairs in split '{}'", allStems.size(), split);
        }

        return metadata;
    }

    /**
     * Executes the complete Stage 2 pipeline:
     * Inspection, validation, tiling, summary generation, and sample visualization.
     */
    public DatasetSummary processDataset() throws IOException {
        long startTime = System.currentTimeMillis();
        DatasetMetadata metadata = inspectDataset();

        DatasetSummary summary = new DatasetSummary();
        summary.setTrainPairsCount(metadata.getTrainPairs().size());
        summary.setValPairsCount(metadata.getValPairs().size());
        summary.setTestPairsCount(metadata.getTestPairs().size());
        summary.setMissingFilesCount(metadata.getMissingFileErrors().size());

        List<ImagePair> allPairs = metadata.getAllPairs();
        logger.info("Validating {} total image pairs...", allPairs.size());

        long totalChangedPixels = 0;
        long totalUnchangedPixels = 0;
        int validCount = 0;
        int invalidCount = 0;
        String detectedDims = "Unknown";

        for (ImagePair pair : allPairs) {
            boolean ok = validator.validate(pair);
            if (ok) {
                validCount++;
                totalChangedPixels += pair.getChangedPixelCount();
                totalUnchangedPixels += pair.getUnchangedPixelCount();
                if ("Unknown".equals(detectedDims) && pair.getWidthA() > 0) {
                    detectedDims = pair.getWidthA() + "x" + pair.getHeightA();
                }
            } else {
                invalidCount++;
                logger.warn("Pair validation failure: {} - {}", pair.getId(), pair.getValidationMessage());
            }
        }

        summary.setValidPairsCount(validCount);
        summary.setInvalidPairsCount(invalidCount);
        summary.setImageDimensions(detectedDims);
        summary.setMaskDimensions(detectedDims);
        summary.setChangedPixelCount(totalChangedPixels);
        summary.setUnchangedPixelCount(totalUnchangedPixels);

        // Tiling and Visualizations
        int totalProcessedTiles = 0;
        Files.createDirectories(config.getProcessedRoot());
        Files.createDirectories(config.getVisualizationsDir());

        for (String split : config.getSplits()) {
            List<ImagePair> splitPairs = "train".equals(split) ? metadata.getTrainPairs() :
                                        ("val".equals(split) ? metadata.getValPairs() : metadata.getTestPairs());

            int pairsToTile = config.getMaxPairsToTilePerSplit();
            int countTiled = 0;
            int countViz = 0;

            for (ImagePair pair : splitPairs) {
                if (!pair.isValid()) {
                    continue;
                }

                // Tile generation
                if (pairsToTile == -1 || countTiled < pairsToTile) {
                    try {
                        int tiles = preprocessor.processAndSaveTiles(pair, config.getProcessedRoot());
                        totalProcessedTiles += tiles;
                        countTiled++;
                        logger.debug("Tiled pair {}: generated {} tiles (256x256)", pair.getId(), tiles);
                    } catch (Exception e) {
                        logger.error("Failed to tile pair {}: {}", pair.getId(), e.getMessage());
                    }
                }

                // Sample visual comparison generation
                if (countViz < config.getMaxVisualizationsPerSplit()) {
                    try {
                        BufferedImage viz = preprocessor.createComparisonVisualization(pair);
                        Path outPath = config.getVisualizationsDir().resolve(pair.getId() + "_comparison.png");
                        ImageIO.write(viz, "PNG", outPath.toFile());
                        countViz++;
                        logger.info("Saved sample visualization to {}", outPath.toAbsolutePath());
                    } catch (Exception e) {
                        logger.error("Failed to create visualization for {}: {}", pair.getId(), e.getMessage());
                    }
                }
            }
        }

        summary.setTotalProcessedTiles(totalProcessedTiles);
        long duration = System.currentTimeMillis() - startTime;
        summary.setProcessingTimeMs(duration);

        // Export summaries to results/
        Path jsonPath = config.getResultsDir().resolve("dataset-summary.json");
        Path csvPath = config.getResultsDir().resolve("dataset-summary.csv");

        summary.toJsonFile(jsonPath);
        summary.toCsvFile(csvPath);

        logger.info("Saved dataset summary JSON to {}", jsonPath.toAbsolutePath());
        logger.info("Saved dataset summary CSV to {}", csvPath.toAbsolutePath());
        logger.info("Stage 2 processing completed in {} ms. Valid pairs: {}/{}, Processed tiles: {}",
                duration, validCount, allPairs.size(), totalProcessedTiles);

        return summary;
    }

    private Map<String, Path> listFilesByStem(Path dir) {
        Map<String, Path> map = new TreeMap<>();
        if (dir == null || !Files.exists(dir) || !Files.isDirectory(dir)) {
            return map;
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path p : stream) {
                if (Files.isRegularFile(p)) {
                    String fileName = p.getFileName().toString();
                    String stem = stripExtension(fileName);
                    map.put(stem, p);
                }
            }
        } catch (IOException e) {
            logger.error("Error reading directory {}: {}", dir, e.getMessage());
        }
        return map;
    }

    private String stripExtension(String name) {
        int idx = name.lastIndexOf('.');
        return idx > 0 ? name.substring(0, idx) : name;
    }

    public static void main(String[] args) {
        try {
            DatasetConfig cfg = DatasetConfig.defaultConfig();
            if (args != null && args.length > 0 && "--full".equalsIgnoreCase(args[0])) {
                cfg.setMaxPairsToTilePerSplit(-1); // Process full dataset
            }
            DatasetLoader loader = new DatasetLoader(cfg);
            DatasetSummary summary = loader.processDataset();
            System.out.println("LEVIR-CD Dataset Preprocessing Complete:");
            System.out.printf("Train Pairs: %d, Val Pairs: %d, Test Pairs: %d%n",
                    summary.getTrainPairsCount(), summary.getValPairsCount(), summary.getTestPairsCount());
            System.out.printf("Valid Pairs: %d, Invalid: %d, Tiles Generated: %d%n",
                    summary.getValidPairsCount(), summary.getInvalidPairsCount(), summary.getTotalProcessedTiles());
            System.out.printf("Changed Pixels: %,d, Unchanged Pixels: %,d (Change Ratio: %.4f%%)%n",
                    summary.getChangedPixelCount(), summary.getUnchangedPixelCount(), summary.getChangeRatio() * 100.0);
        } catch (Exception e) {
            logger.error("DatasetLoader execution failed: {}", e.getMessage(), e);
            System.exit(1);
        }
    }
}
