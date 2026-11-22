package com.satellite.app;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.satellite.change.ChangeDetector;
import com.satellite.classification.OnnxResNetChangeDetector;
import com.satellite.detection.EncroachmentDetector;
import com.satellite.detection.EncroachmentResult;
import com.satellite.evaluation.ChangeMetrics;
import com.satellite.evaluation.ModelComparisonRunner;
import com.satellite.fundamentals.ActivationComparison;
import com.satellite.fundamentals.FundamentalsRunner;
import com.satellite.generative.GenerativeAugmentationDemo;
import com.satellite.learning.LearningRulesDemo;
import com.satellite.optimizers.OptimizerComparison;
import com.satellite.temporal.TemporalLandUseDemo;
import com.satellite.visualization.ChangeVisualizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Satellite Land Use Change and Encroachment Detector
 *
 * Geospatial Intelligence Domain:
 * Multitemporal optical satellite image analysis for detecting Land Use / Land Cover (LULC)
 * transitions and unauthorized human encroachment into environmentally protected zones.
 */
public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static final String APP_NAME = "Satellite Land Use Change and Encroachment Detector";
    public static final String APP_VERSION = "1.0.0-SNAPSHOT";

    public static void main(String[] args) {
        System.out.println("Satellite Land Use Change and Encroachment Detector");
        System.out.println("Java Maven project initialized successfully.");

        logger.info("================================================================================");
        logger.info("Starting {} v{}", APP_NAME, APP_VERSION);
        logger.info("Industry Domain: Geospatial Intelligence");
        logger.info("================================================================================");

        try {
            logRuntimeInfo();
            Properties props = loadProperties();
            logger.info("Loaded configuration: Environment={}, Default GSD={}m, TileSize={}",
                    props.getProperty("app.environment", "development"),
                    props.getProperty("geospatial.default.gsd.meters", "10.0"),
                    props.getProperty("geospatial.tile.size", "256"));

            verifyProjectLayout();

            if (args != null && args.length > 0) {
                handleCommandLine(args);
            } else {
                logger.info("Use --demo to execute benchmark comparison, or --help for available commands.");
            }

            logger.info("================================================================================");
        } catch (IllegalArgumentException e) {
            System.err.println("Command-line argument error: " + e.getMessage());
            logger.error("Command-line argument error: {}", e.getMessage());
            printUsage();
            System.exit(1);
        } catch (Exception e) {
            logger.error("Execution failed due to error: {}", e.getMessage(), e);
            System.exit(1);
        }
    }

    public static void handleCommandLine(String[] args) throws Exception {
        Map<String, String> params = parseArgs(args);

        if (params.containsKey("--demo")) {
            runDemo(params);
        } else if (params.containsKey("--detect-change")) {
            runDetectChange(params);
        } else if (params.containsKey("--fundamentals")) {
            FundamentalsRunner.runAll();
        } else if (params.containsKey("--optimizers")) {
            OptimizerComparison.runComparison(Paths.get("experiments"));
        } else if (params.containsKey("--activations")) {
            ActivationComparison.exportCsv(Paths.get("experiments", "activation_comparison.csv"));
        } else if (params.containsKey("--learning-rules")) {
            LearningRulesDemo.runAll();
        } else if (params.containsKey("--temporal")) {
            TemporalLandUseDemo.main(new String[0]);
        } else if (params.containsKey("--generative")) {
            GenerativeAugmentationDemo.runEducationalDemo();
        } else if (params.containsKey("--help")) {
            printUsage();
        } else {
            throw new IllegalArgumentException("Unknown command. Supported: --demo, --detect-change, --fundamentals, --optimizers, --activations, --learning-rules, --temporal, --generative");
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    map.put(arg, args[i + 1]);
                    i++;
                } else {
                    map.put(arg, "true");
                }
            } else {
                throw new IllegalArgumentException("Unexpected bare argument: " + arg);
            }
        }
        return map;
    }

    /**
     * Executes the comprehensive automated demonstration:
     * - Runs Classical Baseline
     * - Runs Deep Learning Siamese ResNet-18
     * - Evaluates on real LEVIR-CD test tiles
     * - Exports all required artifacts and reports
     */
    public static void runDemo(Map<String, String> params) throws Exception {
        Path baseTestA = Paths.get("data", "processed", "LEVIR-CD", "test", "A");
        Path baseTestB = Paths.get("data", "processed", "LEVIR-CD", "test", "B");
        Path baseTestLabel = Paths.get("data", "processed", "LEVIR-CD", "test", "label");

        if (!Files.exists(baseTestA)) {
            baseTestA = Paths.get("data", "LEVIR-CD", "test", "A");
            baseTestB = Paths.get("data", "LEVIR-CD", "test", "B");
            baseTestLabel = Paths.get("data", "LEVIR-CD", "test", "label");
        }

        if (!Files.exists(baseTestA)) {
            throw new IllegalStateException("Test dataset not found in data/processed/LEVIR-CD/test");
        }

        List<Path> filesA = listSortedFiles(baseTestA);
        List<Path> filesB = new ArrayList<>();
        List<Path> filesL = new ArrayList<>();

        for (Path pA : filesA) {
            String name = pA.getFileName().toString();
            filesB.add(baseTestB.resolve(name));
            filesL.add(baseTestLabel.resolve(name));
        }

        Path resultsDir = Paths.get("results");
        Files.createDirectories(resultsDir);
        Files.createDirectories(Paths.get("experiments"));

        // Generate educational tables
        ActivationComparison.exportCsv(Paths.get("experiments", "activation_comparison.csv"));
        OptimizerComparison.runComparison(Paths.get("experiments"));

        // Select representative sample tile (preferably one with ground truth changes)
        int selectedIndex = 0;
        for (int i = 0; i < filesL.size(); i++) {
            if (filesA.get(i).getFileName().toString().contains("x0256")) {
                selectedIndex = i;
                break;
            }
        }

        Path samplePathA = filesA.get(selectedIndex);
        Path samplePathB = filesB.get(selectedIndex);
        Path samplePathL = filesL.get(selectedIndex);
        String sampleId = samplePathA.getFileName().toString().replaceFirst("\\.[^.]+$", "");

        BufferedImage imgA = ImageIO.read(samplePathA.toFile());
        BufferedImage imgB = ImageIO.read(samplePathB.toFile());
        BufferedImage imgL = Files.exists(samplePathL) ? ImageIO.read(samplePathL.toFile()) : null;

        // 1. Run Classical Baseline
        ChangeDetector baselineDetector = new ChangeDetector(35.0, 30);
        ChangeDetector.DetectionOutput baseOut = baselineDetector.processPair(imgA, imgB, imgL, sampleId);
        EncroachmentResult baseRes = baseOut.getResult();
        ChangeMetrics baseMetrics = baseRes.getMetrics();

        // 2. Run Deep Learning Model (Siamese ResNet-18)
        Path onnxPath = Paths.get("models", "onnx", "resnet18_change_detector.onnx");
        ChangeDetector.DetectionOutput dlOut = null;
        EncroachmentResult dlRes = null;
        ChangeMetrics dlMetrics = null;

        if (Files.exists(onnxPath)) {
            try (OnnxResNetChangeDetector dlDetector = new OnnxResNetChangeDetector(onnxPath, 30)) {
                dlOut = dlDetector.processPair(imgA, imgB, imgL, sampleId);
                dlRes = dlOut.getResult();
                dlMetrics = dlRes.getMetrics();
            }
        }

        // 3. Run Batch Comparison across all available test tiles
        List<ModelComparisonRunner.ModelBenchmarkResult> comparison = ModelComparisonRunner.compareModels(
                filesA, filesB, filesL, resultsDir, filesA.size()
        );

        // 4. Save required individual files in results/
        ChangeVisualizer visualizer = new ChangeVisualizer();

        // Before, After, Ground Truth
        ImageIO.write(imgA, "PNG", resultsDir.resolve("before.png").toFile());
        ImageIO.write(imgB, "PNG", resultsDir.resolve("after.png").toFile());
        if (imgL != null) {
            ImageIO.write(imgL, "PNG", resultsDir.resolve("ground_truth.png").toFile());
        }

        // Baseline change mask
        BufferedImage baseMaskImg = visualizer.createMaskImage(baseOut.getCleanMask());
        ImageIO.write(baseMaskImg, "PNG", resultsDir.resolve("baseline_change_mask.png").toFile());

        // DL prediction mask
        BufferedImage dlMaskImg = null;
        if (dlOut != null) {
            dlMaskImg = visualizer.createMaskImage(dlOut.getCleanMask());
            ImageIO.write(dlMaskImg, "PNG", resultsDir.resolve("dl_predicted_mask.png").toFile());
        }

        // Highlighted changes (bounding boxes on after image)
        BufferedImage activeMask = (dlMaskImg != null) ? dlMaskImg : baseMaskImg;
        List<com.satellite.change.ChangeRegion> activeRegions = (dlRes != null) ? dlRes.getRegions() : baseRes.getRegions();
        EncroachmentResult.Status activeStatus = (dlRes != null) ? dlRes.getStatus() : baseRes.getStatus();

        BufferedImage highlighted = visualizer.renderBoundingBoxes(imgB, activeRegions, activeStatus);
        ImageIO.write(highlighted, "PNG", resultsDir.resolve("highlighted_changes.png").toFile());

        // Comparison diagnostic board
        BufferedImage gtDisplay = (imgL != null) ? imgL : activeMask;
        EncroachmentResult alertRes = (dlRes != null) ? dlRes : baseRes;
        BufferedImage compBoard = visualizer.renderSideBySideBoard(
                imgA, imgB, gtDisplay, activeMask, highlighted, alertRes
        );
        ImageIO.write(compBoard, "PNG", resultsDir.resolve("comparison_image.png").toFile());

        // metrics.csv (Detailed sample metrics)
        Path metricsCsv = resultsDir.resolve("metrics.csv");
        try (BufferedWriter bw = Files.newBufferedWriter(metricsCsv)) {
            bw.write("Method,SampleId,IoU,F1Score,Precision,Recall,ChangedPixels,GroundTruthPixels,AreaErrorPixels,RuntimeMs\n");
            if (baseMetrics != null) {
                bw.write(String.format("Classical Baseline,%s,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n",
                        sampleId, baseMetrics.getIou(), baseMetrics.getF1Score(),
                        baseMetrics.getPrecision(), baseMetrics.getRecall(),
                        baseMetrics.getPredictedChangedPixels(), baseMetrics.getGroundTruthChangedPixels(),
                        baseMetrics.getAreaErrorPixels(), baseMetrics.getRuntimeMs()));
            }
            if (dlMetrics != null) {
                bw.write(String.format("Deep Learning (Siamese ResNet-18),%s,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n",
                        sampleId, dlMetrics.getIou(), dlMetrics.getF1Score(),
                        dlMetrics.getPrecision(), dlMetrics.getRecall(),
                        dlMetrics.getPredictedChangedPixels(), dlMetrics.getGroundTruthChangedPixels(),
                        dlMetrics.getAreaErrorPixels(), dlMetrics.getRuntimeMs()));
            }
        }

        // result.json (Structured results JSON)
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        Map<String, Object> jsonMap = new LinkedHashMap<>();
        jsonMap.put("dataset", "LEVIR-CD");
        jsonMap.put("sampleId", sampleId);
        jsonMap.put("classicalBaseline", baseRes);
        if (dlRes != null) {
            jsonMap.put("deepLearningModel", dlRes);
        }
        jsonMap.put("batchComparison", comparison);
        jsonMap.put("notice", baseRes.getDisclaimer());
        mapper.writeValue(resultsDir.resolve("result.json").toFile(), jsonMap);

        // Concise Console Output as specified in Part 15
        System.out.println();
        System.out.println("Satellite Land Use Change and Encroachment Detector");
        System.out.println();
        System.out.println("Dataset: LEVIR-CD");
        System.out.println("Input: Before + After satellite image (" + sampleId + ")");
        System.out.println();
        System.out.println("Classical Baseline:");
        System.out.printf("Changed Area: %d pixels (%.2f%%)%n", baseRes.getTotalChangedPixels(), baseRes.getChangePercentage());
        if (baseMetrics != null) {
            System.out.printf("IoU: %.4f%n", baseMetrics.getIou());
            System.out.printf("F1: %.4f%n", baseMetrics.getF1Score());
            System.out.printf("Runtime: %d ms%n", baseMetrics.getRuntimeMs());
        }
        System.out.println();
        System.out.println("Deep-Learning Model:");
        System.out.println("Model: Siamese Pretrained ResNet-18 (ONNX Runtime)");
        if (dlMetrics != null) {
            System.out.printf("Changed Area: %d pixels (%.2f%%)%n", dlRes.getTotalChangedPixels(), dlRes.getChangePercentage());
            System.out.printf("IoU: %.4f%n", dlMetrics.getIou());
            System.out.printf("F1: %.4f%n", dlMetrics.getF1Score());
            System.out.printf("Runtime: %d ms%n", dlMetrics.getRuntimeMs());
        } else {
            System.out.println("Status: ONNX model inference not available");
        }
        System.out.println();
        System.out.println("Potential Encroachment:");
        System.out.printf("Screening Status: %s%n", alertRes.getStatus());
        System.out.printf("Confidence: %.1f%%%n", alertRes.getConfidence() * 100.0);
        System.out.printf("Detected Clusters: %d%n", alertRes.getRegionCount());
        System.out.println("Notice: " + alertRes.getDisclaimer());
        System.out.println();
        System.out.println("Outputs saved to:");
        System.out.println(resultsDir.toAbsolutePath());
        System.out.println();
    }

    public static void runDetectChange(Map<String, String> params) throws Exception {
        String beforeStr = params.get("--before");
        String afterStr = params.get("--after");
        String labelStr = params.get("--label");
        String outputDirStr = params.getOrDefault("--output", "results/change-detection");

        if (beforeStr == null || afterStr == null) {
            throw new IllegalArgumentException("--detect-change requires both --before <path> and --after <path>");
        }

        Path pathA = Paths.get(beforeStr);
        Path pathB = Paths.get(afterStr);
        Path pathL = (labelStr != null) ? Paths.get(labelStr) : null;
        Path outDir = Paths.get(outputDirStr);

        if (!Files.exists(pathA)) throw new IllegalArgumentException("Before image does not exist: " + pathA);
        if (!Files.exists(pathB)) throw new IllegalArgumentException("After image does not exist: " + pathB);
        if (pathL != null && !Files.exists(pathL)) throw new IllegalArgumentException("Label mask does not exist: " + pathL);

        double threshold = parseDoubleOrDefault(params.get("--threshold"), 35.0);
        int minArea = parseIntOrDefault(params.get("--min-region-area"), 30);

        BufferedImage imgA = ImageIO.read(pathA.toFile());
        BufferedImage imgB = ImageIO.read(pathB.toFile());
        BufferedImage imgL = (pathL != null) ? ImageIO.read(pathL.toFile()) : null;

        ChangeDetector detector = new ChangeDetector(threshold, minArea);
        String sampleId = stripExtension(pathA.getFileName().toString());

        ChangeDetector.DetectionOutput output = detector.processPair(imgA, imgB, imgL, sampleId);
        detector.saveVisualizations(imgA, imgB, imgL, output, outDir);

        EncroachmentResult res = output.getResult();
        System.out.println("================================================================================");
        System.out.println("  CHANGE DETECTION & ENCROACHMENT RESULT: " + sampleId);
        System.out.println("================================================================================");
        System.out.printf("  Screening Status:        %s%n", res.getStatus());
        System.out.printf("  Confidence:              %.1f%%%n", res.getConfidence() * 100.0);
        System.out.printf("  Changed Area:            %d pixels (%.2f%% of image)%n", res.getTotalChangedPixels(), res.getChangePercentage());
        System.out.printf("  Detected Clusters:       %d%n", res.getRegionCount());
        System.out.printf("  Largest Cluster:         %d pixels%n", res.getLargestRegionPixels());

        if (res.getMetrics() != null) {
            ChangeMetrics m = res.getMetrics();
            System.out.printf("  IoU:                     %.4f%n", m.getIou());
            System.out.printf("  F1-Score:                %.4f%n", m.getF1Score());
            System.out.printf("  Precision:               %.4f%n", m.getPrecision());
            System.out.printf("  Recall (Change Recall):  %.4f%n", m.getRecall());
            System.out.printf("  Area Error:              %d pixels%n", m.getAreaErrorPixels());
        }
        System.out.println("  Notice:                  " + res.getDisclaimer());
        System.out.printf("  Visualizations Saved:    %s%n", outDir.toAbsolutePath());
        System.out.println("================================================================================");
    }

    private static List<Path> listSortedFiles(Path dir) throws Exception {
        List<Path> list = new ArrayList<>();
        try (var stream = Files.list(dir)) {
            stream.filter(p -> {
                String n = p.getFileName().toString().toLowerCase();
                return n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".tif") || n.endsWith(".jpeg");
            }).forEach(list::add);
        }
        list.sort(Comparator.comparing(Path::getFileName));
        return list;
    }

    private static double parseDoubleOrDefault(String val, double def) {
        if (val == null) return def;
        try { return Double.parseDouble(val); } catch (Exception e) { return def; }
    }

    private static int parseIntOrDefault(String val, int def) {
        if (val == null) return def;
        try { return Integer.parseInt(val); } catch (Exception e) { return def; }
    }

    private static String stripExtension(String fn) {
        int dot = fn.lastIndexOf('.');
        return dot > 0 ? fn.substring(0, dot) : fn;
    }

    private static void printUsage() {
        System.out.println("\nUsage: java -jar target/satellite-detector.jar [OPTIONS]");
        System.out.println("Options:");
        System.out.println("  --demo                        Execute automated demonstration (LEVIR-CD)");
        System.out.println("  --detect-change               Execute single-pair change detection");
        System.out.println("    --before <path>             Path to pre-period optical satellite image");
        System.out.println("    --after <path>              Path to post-period optical satellite image");
        System.out.println("    --label <path>              (Optional) Path to ground-truth change mask");
        System.out.println("    --threshold <double>        Radiometric change threshold (default: 35.0)");
        System.out.println("    --min-region-area <int>     Minimum cluster size in pixels (default: 30)");
        System.out.println("    --output <path>             Directory to store results");
        System.out.println("  --fundamentals                Run Course Neural-Network fundamentals demo");
        System.out.println("  --optimizers                  Run 4-optimizer benchmark comparison");
        System.out.println("  --activations                 Export activation function comparison CSV");
        System.out.println("  --learning-rules              Run 5 educational learning rules demo");
        System.out.println("  --temporal                    Run temporal multi-year land change sequence demo");
        System.out.println("  --generative                  Run generative VAE/GAN augmentation demo");
        System.out.println("  --help                        Display this usage guide\n");
    }

    public static void logRuntimeInfo() {
        Runtime rt = Runtime.getRuntime();
        long maxMemoryMB = rt.maxMemory() / (1024 * 1024);
        long totalMemoryMB = rt.totalMemory() / (1024 * 1024);
        int availableProcessors = rt.availableProcessors();

        logger.info("JVM: {} (v{}) by {}",
                System.getProperty("java.vm.name"),
                System.getProperty("java.version"),
                System.getProperty("java.vendor"));
        logger.info("OS: {} {} ({})",
                System.getProperty("os.name"),
                System.getProperty("os.version"),
                System.getProperty("os.arch"));
        logger.info("Hardware: {} available processors, Max Heap: {} MB, Current Heap: {} MB",
                availableProcessors, maxMemoryMB, totalMemoryMB);
    }

    public static Properties loadProperties() {
        Properties properties = new Properties();
        try (InputStream in = Main.class.getClassLoader().getResourceAsStream("application.properties")) {
            if (in != null) {
                properties.load(in);
            } else {
                logger.warn("application.properties not found on classpath, using defaults.");
            }
        } catch (Exception e) {
            logger.warn("Failed to load application.properties: {}. Using fallback defaults.", e.getMessage());
        }
        return properties;
    }

    public static boolean verifyProjectLayout() {
        String[] requiredFolders = {
                "configs", "data", "models", "experiments", "visualizations", "docs"
        };
        boolean allValid = true;
        for (String folder : requiredFolders) {
            Path p = Paths.get(folder);
            if (!Files.exists(p)) {
                try {
                    Files.createDirectories(p);
                } catch (Exception ignored) {}
            }
            if (Files.exists(p) && Files.isDirectory(p)) {
                logger.debug("Project directory verified: {} [OK]", folder);
            } else {
                logger.warn("Project directory missing or inaccessible: {} [WARN]", folder);
                allValid = false;
            }
        }
        return allValid;
    }
}
