package com.satellite.app;

import com.satellite.change.ChangeDetector;
import com.satellite.detection.EncroachmentResult;
import com.satellite.evaluation.ChangeMetrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Entry point for the Satellite Land Use Change and Encroachment Detector.
 * <p>
 * Geospatial Intelligence Domain:
 * Multitemporal optical satellite image analysis for detecting Land Use / Land Cover (LULC)
 * transitions and unauthorized human encroachment into environmentally protected zones.
 * </p>
 */
public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static final String APP_NAME = "Satellite Land Use Change and Encroachment Detector";
    public static final String APP_VERSION = "1.0.0-SNAPSHOT";

    public static void main(String[] args) {
        // Required exact output lines:
        System.out.println("Satellite Land Use Change and Encroachment Detector");
        System.out.println("Java Maven project initialized successfully.");

        logger.info("================================================================================");
        logger.info("Starting {} v{}", APP_NAME, APP_VERSION);
        logger.info("Industry Domain: Geospatial Intelligence");
        logger.info("================================================================================");

        try {
            // 1. Report Runtime Environment
            logRuntimeInfo();

            // 2. Load Properties
            Properties props = loadProperties();
            logger.info("Loaded configuration: Environment={}, Default GSD={}m, TileSize={}",
                    props.getProperty("app.environment", "development"),
                    props.getProperty("geospatial.default.gsd.meters", "10.0"),
                    props.getProperty("geospatial.tile.size", "256"));

            // 3. Verify Directory Infrastructure
            verifyProjectLayout();

            // 4. Dispatch Command-Line Flags
            if (args != null && args.length > 0) {
                handleCommandLine(args);
            } else {
                logger.info("Stage 1 & 2 initialization verified. Use --demo or --detect-change for change detection.");
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

    /**
     * Parses and dispatches CLI options.
     */
    public static void handleCommandLine(String[] args) throws Exception {
        Map<String, String> params = parseArgs(args);

        if (params.containsKey("--demo")) {
            runDemo(params);
        } else if (params.containsKey("--detect-change")) {
            runDetectChange(params);
        } else {
            throw new IllegalArgumentException("Unknown command or flag. Supported: --demo, --detect-change");
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
     * Executes the automatic demonstration using available processed test tiles.
     */
    public static void runDemo(Map<String, String> params) throws Exception {
        System.out.println("\n[DEMO] Starting Automated Change Detection & Encroachment Screening Demo...");
        logger.info("[DEMO] Starting Automated Change Detection & Encroachment Screening Demo...");

        Path baseTestA = Paths.get("data", "processed", "LEVIR-CD", "test", "A");
        Path baseTestB = Paths.get("data", "processed", "LEVIR-CD", "test", "B");
        Path baseTestLabel = Paths.get("data", "processed", "LEVIR-CD", "test", "label");

        // Fallback to raw dataset test split if processed tiles are not present
        if (!Files.exists(baseTestA)) {
            baseTestA = Paths.get("data", "LEVIR-CD", "test", "A");
            baseTestB = Paths.get("data", "LEVIR-CD", "test", "B");
            baseTestLabel = Paths.get("data", "LEVIR-CD", "test", "label");
        }

        if (!Files.exists(baseTestA)) {
            throw new IllegalStateException("Test dataset not found in data/processed/LEVIR-CD/test or data/LEVIR-CD/test");
        }

        List<Path> filesA = listSortedFiles(baseTestA);
        List<Path> filesB = new ArrayList<>();
        List<Path> filesL = new ArrayList<>();

        for (Path pA : filesA) {
            String name = pA.getFileName().toString();
            filesB.add(baseTestB.resolve(name));
            filesL.add(baseTestLabel.resolve(name));
        }

        double threshold = parseDoubleOrDefault(params.get("--threshold"), 35.0);
        int minArea = parseIntOrDefault(params.get("--min-region-area"), 30);
        Path outputDir = Paths.get(params.getOrDefault("--output", "results/change-detection"));
        Path vizDir = Paths.get("visualizations", "change-detection");

        ChangeDetector detector = new ChangeDetector(threshold, minArea);
        ChangeDetector.BatchEvaluationResult evalResult = detector.evaluateBatch(
                filesA, filesB, filesL, outputDir, vizDir, 6
        );

        System.out.println("================================================================================");
        System.out.println("  STAGE 3: CHANGE DETECTION & ENCROACHMENT SCREENING BENCHMARK RESULTS");
        System.out.println("================================================================================");
        System.out.printf("  Samples Evaluated:       %d tiles%n", evalResult.getResults().size());
        System.out.printf("  Mean IoU:                %.4f%n", evalResult.getMeanIou());
        System.out.printf("  Mean F1-Score:           %.4f%n", evalResult.getMeanF1());
        System.out.printf("  Mean Change Recall:      %.4f%n", evalResult.getMeanRecall());
        System.out.printf("  Mean Precision:          %.4f%n", evalResult.getMeanPrecision());
        System.out.printf("  Mean Area Error:         %.1f pixels%n", evalResult.getMeanAreaError());
        System.out.printf("  Total Runtime:           %d ms%n", evalResult.getTotalRuntimeMs());
        System.out.println("--------------------------------------------------------------------------------");
        long countEncroachment = evalResult.getResults().stream().filter(r -> r.getStatus() == EncroachmentResult.Status.POSSIBLE_ENCROACHMENT).count();
        long countChange = evalResult.getResults().stream().filter(r -> r.getStatus() == EncroachmentResult.Status.POSSIBLE_CHANGE).count();
        long countNoChange = evalResult.getResults().stream().filter(r -> r.getStatus() == EncroachmentResult.Status.NO_SIGNIFICANT_CHANGE).count();
        System.out.printf("  Screening Breakdown:     Encroachments: %d | Changes: %d | No Change: %d%n",
                countEncroachment, countChange, countNoChange);
        System.out.printf("  Outputs Generated:       %s%n", outputDir.toAbsolutePath());
        System.out.printf("  Visualizations Saved:    %s%n", vizDir.toAbsolutePath());
        System.out.println("================================================================================");
    }

    /**
     * Executes single-pair change detection via CLI arguments.
     */
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
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.png")) {
            for (Path p : stream) list.add(p);
        }
        list.sort(Comparator.comparing(Path::getFileName));
        return list;
    }

    private static double parseDoubleOrDefault(String str, double def) {
        if (str == null) return def;
        try {
            return Double.parseDouble(str);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid floating-point number: " + str);
        }
    }

    private static int parseIntOrDefault(String str, int def) {
        if (str == null) return def;
        try {
            return Integer.parseInt(str);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid integer number: " + str);
        }
    }

    private static String stripExtension(String name) {
        int idx = name.lastIndexOf('.');
        return idx > 0 ? name.substring(0, idx) : name;
    }

    private static void printUsage() {
        System.out.println("Usage:");
        System.out.println("  java -jar target/satellite-detector.jar --demo");
        System.out.println("  java -jar target/satellite-detector.jar --detect-change --before <A.png> --after <B.png> [--label <label.png>] [--output <dir>] [--threshold <35>] [--min-region-area <30>]");
    }

    /**
     * Inspects and logs JVM and host execution parameters.
     */
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

    /**
     * Loads the core application properties from classpath.
     */
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

    /**
     * Verifies that the required project root folders exist.
     */
    public static boolean verifyProjectLayout() {
        String[] requiredFolders = {
                "configs", "data", "models", "experiments", "visualizations", "docs"
        };
        boolean allValid = true;
        for (String folder : requiredFolders) {
            Path p = Paths.get(folder);
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
