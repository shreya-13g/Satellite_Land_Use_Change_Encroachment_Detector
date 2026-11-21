package com.satellite.preprocessing;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Summary metrics for the LEVIR-CD dataset inspection, validation, and tile preprocessing.
 */
public class DatasetSummary {

    @JsonProperty("trainPairs")
    private int trainPairsCount;

    @JsonProperty("validationPairs")
    private int valPairsCount;

    @JsonProperty("testPairs")
    private int testPairsCount;

    @JsonProperty("validPairs")
    private int validPairsCount;

    @JsonProperty("invalidPairs")
    private int invalidPairsCount;

    @JsonProperty("missingFiles")
    private int missingFilesCount;

    @JsonProperty("imageDimensions")
    private String imageDimensions = "Unknown";

    @JsonProperty("maskDimensions")
    private String maskDimensions = "Unknown";

    @JsonProperty("changedPixelCount")
    private long changedPixelCount;

    @JsonProperty("unchangedPixelCount")
    private long unchangedPixelCount;

    @JsonProperty("totalProcessedTiles")
    private int totalProcessedTiles;

    @JsonProperty("processingTimeMs")
    private long processingTimeMs;

    public DatasetSummary() {
    }

    public int getTrainPairsCount() {
        return trainPairsCount;
    }

    public void setTrainPairsCount(int trainPairsCount) {
        this.trainPairsCount = trainPairsCount;
    }

    public int getValPairsCount() {
        return valPairsCount;
    }

    public void setValPairsCount(int valPairsCount) {
        this.valPairsCount = valPairsCount;
    }

    public int getTestPairsCount() {
        return testPairsCount;
    }

    public void setTestPairsCount(int testPairsCount) {
        this.testPairsCount = testPairsCount;
    }

    public int getValidPairsCount() {
        return validPairsCount;
    }

    public void setValidPairsCount(int validPairsCount) {
        this.validPairsCount = validPairsCount;
    }

    public int getInvalidPairsCount() {
        return invalidPairsCount;
    }

    public void setInvalidPairsCount(int invalidPairsCount) {
        this.invalidPairsCount = invalidPairsCount;
    }

    public int getMissingFilesCount() {
        return missingFilesCount;
    }

    public void setMissingFilesCount(int missingFilesCount) {
        this.missingFilesCount = missingFilesCount;
    }

    public String getImageDimensions() {
        return imageDimensions;
    }

    public void setImageDimensions(String imageDimensions) {
        this.imageDimensions = imageDimensions;
    }

    public String getMaskDimensions() {
        return maskDimensions;
    }

    public void setMaskDimensions(String maskDimensions) {
        this.maskDimensions = maskDimensions;
    }

    public long getChangedPixelCount() {
        return changedPixelCount;
    }

    public void setChangedPixelCount(long changedPixelCount) {
        this.changedPixelCount = changedPixelCount;
    }

    public long getUnchangedPixelCount() {
        return unchangedPixelCount;
    }

    public void setUnchangedPixelCount(long unchangedPixelCount) {
        this.unchangedPixelCount = unchangedPixelCount;
    }

    public int getTotalProcessedTiles() {
        return totalProcessedTiles;
    }

    public void setTotalProcessedTiles(int totalProcessedTiles) {
        this.totalProcessedTiles = totalProcessedTiles;
    }

    public long getProcessingTimeMs() {
        return processingTimeMs;
    }

    public void setProcessingTimeMs(long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }

    public double getChangeRatio() {
        long total = changedPixelCount + unchangedPixelCount;
        return total > 0 ? (double) changedPixelCount / total : 0.0;
    }

    public int getTotalPairsCount() {
        return trainPairsCount + valPairsCount + testPairsCount;
    }

    /**
     * Serializes this summary into a JSON file using Jackson.
     */
    public void toJsonFile(Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(path.toFile(), this);
    }

    /**
     * Serializes this summary into a CSV file.
     */
    public void toCsvFile(Path path) throws IOException {
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            writer.write("metric,value\n");
            writer.write(String.format("number_of_train_pairs,%d\n", trainPairsCount));
            writer.write(String.format("number_of_validation_pairs,%d\n", valPairsCount));
            writer.write(String.format("number_of_test_pairs,%d\n", testPairsCount));
            writer.write(String.format("number_of_valid_pairs,%d\n", validPairsCount));
            writer.write(String.format("number_of_invalid_pairs,%d\n", invalidPairsCount));
            writer.write(String.format("number_of_missing_files,%d\n", missingFilesCount));
            writer.write(String.format("image_dimensions,%s\n", imageDimensions));
            writer.write(String.format("mask_dimensions,%s\n", maskDimensions));
            writer.write(String.format("changed_pixel_count,%d\n", changedPixelCount));
            writer.write(String.format("unchanged_pixel_count,%d\n", unchangedPixelCount));
            writer.write(String.format("change_pixel_ratio,%.6f\n", getChangeRatio()));
            writer.write(String.format("total_processed_tiles,%d\n", totalProcessedTiles));
            writer.write(String.format("processing_time_ms,%d\n", processingTimeMs));
        }
    }
}
