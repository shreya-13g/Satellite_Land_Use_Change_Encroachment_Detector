package com.satellite.preprocessing;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Configuration parameters for the LEVIR-CD dataset handling and satellite image preprocessing.
 */
public class DatasetConfig {

    private Path datasetRoot = Paths.get("data", "LEVIR-CD");
    private Path processedRoot = Paths.get("data", "processed", "LEVIR-CD");
    private Path resultsDir = Paths.get("results");
    private Path visualizationsDir = Paths.get("visualizations", "dataset");

    private List<String> splits = Arrays.asList("train", "val", "test");
    private String subfolderA = "A";
    private String subfolderB = "B";
    private String subfolderLabel = "label";

    private int tileWidth = 256;
    private int tileHeight = 256;
    private int tileStride = 256;

    private int maxPairsToTilePerSplit = 5; // -1 for all; positive for representative export
    private int maxVisualizationsPerSplit = 3;

    private Set<String> supportedExtensions = new HashSet<>(
            Arrays.asList(".png", ".jpg", ".jpeg", ".tif", ".tiff")
    );

    public DatasetConfig() {
    }

    public static DatasetConfig defaultConfig() {
        return new DatasetConfig();
    }

    public Path getDatasetRoot() {
        return datasetRoot;
    }

    public DatasetConfig setDatasetRoot(Path datasetRoot) {
        this.datasetRoot = datasetRoot;
        return this;
    }

    public Path getProcessedRoot() {
        return processedRoot;
    }

    public DatasetConfig setProcessedRoot(Path processedRoot) {
        this.processedRoot = processedRoot;
        return this;
    }

    public Path getResultsDir() {
        return resultsDir;
    }

    public DatasetConfig setResultsDir(Path resultsDir) {
        this.resultsDir = resultsDir;
        return this;
    }

    public Path getVisualizationsDir() {
        return visualizationsDir;
    }

    public DatasetConfig setVisualizationsDir(Path visualizationsDir) {
        this.visualizationsDir = visualizationsDir;
        return this;
    }

    public List<String> getSplits() {
        return Collections.unmodifiableList(splits);
    }

    public DatasetConfig setSplits(List<String> splits) {
        this.splits = splits;
        return this;
    }

    public String getSubfolderA() {
        return subfolderA;
    }

    public DatasetConfig setSubfolderA(String subfolderA) {
        this.subfolderA = subfolderA;
        return this;
    }

    public String getSubfolderB() {
        return subfolderB;
    }

    public DatasetConfig setSubfolderB(String subfolderB) {
        this.subfolderB = subfolderB;
        return this;
    }

    public String getSubfolderLabel() {
        return subfolderLabel;
    }

    public DatasetConfig setSubfolderLabel(String subfolderLabel) {
        this.subfolderLabel = subfolderLabel;
        return this;
    }

    public int getTileWidth() {
        return tileWidth;
    }

    public DatasetConfig setTileWidth(int tileWidth) {
        this.tileWidth = tileWidth;
        return this;
    }

    public int getTileHeight() {
        return tileHeight;
    }

    public DatasetConfig setTileHeight(int tileHeight) {
        this.tileHeight = tileHeight;
        return this;
    }

    public int getTileStride() {
        return tileStride;
    }

    public DatasetConfig setTileStride(int tileStride) {
        this.tileStride = tileStride;
        return this;
    }

    public int getMaxPairsToTilePerSplit() {
        return maxPairsToTilePerSplit;
    }

    public DatasetConfig setMaxPairsToTilePerSplit(int maxPairsToTilePerSplit) {
        this.maxPairsToTilePerSplit = maxPairsToTilePerSplit;
        return this;
    }

    public int getMaxVisualizationsPerSplit() {
        return maxVisualizationsPerSplit;
    }

    public DatasetConfig setMaxVisualizationsPerSplit(int maxVisualizationsPerSplit) {
        this.maxVisualizationsPerSplit = maxVisualizationsPerSplit;
        return this;
    }

    public Set<String> getSupportedExtensions() {
        return Collections.unmodifiableSet(supportedExtensions);
    }

    public DatasetConfig setSupportedExtensions(Set<String> supportedExtensions) {
        this.supportedExtensions = supportedExtensions;
        return this;
    }
}
