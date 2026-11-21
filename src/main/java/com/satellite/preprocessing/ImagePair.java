package com.satellite.preprocessing;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Represents a bi-temporal satellite image pair and its associated ground-truth change mask.
 * <ul>
 *   <li>A: Earlier satellite image (Time T1)</li>
 *   <li>B: Later satellite image (Time T2)</li>
 *   <li>label: Binary ground truth mask indicating building changes (0 = unchanged, 255/1 = changed)</li>
 * </ul>
 */
public class ImagePair {

    private final String id;
    private final String split;
    private final Path pathA;
    private final Path pathB;
    private final Path pathLabel;

    private boolean valid = false;
    private String validationMessage = "Pending validation";

    private int widthA = -1;
    private int heightA = -1;
    private int widthB = -1;
    private int heightB = -1;
    private int widthLabel = -1;
    private int heightLabel = -1;

    private long changedPixelCount = 0;
    private long unchangedPixelCount = 0;

    public ImagePair(String id, String split, Path pathA, Path pathB, Path pathLabel) {
        this.id = id;
        this.split = split;
        this.pathA = pathA;
        this.pathB = pathB;
        this.pathLabel = pathLabel;
    }

    public String getId() {
        return id;
    }

    public String getSplit() {
        return split;
    }

    public Path getPathA() {
        return pathA;
    }

    public Path getPathB() {
        return pathB;
    }

    public Path getPathLabel() {
        return pathLabel;
    }

    public boolean isValid() {
        return valid;
    }

    public void setValid(boolean valid) {
        this.valid = valid;
    }

    public String getValidationMessage() {
        return validationMessage;
    }

    public void setValidationMessage(String validationMessage) {
        this.validationMessage = validationMessage;
    }

    public int getWidthA() {
        return widthA;
    }

    public void setDimensionsA(int width, int height) {
        this.widthA = width;
        this.heightA = height;
    }

    public int getHeightA() {
        return heightA;
    }

    public int getWidthB() {
        return widthB;
    }

    public void setDimensionsB(int width, int height) {
        this.widthB = width;
        this.heightB = height;
    }

    public int getHeightB() {
        return heightB;
    }

    public int getWidthLabel() {
        return widthLabel;
    }

    public void setDimensionsLabel(int width, int height) {
        this.widthLabel = width;
        this.heightLabel = height;
    }

    public int getHeightLabel() {
        return heightLabel;
    }

    public long getChangedPixelCount() {
        return changedPixelCount;
    }

    public void setPixelCounts(long changed, long unchanged) {
        this.changedPixelCount = changed;
        this.unchangedPixelCount = unchanged;
    }

    public long getUnchangedPixelCount() {
        return unchangedPixelCount;
    }

    public double getChangeRatio() {
        long total = changedPixelCount + unchangedPixelCount;
        return total > 0 ? (double) changedPixelCount / total : 0.0;
    }

    public boolean allFilesExist() {
        return pathA != null && Files.exists(pathA) &&
               pathB != null && Files.exists(pathB) &&
               pathLabel != null && Files.exists(pathLabel);
    }

    public boolean hasMatchingDimensions() {
        return widthA > 0 && widthA == widthB && widthB == widthLabel &&
               heightA > 0 && heightA == heightB && heightB == heightLabel;
    }

    @Override
    public String toString() {
        return String.format("ImagePair[%s, split=%s, valid=%s, dims=%dx%d, changed=%d, msg='%s']",
                id, split, valid, widthA, heightA, changedPixelCount, validationMessage);
    }
}
