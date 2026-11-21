package com.satellite.change;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Represents a discrete, contiguous connected component of detected change in a satellite image.
 */
public class ChangeRegion {

    @JsonProperty("id")
    private final int id;

    @JsonProperty("minX")
    private int minX;

    @JsonProperty("minY")
    private int minY;

    @JsonProperty("maxX")
    private int maxX;

    @JsonProperty("maxY")
    private int maxY;

    @JsonProperty("areaPixels")
    private int areaPixels;

    @JsonProperty("percentageOfImage")
    private double percentageOfImage;

    @JsonProperty("centroidX")
    private double centroidX;

    @JsonProperty("centroidY")
    private double centroidY;

    @JsonProperty("classification")
    private String classification = "UNKNOWN";

    public ChangeRegion(int id) {
        this.id = id;
        this.minX = Integer.MAX_VALUE;
        this.minY = Integer.MAX_VALUE;
        this.maxX = Integer.MIN_VALUE;
        this.maxY = Integer.MIN_VALUE;
        this.areaPixels = 0;
    }

    public ChangeRegion(int id, int minX, int minY, int maxX, int maxY, int areaPixels,
                        double percentageOfImage, double centroidX, double centroidY) {
        this.id = id;
        this.minX = minX;
        this.minY = minY;
        this.maxX = maxX;
        this.maxY = maxY;
        this.areaPixels = areaPixels;
        this.percentageOfImage = percentageOfImage;
        this.centroidX = centroidX;
        this.centroidY = centroidY;
    }

    /**
     * Accumulates a pixel into this region, updating bounding box and area.
     */
    public void addPixel(int x, int y) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        areaPixels++;
    }

    public void finalizeRegion(int imageWidth, int imageHeight, double sumX, double sumY) {
        if (areaPixels > 0) {
            this.centroidX = sumX / areaPixels;
            this.centroidY = sumY / areaPixels;
            long totalPixels = (long) imageWidth * imageHeight;
            this.percentageOfImage = totalPixels > 0 ? ((double) areaPixels / totalPixels) * 100.0 : 0.0;
        }
    }

    public int getId() {
        return id;
    }

    public int getMinX() {
        return minX;
    }

    public int getMinY() {
        return minY;
    }

    public int getMaxX() {
        return maxX;
    }

    public int getMaxY() {
        return maxY;
    }

    public int getWidth() {
        return (maxX >= minX) ? (maxX - minX + 1) : 0;
    }

    public int getHeight() {
        return (maxY >= minY) ? (maxY - minY + 1) : 0;
    }

    public int getAreaPixels() {
        return areaPixels;
    }

    public double getPercentageOfImage() {
        return percentageOfImage;
    }

    public double getCentroidX() {
        return centroidX;
    }

    public double getCentroidY() {
        return centroidY;
    }

    public String getClassification() {
        return classification;
    }

    public void setClassification(String classification) {
        this.classification = classification;
    }

    @Override
    public String toString() {
        return String.format("ChangeRegion[id=%d, bbox=[%d,%d -> %d,%d], size=%dx%d, area=%d px (%.2f%%), centroid=(%.1f, %.1f)]",
                id, minX, minY, maxX, maxY, getWidth(), getHeight(), areaPixels, percentageOfImage, centroidX, centroidY);
    }
}
