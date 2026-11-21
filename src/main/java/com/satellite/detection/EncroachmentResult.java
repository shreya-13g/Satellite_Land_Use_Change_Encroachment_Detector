package com.satellite.detection;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.satellite.change.ChangeRegion;
import com.satellite.evaluation.ChangeMetrics;

import java.util.Collections;
import java.util.List;

/**
 * Result data transfer object for automated encroachment screening.
 * Clearly designated as an automated decision-support metric, not a definitive legal adjudication.
 */
public class EncroachmentResult {

    public enum Status {
        NO_SIGNIFICANT_CHANGE,
        POSSIBLE_CHANGE,
        POSSIBLE_ENCROACHMENT
    }

    public static final String LEGAL_DISCLAIMER =
            "Automated screening result based on bi-temporal satellite raster processing. " +
            "Not a legal or final land-ownership determination. Field survey verification required.";

    @JsonProperty("sampleId")
    private final String sampleId;

    @JsonProperty("status")
    private final Status status;

    @JsonProperty("confidence")
    private final double confidence;

    @JsonProperty("totalChangedPixels")
    private final int totalChangedPixels;

    @JsonProperty("totalImagePixels")
    private final int totalImagePixels;

    @JsonProperty("changePercentage")
    private final double changePercentage;

    @JsonProperty("regionCount")
    private final int regionCount;

    @JsonProperty("largestRegionPixels")
    private final int largestRegionPixels;

    @JsonProperty("regions")
    private final List<ChangeRegion> regions;

    @JsonProperty("metrics")
    private ChangeMetrics metrics;

    @JsonProperty("disclaimer")
    private final String disclaimer = LEGAL_DISCLAIMER;

    public EncroachmentResult(String sampleId, Status status, double confidence,
                              int totalChangedPixels, int totalImagePixels,
                              int largestRegionPixels, List<ChangeRegion> regions) {
        this.sampleId = sampleId;
        this.status = status;
        this.confidence = confidence;
        this.totalChangedPixels = totalChangedPixels;
        this.totalImagePixels = totalImagePixels;
        this.changePercentage = totalImagePixels > 0 ? ((double) totalChangedPixels / totalImagePixels) * 100.0 : 0.0;
        this.regions = regions != null ? regions : Collections.emptyList();
        this.regionCount = this.regions.size();
        this.largestRegionPixels = largestRegionPixels;
    }

    public String getSampleId() {
        return sampleId;
    }

    public Status getStatus() {
        return status;
    }

    public double getConfidence() {
        return confidence;
    }

    public int getTotalChangedPixels() {
        return totalChangedPixels;
    }

    public int getTotalImagePixels() {
        return totalImagePixels;
    }

    public double getChangePercentage() {
        return changePercentage;
    }

    public int getRegionCount() {
        return regionCount;
    }

    public int getLargestRegionPixels() {
        return largestRegionPixels;
    }

    public List<ChangeRegion> getRegions() {
        return Collections.unmodifiableList(regions);
    }

    public ChangeMetrics getMetrics() {
        return metrics;
    }

    public void setMetrics(ChangeMetrics metrics) {
        this.metrics = metrics;
    }

    public String getDisclaimer() {
        return disclaimer;
    }

    @Override
    public String toString() {
        return String.format("EncroachmentResult[%s: Status=%s, Conf=%.2f, Changed=%d px (%.2f%%), Clusters=%d, Largest=%d px]",
                sampleId, status, confidence, totalChangedPixels, changePercentage, regionCount, largestRegionPixels);
    }
}
