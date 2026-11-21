package com.satellite.detection;

import com.satellite.change.ChangeRegion;

import java.util.List;

/**
 * Automated encroachment screening engine.
 * Classifies bi-temporal change patterns into:
 * <ul>
 *   <li>NO_SIGNIFICANT_CHANGE</li>
 *   <li>POSSIBLE_CHANGE</li>
 *   <li>POSSIBLE_ENCROACHMENT</li>
 * </ul>
 * Note: These thresholds represent screening criteria for change prioritization and are
 * not legal determinations.
 */
public class EncroachmentDetector {

    public static final double DEFAULT_NO_CHANGE_PERCENT_MAX = 0.08;     // 0.08%
    public static final double DEFAULT_ENCROACHMENT_PERCENT_MIN = 0.40;   // 0.40%
    public static final int DEFAULT_ENCROACHMENT_CLUSTER_MIN_PIXELS = 150; // pixels

    private final double noChangePercentMax;
    private final double encroachmentPercentMin;
    private final int encroachmentClusterMinPixels;

    public EncroachmentDetector(double noChangePercentMax, double encroachmentPercentMin, int encroachmentClusterMinPixels) {
        this.noChangePercentMax = noChangePercentMax;
        this.encroachmentPercentMin = encroachmentPercentMin;
        this.encroachmentClusterMinPixels = encroachmentClusterMinPixels;
    }

    public EncroachmentDetector() {
        this(DEFAULT_NO_CHANGE_PERCENT_MAX, DEFAULT_ENCROACHMENT_PERCENT_MIN, DEFAULT_ENCROACHMENT_CLUSTER_MIN_PIXELS);
    }

    /**
     * Evaluates detected change regions and pixel totals to determine encroachment screening status.
     */
    public EncroachmentResult evaluate(String sampleId, int totalChangedPixels, int totalImagePixels, List<ChangeRegion> regions) {
        double changePercent = totalImagePixels > 0 ? ((double) totalChangedPixels / totalImagePixels) * 100.0 : 0.0;
        int largestCluster = 0;
        for (ChangeRegion r : regions) {
            if (r.getAreaPixels() > largestCluster) {
                largestCluster = r.getAreaPixels();
            }
        }

        EncroachmentResult.Status status;
        double confidence;

        if (largestCluster >= encroachmentClusterMinPixels || changePercent >= encroachmentPercentMin) {
            status = EncroachmentResult.Status.POSSIBLE_ENCROACHMENT;
            // Confidence increases with cluster size and change extent
            confidence = Math.min(0.98, 0.65 + (largestCluster / 1000.0) * 0.25 + (changePercent / 10.0) * 0.10);
            for (ChangeRegion r : regions) {
                if (r.getAreaPixels() >= encroachmentClusterMinPixels) {
                    r.setClassification("POSSIBLE_ENCROACHMENT");
                } else {
                    r.setClassification("MINOR_CHANGE");
                }
            }
        } else if (changePercent >= noChangePercentMax || (!regions.isEmpty() && largestCluster >= 50)) {
            status = EncroachmentResult.Status.POSSIBLE_CHANGE;
            confidence = Math.min(0.85, 0.50 + (changePercent / 2.0) * 0.25);
            for (ChangeRegion r : regions) {
                r.setClassification("POSSIBLE_CHANGE");
            }
        } else {
            status = EncroachmentResult.Status.NO_SIGNIFICANT_CHANGE;
            confidence = Math.max(0.70, 1.0 - (changePercent / 0.10) * 0.30);
            for (ChangeRegion r : regions) {
                r.setClassification("NEGLIGIBLE");
            }
        }

        return new EncroachmentResult(sampleId, status, confidence, totalChangedPixels, totalImagePixels, largestCluster, regions);
    }

    public double getNoChangePercentMax() {
        return noChangePercentMax;
    }

    public double getEncroachmentPercentMin() {
        return encroachmentPercentMin;
    }

    public int getEncroachmentClusterMinPixels() {
        return encroachmentClusterMinPixels;
    }
}
