package com.satellite.evaluation;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Quantitative pixel-level evaluation metrics comparing predicted change masks against ground-truth masks.
 */
public class ChangeMetrics {

    private static final double EPSILON = 1e-7;

    @JsonProperty("sampleId")
    private String sampleId = "sample";

    @JsonProperty("truePositives")
    private long truePositives;

    @JsonProperty("trueNegatives")
    private long trueNegatives;

    @JsonProperty("falsePositives")
    private long falsePositives;

    @JsonProperty("falseNegatives")
    private long falseNegatives;

    @JsonProperty("totalPixels")
    private long totalPixels;

    @JsonProperty("accuracy")
    private double accuracy;

    @JsonProperty("precision")
    private double precision;

    @JsonProperty("recall")
    private double recall; // Change Recall

    @JsonProperty("f1Score")
    private double f1Score;

    @JsonProperty("iou")
    private double iou;

    @JsonProperty("groundTruthChangedPixels")
    private long groundTruthChangedPixels;

    @JsonProperty("predictedChangedPixels")
    private long predictedChangedPixels;

    @JsonProperty("areaErrorPixels")
    private long areaErrorPixels;

    @JsonProperty("areaErrorPercentage")
    private double areaErrorPercentage;

    @JsonProperty("runtimeMs")
    private long runtimeMs;

    public ChangeMetrics() {
    }

    /**
     * Computes pixel-level confusion matrix and derived accuracy metrics.
     */
    public static ChangeMetrics compute(byte[][] predictedMask, byte[][] groundTruthMask, String sampleId, long runtimeMs) {
        int height = predictedMask.length;
        int width = predictedMask[0].length;
        long tp = 0;
        long tn = 0;
        long fp = 0;
        long fn = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pred = (predictedMask[y][x] > 0) ? 1 : 0;
                int gt = (groundTruthMask[y][x] > 0) ? 1 : 0;

                if (pred == 1 && gt == 1) {
                    tp++;
                } else if (pred == 0 && gt == 0) {
                    tn++;
                } else if (pred == 1 && gt == 0) {
                    fp++;
                } else {
                    fn++;
                }
            }
        }

        ChangeMetrics m = new ChangeMetrics();
        m.sampleId = sampleId;
        m.truePositives = tp;
        m.trueNegatives = tn;
        m.falsePositives = fp;
        m.falseNegatives = fn;
        m.totalPixels = (long) height * width;

        m.groundTruthChangedPixels = tp + fn;
        m.predictedChangedPixels = tp + fp;
        m.areaErrorPixels = Math.abs(m.predictedChangedPixels - m.groundTruthChangedPixels);
        m.areaErrorPercentage = (m.totalPixels > 0) ? ((double) m.areaErrorPixels / m.totalPixels) * 100.0 : 0.0;
        m.runtimeMs = runtimeMs;

        // Metrics calculations
        m.accuracy = (double) (tp + tn) / (m.totalPixels + EPSILON);
        m.precision = (double) tp / (tp + fp + EPSILON);
        m.recall = (double) tp / (tp + fn + EPSILON);
        m.f1Score = (2.0 * m.precision * m.recall) / (m.precision + m.recall + EPSILON);
        m.iou = (double) tp / (tp + fp + fn + EPSILON);

        return m;
    }

    public String getSampleId() {
        return sampleId;
    }

    public long getTruePositives() {
        return truePositives;
    }

    public long getTrueNegatives() {
        return trueNegatives;
    }

    public long getFalsePositives() {
        return falsePositives;
    }

    public long getFalseNegatives() {
        return falseNegatives;
    }

    public long getTotalPixels() {
        return totalPixels;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getF1Score() {
        return f1Score;
    }

    public double getIou() {
        return iou;
    }

    public long getGroundTruthChangedPixels() {
        return groundTruthChangedPixels;
    }

    public long getPredictedChangedPixels() {
        return predictedChangedPixels;
    }

    public long getAreaErrorPixels() {
        return areaErrorPixels;
    }

    public double getAreaErrorPercentage() {
        return areaErrorPercentage;
    }

    public long getRuntimeMs() {
        return runtimeMs;
    }

    @Override
    public String toString() {
        return String.format(
                "ChangeMetrics[%s: IoU=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f, AreaErr=%d px (%.2f%%), TP=%d, FP=%d, FN=%d, Time=%dms]",
                sampleId, iou, f1Score, precision, recall, areaErrorPixels, areaErrorPercentage, truePositives, falsePositives, falseNegatives, runtimeMs);
    }
}
