package com.satellite.change;

import java.awt.image.BufferedImage;

/**
 * Generates a binary change mask from a continuous pixel difference map using static or adaptive thresholding.
 */
public class ChangeMaskGenerator {

    public static final double DEFAULT_THRESHOLD = 35.0; // on a 0-255 radiometric scale

    private double threshold;

    public ChangeMaskGenerator(double threshold) {
        this.threshold = threshold;
    }

    public ChangeMaskGenerator() {
        this(DEFAULT_THRESHOLD);
    }

    /**
     * Binarizes a difference map into a 2D byte array where:
     * 1 = change detected (pixel difference >= threshold)
     * 0 = no change
     */
    public byte[][] generateMask(float[][] diffMap) {
        return generateMask(diffMap, false);
    }

    /**
     * Binarizes a difference map into a 2D byte array using either static threshold or Otsu adaptive thresholding.
     */
    public byte[][] generateMask(float[][] diffMap, boolean useOtsu) {
        int height = diffMap.length;
        int width = diffMap[0].length;
        byte[][] mask = new byte[height][width];

        double effectiveThreshold = useOtsu ? computeOtsuThreshold(diffMap) : this.threshold;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mask[y][x] = (diffMap[y][x] >= effectiveThreshold) ? (byte) 1 : (byte) 0;
            }
        }
        return mask;
    }

    /**
     * Computes the Otsu optimal threshold from the difference map histogram.
     */
    public static double computeOtsuThreshold(float[][] diffMap) {
        int height = diffMap.length;
        int width = diffMap[0].length;
        int totalPixels = height * width;

        int[] hist = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int bin = Math.min(255, Math.max(0, Math.round(diffMap[y][x])));
                hist[bin]++;
            }
        }

        double sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += i * hist[i];
        }

        double sumB = 0;
        int wB = 0;
        int wF = 0;
        double maxVariance = 0;
        int optimalThreshold = (int) DEFAULT_THRESHOLD;

        for (int t = 0; t < 256; t++) {
            wB += hist[t];
            if (wB == 0) continue;
            wF = totalPixels - wB;
            if (wF == 0) break;

            sumB += (double) (t * hist[t]);
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;

            double betweenVariance = (double) wB * (double) wF * (mB - mF) * (mB - mF);
            if (betweenVariance > maxVariance) {
                maxVariance = betweenVariance;
                optimalThreshold = t;
            }
        }
        return optimalThreshold;
    }

    /**
     * Converts a binary mask array to a displayable BufferedImage.
     * 0 -> black (RGB: 0,0,0)
     * 1 -> white (RGB: 255,255,255)
     */
    public BufferedImage toMaskImage(byte[][] mask) {
        int height = mask.length;
        int width = mask[0].length;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = (mask[y][x] == 1) ? 255 : 0;
                int rgb = (val << 16) | (val << 8) | val;
                img.setRGB(x, y, rgb);
            }
        }
        return img;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }
}
