package com.satellite.change;

import java.awt.image.BufferedImage;

/**
 * Calculates absolute and Euclidean radiometric differences between multitemporal satellite rasters.
 */
public class DifferenceCalculator {

    public enum Mode {
        EUCLIDEAN_RGB,
        GRAYSCALE_ABSOLUTE
    }

    private final Mode mode;

    public DifferenceCalculator(Mode mode) {
        this.mode = mode != null ? mode : Mode.EUCLIDEAN_RGB;
    }

    public DifferenceCalculator() {
        this(Mode.EUCLIDEAN_RGB);
    }

    /**
     * Calculates the 2D difference intensity map between two aligned images.
     * Returned values are in the range [0.0, 255.0].
     */
    public float[][] calculateDifference(BufferedImage imgA, BufferedImage imgB) {
        int width = imgA.getWidth();
        int height = imgA.getHeight();
        if (imgB.getWidth() != width || imgB.getHeight() != height) {
            throw new IllegalArgumentException(String.format(
                    "Image dimension mismatch: A(%dx%d) vs B(%dx%d)",
                    width, height, imgB.getWidth(), imgB.getHeight()));
        }

        float[][] diff = new float[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgbA = imgA.getRGB(x, y);
                int rgbB = imgB.getRGB(x, y);

                int rA = (rgbA >> 16) & 0xFF;
                int gA = (rgbA >> 8) & 0xFF;
                int bA = rgbA & 0xFF;

                int rB = (rgbB >> 16) & 0xFF;
                int gB = (rgbB >> 8) & 0xFF;
                int bB = rgbB & 0xFF;

                if (mode == Mode.GRAYSCALE_ABSOLUTE) {
                    double grayA = 0.299 * rA + 0.587 * gA + 0.114 * bA;
                    double grayB = 0.299 * rB + 0.587 * gB + 0.114 * bB;
                    diff[y][x] = (float) Math.abs(grayB - grayA);
                } else {
                    // EUCLIDEAN_RGB
                    double dr = rB - rA;
                    double dg = gB - gA;
                    double db = bB - bA;
                    diff[y][x] = (float) Math.sqrt((dr * dr + dg * dg + db * db) / 3.0);
                }
            }
        }
        return diff;
    }

    /**
     * Converts a difference map into a grayscale visual image.
     */
    public BufferedImage toDifferenceImage(float[][] diffMap) {
        int height = diffMap.length;
        int width = diffMap[0].length;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = Math.min(255, Math.max(0, Math.round(diffMap[y][x])));
                int rgb = (val << 16) | (val << 8) | val;
                img.setRGB(x, y, rgb);
            }
        }
        return img;
    }

    public Mode getMode() {
        return mode;
    }
}
