package com.satellite.fundamentals;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Educational 2D and 3D Convolution Visualization Engine.
 *
 * Implements discrete 2D spatial convolution:
 *   (I * K)(y, x) = sum_{i=-k}^{k} sum_{j=-k}^{k} I(y - i, x - j) * K(i, j)
 *
 * And multi-channel 3D convolution concept:
 *   (I *_3D K)(y, x) = sum_c sum_i sum_j I(c, y - i, x - j) * K(c, i, j)
 *
 * Demonstrates:
 * - Sobel Horizontal (dx) & Vertical (dy) gradient filters
 * - Laplacian boundary / edge detection
 * - Gaussian spatial smoothing
 * - Sharpening filter
 * - Non-linear activation (ReLU)
 * - Renders high-resolution diagnostic feature-map sheet using Java AWT Graphics2D
 */
public class ConvolutionDemo {

    // Common standard kernels
    public static final double[][] SOBEL_X = {
            {-1.0, 0.0, 1.0},
            {-2.0, 0.0, 2.0},
            {-1.0, 0.0, 1.0}
    };

    public static final double[][] SOBEL_Y = {
            {-1.0, -2.0, -1.0},
            { 0.0,  0.0,  0.0},
            { 1.0,  2.0,  1.0}
    };

    public static final double[][] LAPLACIAN = {
            { 0.0,  1.0,  0.0},
            { 1.0, -4.0,  1.0},
            { 0.0,  1.0,  0.0}
    };

    public static final double[][] GAUSSIAN_BLUR = {
            {1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0},
            {2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0},
            {1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0}
    };

    public static final double[][] SHARPEN = {
            { 0.0, -1.0,  0.0},
            {-1.0,  5.0, -1.0},
            { 0.0, -1.0,  0.0}
    };

    /**
     * Executes discrete 2D spatial convolution with zero-padding (SAME size output).
     */
    public static double[][] convolve2D(double[][] input, double[][] kernel) {
        int h = input.length;
        int w = input[0].length;
        int kh = kernel.length;
        int kw = kernel[0].length;
        int padY = kh / 2;
        int padX = kw / 2;

        double[][] output = new double[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double sum = 0.0;
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        int iy = y + ky - padY;
                        int ix = x + kx - padX;
                        double val = (iy >= 0 && iy < h && ix >= 0 && ix < w) ? input[iy][ix] : 0.0;
                        sum += val * kernel[ky][kx];
                    }
                }
                output[y][x] = sum;
            }
        }
        return output;
    }

    /**
     * 3D Convolution across multitemporal or multispectral channels (C x H x W).
     */
    public static double[][] convolve3D(double[][][] channels, double[][][] kernel3D) {
        int numChannels = channels.length;
        int h = channels[0].length;
        int w = channels[0][0].length;
        double[][] output = new double[h][w];

        for (int c = 0; c < numChannels; c++) {
            double[][] conv2d = convolve2D(channels[c], kernel3D[c]);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    output[y][x] += conv2d[y][x];
                }
            }
        }
        return output;
    }

    /**
     * Applies Rectified Linear Unit (ReLU) activation: max(0, x).
     */
    public static double[][] applyReLU(double[][] featureMap) {
        int h = featureMap.length;
        int w = featureMap[0].length;
        double[][] activated = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                activated[y][x] = Math.max(0.0, featureMap[y][x]);
            }
        }
        return activated;
    }

    /**
     * Generates a synthetic satellite image patch representing road, building boundary, and forest canopy.
     */
    public static double[][] generateSyntheticPatch(int size) {
        double[][] patch = new double[size][size];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // Background forest texture (low frequency)
                double bg = 0.3 + 0.1 * Math.sin(x * 0.1) * Math.cos(y * 0.1);

                // Road line (vertical stripe)
                if (x >= size / 2 - 3 && x <= size / 2 + 3) {
                    bg = 0.75;
                }

                // Building rectangular structure
                if (x >= size / 5 && x <= size / 5 + size / 3 &&
                    y >= size / 4 && y <= size / 4 + size / 4) {
                    bg = 0.95;
                }

                // Encroachment trench / excavation
                if (y >= size * 3 / 4 && x >= size / 3 && x <= size * 2 / 3) {
                    bg = 0.15;
                }

                patch[y][x] = Math.max(0.0, Math.min(1.0, bg));
            }
        }
        return patch;
    }

    /**
     * Converts a 2D float/double array into a displayable grayscale BufferedImage.
     */
    public static BufferedImage arrayToImage(double[][] arr, boolean normalize) {
        int h = arr.length;
        int w = arr[0].length;
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        double minVal = Double.MAX_VALUE;
        double maxVal = -Double.MAX_VALUE;
        if (normalize) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (arr[y][x] < minVal) minVal = arr[y][x];
                    if (arr[y][x] > maxVal) maxVal = arr[y][x];
                }
            }
        } else {
            minVal = 0.0;
            maxVal = 1.0;
        }

        double range = Math.max(1e-5, maxVal - minVal);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double norm = (arr[y][x] - minVal) / range;
                int gray = (int) Math.round(Math.max(0.0, Math.min(1.0, norm)) * 255);
                int rgb = (gray << 16) | (gray << 8) | gray;
                img.setRGB(x, y, rgb);
            }
        }
        return img;
    }

    /**
     * Renders a comprehensive 6-panel convolution visualization board to PNG.
     */
    public static void renderConvolutionDashboard(Path outputPath) throws IOException {
        int size = 128;
        double[][] input = generateSyntheticPatch(size);

        double[][] sobelX = convolve2D(input, SOBEL_X);
        double[][] sobelY = convolve2D(input, SOBEL_Y);
        double[][] laplace = convolve2D(input, LAPLACIAN);
        double[][] reluLaplace = applyReLU(laplace);
        double[][] blurred = convolve2D(input, GAUSSIAN_BLUR);
        double[][] sharpened = convolve2D(input, SHARPEN);

        // Multi-channel 3D convolution demo (3 channels: Visible Red, NIR, Temporal Difference)
        double[][][] channels = new double[3][size][size];
        channels[0] = input;
        channels[1] = blurred;
        channels[2] = laplace;

        double[][][] kernel3D = new double[3][3][3];
        kernel3D[0] = SOBEL_X;
        kernel3D[1] = SOBEL_Y;
        kernel3D[2] = LAPLACIAN;
        double[][] conv3dOutput = convolve3D(channels, kernel3D);

        int panelW = 160;
        int panelH = 160;
        int cols = 4;
        int rows = 2;
        int headerHeight = 70;
        int totalW = panelW * cols + 50;
        int totalH = panelH * rows + headerHeight + 50;

        BufferedImage board = new BufferedImage(totalW, totalH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = board.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Background
        g.setColor(new Color(24, 27, 36));
        g.fillRect(0, 0, totalW, totalH);

        // Header
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.setColor(Color.WHITE);
        g.drawString("Representative 2D & 3D Convolution Operations in Remote Sensing", 25, 32);

        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g.setColor(new Color(150, 160, 180));
        g.drawString("Visual Demonstration of Spatial Filtering, Gradient Extraction, ReLU Activation & 3D Spatio-Temporal Fusion", 25, 52);

        // Panel definitions
        String[] titles = {
                "Input Satellite Patch",
                "Sobel-X (Vertical Edges)",
                "Sobel-Y (Horizontal Edges)",
                "Laplacian (Boundaries)",
                "Post-Activation (ReLU)",
                "Gaussian Blur (Smoothing)",
                "Sharpen (Boundary Boost)",
                "3D Conv (Multispectral Fusion)"
        };

        BufferedImage[] images = {
                arrayToImage(input, false),
                arrayToImage(sobelX, true),
                arrayToImage(sobelY, true),
                arrayToImage(laplace, true),
                arrayToImage(reluLaplace, true),
                arrayToImage(blurred, false),
                arrayToImage(sharpened, true),
                arrayToImage(conv3dOutput, true)
        };

        for (int i = 0; i < titles.length; i++) {
            int r = i / cols;
            int c = i % cols;
            int px = 25 + c * (panelW + 10);
            int py = headerHeight + r * (panelH + 25);

            // Draw image scaled
            g.drawImage(images[i], px, py, panelW, panelH - 25, null);

            // Border
            g.setColor(new Color(60, 70, 90));
            g.drawRect(px, py, panelW, panelH - 25);

            // Caption
            g.setFont(new Font("SansSerif", Font.BOLD, 10));
            g.setColor(new Color(220, 230, 245));
            g.drawString(titles[i], px + 2, py + panelH - 10);
        }

        g.dispose();
        Files.createDirectories(outputPath.getParent());
        ImageIO.write(board, "PNG", outputPath.toFile());
        System.out.println("  Convolution visualization board rendered to: " + outputPath.toAbsolutePath());
    }

    public static void runDemo() throws IOException {
        System.out.println("================================================================================");
        System.out.println("  CONVOLUTION MATHEMATICS & VISUALIZATION (Requirement 7)");
        System.out.println("  2D Spatial Filtering & 3D Multi-Spectral/Temporal Tensor Convolutions");
        System.out.println("================================================================================");
        Path vizPath = Paths.get("visualizations", "convolution_feature_maps.png");
        renderConvolutionDashboard(vizPath);
        System.out.println("  Generated 8 feature maps showcasing Sobel, Laplacian, ReLU, Blur, Sharpen & 3D Conv.");
        System.out.println("================================================================================");
    }

    public static void main(String[] args) throws IOException {
        runDemo();
    }
}
