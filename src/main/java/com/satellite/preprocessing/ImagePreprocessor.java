package com.satellite.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles image loading, normalization (0.0-1.0), binary mask preservation,
 * 256x256 tiling, tile exporting, and visual comparisons.
 */
public class ImagePreprocessor {

    private static final Logger logger = LoggerFactory.getLogger(ImagePreprocessor.class);

    private final int tileWidth;
    private final int tileHeight;
    private final int tileStride;

    public ImagePreprocessor(int tileWidth, int tileHeight, int tileStride) {
        this.tileWidth = tileWidth;
        this.tileHeight = tileHeight;
        this.tileStride = tileStride;
    }

    public ImagePreprocessor() {
        this(256, 256, 256);
    }

    /**
     * Loads a BufferedImage from file.
     */
    public BufferedImage loadImage(Path path) throws IOException {
        BufferedImage img = ImageIO.read(path.toFile());
        if (img == null) {
            throw new IOException("Failed to decode image from path: " + path);
        }
        return img;
    }

    /**
     * Normalizes an RGB BufferedImage into a 3D float array with values scaled to [0.0, 1.0].
     * Dimensions: [height][width][3] (channels: R, G, B)
     */
    public float[][][] normalizeRGB(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[][][] normalized = new float[height][width][3];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                normalized[y][x][0] = ((rgb >> 16) & 0xFF) / 255.0f; // Red
                normalized[y][x][1] = ((rgb >> 8) & 0xFF) / 255.0f;  // Green
                normalized[y][x][2] = (rgb & 0xFF) / 255.0f;         // Blue
            }
        }
        return normalized;
    }

    /**
     * Extracts a binary mask as a 2D byte array with values 0 (unchanged) and 1 (changed).
     */
    public byte[][] extractBinaryMask(BufferedImage maskImage) {
        int width = maskImage.getWidth();
        int height = maskImage.getHeight();
        byte[][] mask = new byte[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = maskImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                mask[y][x] = (r > 127 || r == 1) ? (byte) 1 : (byte) 0;
            }
        }
        return mask;
    }

    /**
     * Resizes an image to specified width and height using high-quality bilinear interpolation.
     */
    public BufferedImage resizeImage(BufferedImage original, int targetWidth, int targetHeight) {
        if (original.getWidth() == targetWidth && original.getHeight() == targetHeight) {
            return original;
        }
        int type = original.getType() == 0 ? BufferedImage.TYPE_INT_RGB : original.getType();
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, type);
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.drawImage(original, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();
        return resized;
    }

    /**
     * Splits an ImagePair into matching 256x256 tiles and saves them under outputRoot.
     * Returns the total count of tiles generated for this pair.
     */
    public int processAndSaveTiles(ImagePair pair, Path outputRoot) throws IOException {
        BufferedImage imgA = loadImage(pair.getPathA());
        BufferedImage imgB = loadImage(pair.getPathB());
        BufferedImage imgLabel = loadImage(pair.getPathLabel());

        Path splitDir = outputRoot.resolve(pair.getSplit());
        Path dirA = splitDir.resolve("A");
        Path dirB = splitDir.resolve("B");
        Path dirLabel = splitDir.resolve("label");

        Files.createDirectories(dirA);
        Files.createDirectories(dirB);
        Files.createDirectories(dirLabel);

        int width = imgA.getWidth();
        int height = imgA.getHeight();
        int tileCount = 0;

        for (int y = 0; y + tileHeight <= height; y += tileStride) {
            for (int x = 0; x + tileWidth <= width; x += tileStride) {
                BufferedImage tileA = imgA.getSubimage(x, y, tileWidth, tileHeight);
                BufferedImage tileB = imgB.getSubimage(x, y, tileWidth, tileHeight);
                BufferedImage rawTileLabel = imgLabel.getSubimage(x, y, tileWidth, tileHeight);

                // Ensure tile label is strictly binary (0 or 255)
                BufferedImage tileLabel = toBinaryMaskImage(rawTileLabel);

                String tileName = String.format("%s_tile_y%04d_x%04d.png", pair.getId(), y, x);

                ImageIO.write(tileA, "PNG", dirA.resolve(tileName).toFile());
                ImageIO.write(tileB, "PNG", dirB.resolve(tileName).toFile());
                ImageIO.write(tileLabel, "PNG", dirLabel.resolve(tileName).toFile());

                tileCount++;
            }
        }
        return tileCount;
    }

    /**
     * Converts any mask tile into a clean 1-channel or RGB binary mask image (0 = black, 255 = white).
     */
    public BufferedImage toBinaryMaskImage(BufferedImage rawMask) {
        int w = rawMask.getWidth();
        int h = rawMask.getHeight();
        BufferedImage binaryImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = rawMask.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int val = (r > 127 || r == 1) ? 255 : 0;
                int grayRgb = (val << 16) | (val << 8) | val;
                binaryImg.setRGB(x, y, grayRgb);
            }
        }
        return binaryImg;
    }

    /**
     * Generates a side-by-side composite visualization showing:
     * [Image A (T1)] | [Image B (T2)] | [Ground Truth Mask] | [Change Overlay on B]
     */
    public BufferedImage createComparisonVisualization(ImagePair pair) throws IOException {
        BufferedImage imgA = loadImage(pair.getPathA());
        BufferedImage imgB = loadImage(pair.getPathB());
        BufferedImage imgLabel = loadImage(pair.getPathLabel());

        int w = imgA.getWidth();
        int h = imgA.getHeight();
        int panelWidth = w;
        int panelHeight = h;

        // Downsample for large 1024x1024 panels if needed, or keep 512 for clear display
        int displaySize = 512;
        BufferedImage dispA = resizeImage(imgA, displaySize, displaySize);
        BufferedImage dispB = resizeImage(imgB, displaySize, displaySize);
        BufferedImage dispLabel = toBinaryMaskImage(resizeImage(imgLabel, displaySize, displaySize));

        // Create overlay of change (translucent red) over image B
        BufferedImage overlay = createOverlayImage(dispB, dispLabel);

        int totalWidth = displaySize * 4;
        int bannerHeight = 50;
        int totalHeight = displaySize + bannerHeight;

        BufferedImage composite = new BufferedImage(totalWidth, totalHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = composite.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Fill background
        g.setColor(new Color(24, 28, 36));
        g.fillRect(0, 0, totalWidth, totalHeight);

        // Draw sub-images
        g.drawImage(dispA, 0, bannerHeight, null);
        g.drawImage(dispB, displaySize, bannerHeight, null);
        g.drawImage(dispLabel, displaySize * 2, bannerHeight, null);
        g.drawImage(overlay, displaySize * 3, bannerHeight, null);

        // Draw divider lines
        g.setColor(new Color(60, 65, 80));
        for (int i = 1; i <= 3; i++) {
            g.drawLine(displaySize * i, 0, displaySize * i, totalHeight);
        }

        // Draw Headers
        g.setFont(new Font("SansSerif", Font.BOLD, 16));
        g.setColor(Color.WHITE);

        drawCenteredString(g, "Time T1: Earlier Image (A)", 0, displaySize, 30);
        drawCenteredString(g, "Time T2: Later Image (B)", displaySize, displaySize, 30);
        drawCenteredString(g, "Ground Truth Change Mask", displaySize * 2, displaySize, 30);
        drawCenteredString(g, "Encroachment Overlay (Red on B)", displaySize * 3, displaySize, 30);

        g.dispose();
        return composite;
    }

    /**
     * Creates a translucent red overlay where mask == 255 onto base image.
     */
    public BufferedImage createOverlayImage(BufferedImage baseImage, BufferedImage mask) {
        int w = baseImage.getWidth();
        int h = baseImage.getHeight();
        BufferedImage overlay = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        Color redOverlay = new Color(255, 30, 30, 180);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int baseRgb = baseImage.getRGB(x, y);
                int maskRgb = mask.getRGB(x, y);
                int maskVal = (maskRgb >> 16) & 0xFF;

                if (maskVal > 127) {
                    // Blend base with red
                    int r = (baseRgb >> 16) & 0xFF;
                    int g = (baseRgb >> 8) & 0xFF;
                    int b = baseRgb & 0xFF;

                    int blendedR = Math.min(255, (int) (r * 0.3 + redOverlay.getRed() * 0.7));
                    int blendedG = (int) (g * 0.3);
                    int blendedB = (int) (b * 0.3);

                    overlay.setRGB(x, y, (blendedR << 16) | (blendedG << 8) | blendedB);
                } else {
                    overlay.setRGB(x, y, baseRgb);
                }
            }
        }
        return overlay;
    }

    private void drawCenteredString(Graphics2D g, String text, int startX, int width, int y) {
        int strWidth = g.getFontMetrics().stringWidth(text);
        int x = startX + (width - strWidth) / 2;
        g.drawString(text, x, y);
    }
}
