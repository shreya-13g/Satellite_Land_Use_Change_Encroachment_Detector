package com.satellite.visualization;

import com.satellite.change.ChangeRegion;
import com.satellite.detection.EncroachmentResult;
import com.satellite.evaluation.ChangeMetrics;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Generates visual overlays, bounding-box annotations, error heatmaps, and multi-panel comparison boards.
 */
public class ChangeVisualizer {

    private static final Font HEADER_FONT = new Font("SansSerif", Font.BOLD, 15);
    private static final Font SUBHEADER_FONT = new Font("SansSerif", Font.PLAIN, 12);
    private static final Font LABEL_FONT = new Font("SansSerif", Font.BOLD, 11);

    /**
     * Converts a binary mask array to a displayable black-and-white BufferedImage.
     */
    public BufferedImage createMaskImage(byte[][] mask) {
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

    /**
     * Draws bounding boxes, IDs, and centroid crosshairs around detected change regions on top of base image.
     */
    public BufferedImage renderBoundingBoxes(BufferedImage baseImage, List<ChangeRegion> regions, EncroachmentResult.Status status) {
        int w = baseImage.getWidth();
        int h = baseImage.getHeight();
        BufferedImage annotated = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = annotated.createGraphics();
        g.drawImage(baseImage, 0, 0, null);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        Color boxColor;
        if (status == EncroachmentResult.Status.POSSIBLE_ENCROACHMENT) {
            boxColor = new Color(255, 30, 30); // Red
        } else if (status == EncroachmentResult.Status.POSSIBLE_CHANGE) {
            boxColor = new Color(255, 200, 0); // Yellow/Gold
        } else {
            boxColor = new Color(0, 200, 255); // Cyan
        }

        g.setStroke(new BasicStroke(2.0f));
        g.setFont(LABEL_FONT);

        for (ChangeRegion r : regions) {
            g.setColor(boxColor);
            int rx = r.getMinX();
            int ry = r.getMinY();
            int rw = r.getWidth();
            int rh = r.getHeight();

            // Draw bounding box
            g.drawRect(rx, ry, rw, rh);

            // Centroid crosshair
            int cx = (int) Math.round(r.getCentroidX());
            int cy = (int) Math.round(r.getCentroidY());
            g.drawLine(cx - 3, cy, cx + 3, cy);
            g.drawLine(cx, cy - 3, cx, cy + 3);

            // Tag label background & text
            String tag = String.format("#%d (%dpx)", r.getId(), r.getAreaPixels());
            int tagWidth = g.getFontMetrics().stringWidth(tag);
            int tagHeight = g.getFontMetrics().getHeight();

            int labelY = Math.max(tagHeight, ry - 3);
            g.setColor(new Color(0, 0, 0, 180));
            g.fillRect(rx, labelY - tagHeight + 2, tagWidth + 4, tagHeight);

            g.setColor(Color.WHITE);
            g.drawString(tag, rx + 2, labelY - 2);
        }

        g.dispose();
        return annotated;
    }

    /**
     * Renders a color-coded confusion overlay:
     * - Green: True Positive (Correct change detection)
     * - Red: False Positive (False alarm)
     * - Cyan: False Negative (Missed change)
     */
    public BufferedImage renderConfusionOverlay(BufferedImage baseImage, byte[][] predictedMask, byte[][] groundTruthMask) {
        int w = baseImage.getWidth();
        int h = baseImage.getHeight();
        BufferedImage overlay = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int baseRgb = baseImage.getRGB(x, y);
                int pred = (predictedMask[y][x] > 0) ? 1 : 0;
                int gt = (groundTruthMask != null && groundTruthMask[y][x] > 0) ? 1 : 0;

                if (pred == 1 && gt == 1) {
                    // True Positive -> Vibrant Green
                    overlay.setRGB(x, y, blendRGB(baseRgb, 0, 230, 60, 0.70));
                } else if (pred == 1 && gt == 0) {
                    // False Positive -> Red
                    overlay.setRGB(x, y, blendRGB(baseRgb, 240, 30, 30, 0.70));
                } else if (pred == 0 && gt == 1) {
                    // False Negative -> Cyan / Blue
                    overlay.setRGB(x, y, blendRGB(baseRgb, 0, 150, 255, 0.70));
                } else {
                    overlay.setRGB(x, y, baseRgb);
                }
            }
        }
        return overlay;
    }

    /**
     * Generates a comprehensive 5-panel side-by-side diagnostic board:
     * [Image A (T1)] | [Image B (T2)] | [Ground Truth] | [Predicted Mask] | [Regions & Bounding Boxes]
     */
    public BufferedImage renderSideBySideBoard(
            BufferedImage imgA, BufferedImage imgB,
            BufferedImage gtMaskImg, BufferedImage predMaskImg,
            BufferedImage bboxAnnotatedImg,
            EncroachmentResult result) {

        int panelSize = 256;
        int numPanels = 5;
        int bannerHeight = 70;
        int totalWidth = panelSize * numPanels;
        int totalHeight = panelSize + bannerHeight;

        BufferedImage board = new BufferedImage(totalWidth, totalHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = board.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Header Background
        g.setColor(new Color(20, 24, 33));
        g.fillRect(0, 0, totalWidth, totalHeight);

        // Header Banner Information
        g.setFont(HEADER_FONT);
        g.setColor(Color.WHITE);
        String title = String.format("Change & Encroachment Screening: %s", result.getSampleId());
        g.drawString(title, 20, 26);

        // Status badge
        g.setFont(SUBHEADER_FONT);
        Color badgeColor;
        if (result.getStatus() == EncroachmentResult.Status.POSSIBLE_ENCROACHMENT) {
            badgeColor = new Color(255, 60, 60);
        } else if (result.getStatus() == EncroachmentResult.Status.POSSIBLE_CHANGE) {
            badgeColor = new Color(255, 190, 0);
        } else {
            badgeColor = new Color(60, 200, 100);
        }
        g.setColor(badgeColor);
        String statusText = String.format("STATUS: %s (Confidence: %.1f%%)", result.getStatus(), result.getConfidence() * 100.0);
        g.drawString(statusText, 20, 48);

        // Metrics Summary if present
        ChangeMetrics m = result.getMetrics();
        g.setColor(new Color(200, 210, 230));
        String metricsText;
        if (m != null) {
            metricsText = String.format("IoU: %.3f | F1: %.3f | Prec: %.3f | Rec: %.3f | Changed Area: %d px (%.2f%%)",
                    m.getIou(), m.getF1Score(), m.getPrecision(), m.getRecall(), result.getTotalChangedPixels(), result.getChangePercentage());
        } else {
            metricsText = String.format("Changed Pixels: %d (%.2f%%) | Clusters: %d | Largest: %d px",
                    result.getTotalChangedPixels(), result.getChangePercentage(), result.getRegionCount(), result.getLargestRegionPixels());
        }
        g.drawString(metricsText, totalWidth - g.getFontMetrics().stringWidth(metricsText) - 20, 36);

        // Panels
        int yOffset = bannerHeight;
        g.drawImage(imgA, 0, yOffset, panelSize, panelSize, null);
        g.drawImage(imgB, panelSize, yOffset, panelSize, panelSize, null);
        g.drawImage(gtMaskImg, panelSize * 2, yOffset, panelSize, panelSize, null);
        g.drawImage(predMaskImg, panelSize * 3, yOffset, panelSize, panelSize, null);
        g.drawImage(bboxAnnotatedImg, panelSize * 4, yOffset, panelSize, panelSize, null);

        // Vertical divider lines
        g.setColor(new Color(50, 55, 70));
        for (int i = 1; i < numPanels; i++) {
            g.drawLine(panelSize * i, bannerHeight, panelSize * i, totalHeight);
        }

        // Sub-panel Footers
        g.setFont(LABEL_FONT);
        g.setColor(Color.WHITE);
        drawPanelLabel(g, "Time T1: Earlier (A)", 0, panelSize, totalHeight - 8);
        drawPanelLabel(g, "Time T2: Later (B)", panelSize, panelSize, totalHeight - 8);
        drawPanelLabel(g, "Ground Truth Mask", panelSize * 2, panelSize, totalHeight - 8);
        drawPanelLabel(g, "Predicted Change Mask", panelSize * 3, panelSize, totalHeight - 8);
        drawPanelLabel(g, "Encroachment Clusters", panelSize * 4, panelSize, totalHeight - 8);

        g.dispose();
        return board;
    }

    private void drawPanelLabel(Graphics2D g, String text, int startX, int width, int y) {
        int strWidth = g.getFontMetrics().stringWidth(text);
        int x = startX + (width - strWidth) / 2;
        g.setColor(new Color(0, 0, 0, 160));
        g.fillRect(startX + 4, y - 14, width - 8, 18);
        g.setColor(Color.WHITE);
        g.drawString(text, x, y);
    }

    private int blendRGB(int baseRgb, int targetR, int targetG, int targetB, double alpha) {
        int r = (baseRgb >> 16) & 0xFF;
        int g = (baseRgb >> 8) & 0xFF;
        int b = baseRgb & 0xFF;

        int blendedR = Math.min(255, (int) (r * (1.0 - alpha) + targetR * alpha));
        int blendedG = Math.min(255, (int) (g * (1.0 - alpha) + targetG * alpha));
        int blendedB = Math.min(255, (int) (b * (1.0 - alpha) + targetB * alpha));

        return (blendedR << 16) | (blendedG << 8) | blendedB;
    }
}
