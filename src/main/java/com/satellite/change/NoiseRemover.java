package com.satellite.change;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

/**
 * Removes high-frequency radiometric noise, sensor artifacts, and insignificant pixel specks
 * using mathematical morphology (opening/closing) and small-component area filtering.
 */
public class NoiseRemover {

    public static final int DEFAULT_MIN_REGION_AREA = 30; // pixels

    private final int minRegionArea;
    private final boolean applyMorphology;

    public NoiseRemover(int minRegionArea, boolean applyMorphology) {
        this.minRegionArea = minRegionArea;
        this.applyMorphology = applyMorphology;
    }

    public NoiseRemover(int minRegionArea) {
        this(minRegionArea, true);
    }

    public NoiseRemover() {
        this(DEFAULT_MIN_REGION_AREA, true);
    }

    /**
     * Cleans a binary change mask by applying morphological opening/closing and small-region filtering.
     */
    public byte[][] cleanMask(byte[][] rawMask) {
        if (rawMask == null || rawMask.length == 0 || rawMask[0].length == 0) {
            return rawMask;
        }

        byte[][] mask = rawMask;
        if (applyMorphology) {
            // Opening (Erode -> Dilate) removes tiny spurious spikes
            mask = opening(mask);
            // Closing (Dilate -> Erode) fills small internal voids
            mask = closing(mask);
        }

        // Area filtering: eliminate components with area < minRegionArea
        if (minRegionArea > 1) {
            mask = filterSmallRegions(mask, minRegionArea);
        }

        return mask;
    }

    /**
     * Morphological erosion with a 3x3 structuring element.
     */
    public byte[][] erode(byte[][] mask) {
        int h = mask.length;
        int w = mask[0].length;
        byte[][] result = new byte[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mask[y][x] == 1) {
                    boolean allNeighborsOne = true;
                    for (int dy = -1; dy <= 1 && allNeighborsOne; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                                if (mask[ny][nx] == 0) {
                                    allNeighborsOne = false;
                                    break;
                                }
                            }
                        }
                    }
                    result[y][x] = allNeighborsOne ? (byte) 1 : (byte) 0;
                }
            }
        }
        return result;
    }

    /**
     * Morphological dilation with a 3x3 structuring element.
     */
    public byte[][] dilate(byte[][] mask) {
        int h = mask.length;
        int w = mask[0].length;
        byte[][] result = new byte[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (mask[y][x] == 1) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                                result[ny][nx] = 1;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Morphological Opening: Erosion followed by Dilation.
     */
    public byte[][] opening(byte[][] mask) {
        return dilate(erode(mask));
    }

    /**
     * Morphological Closing: Dilation followed by Erosion.
     */
    public byte[][] closing(byte[][] mask) {
        return erode(dilate(mask));
    }

    /**
     * Eliminates connected components whose pixel area is strictly below minArea.
     */
    public byte[][] filterSmallRegions(byte[][] mask, int minArea) {
        int h = mask.length;
        int w = mask[0].length;
        byte[][] result = new byte[h][w];
        for (int y = 0; y < h; y++) {
            System.arraycopy(mask[y], 0, result[y], 0, w);
        }

        boolean[][] visited = new boolean[h][w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (result[y][x] == 1 && !visited[y][x]) {
                    List<int[]> component = new ArrayList<>();
                    Queue<int[]> queue = new ArrayDeque<>();

                    queue.add(new int[]{x, y});
                    visited[y][x] = true;

                    while (!queue.isEmpty()) {
                        int[] pt = queue.poll();
                        component.add(pt);
                        int px = pt[0];
                        int py = pt[1];

                        // 8-neighborhood
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0) continue;
                                int nx = px + dx;
                                int ny = py + dy;
                                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                                    if (result[ny][nx] == 1 && !visited[ny][nx]) {
                                        visited[ny][nx] = true;
                                        queue.add(new int[]{nx, ny});
                                    }
                                }
                            }
                        }
                    }

                    if (component.size() < minArea) {
                        for (int[] pt : component) {
                            result[pt[1]][pt[0]] = 0;
                        }
                    }
                }
            }
        }
        return result;
    }

    public int getMinRegionArea() {
        return minRegionArea;
    }

    public boolean isApplyMorphology() {
        return applyMorphology;
    }
}
