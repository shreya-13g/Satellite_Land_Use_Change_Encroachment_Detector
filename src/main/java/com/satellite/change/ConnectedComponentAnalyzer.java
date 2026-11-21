package com.satellite.change;

import java.util.*;

/**
 * Extracts connected components from a binary change mask using 8-connectivity BFS
 * and computes bounding boxes, areas, centroids, and image percentages.
 */
public class ConnectedComponentAnalyzer {

    private final int minRegionArea;

    public ConnectedComponentAnalyzer(int minRegionArea) {
        this.minRegionArea = minRegionArea;
    }

    public ConnectedComponentAnalyzer() {
        this(NoiseRemover.DEFAULT_MIN_REGION_AREA);
    }

    /**
     * Analyzes the binary mask and returns all detected ChangeRegions meeting the minimum area threshold.
     */
    public List<ChangeRegion> findRegions(byte[][] mask) {
        List<ChangeRegion> regions = new ArrayList<>();
        if (mask == null || mask.length == 0 || mask[0].length == 0) {
            return regions;
        }

        int height = mask.length;
        int width = mask[0].length;
        boolean[][] visited = new boolean[height][width];
        int regionCounter = 1;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x] == 1 && !visited[y][x]) {
                    ChangeRegion region = new ChangeRegion(regionCounter);
                    double sumX = 0;
                    double sumY = 0;

                    Queue<int[]> queue = new ArrayDeque<>();
                    queue.add(new int[]{x, y});
                    visited[y][x] = true;

                    while (!queue.isEmpty()) {
                        int[] pt = queue.poll();
                        int px = pt[0];
                        int py = pt[1];

                        region.addPixel(px, py);
                        sumX += px;
                        sumY += py;

                        // 8-connectivity
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0) continue;
                                int nx = px + dx;
                                int ny = py + dy;

                                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                    if (mask[ny][nx] == 1 && !visited[ny][nx]) {
                                        visited[ny][nx] = true;
                                        queue.add(new int[]{nx, ny});
                                    }
                                }
                            }
                        }
                    }

                    if (region.getAreaPixels() >= minRegionArea) {
                        region.finalizeRegion(width, height, sumX, sumY);
                        regions.add(region);
                        regionCounter++;
                    }
                }
            }
        }

        // Sort descending by area (largest clusters first)
        regions.sort((a, b) -> Integer.compare(b.getAreaPixels(), a.getAreaPixels()));
        return regions;
    }

    public int getMinRegionArea() {
        return minRegionArea;
    }
}
