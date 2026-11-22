package com.satellite.temporal;

import java.util.Arrays;

/**
 * Educational Temporal Sequence Demonstration for Multitemporal Land-Use Change.
 *
 * Sequence Concept:
 *   Year:                2018   2020   2022   2024   2026
 *   Built-up Area (%):     10     13     17     23     31
 *
 * Implements a lightweight Recurrent Transition Model (RNN / Linear Auto-regressive unit)
 * in pure Java to model temporal dynamics and forecast the next time-step.
 */
public class TemporalLandUseDemo {

    public static class ForecastResult {
        public final int[] years;
        public final double[] observedValues;
        public final int forecastYear;
        public final double forecastValue;
        public final double meanAnnualGrowthRate;
        public final double recurrentHiddenWeight;

        public ForecastResult(int[] years, double[] observedValues, int forecastYear,
                              double forecastValue, double growthRate, double weight) {
            this.years = years;
            this.observedValues = observedValues;
            this.forecastYear = forecastYear;
            this.forecastValue = forecastValue;
            this.meanAnnualGrowthRate = growthRate;
            this.recurrentHiddenWeight = weight;
        }
    }

    /**
     * Executes temporal sequence modeling on a multi-year land-use observation sequence.
     * Uses a single-cell recurrent state transition: h_t = tanh(w_h * h_{t-1} + w_x * x_t + b)
     */
    public static ForecastResult runTemporalDemo(int[] years, double[] builtUpPercentages) {
        if (years.length != builtUpPercentages.length || years.length < 2) {
            throw new IllegalArgumentException("Sequence must contain at least 2 matching time points");
        }

        int n = years.length;
        // Compute average bi-annual percentage growth
        double totalGrowth = 0.0;
        for (int i = 1; i < n; i++) {
            totalGrowth += (builtUpPercentages[i] - builtUpPercentages[i - 1]) / (years[i] - years[i - 1]);
        }
        double annualGrowthRate = totalGrowth / (n - 1);

        // Simple Recurrent Hidden State transition modeling
        double h = 0.0;
        double wH = 0.65; // Recurrent persistence weight
        double wX = 0.35; // New observation input weight

        for (int i = 0; i < n; i++) {
            // Normalized input in [0, 1]
            double normX = builtUpPercentages[i] / 100.0;
            h = Math.tanh(wH * h + wX * normX);
        }

        // Extrapolate next period (e.g. 2028 from 2026)
        int stepYears = years[n - 1] - years[n - 2];
        int forecastYear = years[n - 1] + stepYears;
        double predictedNext = builtUpPercentages[n - 1] + (annualGrowthRate * stepYears) * (1.0 + 0.1 * h);

        return new ForecastResult(years, builtUpPercentages, forecastYear, predictedNext, annualGrowthRate, wH);
    }

    public static void printSummary(ForecastResult res) {
        System.out.println("================================================================================");
        System.out.println("  TEMPORAL SEQUENCE ANALYSIS: MULTITEMPORAL BUILT-UP LAND CHANGE");
        System.out.println("================================================================================");
        for (int i = 0; i < res.years.length; i++) {
            System.out.printf("  Year %d: %.1f%% Built-up%n", res.years[i], res.observedValues[i]);
        }
        System.out.printf("  Mean Annual Expansion Rate: +%.2f%% / year%n", res.meanAnnualGrowthRate);
        System.out.printf("  Recurrent Model Forecast for Year %d: %.1f%% Built-up%n",
                res.forecastYear, res.forecastValue);
        System.out.println("================================================================================");
    }

    public static void main(String[] args) {
        int[] years = {2018, 2020, 2022, 2024, 2026};
        double[] builtUp = {10.0, 13.0, 17.0, 23.0, 31.0};
        ForecastResult res = runTemporalDemo(years, builtUp);
        printSummary(res);
    }
}

