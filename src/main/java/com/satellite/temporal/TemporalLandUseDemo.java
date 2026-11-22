package com.satellite.temporal;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Multitemporal Sequence Modeling for Land-Use / Built-up Change Analysis.
 *
 * Implements:
 * 1. Deep Recurrent Gated LSTM Network (LSTMCell) with forget/input/output gates
 * 2. Transformer Self-Attention mechanism (SimpleAttentionDemo)
 * 3. Future extrapolation forecasting and CSV experiment export
 */
public class TemporalLandUseDemo {

    public static class ForecastResult {
        public final int[] years;
        public final double[] observedValues;
        public final int forecastYear;
        public final double forecastValue;
        public final double meanAnnualGrowthRate;
        public final LSTMCell.LSTMState[] lstmStates;
        public final SimpleAttentionDemo.AttentionResult attentionResult;

        public ForecastResult(int[] years, double[] observedValues, int forecastYear,
                              double forecastValue, double growthRate,
                              LSTMCell.LSTMState[] lstmStates,
                              SimpleAttentionDemo.AttentionResult attentionResult) {
            this.years = years;
            this.observedValues = observedValues;
            this.forecastYear = forecastYear;
            this.forecastValue = forecastValue;
            this.meanAnnualGrowthRate = growthRate;
            this.lstmStates = lstmStates;
            this.attentionResult = attentionResult;
        }
    }

    /**
     * Executes sequence modeling using both an LSTM Gated Cell and Transformer Self-Attention.
     */
    public static ForecastResult runTemporalDemo(int[] years, double[] builtUpPercentages) {
        if (years.length != builtUpPercentages.length || years.length < 2) {
            throw new IllegalArgumentException("Sequence must contain at least 2 matching time points");
        }

        int n = years.length;
        // Compute average annual percentage growth
        double totalGrowth = 0.0;
        for (int i = 1; i < n; i++) {
            totalGrowth += (builtUpPercentages[i] - builtUpPercentages[i - 1]) / (years[i] - years[i - 1]);
        }
        double annualGrowthRate = totalGrowth / (n - 1);

        // 1. Process temporal observations through LSTM Cell (Hidden Dim = 4)
        int hiddenDim = 4;
        LSTMCell lstm = new LSTMCell(1, hiddenDim, 42L);
        double[][] sequence = new double[n][1];
        for (int i = 0; i < n; i++) {
            sequence[i][0] = builtUpPercentages[i] / 100.0; // Normalized input [0, 1]
        }
        LSTMCell.LSTMState[] states = lstm.processSequence(sequence);

        // 2. Transformer Self-Attention across temporal embeddings
        // Construct 4-dim embedding per time step: [normValue, deltaFromPrev, yearIndexNorm, lstmH0]
        double[][] embeddings = new double[n][4];
        for (int i = 0; i < n; i++) {
            embeddings[i][0] = sequence[i][0];
            embeddings[i][1] = (i == 0) ? 0.0 : (builtUpPercentages[i] - builtUpPercentages[i - 1]) / 100.0;
            embeddings[i][2] = (double) i / n;
            embeddings[i][3] = states[i].h[0];
        }
        SimpleAttentionDemo.AttentionResult attnResult = SimpleAttentionDemo.computeSelfAttention(embeddings, years);

        // 3. Extrapolate next period (e.g. 2028 from 2026) using LSTM final hidden state
        int stepYears = years[n - 1] - years[n - 2];
        int forecastYear = years[n - 1] + stepYears;

        // LSTM hidden gate modulates the momentum of expansion
        double lstmModulation = 1.0 + 0.15 * states[n - 1].h[0];
        double predictedNext = builtUpPercentages[n - 1] + (annualGrowthRate * stepYears) * lstmModulation;

        return new ForecastResult(years, builtUpPercentages, forecastYear, predictedNext, annualGrowthRate, states, attnResult);
    }

    public static void exportCsv(ForecastResult res, Path csvPath) throws IOException {
        Files.createDirectories(csvPath.getParent());
        try (BufferedWriter bw = Files.newBufferedWriter(csvPath)) {
            bw.write("year,observed_builtup_pct,lstm_cell_state_c0,lstm_hidden_state_h0,lstm_forget_gate_f0,lstm_input_gate_i0\n");
            for (int i = 0; i < res.years.length; i++) {
                LSTMCell.LSTMState s = res.lstmStates[i];
                bw.write(String.format("%d,%.2f,%.4f,%.4f,%.4f,%.4f\n",
                        res.years[i], res.observedValues[i], s.c[0], s.h[0], s.f[0], s.i[0]));
            }
            bw.write(String.format("%d,%.2f(forecast),,,,\n", res.forecastYear, res.forecastValue));
        }
    }

    public static void printSummary(ForecastResult res) {
        System.out.println("================================================================================");
        System.out.println("  TEMPORAL SEQUENCE ANALYSIS: MULTITEMPORAL BUILT-UP LAND CHANGE");
        System.out.println("  Architectures: Long Short-Term Memory (LSTM) + Transformer Self-Attention");
        System.out.println("================================================================================");
        for (int i = 0; i < res.years.length; i++) {
            LSTMCell.LSTMState s = res.lstmStates[i];
            System.out.printf("  Year %d: %5.1f%% Built-up | LSTM [h0=%.3f, c0=%.3f, f0=%.3f, i0=%.3f]%n",
                    res.years[i], res.observedValues[i], s.h[0], s.c[0], s.f[0], s.i[0]);
        }
        System.out.printf("  Mean Annual Expansion Rate: +%.2f%% / year%n", res.meanAnnualGrowthRate);
        System.out.printf("  LSTM Gated Forecast for Year %d: %.1f%% Built-up%n",
                res.forecastYear, res.forecastValue);
        System.out.println("--------------------------------------------------------------------------------");
        SimpleAttentionDemo.printAttentionMatrix(res.attentionResult);
        System.out.println("================================================================================");
    }

    public static void main(String[] args) {
        int[] years = {2018, 2020, 2022, 2024, 2026};
        double[] builtUp = {10.0, 13.0, 17.0, 23.0, 31.0};
        ForecastResult res = runTemporalDemo(years, builtUp);
        printSummary(res);

        try {
            Path outPath = Paths.get("experiments", "temporal_forecast.csv");
            exportCsv(res, outPath);
            System.out.println("  Saved temporal forecast results to: " + outPath.toAbsolutePath());
        } catch (Exception e) {
            System.err.println("Failed to export temporal CSV: " + e.getMessage());
        }
    }
}
