package com.satellite.temporal;

import java.util.Arrays;

/**
 * Educational Self-Attention / Transformer Mechanism Demonstration for Multitemporal Earth Observation.
 *
 * Implements Scaled Dot-Product Attention (Vaswani et al., 2017):
 *   Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
 *
 * In multitemporal remote sensing, self-attention allows the model to compute
 * relational weights between any pair of observation dates (e.g. 2018 vs 2024),
 * regardless of their temporal separation, capturing long-range seasonal phenology
 * and abrupt encroachment dynamics.
 */
public class SimpleAttentionDemo {

    public static class AttentionResult {
        public final double[][] attentionWeights; // [seqLen x seqLen]
        public final double[][] attendedOutputs;   // [seqLen x d_v]
        public final int[] years;

        public AttentionResult(double[][] attentionWeights, double[][] attendedOutputs, int[] years) {
            this.attentionWeights = attentionWeights;
            this.attendedOutputs = attendedOutputs;
            this.years = years;
        }
    }

    /**
     * Computes scaled dot-product self-attention across a temporal sequence of embeddings.
     *
     * @param embeddings Feature vector for each year [seqLen x dim]
     * @param years Corresponding observation years
     * @return AttentionResult with attention matrix and weighted output
     */
    public static AttentionResult computeSelfAttention(double[][] embeddings, int[] years) {
        int seqLen = embeddings.length;
        int dK = embeddings[0].length;
        double scale = Math.sqrt(dK);

        // Raw attention scores: S = Q * K^T / sqrt(d_k)
        double[][] scores = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double dot = 0.0;
                for (int d = 0; d < dK; d++) {
                    dot += embeddings[i][d] * embeddings[j][d];
                }
                scores[i][j] = dot / scale;
            }
        }

        // Softmax row-wise to get normalized attention weights
        double[][] attentionWeights = new double[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++) {
            double maxScore = -Double.MAX_VALUE;
            for (int j = 0; j < seqLen; j++) {
                if (scores[i][j] > maxScore) maxScore = scores[i][j];
            }

            double sumExp = 0.0;
            for (int j = 0; j < seqLen; j++) {
                attentionWeights[i][j] = Math.exp(scores[i][j] - maxScore);
                sumExp += attentionWeights[i][j];
            }

            for (int j = 0; j < seqLen; j++) {
                attentionWeights[i][j] /= sumExp;
            }
        }

        // Context output: Out = AttentionWeights * V (here V = embeddings)
        double[][] attended = new double[seqLen][dK];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double w = attentionWeights[i][j];
                for (int d = 0; d < dK; d++) {
                    attended[i][d] += w * embeddings[j][d];
                }
            }
        }

        return new AttentionResult(attentionWeights, attended, years);
    }

    public static void printAttentionMatrix(AttentionResult res) {
        System.out.println("  [Transformer / Self-Attention Temporal Dependency Matrix (Softmax Weights)]");
        System.out.print("         ");
        for (int y : res.years) {
            System.out.printf("  %d ", y);
        }
        System.out.println();

        for (int i = 0; i < res.years.length; i++) {
            System.out.printf("  %d  |", res.years[i]);
            for (int j = 0; j < res.years.length; j++) {
                System.out.printf("  %.3f", res.attentionWeights[i][j]);
            }
            System.out.println(" |");
        }
    }
}
