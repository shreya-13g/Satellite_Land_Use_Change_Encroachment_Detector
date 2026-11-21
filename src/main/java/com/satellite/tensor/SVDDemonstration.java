package com.satellite.tensor;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import java.util.ArrayList;
import java.util.List;

/**
 * Demonstrates Singular Value Decomposition (SVD) for dimensionality reduction,
 * feature extraction, and low-rank approximation of geospatial satellite feature matrices.
 * Uses Apache Commons Math for the linear algebra decomposition.
 */
public class SVDDemonstration {

    public static class SVDResult {
        private final double[] singularValues;
        private final double totalEnergy;
        private final List<LowRankApproximation> approximations;

        public SVDResult(double[] singularValues, double totalEnergy, List<LowRankApproximation> approximations) {
            this.singularValues = singularValues;
            this.totalEnergy = totalEnergy;
            this.approximations = approximations;
        }

        public double[] getSingularValues() {
            return singularValues;
        }

        public double getTotalEnergy() {
            return totalEnergy;
        }

        public List<LowRankApproximation> getApproximations() {
            return approximations;
        }
    }

    public static class LowRankApproximation {
        private final int rank;
        private final double energyRetainedRatio;
        private final double frobeniusError;

        public LowRankApproximation(int rank, double energyRetainedRatio, double frobeniusError) {
            this.rank = rank;
            this.energyRetainedRatio = energyRetainedRatio;
            this.frobeniusError = frobeniusError;
        }

        public int getRank() {
            return rank;
        }

        public double getEnergyRetainedRatio() {
            return energyRetainedRatio;
        }

        public double getFrobeniusError() {
            return frobeniusError;
        }
    }

    /**
     * Executes SVD on an input Tensor (2D matrix) and computes rank-k approximations.
     */
    public static SVDResult analyze(Tensor tensor, int[] ranksToEvaluate) {
        if (tensor.getRank() != 2) {
            throw new IllegalArgumentException("SVD requires rank-2 Tensor");
        }

        double[][] raw = tensor.toArray2D();
        RealMatrix matrix = new Array2DRowRealMatrix(raw);
        SingularValueDecomposition svd = new SingularValueDecomposition(matrix);

        double[] s = svd.getSingularValues();
        RealMatrix u = svd.getU();
        RealMatrix v = svd.getV();

        double totalEnergy = 0.0;
        for (double val : s) {
            totalEnergy += val * val;
        }

        List<LowRankApproximation> approxList = new ArrayList<>();
        for (int k : ranksToEvaluate) {
            int actualRank = Math.min(k, s.length);
            double kEnergy = 0.0;
            for (int i = 0; i < actualRank; i++) {
                kEnergy += s[i] * s[i];
            }
            double energyRatio = (totalEnergy > 0) ? (kEnergy / totalEnergy) : 1.0;

            // Reconstruct A_k = U_k * Sigma_k * V_k^T
            RealMatrix recon = new Array2DRowRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
            for (int i = 0; i < actualRank; i++) {
                RealMatrix uCol = u.getColumnMatrix(i);
                RealMatrix vCol = v.getColumnMatrix(i);
                RealMatrix outer = uCol.multiply(vCol.transpose()).scalarMultiply(s[i]);
                recon = recon.add(outer);
            }

            double frobeniusError = matrix.subtract(recon).getFrobeniusNorm();
            approxList.add(new LowRankApproximation(actualRank, energyRatio, frobeniusError));
        }

        return new SVDResult(s, totalEnergy, approxList);
    }
}
