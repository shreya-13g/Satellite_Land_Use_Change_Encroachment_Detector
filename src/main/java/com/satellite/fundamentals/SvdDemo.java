package com.satellite.fundamentals;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

/**
 * Educational demonstration of Singular Value Decomposition (SVD):
 * A = U * S * V^T
 * Shows matrix decomposition and verifies numerical reconstruction.
 */
public class SvdDemo {

    public static class SvdResult {
        public final double[][] uMatrix;
        public final double[] singularValues;
        public final double[][] vMatrix;
        public final double reconstructionFrobeniusError;

        public SvdResult(double[][] uMatrix, double[] singularValues, double[][] vMatrix, double error) {
            this.uMatrix = uMatrix;
            this.singularValues = singularValues;
            this.vMatrix = vMatrix;
            this.reconstructionFrobeniusError = error;
        }
    }

    /**
     * Computes SVD of matrix A: A = U * S * V^T using Apache Commons Math.
     */
    public static SvdResult decompose(double[][] matrixData) {
        RealMatrix a = new Array2DRowRealMatrix(matrixData);
        SingularValueDecomposition svd = new SingularValueDecomposition(a);

        RealMatrix u = svd.getU();
        RealMatrix s = svd.getS();
        RealMatrix v = svd.getV();

        // Reconstruct A_recon = U * S * V^T
        RealMatrix reconstructed = u.multiply(s).multiply(v.transpose());
        double error = a.subtract(reconstructed).getFrobeniusNorm();

        return new SvdResult(
                u.getData(),
                svd.getSingularValues(),
                v.getData(),
                error
        );
    }
}

