package com.satellite.fundamentals;

import java.util.Arrays;

/**
 * Educational N-dimensional Tensor for learning neural-network linear algebra.
 * Uses plain Java arrays with clear loops and short methods.
 */
public class SimpleTensor {

    private final double[] data;
    private final int[] shape;

    public SimpleTensor(double[] data, int... shape) {
        int total = 1;
        for (int s : shape) total *= s;
        if (data.length != total) {
            throw new IllegalArgumentException("Data length does not match shape total: " + data.length + " vs " + total);
        }
        this.data = Arrays.copyOf(data, data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
    }

    public SimpleTensor(int... shape) {
        int total = 1;
        for (int s : shape) total *= s;
        this.data = new double[total];
        this.shape = Arrays.copyOf(shape, shape.length);
    }

    public static SimpleTensor of(double... values) {
        return new SimpleTensor(values, values.length);
    }

    public static SimpleTensor of2D(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        return new SimpleTensor(flat, rows, cols);
    }

    public static SimpleTensor zeros(int... shape) {
        return new SimpleTensor(shape);
    }

    public static SimpleTensor ones(int... shape) {
        SimpleTensor t = new SimpleTensor(shape);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public double[] getData() {
        return Arrays.copyOf(data, data.length);
    }

    public double get(int r, int c) {
        return data[r * shape[1] + c];
    }

    public void set(int r, int c, double val) {
        data[r * shape[1] + c] = val;
    }

    /**
     * Reshape tensor into new dimensions with identical total elements.
     */
    public SimpleTensor reshape(int... newShape) {
        int total = 1;
        for (int s : newShape) total *= s;
        if (total != this.data.length) {
            throw new IllegalArgumentException("Reshape total mismatch");
        }
        return new SimpleTensor(this.data, newShape);
    }

    /**
     * Flatten tensor into a 1D vector.
     */
    public SimpleTensor flatten() {
        return new SimpleTensor(this.data, this.data.length);
    }

    /**
     * 2D Matrix transpose: A[i][j] -> A[j][i].
     */
    public SimpleTensor transpose() {
        if (shape.length != 2) {
            throw new IllegalStateException("Transpose requires a 2D tensor");
        }
        int rows = shape[0];
        int cols = shape[1];
        double[] transposed = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        return new SimpleTensor(transposed, cols, rows);
    }

    /**
     * Element-wise addition of two tensors with identical shapes.
     */
    public SimpleTensor add(SimpleTensor other) {
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Shapes must match for addition");
        }
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] + other.data[i];
        }
        return new SimpleTensor(result, this.shape);
    }

    /**
     * Multiply all elements by a scalar constant.
     */
    public SimpleTensor scalarMultiply(double scalar) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] * scalar;
        }
        return new SimpleTensor(result, this.shape);
    }

    /**
     * 2D Matrix multiplication: C[i][j] = sum_k A[i][k] * B[k][j].
     */
    public SimpleTensor matmul(SimpleTensor other) {
        if (this.shape.length != 2 || other.shape.length != 2) {
            throw new IllegalArgumentException("Matmul requires two 2D tensors");
        }
        int rA = this.shape[0];
        int cA = this.shape[1];
        int rB = other.shape[0];
        int cB = other.shape[1];
        if (cA != rB) {
            throw new IllegalArgumentException("Incompatible inner dimensions: " + cA + " vs " + rB);
        }

        double[] out = new double[rA * cB];
        for (int i = 0; i < rA; i++) {
            for (int k = 0; k < cA; k++) {
                double aVal = this.data[i * cA + k];
                for (int j = 0; j < cB; j++) {
                    out[i * cB + j] += aVal * other.data[k * cB + j];
                }
            }
        }
        return new SimpleTensor(out, rA, cB);
    }

    /**
     * Vector inner product (dot product): sum_i a_i * b_i.
     */
    public double innerProduct(SimpleTensor other) {
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException("Vectors must have identical length");
        }
        double sum = 0.0;
        for (int i = 0; i < this.data.length; i++) {
            sum += this.data[i] * other.data[i];
        }
        return sum;
    }

    /**
     * Demonstrates simple row-wise broadcasting: adding a 1D vector (1 x C) to a 2D matrix (R x C).
     */
    public SimpleTensor broadcastAddRow(SimpleTensor rowVector) {
        if (this.shape.length != 2 || rowVector.data.length != this.shape[1]) {
            throw new IllegalArgumentException("rowVector length must match matrix columns");
        }
        int rows = this.shape[0];
        int cols = this.shape[1];
        double[] out = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                out[r * cols + c] = this.data[r * cols + c] + rowVector.data[c];
            }
        }
        return new SimpleTensor(out, rows, cols);
    }
}

