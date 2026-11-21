package com.satellite.tensor;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * High-performance, pure Java multi-dimensional Tensor engine.
 * Supports 1D vectors and 2D matrices, broadcasting, linear algebra,
 * Hadamard product, matrix multiplication, reductions, and numerical stability.
 */
public class Tensor {

    private final double[] data;
    private final int[] shape;
    private final int[] strides;
    private final int size;

    // ------------------------------------------------------------------------
    // Constructors & Factory Methods
    // ------------------------------------------------------------------------

    public Tensor(int... shape) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Tensor shape must have at least one dimension");
        }
        int total = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Dimension sizes must be strictly positive: " + Arrays.toString(shape));
            }
            total *= dim;
        }
        this.shape = Arrays.copyOf(shape, shape.length);
        this.size = total;
        this.data = new double[total];
        this.strides = computeStrides(this.shape);
    }

    public Tensor(double[] data, int... shape) {
        this(shape);
        if (data.length != this.size) {
            throw new IllegalArgumentException(String.format("Data length (%d) does not match shape %s size (%d)",
                    data.length, Arrays.toString(shape), this.size));
        }
        System.arraycopy(data, 0, this.data, 0, this.size);
    }

    public static Tensor of(double... values) {
        return new Tensor(values, values.length);
    }

    public static Tensor of(double[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("Matrix must be non-empty");
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            if (matrix[i].length != cols) {
                throw new IllegalArgumentException("All matrix rows must have identical length");
            }
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        return new Tensor(flat, rows, cols);
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    public static Tensor fill(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, value);
        return t;
    }

    public static Tensor random(int[] shape, double min, double max, Random rng) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.size; i++) {
            t.data[i] = min + (max - min) * rng.nextDouble();
        }
        return t;
    }

    public static Tensor randn(int[] shape, double mean, double std, Random rng) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.size; i++) {
            t.data[i] = mean + std * rng.nextGaussian();
        }
        return t;
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int s = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    // ------------------------------------------------------------------------
    // Shape & Inspection
    // ------------------------------------------------------------------------

    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public int getRank() {
        return shape.length;
    }

    public int size() {
        return size;
    }

    public int getRows() {
        if (shape.length == 1) return 1;
        return shape[0];
    }

    public int getCols() {
        if (shape.length == 1) return shape[0];
        return shape[1];
    }

    public double[] getData() {
        return data;
    }

    private int getFlatIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(String.format("Expected %d indices, got %d", shape.length, indices.length));
        }
        int flat = 0;
        for (int i = 0; i < indices.length; i++) {
            int idx = indices[i];
            if (idx < 0 || idx >= shape[i]) {
                throw new IndexOutOfBoundsException(String.format("Index %d out of bounds for axis %d with size %d", idx, i, shape[i]));
            }
            flat += idx * strides[i];
        }
        return flat;
    }

    public double get(int... indices) {
        return data[getFlatIndex(indices)];
    }

    public void set(double value, int... indices) {
        data[getFlatIndex(indices)] = value;
    }

    // ------------------------------------------------------------------------
    // Transformations: Reshape, Flatten, Transpose
    // ------------------------------------------------------------------------

    public Tensor reshape(int... newShape) {
        if (newShape == null || newShape.length == 0) {
            throw new IllegalArgumentException("Target shape must have at least one dimension");
        }
        int total = 1;
        int inferIdx = -1;
        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] == -1) {
                if (inferIdx != -1) {
                    throw new IllegalArgumentException("Only one dimension can be inferred (-1)");
                }
                inferIdx = i;
            } else if (newShape[i] <= 0) {
                throw new IllegalArgumentException("Invalid dimension size: " + newShape[i]);
            } else {
                total *= newShape[i];
            }
        }

        int[] resolvedShape = Arrays.copyOf(newShape, newShape.length);
        if (inferIdx != -1) {
            if (size % total != 0) {
                throw new IllegalArgumentException(String.format("Cannot infer dimension: size %d not divisible by %d", size, total));
            }
            resolvedShape[inferIdx] = size / total;
            total *= resolvedShape[inferIdx];
        }

        if (total != size) {
            throw new IllegalArgumentException(String.format("Cannot reshape tensor of size %d to shape %s (size %d)",
                    size, Arrays.toString(resolvedShape), total));
        }

        Tensor result = new Tensor(resolvedShape);
        System.arraycopy(this.data, 0, result.data, 0, this.size);
        return result;
    }

    public Tensor flatten() {
        return reshape(size);
    }

    public Tensor transpose() {
        if (shape.length == 1) {
            // Convert 1D vector of shape [N] to column vector [N, 1]
            Tensor t = new Tensor(shape[0], 1);
            System.arraycopy(data, 0, t.data, 0, size);
            return t;
        } else if (shape.length == 2) {
            int r = shape[0];
            int c = shape[1];
            Tensor t = new Tensor(c, r);
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    t.data[j * r + i] = this.data[i * c + j];
                }
            }
            return t;
        } else {
            throw new UnsupportedOperationException("Transpose currently supports 1D and 2D tensors");
        }
    }

    // ------------------------------------------------------------------------
    // Arithmetic Operations & Broadcasting
    // ------------------------------------------------------------------------

    public Tensor add(Tensor other) {
        return elementWiseBinaryOp(other, Double::sum, "add");
    }

    public Tensor add(double scalar) {
        return apply(x -> x + scalar);
    }

    public Tensor subtract(Tensor other) {
        return elementWiseBinaryOp(other, (a, b) -> a - b, "subtract");
    }

    public Tensor subtract(double scalar) {
        return apply(x -> x - scalar);
    }

    public Tensor multiply(double scalar) {
        return apply(x -> x * scalar);
    }

    public Tensor multiply(Tensor other) {
        return elementWiseBinaryOp(other, (a, b) -> a * b, "multiply");
    }

    public Tensor divide(double scalar) {
        if (scalar == 0.0) {
            throw new ArithmeticException("Division by zero in Tensor.divide(scalar)");
        }
        return apply(x -> x / scalar);
    }

    public Tensor divide(Tensor other) {
        return elementWiseBinaryOp(other, (a, b) -> {
            if (b == 0.0) throw new ArithmeticException("Division by zero in Tensor element-wise division");
            return a / b;
        }, "divide");
    }

    private Tensor elementWiseBinaryOp(Tensor other, DoubleBinaryOperator op, String opName) {
        // Case 1: Identical shapes
        if (Arrays.equals(this.shape, other.shape)) {
            Tensor out = new Tensor(this.shape);
            for (int i = 0; i < this.size; i++) {
                out.data[i] = op.applyAsDouble(this.data[i], other.data[i]);
            }
            return out;
        }

        // Case 2: 2D with 1D broadcasting (Row or Column broadcasting)
        if (this.shape.length == 2 && other.shape.length == 1) {
            int r = this.shape[0];
            int c = this.shape[1];
            if (other.shape[0] == c) {
                // Broadcast 1D vector across each row
                Tensor out = new Tensor(r, c);
                for (int i = 0; i < r; i++) {
                    int offset = i * c;
                    for (int j = 0; j < c; j++) {
                        out.data[offset + j] = op.applyAsDouble(this.data[offset + j], other.data[j]);
                    }
                }
                return out;
            } else if (other.shape[0] == r) {
                // Broadcast 1D vector across each column
                Tensor out = new Tensor(r, c);
                for (int i = 0; i < r; i++) {
                    int offset = i * c;
                    double val = other.data[i];
                    for (int j = 0; j < c; j++) {
                        out.data[offset + j] = op.applyAsDouble(this.data[offset + j], val);
                    }
                }
                return out;
            }
        }

        // Case 3: 2D with 2D broadcasting ([r, c] with [1, c] or [r, 1])
        if (this.shape.length == 2 && other.shape.length == 2) {
            int r1 = this.shape[0], c1 = this.shape[1];
            int r2 = other.shape[0], c2 = other.shape[1];

            if ((r1 == r2 || r2 == 1) && (c1 == c2 || c2 == 1)) {
                int outR = Math.max(r1, r2);
                int outC = Math.max(c1, c2);
                Tensor out = new Tensor(outR, outC);
                for (int i = 0; i < outR; i++) {
                    int i1 = (r1 == 1) ? 0 : i;
                    int i2 = (r2 == 1) ? 0 : i;
                    for (int j = 0; j < outC; j++) {
                        int j1 = (c1 == 1) ? 0 : j;
                        int j2 = (c2 == 1) ? 0 : j;
                        double v1 = this.data[i1 * c1 + j1];
                        double v2 = other.data[i2 * c2 + j2];
                        out.data[i * outC + j] = op.applyAsDouble(v1, v2);
                    }
                }
                return out;
            }
        }

        throw new IllegalArgumentException(String.format("Incompatible shapes for operation '%s': %s and %s",
                opName, Arrays.toString(this.shape), Arrays.toString(other.shape)));
    }

    // ------------------------------------------------------------------------
    // Linear Algebra: Matrix Multiplication & Inner Product
    // ------------------------------------------------------------------------

    /**
     * Standard matrix multiplication: (M x K) * (K x N) -> (M x N).
     */
    public Tensor matmul(Tensor other) {
        if (this.shape.length != 2 || other.shape.length != 2) {
            throw new IllegalArgumentException(String.format("matmul requires 2D tensors, got %s and %s",
                    Arrays.toString(this.shape), Arrays.toString(other.shape)));
        }
        int m = this.shape[0];
        int k1 = this.shape[1];
        int k2 = other.shape[0];
        int n = other.shape[1];

        if (k1 != k2) {
            throw new IllegalArgumentException(String.format("Inner dimensions mismatch for matmul: (%d x %d) * (%d x %d)",
                    m, k1, k2, n));
        }

        Tensor out = new Tensor(m, n);
        // Cache-friendly matrix multiplication
        for (int i = 0; i < m; i++) {
            int aOffset = i * k1;
            int cOffset = i * n;
            for (int k = 0; k < k1; k++) {
                double aVal = this.data[aOffset + k];
                int bOffset = k * n;
                for (int j = 0; j < n; j++) {
                    out.data[cOffset + j] += aVal * other.data[bOffset + j];
                }
            }
        }
        return out;
    }

    /**
     * Inner product (dot product) of two 1D vectors: sum(a_i * b_i).
     */
    public double dot(Tensor other) {
        if (this.shape.length != 1 || other.shape.length != 1) {
            throw new IllegalArgumentException("dot() requires 1D tensors");
        }
        if (this.size != other.size) {
            throw new IllegalArgumentException(String.format("dot() vector size mismatch: %d vs %d", this.size, other.size));
        }
        double sum = 0.0;
        for (int i = 0; i < this.size; i++) {
            sum += this.data[i] * other.data[i];
        }
        return sum;
    }

    // ------------------------------------------------------------------------
    // Reductions: Mean, Sum, Max, Min
    // ------------------------------------------------------------------------

    public double sum() {
        double s = 0.0;
        for (double v : data) s += v;
        return s;
    }

    public double mean() {
        return sum() / size;
    }

    public double max() {
        double m = -Double.MAX_VALUE;
        for (double v : data) {
            if (v > m) m = v;
        }
        return m;
    }

    public double min() {
        double m = Double.MAX_VALUE;
        for (double v : data) {
            if (v < m) m = v;
        }
        return m;
    }

    /**
     * Sum across axis 0 (rows -> result shape [1, cols] or [cols])
     * or axis 1 (cols -> result shape [rows, 1]).
     */
    public Tensor sum(int axis) {
        if (shape.length != 2) {
            throw new UnsupportedOperationException("Axis-wise sum currently supports 2D tensors");
        }
        int r = shape[0];
        int c = shape[1];
        if (axis == 0) {
            Tensor out = new Tensor(1, c);
            for (int i = 0; i < r; i++) {
                int offset = i * c;
                for (int j = 0; j < c; j++) {
                    out.data[j] += this.data[offset + j];
                }
            }
            return out;
        } else if (axis == 1) {
            Tensor out = new Tensor(r, 1);
            for (int i = 0; i < r; i++) {
                int offset = i * c;
                double s = 0.0;
                for (int j = 0; j < c; j++) {
                    s += this.data[offset + j];
                }
                out.data[i] = s;
            }
            return out;
        } else {
            throw new IllegalArgumentException("Invalid axis: " + axis + " for 2D tensor");
        }
    }

    public Tensor mean(int axis) {
        Tensor s = sum(axis);
        int divisor = (axis == 0) ? shape[0] : shape[1];
        return s.divide(divisor);
    }

    // ------------------------------------------------------------------------
    // Element-wise Operators & Numerical Stability
    // ------------------------------------------------------------------------

    public Tensor apply(DoubleUnaryOperator op) {
        Tensor out = new Tensor(this.shape);
        for (int i = 0; i < size; i++) {
            out.data[i] = op.applyAsDouble(this.data[i]);
        }
        return out;
    }

    public Tensor clip(double minVal, double maxVal) {
        return apply(x -> Math.max(minVal, Math.min(maxVal, x)));
    }

    public void checkFinite() {
        for (int i = 0; i < size; i++) {
            if (Double.isNaN(data[i])) {
                throw new ArithmeticException("Tensor contains NaN at element " + i);
            }
            if (Double.isInfinite(data[i])) {
                throw new ArithmeticException("Tensor contains Infinite value at element " + i);
            }
        }
    }

    public Tensor copy() {
        Tensor t = new Tensor(this.shape);
        System.arraycopy(this.data, 0, t.data, 0, this.size);
        return t;
    }

    public double[] toArray1D() {
        return Arrays.copyOf(data, size);
    }

    public double[][] toArray2D() {
        if (shape.length != 2) {
            throw new IllegalStateException("toArray2D requires rank-2 Tensor, current rank: " + shape.length);
        }
        int r = shape[0];
        int c = shape[1];
        double[][] mat = new double[r][c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(data, i * c, mat[i], 0, c);
        }
        return mat;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(shape=").append(Arrays.toString(shape)).append(", data=");
        if (shape.length == 1) {
            sb.append(Arrays.toString(data));
        } else if (shape.length == 2) {
            sb.append("[\n");
            int r = shape[0];
            int c = shape[1];
            for (int i = 0; i < Math.min(r, 6); i++) {
                sb.append("  [");
                for (int j = 0; j < Math.min(c, 8); j++) {
                    sb.append(String.format("%8.4f", data[i * c + j]));
                    if (j < Math.min(c, 8) - 1) sb.append(", ");
                }
                if (c > 8) sb.append(", ...");
                sb.append("]\n");
            }
            if (r > 6) sb.append("  ...\n");
            sb.append("]");
        } else {
            sb.append("[... ").append(size).append(" elements ...]");
        }
        sb.append(")");
        return sb.toString();
    }
}
