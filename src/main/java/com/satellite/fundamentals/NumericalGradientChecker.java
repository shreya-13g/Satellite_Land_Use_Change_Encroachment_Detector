package com.satellite.fundamentals;

import java.util.function.DoubleUnaryOperator;

/**
 * Educational numerical gradient checking using central finite differences:
 * f'(x) approx (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon).
 */
public class NumericalGradientChecker {

    public static final double DEFAULT_EPSILON = 1e-5;

    public static class CheckResult {
        public final double x;
        public final double numericalGradient;
        public final double analyticalGradient;
        public final double relativeError;
        public final boolean passed;

        public CheckResult(double x, double numGrad, double anaGrad, double relError, boolean passed) {
            this.x = x;
            this.numericalGradient = numGrad;
            this.analyticalGradient = anaGrad;
            this.relativeError = relError;
            this.passed = passed;
        }
    }

    /**
     * Approximates the derivative of f at point x using the central-difference formula:
     * (f(x + eps) - f(x - eps)) / (2 * eps).
     */
    public static double computeNumericalGradient(DoubleUnaryOperator f, double x, double epsilon) {
        return (f.applyAsDouble(x + epsilon) - f.applyAsDouble(x - epsilon)) / (2.0 * epsilon);
    }

    /**
     * Checks numerical gradient against analytical gradient and verifies relative error.
     */
    public static CheckResult check(DoubleUnaryOperator f, DoubleUnaryOperator analyticalDerivative,
                                    double x, double epsilon, double tolerance) {
        double numGrad = computeNumericalGradient(f, x, epsilon);
        double anaGrad = analyticalDerivative.applyAsDouble(x);

        double numerator = Math.abs(numGrad - anaGrad);
        double denominator = Math.max(Math.max(Math.abs(numGrad), Math.abs(anaGrad)), 1e-7);
        double relError = numerator / denominator;

        return new CheckResult(x, numGrad, anaGrad, relError, relError <= tolerance);
    }
}

