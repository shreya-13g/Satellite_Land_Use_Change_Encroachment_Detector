package com.satellite.mlp;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.satellite.tensor.Tensor;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Validates analytical neural network gradients against two-sided numerical finite differences:
 * dL/d_theta_i approx (L(theta + eps) - L(theta - eps)) / (2 * eps).
 */
public class GradientChecker {

    public static final double DEFAULT_EPSILON = 1e-5;
    public static final double DEFAULT_TOLERANCE = 1e-5;

    public static class CheckItem {
        private final String parameterName;
        private final int index;
        private final double numericalGradient;
        private final double analyticalGradient;
        private final double relativeError;
        private final boolean passed;

        public CheckItem(String parameterName, int index, double numericalGradient, double analyticalGradient, double relativeError, boolean passed) {
            this.parameterName = parameterName;
            this.index = index;
            this.numericalGradient = numericalGradient;
            this.analyticalGradient = analyticalGradient;
            this.relativeError = relativeError;
            this.passed = passed;
        }

        public String getParameterName() { return parameterName; }
        public int getIndex() { return index; }
        public double getNumericalGradient() { return numericalGradient; }
        public double getAnalyticalGradient() { return analyticalGradient; }
        public double getRelativeError() { return relativeError; }
        public boolean isPassed() { return passed; }
    }

    public static class CheckSummary {
        private final String testCase;
        private final double maxRelativeError;
        private final double meanRelativeError;
        private final boolean allPassed;
        private final List<CheckItem> items;

        public CheckSummary(String testCase, double maxRelativeError, double meanRelativeError, boolean allPassed, List<CheckItem> items) {
            this.testCase = testCase;
            this.maxRelativeError = maxRelativeError;
            this.meanRelativeError = meanRelativeError;
            this.allPassed = allPassed;
            this.items = items;
        }

        public String getTestCase() { return testCase; }
        public double getMaxRelativeError() { return maxRelativeError; }
        public double getMeanRelativeError() { return meanRelativeError; }
        public boolean isAllPassed() { return allPassed; }
        public List<CheckItem> getItems() { return items; }
    }

    /**
     * Checks numerical vs analytical gradients for a scalar-valued function f(theta).
     */
    public static CheckSummary checkFunction(String name, Tensor theta,
                                             Function<Tensor, Double> lossFunc,
                                             Tensor analyticalGrad,
                                             double epsilon, double tolerance) {
        List<CheckItem> items = new ArrayList<>();
        double[] paramData = theta.getData();
        double[] gradData = analyticalGrad.getData();
        double maxRel = 0.0;
        double sumRel = 0.0;
        boolean allPass = true;

        for (int i = 0; i < paramData.length; i++) {
            double orig = paramData[i];

            // theta + eps
            paramData[i] = orig + epsilon;
            double lossPlus = lossFunc.apply(theta);

            // theta - eps
            paramData[i] = orig - epsilon;
            double lossMinus = lossFunc.apply(theta);

            // Restore original
            paramData[i] = orig;

            double numGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
            double anaGrad = gradData[i];

            double diff = Math.abs(numGrad - anaGrad);
            double denom = Math.max(Math.abs(numGrad) + Math.abs(anaGrad), 1e-8);
            double relError = diff / denom;

            boolean pass = relError <= tolerance;
            if (!pass) allPass = false;

            if (relError > maxRel) maxRel = relError;
            sumRel += relError;

            items.add(new CheckItem(name, i, numGrad, anaGrad, relError, pass));
        }

        double meanRel = sumRel / paramData.length;
        return new CheckSummary(name, maxRel, meanRel, allPass, items);
    }

    /**
     * Checks gradients of all parameters (W1, b1, W2, b2) in a MultilayerPerceptron instance.
     */
    public static CheckSummary checkMLP(MultilayerPerceptron mlp, Tensor x, Tensor y, double epsilon, double tolerance) {
        MultilayerPerceptron.ForwardCache cache = mlp.forward(x);
        MultilayerPerceptron.Gradients analyticalGrads = mlp.backward(cache, y, 0.0);

        List<Tensor> params = mlp.getParameters();
        List<Tensor> grads = analyticalGrads.toList();
        String[] paramNames = {"W1", "b1", "W2", "b2"};

        List<CheckItem> allItems = new ArrayList<>();
        double maxRel = 0.0;
        double sumRel = 0.0;
        boolean allPass = true;

        for (int pIdx = 0; pIdx < params.size(); pIdx++) {
            Tensor p = params.get(pIdx);
            Tensor g = grads.get(pIdx);
            String name = paramNames[pIdx];

            double[] pData = p.getData();
            double[] gData = g.getData();

            for (int i = 0; i < pData.length; i++) {
                double orig = pData[i];

                pData[i] = orig + epsilon;
                double lossPlus = mlp.getLossFunction().compute(mlp.forward(x).a2, y);

                pData[i] = orig - epsilon;
                double lossMinus = mlp.getLossFunction().compute(mlp.forward(x).a2, y);

                pData[i] = orig;

                double numGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
                double anaGrad = gData[i];

                double diff = Math.abs(numGrad - anaGrad);
                double denom = Math.max(Math.abs(numGrad) + Math.abs(anaGrad), 1e-8);
                double relError = diff / denom;

                boolean pass = relError <= tolerance;
                if (!pass) allPass = false;

                if (relError > maxRel) maxRel = relError;
                sumRel += relError;

                allItems.add(new CheckItem(name, i, numGrad, anaGrad, relError, pass));
            }
        }

        double meanRel = sumRel / allItems.size();
        return new CheckSummary("MLP (W1, b1, W2, b2)", maxRel, meanRel, allPass, allItems);
    }

    public static void exportResults(CheckSummary summary, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);

        // 1. JSON Export
        Path jsonPath = outputDir.resolve("gradient-check-results.json");
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(jsonPath.toFile(), summary);

        // 2. CSV Export
        Path csvPath = outputDir.resolve("gradient-check-results.csv");
        try (BufferedWriter w = Files.newBufferedWriter(csvPath)) {
            w.write("parameter,index,numerical_gradient,analytical_gradient,relative_error,passed\n");
            for (CheckItem item : summary.getItems()) {
                w.write(String.format("%s,%d,%.8e,%.8e,%.8e,%b\n",
                        item.getParameterName(), item.getIndex(),
                        item.getNumericalGradient(), item.getAnalyticalGradient(),
                        item.getRelativeError(), item.isPassed()));
            }
        }
    }
}
