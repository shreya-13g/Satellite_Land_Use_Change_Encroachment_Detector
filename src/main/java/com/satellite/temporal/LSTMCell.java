package com.satellite.temporal;

import java.util.Arrays;
import java.util.Random;

/**
 * Pure Java Long Short-Term Memory (LSTM) Cell implementation.
 *
 * Implements the standard Hochreiter & Schmidhuber gating equations:
 *   f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)   (Forget Gate)
 *   i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)   (Input Gate)
 *   c_tilde_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c) (Candidate Cell State)
 *   c_t = f_t * c_{t-1} + i_t * c_tilde_t            (Cell State Update)
 *   o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)   (Output Gate)
 *   h_t = o_t * tanh(c_t)                            (Hidden State Output)
 *
 * Designed for multitemporal satellite sequence modeling (e.g. multi-year built-up expansion).
 */
public class LSTMCell {

    private final int inputDim;
    private final int hiddenDim;

    // Weight matrices: input-to-hidden (W), hidden-to-hidden (U), biases (b)
    // Forget gate
    private final double[][] Wf;
    private final double[][] Uf;
    private final double[] bf;

    // Input gate
    private final double[][] Wi;
    private final double[][] Ui;
    private final double[] bi;

    // Candidate state
    private final double[][] Wc;
    private final double[][] Uc;
    private final double[] bc;

    // Output gate
    private final double[][] Wo;
    private final double[][] Uo;
    private final double[] bo;

    public LSTMCell(int inputDim, int hiddenDim, long seed) {
        this.inputDim = inputDim;
        this.hiddenDim = hiddenDim;
        Random rng = new Random(seed);

        double scale = Math.sqrt(2.0 / (inputDim + hiddenDim));

        this.Wf = initMatrix(hiddenDim, inputDim, scale, rng);
        this.Uf = initMatrix(hiddenDim, hiddenDim, scale, rng);
        this.bf = new double[hiddenDim];
        Arrays.fill(this.bf, 1.0); // Forget gate bias initialized to 1.0 (Jozefowicz et al., 2015)

        this.Wi = initMatrix(hiddenDim, inputDim, scale, rng);
        this.Ui = initMatrix(hiddenDim, hiddenDim, scale, rng);
        this.bi = new double[hiddenDim];

        this.Wc = initMatrix(hiddenDim, inputDim, scale, rng);
        this.Uc = initMatrix(hiddenDim, hiddenDim, scale, rng);
        this.bc = new double[hiddenDim];

        this.Wo = initMatrix(hiddenDim, inputDim, scale, rng);
        this.Uo = initMatrix(hiddenDim, hiddenDim, scale, rng);
        this.bo = new double[hiddenDim];
    }

    public static class LSTMState {
        public final double[] h; // Hidden state
        public final double[] c; // Cell state
        public final double[] f; // Forget gate activation
        public final double[] i; // Input gate activation
        public final double[] o; // Output gate activation

        public LSTMState(double[] h, double[] c, double[] f, double[] i, double[] o) {
            this.h = h;
            this.c = c;
            this.f = f;
            this.i = i;
            this.o = o;
        }

        public static LSTMState zero(int hiddenDim) {
            return new LSTMState(new double[hiddenDim], new double[hiddenDim],
                    new double[hiddenDim], new double[hiddenDim], new double[hiddenDim]);
        }
    }

    /**
     * Executes one time-step forward pass through the LSTM cell.
     *
     * @param x Input vector at time t [inputDim]
     * @param prev Previous LSTMState at time t-1
     * @return New LSTMState at time t
     */
    public LSTMState step(double[] x, LSTMState prev) {
        if (x.length != inputDim) {
            throw new IllegalArgumentException("Expected input dimension " + inputDim + ", got " + x.length);
        }

        double[] f = new double[hiddenDim];
        double[] i = new double[hiddenDim];
        double[] cTilde = new double[hiddenDim];
        double[] o = new double[hiddenDim];
        double[] c = new double[hiddenDim];
        double[] h = new double[hiddenDim];

        for (int k = 0; k < hiddenDim; k++) {
            // Forget gate: f_t = sigmoid(W_f * x + U_f * h_{t-1} + b_f)
            double actF = bf[k];
            for (int d = 0; d < inputDim; d++) actF += Wf[k][d] * x[d];
            for (int d = 0; d < hiddenDim; d++) actF += Uf[k][d] * prev.h[d];
            f[k] = sigmoid(actF);

            // Input gate: i_t = sigmoid(W_i * x + U_i * h_{t-1} + b_i)
            double actI = bi[k];
            for (int d = 0; d < inputDim; d++) actI += Wi[k][d] * x[d];
            for (int d = 0; d < hiddenDim; d++) actI += Ui[k][d] * prev.h[d];
            i[k] = sigmoid(actI);

            // Candidate cell state: c_tilde = tanh(W_c * x + U_c * h_{t-1} + b_c)
            double actC = bc[k];
            for (int d = 0; d < inputDim; d++) actC += Wc[k][d] * x[d];
            for (int d = 0; d < hiddenDim; d++) actC += Uc[k][d] * prev.h[d];
            cTilde[k] = Math.tanh(actC);

            // Cell state update: c_t = f_t * c_{t-1} + i_t * c_tilde
            c[k] = f[k] * prev.c[k] + i[k] * cTilde[k];

            // Output gate: o_t = sigmoid(W_o * x + U_o * h_{t-1} + b_o)
            double actO = bo[k];
            for (int d = 0; d < inputDim; d++) actO += Wo[k][d] * x[d];
            for (int d = 0; d < hiddenDim; d++) actO += Uo[k][d] * prev.h[d];
            o[k] = sigmoid(actO);

            // Hidden state: h_t = o_t * tanh(c_t)
            h[k] = o[k] * Math.tanh(c[k]);
        }

        return new LSTMState(h, c, f, i, o);
    }

    /**
     * Processes a full multitemporal sequence of observations through the LSTM.
     *
     * @param sequence Array of shape [sequenceLength][inputDim]
     * @return Array of LSTMState for each time step
     */
    public LSTMState[] processSequence(double[][] sequence) {
        int seqLen = sequence.length;
        LSTMState[] states = new LSTMState[seqLen];
        LSTMState curr = LSTMState.zero(hiddenDim);

        for (int t = 0; t < seqLen; t++) {
            curr = step(sequence[t], curr);
            states[t] = curr;
        }

        return states;
    }

    public int getInputDim() { return inputDim; }
    public int getHiddenDim() { return hiddenDim; }

    private static double sigmoid(double z) {
        if (z >= 45.0) return 1.0;
        if (z <= -45.0) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static double[][] initMatrix(int rows, int cols, double scale, Random rng) {
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m[r][c] = rng.nextGaussian() * scale;
            }
        }
        return m;
    }
}
