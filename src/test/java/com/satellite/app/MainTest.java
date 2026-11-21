package com.satellite.app;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit test for Stage 1 project startup and verification.
 */
class MainTest {

    @Test
    @DisplayName("Verify Project Starts Correctly via Main Entry Point")
    void testProjectStartsCorrectly() {
        // Capture standard output to verify required Stage 1 console messages
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(outContent));
            assertDoesNotThrow(() -> Main.main(new String[]{}), "Main.main execution must not throw any exception");
            String output = outContent.toString();
            assertTrue(output.contains("Satellite Land Use Change and Encroachment Detector"),
                    "Output should include project title");
            assertTrue(output.contains("Java Maven project initialized successfully."),
                    "Output should include initialization success message");
        } finally {
            System.setOut(originalOut);
        }
    }

    @Test
    @DisplayName("Verify Project Metadata Constants")
    void testProjectMetadata() {
        assertNotNull(Main.APP_NAME, "Application name must not be null");
        assertEquals("Satellite Land Use Change and Encroachment Detector", Main.APP_NAME);
        assertNotNull(Main.APP_VERSION, "Version must not be null");
    }

    @Test
    @DisplayName("Verify Properties Loading from Classpath")
    void testLoadProperties() {
        Properties props = Main.loadProperties();
        assertNotNull(props, "Properties instance should not be null");
        assertEquals("256", props.getProperty("geospatial.tile.size"));
        assertEquals("10.0", props.getProperty("geospatial.default.gsd.meters"));
    }

    @Test
    @DisplayName("Verify Project Directory Layout")
    void testProjectLayout() {
        boolean valid = Main.verifyProjectLayout();
        assertTrue(valid, "All primary project directories (configs, data, models, docs, etc.) must exist");
    }

    @Test
    @DisplayName("Verify Apache Commons Math Integration")
    void testCommonsMathSanity() {
        double[][] data = {
                {1.0, 2.0},
                {3.0, 4.0}
        };
        RealMatrix matrix = new Array2DRowRealMatrix(data);
        assertEquals(2, matrix.getRowDimension());
        assertEquals(2, matrix.getColumnDimension());
        // [1, 2] * [2, 4]^T = 1*2 + 2*4 = 10.0
        assertEquals(10.0, matrix.multiply(matrix).getEntry(0, 1), 1e-6);
    }
}
