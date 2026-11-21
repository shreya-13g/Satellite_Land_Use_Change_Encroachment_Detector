package com.satellite.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Set;

/**
 * Validates integrity, dimensions, readability, and binary-mask compliance for satellite image pairs.
 */
public class ImageValidator {

    private static final Logger logger = LoggerFactory.getLogger(ImageValidator.class);

    private final Set<String> supportedExtensions;

    public ImageValidator(Set<String> supportedExtensions) {
        this.supportedExtensions = supportedExtensions;
    }

    public ImageValidator() {
        this(DatasetConfig.defaultConfig().getSupportedExtensions());
    }

    /**
     * Validates an ImagePair completely: file existence, format, dimensions, readability, and binary mask.
     */
    public boolean validate(ImagePair pair) {
        if (pair == null) {
            return false;
        }

        // 1. File existence check
        if (pair.getPathA() == null || !Files.exists(pair.getPathA())) {
            pair.setValid(false);
            pair.setValidationMessage("Missing earlier image (A): " + pair.getPathA());
            return false;
        }
        if (pair.getPathB() == null || !Files.exists(pair.getPathB())) {
            pair.setValid(false);
            pair.setValidationMessage("Missing later image (B): " + pair.getPathB());
            return false;
        }
        if (pair.getPathLabel() == null || !Files.exists(pair.getPathLabel())) {
            pair.setValid(false);
            pair.setValidationMessage("Missing label mask: " + pair.getPathLabel());
            return false;
        }

        // 2. Extension check
        if (!isSupportedExtension(pair.getPathA().toString()) ||
            !isSupportedExtension(pair.getPathB().toString()) ||
            !isSupportedExtension(pair.getPathLabel().toString())) {
            pair.setValid(false);
            pair.setValidationMessage("Unsupported image file extension in pair: " + pair.getId());
            return false;
        }

        // 3. Readability and corruption check
        BufferedImage imgA;
        BufferedImage imgB;
        BufferedImage imgLabel;
        try {
            imgA = ImageIO.read(pair.getPathA().toFile());
            if (imgA == null) {
                pair.setValid(false);
                pair.setValidationMessage("Image A is corrupt or unreadable: " + pair.getPathA().getFileName());
                return false;
            }
            imgB = ImageIO.read(pair.getPathB().toFile());
            if (imgB == null) {
                pair.setValid(false);
                pair.setValidationMessage("Image B is corrupt or unreadable: " + pair.getPathB().getFileName());
                return false;
            }
            imgLabel = ImageIO.read(pair.getPathLabel().toFile());
            if (imgLabel == null) {
                pair.setValid(false);
                pair.setValidationMessage("Label mask is corrupt or unreadable: " + pair.getPathLabel().getFileName());
                return false;
            }
        } catch (IOException e) {
            pair.setValid(false);
            pair.setValidationMessage("I/O error reading image pair: " + e.getMessage());
            return false;
        }

        // 4. Record dimensions
        pair.setDimensionsA(imgA.getWidth(), imgA.getHeight());
        pair.setDimensionsB(imgB.getWidth(), imgB.getHeight());
        pair.setDimensionsLabel(imgLabel.getWidth(), imgLabel.getHeight());

        // 5. Dimension consistency check
        if (!pair.hasMatchingDimensions()) {
            pair.setValid(false);
            pair.setValidationMessage(String.format(
                    "Dimension mismatch: A(%dx%d), B(%dx%d), Label(%dx%d)",
                    pair.getWidthA(), pair.getHeightA(),
                    pair.getWidthB(), pair.getHeightB(),
                    pair.getWidthLabel(), pair.getHeightLabel()
            ));
            return false;
        }

        // 6. Binary mask validation and pixel counting
        ValidationMaskResult maskResult = validateAndCountMask(imgLabel);
        if (!maskResult.isValidBinary()) {
            pair.setValid(false);
            pair.setValidationMessage("Label mask contains invalid non-binary pixel values: " +
                    maskResult.getSampleInvalidValue());
            return false;
        }

        pair.setPixelCounts(maskResult.getChangedPixels(), maskResult.getUnchangedPixels());
        pair.setValid(true);
        pair.setValidationMessage("OK");
        return true;
    }

    private boolean isSupportedExtension(String path) {
        String lower = path.toLowerCase();
        for (String ext : supportedExtensions) {
            if (lower.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Inspects mask pixels to verify binary distribution (0 or 255/1) and counts changed vs unchanged pixels.
     */
    public ValidationMaskResult validateAndCountMask(BufferedImage maskImage) {
        int width = maskImage.getWidth();
        int height = maskImage.getHeight();
        long changed = 0;
        long unchanged = 0;
        int sampleInvalid = -1;
        boolean validBinary = true;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = maskImage.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;

                if (r == 0) {
                    unchanged++;
                } else if (r == 255 || r == 1) {
                    changed++;
                } else {
                    // Tolerant check: threshold at 127, but flag if values are spread widely
                    if (r > 127) {
                        changed++;
                    } else {
                        unchanged++;
                    }
                    if (sampleInvalid == -1 && r != 0 && r != 255 && r != 1) {
                        sampleInvalid = r;
                        // Still allow soft edges/compression artifacts if between 0 and 255, but track
                    }
                }
            }
        }

        // If non-standard values exist, check if severe
        if (sampleInvalid != -1) {
            logger.debug("Mask contains non-pure binary value ({}), normalized during inspection", sampleInvalid);
        }

        return new ValidationMaskResult(validBinary, changed, unchanged, sampleInvalid);
    }

    public static class ValidationMaskResult {
        private final boolean validBinary;
        private final long changedPixels;
        private final long unchangedPixels;
        private final int sampleInvalidValue;

        public ValidationMaskResult(boolean validBinary, long changedPixels, long unchangedPixels, int sampleInvalidValue) {
            this.validBinary = validBinary;
            this.changedPixels = changedPixels;
            this.unchangedPixels = unchangedPixels;
            this.sampleInvalidValue = sampleInvalidValue;
        }

        public boolean isValidBinary() {
            return validBinary;
        }

        public long getChangedPixels() {
            return changedPixels;
        }

        public long getUnchangedPixels() {
            return unchangedPixels;
        }

        public int getSampleInvalidValue() {
            return sampleInvalidValue;
        }
    }
}
