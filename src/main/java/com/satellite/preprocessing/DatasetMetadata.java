package com.satellite.preprocessing;

import java.nio.file.Path;
import java.util.*;

/**
 * Stores catalog metadata discovered by scanning the LEVIR-CD dataset directories.
 */
public class DatasetMetadata {

    private final Path datasetRoot;
    private final Map<String, Boolean> zipArchivesDetected = new HashMap<>();
    private final Map<String, Boolean> zipArchivesExtracted = new HashMap<>();
    private final List<String> unextractedArchives = new ArrayList<>();

    private final Map<String, List<ImagePair>> splitMap = new LinkedHashMap<>();
    private final List<ImagePair> allPairs = new ArrayList<>();
    private final List<ImagePair> invalidPairs = new ArrayList<>();
    private final List<String> missingFileErrors = new ArrayList<>();

    public DatasetMetadata(Path datasetRoot) {
        this.datasetRoot = datasetRoot;
    }

    public Path getDatasetRoot() {
        return datasetRoot;
    }

    public void recordArchive(String name, boolean exists, boolean extracted) {
        zipArchivesDetected.put(name, exists);
        zipArchivesExtracted.put(name, extracted);
        if (exists && !extracted) {
            unextractedArchives.add(name);
        }
    }

    public Map<String, Boolean> getZipArchivesDetected() {
        return Collections.unmodifiableMap(zipArchivesDetected);
    }

    public Map<String, Boolean> getZipArchivesExtracted() {
        return Collections.unmodifiableMap(zipArchivesExtracted);
    }

    public List<String> getUnextractedArchives() {
        return Collections.unmodifiableList(unextractedArchives);
    }

    public void addPair(ImagePair pair) {
        allPairs.add(pair);
        splitMap.computeIfAbsent(pair.getSplit().toLowerCase(), k -> new ArrayList<>()).add(pair);
        if (!pair.isValid()) {
            invalidPairs.add(pair);
        }
    }

    public void addMissingFileError(String error) {
        missingFileErrors.add(error);
    }

    public List<ImagePair> getTrainPairs() {
        return Collections.unmodifiableList(splitMap.getOrDefault("train", Collections.emptyList()));
    }

    public List<ImagePair> getValPairs() {
        List<ImagePair> val = splitMap.get("val");
        if (val == null) {
            val = splitMap.getOrDefault("validation", Collections.emptyList());
        }
        return Collections.unmodifiableList(val);
    }

    public List<ImagePair> getTestPairs() {
        return Collections.unmodifiableList(splitMap.getOrDefault("test", Collections.emptyList()));
    }

    public List<ImagePair> getPairsBySplit(String split) {
        return Collections.unmodifiableList(splitMap.getOrDefault(split.toLowerCase(), Collections.emptyList()));
    }

    public List<ImagePair> getAllPairs() {
        return Collections.unmodifiableList(allPairs);
    }

    public List<ImagePair> getValidPairs() {
        List<ImagePair> valid = new ArrayList<>();
        for (ImagePair p : allPairs) {
            if (p.isValid()) {
                valid.add(p);
            }
        }
        return Collections.unmodifiableList(valid);
    }

    public List<ImagePair> getInvalidPairs() {
        return Collections.unmodifiableList(invalidPairs);
    }

    public List<String> getMissingFileErrors() {
        return Collections.unmodifiableList(missingFileErrors);
    }

    public int getTotalPairsCount() {
        return allPairs.size();
    }
}
