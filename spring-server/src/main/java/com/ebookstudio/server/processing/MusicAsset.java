package com.ebookstudio.server.processing;

import java.util.Objects;

public record MusicAsset(
        Long id,
        String signature,
        AssetSource assetSource,
        String prompt,
        String genre,
        int bpm,
        String keywordsJson,
        String modelName,
        String modelVersion,
        String generationParamsJson,
        String storageKey,
        String checksum,
        Integer durationSeconds,
        Status status,
        int reuseCount,
        long createdAt,
        long updatedAt,
        Long lastUsedAt) {

    public MusicAsset {
        if (signature == null || signature.isBlank()) throw new IllegalArgumentException("signature is required");
        Objects.requireNonNull(assetSource, "assetSource");
        if (prompt == null || prompt.isBlank()) throw new IllegalArgumentException("prompt is required");
        if (genre == null || genre.isBlank()) throw new IllegalArgumentException("genre is required");
        if (bpm < 20 || bpm > 300) throw new IllegalArgumentException("bpm must be between 20 and 300");
        if (keywordsJson == null || keywordsJson.isBlank()) keywordsJson = "[]";
        if (modelName == null || modelName.isBlank()) throw new IllegalArgumentException("modelName is required");
        if (modelVersion == null || modelVersion.isBlank()) throw new IllegalArgumentException("modelVersion is required");
        if (generationParamsJson == null || generationParamsJson.isBlank()) generationParamsJson = "{}";
        Objects.requireNonNull(status, "status");
        if (durationSeconds != null && durationSeconds <= 0) {
            throw new IllegalArgumentException("durationSeconds must be positive");
        }
        if (reuseCount < 0) throw new IllegalArgumentException("reuseCount cannot be negative");
        if (status == Status.READY && (storageKey == null || storageKey.isBlank())) {
            throw new IllegalArgumentException("READY music requires storageKey");
        }
    }

    public static MusicAsset generating(String signature, AssetSource source, String prompt,
                                        String genre, int bpm, String keywordsJson,
                                        String modelName, String modelVersion,
                                        String generationParamsJson, long now) {
        return new MusicAsset(null, signature, source, prompt, genre, bpm, keywordsJson,
                modelName, modelVersion, generationParamsJson, null, null, null,
                Status.GENERATING, 0, now, now, null);
    }

    public enum AssetSource {
        AI_GENERATED,
        DEFAULT
    }

    public enum Status {
        GENERATING,
        READY,
        FAILED
    }
}
