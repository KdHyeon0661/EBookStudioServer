package com.ebookstudio.server.processing;

import java.util.Objects;

public record BookArtifact(
        Long id,
        long processingRunId,
        ArtifactType artifactType,
        String storageKey,
        String fileName,
        String checksum,
        Long fileSize,
        int version,
        long createdAt) {

    public BookArtifact {
        if (processingRunId <= 0) throw new IllegalArgumentException("processingRunId must be positive");
        Objects.requireNonNull(artifactType, "artifactType");
        if (storageKey == null || storageKey.isBlank()) throw new IllegalArgumentException("storageKey is required");
        if (fileName == null || fileName.isBlank()) throw new IllegalArgumentException("fileName is required");
        if (fileSize != null && fileSize < 0) throw new IllegalArgumentException("fileSize cannot be negative");
        if (version <= 0) throw new IllegalArgumentException("version must be positive");
    }

    public enum ArtifactType {
        SOURCE_PDF,
        BOOK_JSON,
        COVER_IMAGE,
        MUSIC_INDEX
    }
}
