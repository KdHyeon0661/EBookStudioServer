package com.ebookstudio.server.processing;

import java.util.Objects;

public record BookMusicBinding(
        long bookId,
        String segmentKey,
        long musicAssetId,
        Long processingRunId,
        BindingType bindingType,
        long createdAt,
        long updatedAt) {

    public BookMusicBinding {
        if (bookId <= 0) throw new IllegalArgumentException("bookId must be positive");
        if (segmentKey == null || segmentKey.isBlank()) throw new IllegalArgumentException("segmentKey is required");
        if (musicAssetId <= 0) throw new IllegalArgumentException("musicAssetId must be positive");
        Objects.requireNonNull(bindingType, "bindingType");
    }

    public enum BindingType {
        GENERATED,
        CACHE_REUSED,
        DEFAULT
    }
}
