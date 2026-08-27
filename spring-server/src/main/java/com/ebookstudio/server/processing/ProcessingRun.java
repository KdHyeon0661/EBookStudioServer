package com.ebookstudio.server.processing;

import java.util.Objects;
import java.util.UUID;

public record ProcessingRun(
        Long id,
        long bookId,
        UUID requestId,
        String queueJobId,
        ProcessType processType,
        Status status,
        int attemptCount,
        int maxAttempts,
        String modelVersion,
        Long startedAt,
        Long finishedAt,
        String errorCode,
        String errorMessage,
        long createdAt,
        long updatedAt) {

    public ProcessingRun {
        if (bookId <= 0) throw new IllegalArgumentException("bookId must be positive");
        Objects.requireNonNull(requestId, "requestId");
        Objects.requireNonNull(processType, "processType");
        Objects.requireNonNull(status, "status");
        if (attemptCount < 0 || maxAttempts <= 0 || attemptCount > maxAttempts) {
            throw new IllegalArgumentException("Invalid processing attempt counts");
        }
        if (finishedAt != null && startedAt != null && finishedAt < startedAt) {
            throw new IllegalArgumentException("finishedAt cannot precede startedAt");
        }
    }

    public static ProcessingRun queued(long bookId, UUID requestId, String queueJobId,
                                       ProcessType processType, int maxAttempts,
                                       String modelVersion, long now) {
        return new ProcessingRun(null, bookId, requestId, queueJobId, processType,
                Status.QUEUED, 0, maxAttempts, modelVersion, null, null,
                null, null, now, now);
    }

    public enum ProcessType {
        ANALYZE,
        MUSIC_GENERATION
    }

    public enum Status {
        QUEUED,
        RUNNING,
        CANCEL_REQUESTED,
        CANCELLED,
        SUCCEEDED,
        FAILED
    }
}
