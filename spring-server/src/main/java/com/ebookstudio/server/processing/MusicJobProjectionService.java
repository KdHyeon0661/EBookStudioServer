package com.ebookstudio.server.processing;

import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

@Service
public class MusicJobProjectionService {
    private final JdbcTemplate appJdbc;
    private final ProcessingRunRepository runs;
    private final ProcessingCatalogService processingCatalog;
    private final MusicCatalogImportService musicCatalog;

    public MusicJobProjectionService(JdbcTemplate appJdbc,
                                     ProcessingRunRepository runs,
                                     ProcessingCatalogService processingCatalog,
                                     MusicCatalogImportService musicCatalog) {
        this.appJdbc = appJdbc;
        this.runs = runs;
        this.processingCatalog = processingCatalog;
        this.musicCatalog = musicCatalog;
    }

    public ProcessingRun ensureQueued(String ownerPublicId, String bookFolder,
                                      String jobId, long createdAt, int maxAttempts) {
        ProcessingRun existing = runs.findByQueueJobId(jobId).orElse(null);
        if (existing != null) return existing;
        Long bookId = appJdbc.query("""
                SELECT id FROM books WHERE owner_public_id=? AND folder=?
                """, rs -> rs.next() ? rs.getLong("id") : null, ownerPublicId, bookFolder);
        if (bookId == null) throw new IllegalStateException("Music job book was not found");

        UUID requestId = requestId(jobId);
        try {
            return processingCatalog.createRun(ProcessingRun.queued(bookId, requestId, jobId,
                    ProcessingRun.ProcessType.MUSIC_GENERATION,
                    Math.max(1, maxAttempts), null, createdAt), List.of());
        } catch (DataIntegrityViolationException race) {
            return runs.findByQueueJobId(jobId).orElseThrow(() -> race);
        }
    }

    public void synchronize(MusicQueueJob job) {
        ProcessingRun run = ensureQueued(job.ownerPublicId(), job.bookFolder(), job.jobId(),
                job.createdAt(), job.maxAttempts());
        long now = Instant.now().getEpochSecond();
        long startedAt = job.startedAt() != null ? job.startedAt()
                : job.finishedAt() != null ? job.finishedAt() : now;
        if (requiresStartedState(job.status(), job.startedAt())
                && run.status() == ProcessingRun.Status.QUEUED) {
            runs.markRunning(run.id(), null, startedAt);
            run = reload(run.id());
        }

        int observedAttempts = Math.max(job.attemptCount(),
                run.status() == ProcessingRun.Status.QUEUED ? 0 : 1);
        if (observedAttempts > run.attemptCount()) {
            runs.synchronizeAttemptCount(run.id(), observedAttempts, now);
            run = reload(run.id());
        }

        long finishedAt = job.finishedAt() == null ? now : job.finishedAt();
        switch (job.status()) {
            case "queued", "running" -> {
            }
            case "cancel_requested" -> runs.requestCancellation(run.id(), now);
            case "cancelled" -> {
                if (run.status() == ProcessingRun.Status.RUNNING) {
                    runs.requestCancellation(run.id(), finishedAt);
                }
                runs.markCancelled(run.id(), finishedAt);
            }
            case "done" -> musicCatalog.completeMusicRun(run.id(), run.bookId(),
                    job.jobId(), finishedAt);
            case "error" -> runs.markFailed(run.id(), "MUSIC_WORKER_ERROR",
                    job.errorMessage(), finishedAt);
            default -> throw new IllegalStateException("Unknown music queue status");
        }
    }

    private ProcessingRun reload(long id) {
        return runs.findById(id)
                .orElseThrow(() -> new IllegalStateException("Music processing history disappeared"));
    }

    private static UUID requestId(String jobId) {
        try {
            return UUID.fromString(jobId);
        } catch (IllegalArgumentException legacyId) {
            return UUID.nameUUIDFromBytes(("music-job:" + jobId)
                    .getBytes(StandardCharsets.UTF_8));
        }
    }

    private static boolean requiresStartedState(String status, Long startedAt) {
        return startedAt != null || status.equals("running") || status.equals("done")
                || status.equals("error") || status.equals("cancel_requested");
    }

    public record MusicQueueJob(
            String ownerPublicId,
            String bookFolder,
            String jobId,
            String status,
            long createdAt,
            Long startedAt,
            Long finishedAt,
            String errorMessage,
            int attemptCount,
            int maxAttempts) {
    }
}
