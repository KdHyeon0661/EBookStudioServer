package com.ebookstudio.server.processing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.nio.file.Path;

@Service
public class ProcessingHistorySynchronizer {
    private static final Logger log = LoggerFactory.getLogger(ProcessingHistorySynchronizer.class);
    private final ProcessingRunRepository runs;
    private final AnalysisArtifactCatalogService analysisArtifacts;

    public ProcessingHistorySynchronizer(ProcessingRunRepository runs,
                                         AnalysisArtifactCatalogService analysisArtifacts) {
        this.runs = runs;
        this.analysisArtifacts = analysisArtifacts;
    }

    public void synchronize(QueueJobSnapshot job) {
        ProcessingRun run = runs.findByQueueJobId(job.jobId()).orElse(null);
        if (run == null) {
            log.debug("No PostgreSQL processing history for queue job {}", job.jobId());
            return;
        }

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
                // SQLite remains the live queue; PostgreSQL keeps the durable lifecycle projection.
            }
            case "cancel_requested" -> runs.requestCancellation(run.id(), now);
            case "cancelled" -> {
                if (run.status() == ProcessingRun.Status.RUNNING) {
                    runs.requestCancellation(run.id(), finishedAt);
                }
                runs.markCancelled(run.id(), finishedAt);
            }
            case "done" -> analysisArtifacts.completeAnalysis(run.id(), job.bookRoot(),
                    job.outputJsonFile(), job.coverFile(), finishedAt);
            case "error" -> runs.markFailed(run.id(), "ANALYSIS_WORKER_ERROR",
                    job.errorMessage(), finishedAt);
            default -> log.warn("Unknown queue status '{}' for job {}", job.status(), job.jobId());
        }
    }

    private ProcessingRun reload(long id) {
        return runs.findById(id)
                .orElseThrow(() -> new IllegalStateException("Processing history disappeared"));
    }

    private static boolean requiresStartedState(String status, Long startedAt) {
        return startedAt != null || status.equals("running") || status.equals("done")
                || status.equals("error") || status.equals("cancel_requested");
    }

    public record QueueJobSnapshot(
            String jobId,
            String status,
            int attemptCount,
            Long startedAt,
            Long finishedAt,
            String errorMessage,
            Path bookRoot,
            String outputJsonFile,
            String coverFile) {
    }
}
