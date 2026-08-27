package com.ebookstudio.server.processing;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class ProcessingCatalogService {
    private final ProcessingRunRepository runs;
    private final BookArtifactRepository artifacts;

    public ProcessingCatalogService(ProcessingRunRepository runs,
                                    BookArtifactRepository artifacts) {
        this.runs = runs;
        this.artifacts = artifacts;
    }

    @Transactional
    public ProcessingRun createRun(ProcessingRun queuedRun, List<ArtifactDraft> initialArtifacts) {
        if (queuedRun.status() != ProcessingRun.Status.QUEUED) {
            throw new IllegalArgumentException("A new processing run must be QUEUED");
        }
        ProcessingRun saved = runs.insert(queuedRun);
        for (ArtifactDraft draft : List.copyOf(initialArtifacts)) {
            artifacts.insert(draft.forRun(saved.id()));
        }
        return saved;
    }

    public record ArtifactDraft(
            BookArtifact.ArtifactType artifactType,
            String storageKey,
            String fileName,
            String checksum,
            Long fileSize,
            int version,
            long createdAt) {

        BookArtifact forRun(long runId) {
            return new BookArtifact(null, runId, artifactType, storageKey,
                    fileName, checksum, fileSize, version, createdAt);
        }
    }
}
