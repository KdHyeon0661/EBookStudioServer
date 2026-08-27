package com.ebookstudio.server.book;

import com.ebookstudio.server.processing.BookArtifact;
import com.ebookstudio.server.processing.ProcessingCatalogService;
import com.ebookstudio.server.processing.ProcessingRun;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;

@Service
public class BookUploadCatalogService {
    private final JdbcTemplate appJdbc;
    private final ProcessingCatalogService processingCatalog;

    public BookUploadCatalogService(JdbcTemplate appJdbc,
                                    ProcessingCatalogService processingCatalog) {
        this.appJdbc = appJdbc;
        this.processingCatalog = processingCatalog;
    }

    @Transactional
    public RegisteredUpload register(String ownerPublicId, String bookFolder,
                                     String jobId, String sourcePdfFileName,
                                     String sourceStorageKey, String checksum,
                                     long fileSize, long createdAt) {
        appJdbc.update("""
                INSERT INTO books(owner_public_id, folder, job_id, status,
                                  source_pdf, created_at, updated_at)
                VALUES (?, ?, ?, 'PROCESSING', ?, ?, ?)
                """, ownerPublicId, bookFolder, jobId, sourcePdfFileName,
                createdAt, createdAt);
        Long bookId = appJdbc.queryForObject("""
                SELECT id FROM books WHERE owner_public_id=? AND folder=?
                """, Long.class, ownerPublicId, bookFolder);
        if (bookId == null) throw new IllegalStateException("Registered book was not found");

        ProcessingRun run = processingCatalog.createRun(
                ProcessingRun.queued(bookId, UUID.fromString(jobId), jobId,
                        ProcessingRun.ProcessType.ANALYZE, 3, null, createdAt),
                List.of(new ProcessingCatalogService.ArtifactDraft(
                        BookArtifact.ArtifactType.SOURCE_PDF,
                        sourceStorageKey, sourcePdfFileName, checksum,
                        fileSize, 1, createdAt)));
        return new RegisteredUpload(bookId, run.id());
    }

    @Transactional
    public void compensateFailedQueueRegistration(String ownerPublicId, String bookFolder) {
        appJdbc.update("DELETE FROM books WHERE owner_public_id=? AND folder=?",
                ownerPublicId, bookFolder);
    }

    public record RegisteredUpload(long bookId, long processingRunId) {
    }
}
