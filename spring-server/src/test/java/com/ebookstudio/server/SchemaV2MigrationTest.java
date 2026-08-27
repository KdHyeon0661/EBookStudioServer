package com.ebookstudio.server;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class SchemaV2MigrationTest {
    private static final Path ROOT = createRoot();

    @Autowired
    JdbcTemplate appJdbc;

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", () ->
                "jdbc:h2:mem:ebookstudio_v2;MODE=PostgreSQL;DB_CLOSE_DELAY=-1;DATABASE_TO_LOWER=TRUE");
        registry.add("spring.datasource.driver-class-name", () -> "org.h2.Driver");
        registry.add("spring.datasource.username", () -> "sa");
        registry.add("spring.datasource.password", () -> "");
        registry.add("ebookstudio.queue-db-path", () -> ROOT.resolve("jobs.db").toString());
        registry.add("ebookstudio.storage-root", () -> ROOT.resolve("storage").toString());
        registry.add("ebookstudio.jwt-secret", () -> "schema-v2-test-secret");
        registry.add("ebookstudio.email-delivery-enabled", () -> "false");
    }

    @Test
    void migrationCreatesRelationalProcessingArtifactAndMusicSchema() {
        assertThat(appJdbc.queryForObject("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema='public'
                  AND table_name IN ('processing_runs', 'book_artifacts',
                                     'music_assets', 'book_music_bindings')
                """, Integer.class)).isEqualTo(4);

        long now = System.currentTimeMillis() / 1000;
        String userPublicId = UUID.randomUUID().toString();
        appJdbc.update("""
                INSERT INTO users(public_id, username, email, password_hash, auth_version)
                VALUES (?, ?, ?, ?, 0)
                """, userPublicId, "schema-user", "schema@example.com", "hash");
        appJdbc.update("""
                INSERT INTO books(owner_public_id, folder, job_id, status, title,
                                  created_at, updated_at)
                VALUES (?, ?, ?, 'PROCESSING', ?, ?, ?)
                """, userPublicId, "schema-book", UUID.randomUUID().toString(),
                "Schema Book", now, now);
        Long bookId = appJdbc.queryForObject(
                "SELECT id FROM books WHERE owner_public_id=? AND folder=?",
                Long.class, userPublicId, "schema-book");

        String requestId = UUID.randomUUID().toString();
        String queueJobId = UUID.randomUUID().toString();
        appJdbc.update("""
                INSERT INTO processing_runs(book_id, request_id, queue_job_id, process_type,
                                            status, attempt_count, max_attempts, created_at, updated_at)
                VALUES (?, ?, ?, 'ANALYZE', 'QUEUED', 0, 3, ?, ?)
                """, bookId, requestId, queueJobId, now, now);
        Long runId = appJdbc.queryForObject(
                "SELECT id FROM processing_runs WHERE request_id=?", Long.class, requestId);

        appJdbc.update("""
                INSERT INTO book_artifacts(processing_run_id, artifact_type, storage_key,
                                           file_name, checksum, file_size, version, created_at)
                VALUES (?, 'BOOK_JSON', ?, 'schema_full.json', ?, 128, 1, ?)
                """, runId, "users/schema/schema-book/schema_full.json", "a".repeat(64), now);

        appJdbc.update("""
                INSERT INTO music_assets(signature, asset_source, prompt, genre, bpm,
                                         model_name, model_version, storage_key, duration_seconds,
                                         status, created_at, updated_at)
                VALUES (?, 'AI_GENERATED', ?, 'ambient', 90, 'facebook/musicgen-small',
                        '1', ?, 30, 'READY', ?, ?)
                """, "b".repeat(64), "calm instrumental background music",
                "defaults/music/schema.wav", now, now);
        Long musicAssetId = appJdbc.queryForObject(
                "SELECT id FROM music_assets WHERE signature=?", Long.class, "b".repeat(64));

        appJdbc.update("""
                INSERT INTO book_music_bindings(book_id, segment_key, music_asset_id,
                                                processing_run_id, binding_type, created_at, updated_at)
                VALUES (?, 'chapter-1:segment-1', ?, ?, 'GENERATED', ?, ?)
                """, bookId, musicAssetId, runId, now, now);

        assertThat(appJdbc.queryForObject("""
                SELECT COUNT(*)
                FROM books b
                JOIN processing_runs r ON r.book_id=b.id
                JOIN book_artifacts a ON a.processing_run_id=r.id
                JOIN book_music_bindings bm ON bm.book_id=b.id
                JOIN music_assets m ON m.id=bm.music_asset_id
                WHERE b.id=?
                """, Integer.class, bookId)).isEqualTo(1);

        assertThatThrownBy(() -> appJdbc.update("""
                INSERT INTO processing_runs(book_id, request_id, process_type, status,
                                            attempt_count, max_attempts, created_at, updated_at)
                VALUES (?, ?, 'ANALYZE', 'QUEUED', 0, 3, ?, ?)
                """, bookId, requestId, now, now))
                .isInstanceOf(DataIntegrityViolationException.class);

        assertThatThrownBy(() -> appJdbc.update("""
                INSERT INTO processing_runs(book_id, request_id, process_type, status,
                                            attempt_count, max_attempts, created_at, updated_at)
                VALUES (?, ?, 'UNKNOWN', 'QUEUED', 0, 3, ?, ?)
                """, bookId, UUID.randomUUID().toString(), now, now))
                .isInstanceOf(DataIntegrityViolationException.class);

        assertThatThrownBy(() -> appJdbc.update(
                "DELETE FROM music_assets WHERE id=?", musicAssetId))
                .isInstanceOf(DataIntegrityViolationException.class);

        appJdbc.update("DELETE FROM books WHERE id=?", bookId);
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM processing_runs WHERE id=?", Integer.class, runId)).isZero();
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM book_artifacts WHERE processing_run_id=?",
                Integer.class, runId)).isZero();
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM book_music_bindings WHERE book_id=?",
                Integer.class, bookId)).isZero();
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM music_assets WHERE id=?",
                Integer.class, musicAssetId)).isEqualTo(1);
    }

    private static Path createRoot() {
        try {
            return Files.createTempDirectory("ebookstudio-schema-v2-test-");
        } catch (Exception error) {
            throw new IllegalStateException(error);
        }
    }
}
