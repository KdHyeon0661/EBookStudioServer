package com.ebookstudio.server;

import com.ebookstudio.server.processing.BookArtifact;
import com.ebookstudio.server.processing.BookArtifactRepository;
import com.ebookstudio.server.processing.BookMusicBinding;
import com.ebookstudio.server.processing.BookMusicBindingRepository;
import com.ebookstudio.server.processing.BookMusicTrack;
import com.ebookstudio.server.processing.MusicAsset;
import com.ebookstudio.server.processing.MusicAssetRepository;
import com.ebookstudio.server.processing.ProcessingCatalogService;
import com.ebookstudio.server.processing.ProcessingRun;
import com.ebookstudio.server.processing.ProcessingRunRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ProcessingRepositoryIntegrationTest {
    private static final Path ROOT = createRoot();

    @Autowired
    JdbcTemplate appJdbc;
    @Autowired
    ProcessingRunRepository runs;
    @Autowired
    BookArtifactRepository artifacts;
    @Autowired
    MusicAssetRepository musicAssets;
    @Autowired
    BookMusicBindingRepository musicBindings;
    @Autowired
    ProcessingCatalogService catalog;

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", () ->
                "jdbc:h2:mem:ebookstudio_processing_repository;MODE=PostgreSQL;"
                        + "DB_CLOSE_DELAY=-1;DATABASE_TO_LOWER=TRUE");
        registry.add("spring.datasource.driver-class-name", () -> "org.h2.Driver");
        registry.add("spring.datasource.username", () -> "sa");
        registry.add("spring.datasource.password", () -> "");
        registry.add("ebookstudio.queue-db-path", () -> ROOT.resolve("jobs.db").toString());
        registry.add("ebookstudio.storage-root", () -> ROOT.resolve("storage").toString());
        registry.add("ebookstudio.jwt-secret", () -> "processing-repository-test-secret");
        registry.add("ebookstudio.email-delivery-enabled", () -> "false");
    }

    @Test
    void storesRunArtifactsAndAppliesGuardedStateTransitions() {
        long bookId = insertBook("repository-flow");
        long now = Instant.now().getEpochSecond() - 10;
        UUID requestId = UUID.randomUUID();
        String queueJobId = UUID.randomUUID().toString();
        ProcessingRun queued = ProcessingRun.queued(bookId, requestId, queueJobId,
                ProcessingRun.ProcessType.ANALYZE, 3, "analyzer-v2", now);

        ProcessingRun saved = catalog.createRun(queued, List.of(
                artifact(BookArtifact.ArtifactType.SOURCE_PDF,
                        "users/repository-flow/source.pdf", "source.pdf", 512L, now)));

        assertThat(saved.id()).isPositive();
        assertThat(runs.findByRequestId(requestId)).contains(saved);
        assertThat(runs.findByQueueJobId(queueJobId)).contains(saved);
        assertThat(artifacts.findByProcessingRunId(saved.id()))
                .extracting(BookArtifact::artifactType)
                .containsExactly(BookArtifact.ArtifactType.SOURCE_PDF);

        assertThat(runs.markRunning(saved.id(), "analyzer-v2.1", now + 1)).isTrue();
        assertThat(runs.markRunning(saved.id(), "duplicate-worker", now + 2)).isFalse();

        artifacts.insert(new BookArtifact(null, saved.id(),
                BookArtifact.ArtifactType.BOOK_JSON, "users/repository-flow/book.json",
                "book.json", null, 2048L, 1, now + 2));
        artifacts.insert(new BookArtifact(null, saved.id(),
                BookArtifact.ArtifactType.COVER_IMAGE, "users/repository-flow/cover.jpg",
                "cover.jpg", null, 1024L, 1, now + 2));
        assertThat(runs.markSucceeded(saved.id(), now + 3)).isTrue();
        ProcessingRun completed = runs.findById(saved.id()).orElseThrow();

        assertThat(completed.status()).isEqualTo(ProcessingRun.Status.SUCCEEDED);
        assertThat(completed.attemptCount()).isEqualTo(1);
        assertThat(completed.modelVersion()).isEqualTo("analyzer-v2.1");
        assertThat(runs.markFailed(saved.id(), "LATE_FAILURE", "too late", now + 3)).isFalse();
        assertThat(artifacts.findByBookId(bookId))
                .extracting(BookArtifact::artifactType)
                .containsExactlyInAnyOrder(BookArtifact.ArtifactType.SOURCE_PDF,
                        BookArtifact.ArtifactType.BOOK_JSON,
                        BookArtifact.ArtifactType.COVER_IMAGE);
    }

    @Test
    void rollsBackRunWhenInitialArtifactSetViolatesUniqueness() {
        long bookId = insertBook("repository-rollback");
        long now = Instant.now().getEpochSecond();
        UUID requestId = UUID.randomUUID();
        ProcessingRun queued = ProcessingRun.queued(bookId, requestId,
                UUID.randomUUID().toString(), ProcessingRun.ProcessType.ANALYZE,
                3, "analyzer-v2", now);
        ProcessingCatalogService.ArtifactDraft duplicate = artifact(
                BookArtifact.ArtifactType.SOURCE_PDF,
                "users/repository-rollback/source.pdf", "source.pdf", 512L, now);

        assertThatThrownBy(() -> catalog.createRun(queued, List.of(duplicate, duplicate)))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThat(runs.findByRequestId(requestId)).isEmpty();
    }

    @Test
    void joinsBookBindingsWithReusableMusicAssets() {
        long bookId = insertBook("repository-music");
        long now = Instant.now().getEpochSecond();
        ProcessingRun run = catalog.createRun(ProcessingRun.queued(bookId, UUID.randomUUID(),
                UUID.randomUUID().toString(), ProcessingRun.ProcessType.MUSIC_GENERATION,
                3, "musicgen-small-v1", now), List.of());

        MusicAsset music = musicAssets.saveSnapshot(MusicAsset.generating(
                "c".repeat(64), MusicAsset.AssetSource.AI_GENERATED,
                "calm fantasy instrumental", "ambient", 88,
                "[\"calm\",\"fantasy\"]", "facebook/musicgen-small", "1.0",
                "{\"duration\":30}", now));
        assertThat(music.storageKey()).isNull();
        assertThat(music.durationSeconds()).isNull();
        assertThat(music.lastUsedAt()).isNull();

        assertThat(musicAssets.markReady(music.id(), "defaults/music/calm.wav",
                "d".repeat(64), 30, now + 1)).isTrue();
        musicBindings.save(new BookMusicBinding(bookId, "chapter-1:segment-1",
                music.id(), run.id(), BookMusicBinding.BindingType.GENERATED,
                now + 1, now + 1));

        List<BookMusicTrack> tracks = musicBindings.findTracksByBookId(bookId);
        assertThat(tracks).hasSize(1);
        assertThat(tracks.get(0).binding().segmentKey()).isEqualTo("chapter-1:segment-1");
        assertThat(tracks.get(0).asset().status()).isEqualTo(MusicAsset.Status.READY);
        assertThat(tracks.get(0).asset().storageKey()).isEqualTo("defaults/music/calm.wav");

        assertThat(musicAssets.recordReuse(music.id(), now + 2)).isTrue();
        musicBindings.save(new BookMusicBinding(bookId, "chapter-1:segment-1",
                music.id(), run.id(), BookMusicBinding.BindingType.CACHE_REUSED,
                now + 1, now + 2));
        BookMusicTrack reused = musicBindings.findTracksByBookId(bookId).get(0);
        assertThat(reused.binding().bindingType())
                .isEqualTo(BookMusicBinding.BindingType.CACHE_REUSED);
        assertThat(reused.binding().createdAt()).isEqualTo(now + 1);
        assertThat(reused.asset().reuseCount()).isEqualTo(1);
        assertThat(reused.asset().lastUsedAt()).isEqualTo(now + 2);
    }

    private long insertBook(String prefix) {
        String suffix = UUID.randomUUID().toString().substring(0, 8);
        String publicId = UUID.randomUUID().toString();
        long now = Instant.now().getEpochSecond();
        appJdbc.update("""
                INSERT INTO users(public_id, username, email, password_hash, auth_version)
                VALUES (?, ?, ?, 'hash', 0)
                """, publicId, prefix + "-" + suffix, prefix + "-" + suffix + "@example.com");
        appJdbc.update("""
                INSERT INTO books(owner_public_id, folder, status, title, created_at, updated_at)
                VALUES (?, ?, 'PROCESSING', ?, ?, ?)
                """, publicId, prefix, prefix, now, now);
        return appJdbc.queryForObject(
                "SELECT id FROM books WHERE owner_public_id=? AND folder=?",
                Long.class, publicId, prefix);
    }

    private static ProcessingCatalogService.ArtifactDraft artifact(
            BookArtifact.ArtifactType type, String storageKey, String fileName,
            Long fileSize, long createdAt) {
        return new ProcessingCatalogService.ArtifactDraft(type, storageKey, fileName,
                null, fileSize, 1, createdAt);
    }

    private static Path createRoot() {
        try {
            return Files.createTempDirectory("ebookstudio-processing-repository-test-");
        } catch (Exception error) {
            throw new IllegalStateException(error);
        }
    }
}
