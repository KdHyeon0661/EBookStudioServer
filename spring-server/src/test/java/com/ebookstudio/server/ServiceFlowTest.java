package com.ebookstudio.server;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import tools.jackson.databind.ObjectMapper;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ServiceFlowTest {
    private static final Path ROOT = createRoot();

    @LocalServerPort
    int port;

    @Autowired
    ObjectMapper mapper;

    @Autowired
    JdbcTemplate appJdbc;

    @Autowired
    @Qualifier("queueJdbcTemplate")
    JdbcTemplate queueJdbc;

    private final HttpClient http = HttpClient.newHttpClient();

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", () ->
                "jdbc:h2:mem:ebookstudio;MODE=PostgreSQL;DB_CLOSE_DELAY=-1;DATABASE_TO_LOWER=TRUE");
        registry.add("spring.datasource.driver-class-name", () -> "org.h2.Driver");
        registry.add("spring.datasource.username", () -> "sa");
        registry.add("spring.datasource.password", () -> "");
        registry.add("ebookstudio.queue-db-path", () -> ROOT.resolve("jobs.db").toString());
        registry.add("ebookstudio.storage-root", () -> ROOT.resolve("storage").toString());
        registry.add("ebookstudio.jwt-secret", () -> "integration-test-secret");
        registry.add("ebookstudio.email-delivery-enabled", () -> "false");
        registry.add("ebookstudio.email-expose-development-code", () -> "true");
        registry.add("ebookstudio.verification-send-cooldown-seconds", () -> "0");
        registry.add("ebookstudio.login-ip-limit", () -> "100");
        registry.add("ebookstudio.login-account-limit", () -> "2");
        registry.add("ebookstudio.login-rate-window-seconds", () -> "3600");
        registry.add("ebookstudio.verification-ip-limit", () -> "100");
        registry.add("ebookstudio.verification-email-limit", () -> "2");
        registry.add("ebookstudio.verification-rate-window-seconds", () -> "3600");
    }

    @Test
    void accountUploadRecoveryAndDeletionFlow() throws Exception {
        ApiResponse health = json("GET", "/health", null, null);
        assertThat(health.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(health.body().get("database")).isEqualTo("ok");
        assertThat(queueJdbc.queryForObject(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='music_prompt_cache'",
                Integer.class)).isEqualTo(1);
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM usage_events", Integer.class)).isZero();

        String email = "flow@example.com";
        String username = "flow-user";
        String registrationCode = sendCode(email, "register");
        ApiResponse registration = json("POST", "/register", Map.of(
                "username", username, "password", "password123", "email", email, "code", registrationCode), null);
        assertThat(registration.status()).isEqualTo(HttpStatus.CREATED.value());

        Map login = json("POST", "/login", Map.of("username", username, "password", "password123"), null).body();
        String access = (String) login.get("access_token");
        String refresh = (String) login.get("refresh_token");
        String publicId = (String) login.get("public_id");

        long occurredAt = System.currentTimeMillis() / 1000;
        Map<String, Object> readingUsage = Map.of(
                "event_id", java.util.UUID.randomUUID().toString(),
                "event_type", "reading_session", "book_id", "offline-book-1",
                "occurred_at", occurredAt, "duration_seconds", 120,
                "page_turns", 4, "progress_percent", 35);
        Map<String, Object> appUsage = Map.of(
                "event_id", java.util.UUID.randomUUID().toString(),
                "event_type", "app_session", "occurred_at", occurredAt,
                "duration_seconds", 300, "page_turns", 0, "progress_percent", 0);
        Map<String, Object> usageBatch = Map.of("events", List.of(readingUsage, appUsage));
        ApiResponse usageInserted = json("POST", "/usage/events", usageBatch, access);
        assertThat(usageInserted.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(usageInserted.body().get("received_count")).isEqualTo(2);
        assertThat(usageInserted.body().get("inserted_count")).isEqualTo(2);
        assertThat(json("POST", "/usage/events", usageBatch, access).body().get("inserted_count"))
                .isEqualTo(0);
        ApiResponse usageSummary = json("GET", "/usage/summary", null, access);
        assertThat(usageSummary.body().get("total_app_seconds")).isEqualTo(300);
        assertThat(usageSummary.body().get("total_reading_seconds")).isEqualTo(120);
        assertThat(usageSummary.body().get("reading_session_count")).isEqualTo(1);
        assertThat(usageSummary.body().get("page_turn_count")).isEqualTo(4);
        assertThat(usageSummary.body().get("books_read_count")).isEqualTo(1);
        ApiResponse usageBooks = json("GET", "/usage/books", null, access);
        assertThat(usageBooks.status()).isEqualTo(HttpStatus.OK.value());
        List<?> bookUsageItems = (List<?>) usageBooks.body().get("books");
        assertThat(bookUsageItems).hasSize(1);
        Map<?, ?> bookUsage = (Map<?, ?>) bookUsageItems.get(0);
        assertThat(bookUsage.get("book_id")).isEqualTo("offline-book-1");
        assertThat(bookUsage.get("total_reading_seconds")).isEqualTo(120);
        assertThat(bookUsage.get("highest_progress_percent")).isEqualTo(35);
        ApiResponse dailyUsage = json("GET", "/usage/daily?days=7", null, access);
        assertThat(dailyUsage.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(dailyUsage.body().get("window_days")).isEqualTo(7);
        assertThat(dailyUsage.body().get("timezone")).isEqualTo("UTC");
        List<?> dailyItems = (List<?>) dailyUsage.body().get("daily");
        assertThat(dailyItems).hasSize(7).anySatisfy(item ->
                assertThat(((Map<?, ?>) item).get("reading_seconds")).isEqualTo(120));
        assertThat(json("GET", "/usage/daily?days=0", null, access).status())
                .isEqualTo(HttpStatus.BAD_REQUEST.value());

        String uploadRequestId = java.util.UUID.randomUUID().toString();
        ApiResponse accepted = upload(access, uploadRequestId);
        assertThat(accepted.status()).isEqualTo(HttpStatus.ACCEPTED.value());
        ApiResponse duplicateAccepted = upload(access, uploadRequestId);
        assertThat(duplicateAccepted.body().get("job_id")).isEqualTo(accepted.body().get("job_id"));
        assertThat(duplicateAccepted.body().get("book_folder")).isEqualTo(accepted.body().get("book_folder"));
        assertThat(queueJdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE id=?", Integer.class,
                uploadRequestId)).isEqualTo(1);
        String bookFolder = (String) accepted.body().get("book_folder");
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM books WHERE owner_public_id=? AND folder=?", Integer.class,
                publicId, bookFolder)).isEqualTo(1);
        Long uploadedBookId = appJdbc.queryForObject(
                "SELECT id FROM books WHERE owner_public_id=? AND folder=?", Long.class,
                publicId, bookFolder);
        String jobId = (String) accepted.body().get("job_id");
        Long uploadedRunId = appJdbc.queryForObject(
                "SELECT id FROM processing_runs WHERE book_id=?", Long.class, uploadedBookId);
        assertThat(appJdbc.queryForObject(
                "SELECT status FROM processing_runs WHERE id=?", String.class, uploadedRunId))
                .isEqualTo("QUEUED");
        assertThat(appJdbc.queryForObject(
                "SELECT request_id FROM processing_runs WHERE id=?", String.class, uploadedRunId))
                .isEqualTo(jobId);
        assertThat(appJdbc.queryForObject(
                "SELECT queue_job_id FROM processing_runs WHERE id=?", String.class, uploadedRunId))
                .isEqualTo(jobId);
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM book_artifacts WHERE processing_run_id=? AND artifact_type='SOURCE_PDF'",
                Integer.class, uploadedRunId)).isEqualTo(1);
        assertThat(appJdbc.queryForObject(
                "SELECT LENGTH(checksum) FROM book_artifacts WHERE processing_run_id=?",
                Integer.class, uploadedRunId)).isEqualTo(64);
        ApiResponse status = json("GET", "/check_status/" + jobId, null, access);
        assertThat(status.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(status.body().get("status")).isEqualTo("queued");
        assertThat(status.body().get("type")).isEqualTo("analyze");
        assertThat(status.body().get("attempt_count")).isEqualTo(0);
        ApiResponse cancelled = json("DELETE", "/jobs/" + jobId, null, access);
        assertThat(cancelled.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(cancelled.body().get("status")).isEqualTo("cancelled");
        assertThat(json("DELETE", "/jobs/" + jobId, null, access).body().get("status"))
                .isEqualTo("cancelled");
        assertThat(appJdbc.queryForObject(
                "SELECT status FROM processing_runs WHERE id=?", String.class, uploadedRunId))
                .isEqualTo("CANCELLED");

        ApiResponse runningAccepted = upload(access);
        String runningJobId = (String) runningAccepted.body().get("job_id");
        queueJdbc.update("""
                UPDATE jobs SET status='running', worker_id='test-worker',
                                started_at=?, attempt_count=1 WHERE id=?
                """, System.currentTimeMillis() / 1000, runningJobId);
        ApiResponse requested = json("DELETE", "/jobs/" + runningJobId, null, access);
        assertThat(requested.body().get("status")).isEqualTo("cancel_requested");
        assertThat(appJdbc.queryForObject("""
                SELECT r.status FROM processing_runs r JOIN books b ON b.id=r.book_id
                WHERE b.job_id=?
                """, String.class, runningJobId)).isEqualTo("CANCEL_REQUESTED");
        queueJdbc.update("UPDATE jobs SET status='cancelled', finished_at=?, worker_id=NULL WHERE id=?",
                System.currentTimeMillis() / 1000, runningJobId);
        assertThat(json("GET", "/check_status/" + runningJobId, null, access).body().get("status"))
                .isEqualTo("cancelled");
        assertThat(appJdbc.queryForObject("""
                SELECT r.status FROM processing_runs r JOIN books b ON b.id=r.book_id
                WHERE b.job_id=?
                """, String.class, runningJobId)).isEqualTo("CANCELLED");

        ApiResponse completedAccepted = upload(access);
        String completedJobId = (String) completedAccepted.body().get("job_id");
        String completedFolder = (String) completedAccepted.body().get("book_folder");
        Path completedRoot = ROOT.resolve("storage").resolve("users")
                .resolve(publicId).resolve(completedFolder);
        Files.writeString(completedRoot.resolve("completed_full.json"),
                "{\"book_info\":{\"title\":\"Completed Book\"},"
                        + "\"chapters\":[{\"segments\":[{"
                        + "\"music_filename\":\"default_ambient.wav\","
                        + "\"music_source\":\"system_default\",\"bpm\":80,"
                        + "\"generation_hint\":{\"target_emotion\":\"calm\"}}]}]}");
        Files.write(completedRoot.resolve("cover.png"), new byte[]{
                (byte) 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a});
        Path defaultsRoot = ROOT.resolve("storage").resolve("defaults");
        Files.createDirectories(defaultsRoot);
        Files.writeString(defaultsRoot.resolve("music_index.json"), "{}");
        long completedAt = System.currentTimeMillis() / 1000;
        String musicJobId = java.util.UUID.randomUUID().toString();
        queueJdbc.update("""
                INSERT INTO jobs(id, type, user_uuid, book_id, status, created_at,
                                 json_path, music_folder, web_path_prefix,
                                 parent_job_id, max_attempts)
                VALUES (?, 'music_generation', ?, ?, 'queued', ?, ?, ?, ?, ?, 3)
                """, musicJobId, publicId, completedFolder, completedAt,
                completedRoot.resolve("completed_full.json").toString(),
                defaultsRoot.resolve("music").toString(),
                "/files/" + username + "/" + completedFolder, completedJobId);
        queueJdbc.update("""
                UPDATE jobs SET status='done', started_at=?, finished_at=?, attempt_count=1,
                                output_json='completed_full.json', cover_file='cover.png',
                                book_title='Completed Book', author='Test Author', music_job_id=?
                WHERE id=?
                """, completedAt - 1, completedAt, musicJobId, completedJobId);
        assertThat(json("GET", "/check_status/" + completedJobId, null, access).body().get("status"))
                .isEqualTo("done");
        assertThat(appJdbc.queryForObject("""
                SELECT r.status FROM processing_runs r JOIN books b ON b.id=r.book_id
                WHERE b.job_id=? AND r.process_type='ANALYZE'
                """, String.class, completedJobId)).isEqualTo("SUCCEEDED");
        assertThat(appJdbc.queryForObject(
                "SELECT status FROM books WHERE job_id=?", String.class, completedJobId))
                .isEqualTo("READY");
        Long completedRunId = appJdbc.queryForObject("""
                SELECT r.id FROM processing_runs r JOIN books b ON b.id=r.book_id
                WHERE b.job_id=? AND r.process_type='ANALYZE'
                """, Long.class, completedJobId);
        assertThat(appJdbc.queryForList("""
                SELECT artifact_type FROM book_artifacts
                WHERE processing_run_id=? ORDER BY artifact_type
                """, String.class, completedRunId)).containsExactly(
                "BOOK_JSON", "COVER_IMAGE", "MUSIC_INDEX", "SOURCE_PDF");
        assertThat(appJdbc.queryForObject("""
                SELECT COUNT(*) FROM book_artifacts
                WHERE processing_run_id=? AND LENGTH(checksum)=64 AND file_size>0
                """, Integer.class, completedRunId)).isEqualTo(4);
        assertThat(json("GET", "/check_status/" + completedJobId, null, access).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM book_artifacts WHERE processing_run_id=?",
                Integer.class, completedRunId)).isEqualTo(4);
        Files.writeString(defaultsRoot.resolve("music_index.json"),
                "{\"changed-after-completion\":{}}");
        assertThat(json("GET", "/check_status/" + completedJobId, null, access).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM book_artifacts WHERE processing_run_id=?",
                Integer.class, completedRunId)).isEqualTo(4);
        Long musicRunId = appJdbc.queryForObject(
                "SELECT id FROM processing_runs WHERE queue_job_id=?", Long.class, musicJobId);
        assertThat(appJdbc.queryForObject(
                "SELECT status FROM processing_runs WHERE id=?", String.class, musicRunId))
                .isEqualTo("QUEUED");

        String musicSignature = "e".repeat(64);
        Path generatedMusic = defaultsRoot.resolve("music").resolve(musicSignature + ".wav");
        Files.writeString(generatedMusic, "generated-music");
        Files.writeString(completedRoot.resolve("completed_full.json"),
                "{\"book_info\":{\"title\":\"Completed Book\"},"
                        + "\"chapters\":[{\"segments\":["
                        + "{\"music_filename\":\"" + musicSignature
                        + ".wav\",\"music_source\":\"ai_generated\",\"bpm\":88},"
                        + "{\"music_filename\":\"" + musicSignature
                        + ".wav\",\"music_source\":\"ai_reused\",\"bpm\":88}]}]}");
        queueJdbc.update("""
                INSERT INTO music_prompt_cache(
                    signature, prompt, genre, bpm, keywords_json,
                    target_duration_sec, segment_duration_sec, generator_version,
                    status, filename, relative_path, created_at, updated_at,
                    generated_at, last_used_at, reuse_count, owner_job_id)
                VALUES (?, 'calm fantasy instrumental', 'ambient', 88, '["calm","fantasy"]',
                        120, 30, 'facebook/musicgen-small:v1', 'ready', ?, ?, ?, ?, ?, ?, 1, ?)
                """, musicSignature, musicSignature + ".wav", musicSignature + ".wav",
                completedAt, completedAt + 1, completedAt, completedAt + 1, musicJobId);
        queueJdbc.update("""
                UPDATE jobs SET status='done', started_at=?, finished_at=?, attempt_count=1
                WHERE id=?
                """, completedAt, completedAt + 2, musicJobId);
        ApiResponse musicDone = json("GET", "/check_status/" + musicJobId, null, access);
        assertThat(musicDone.status()).as(musicDone.body().toString())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(musicDone.body().get("status")).isEqualTo("done");
        assertThat(appJdbc.queryForObject(
                "SELECT status FROM processing_runs WHERE id=?", String.class, musicRunId))
                .isEqualTo("SUCCEEDED");
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM music_assets WHERE signature=? AND status='READY'",
                Integer.class, musicSignature)).isEqualTo(1);
        assertThat(appJdbc.queryForObject(
                "SELECT reuse_count FROM music_assets WHERE signature=?",
                Integer.class, musicSignature)).isEqualTo(1);
        assertThat(appJdbc.queryForList("""
                SELECT bm.binding_type FROM book_music_bindings bm
                JOIN music_assets m ON m.id=bm.music_asset_id
                WHERE m.signature=? ORDER BY bm.segment_key
                """, String.class, musicSignature)).containsExactly("GENERATED", "CACHE_REUSED");
        assertThat(appJdbc.queryForList("""
                SELECT artifact_type FROM book_artifacts
                WHERE processing_run_id=? ORDER BY artifact_type
                """, String.class, musicRunId)).containsExactly("BOOK_JSON", "MUSIC_INDEX");
        assertThat(json("GET", "/check_status/" + musicJobId, null, access).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(appJdbc.queryForObject(
                "SELECT COUNT(*) FROM book_music_bindings WHERE processing_run_id=?",
                Integer.class, musicRunId)).isEqualTo(2);

        ApiResponse processingHistory = json("GET", "/api/books/" + completedFolder
                + "/processing-history", null, access);
        assertThat(processingHistory.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(processingHistory.body().get("book_folder")).isEqualTo(completedFolder);
        assertThat(processingHistory.body().get("book_status")).isEqualTo("ready");
        List<?> historyRuns = (List<?>) processingHistory.body().get("runs");
        assertThat(historyRuns).hasSize(2);
        List<String> processTypes = historyRuns.stream()
                .map(item -> String.valueOf(((Map<?, ?>) item).get("process_type"))).toList();
        assertThat(processTypes).containsExactly("music_generation", "analyze");
        Map<?, ?> analysisHistory = historyRuns.stream().map(item -> (Map<?, ?>) item)
                .filter(item -> "analyze".equals(item.get("process_type"))).findFirst().orElseThrow();
        Map<?, ?> musicHistory = historyRuns.stream().map(item -> (Map<?, ?>) item)
                .filter(item -> "music_generation".equals(item.get("process_type"))).findFirst().orElseThrow();
        assertThat((List<?>) analysisHistory.get("artifacts")).hasSize(4);
        assertThat((List<?>) musicHistory.get("artifacts")).hasSize(2);

        ApiResponse musicTracks = json("GET", "/api/books/" + completedFolder
                + "/music-tracks", null, access);
        assertThat(musicTracks.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(musicTracks.body().get("track_count")).isEqualTo(2);
        assertThat(musicTracks.body().get("unique_asset_count")).isEqualTo(1);
        List<?> trackItems = (List<?>) musicTracks.body().get("tracks");
        List<String> bindingTypes = trackItems.stream()
                .map(item -> String.valueOf(((Map<?, ?>) item).get("binding_type"))).toList();
        assertThat(bindingTypes).containsExactly("generated", "cache_reused");
        Map<?, ?> firstTrack = (Map<?, ?>) trackItems.get(0);
        assertThat(firstTrack.containsKey("prompt")).isFalse();
        assertThat(firstTrack.containsKey("storage_key")).isFalse();
        assertThat(firstTrack.containsKey("generation_params_json")).isFalse();
        assertThat(json("GET", "/api/books/" + completedFolder
                + "/processing-history", null, null).status())
                .isEqualTo(HttpStatus.UNAUTHORIZED.value());

        String otherEmail = "other-flow@example.com";
        String otherUsername = "other-flow-user";
        String otherCode = sendCode(otherEmail, "register");
        assertThat(json("POST", "/register", Map.of(
                "username", otherUsername, "password", "password123",
                "email", otherEmail, "code", otherCode), null).status())
                .isEqualTo(HttpStatus.CREATED.value());
        String otherAccess = (String) json("POST", "/login", Map.of(
                "username", otherUsername, "password", "password123"), null)
                .body().get("access_token");
        assertThat(json("GET", "/api/books/" + completedFolder
                + "/processing-history", null, otherAccess).status())
                .isEqualTo(HttpStatus.NOT_FOUND.value());
        assertThat(json("GET", "/api/books/" + completedFolder
                + "/music-tracks", null, otherAccess).status())
                .isEqualTo(HttpStatus.NOT_FOUND.value());

        ApiResponse missingArtifactAccepted = upload(access);
        String missingJobId = (String) missingArtifactAccepted.body().get("job_id");
        long missingFinishedAt = System.currentTimeMillis() / 1000;
        queueJdbc.update("""
                UPDATE jobs SET status='done', started_at=?, finished_at=?, attempt_count=1,
                                output_json='missing_full.json', cover_file='missing.png',
                                book_title='Broken Book', author='Test Author'
                WHERE id=?
                """, missingFinishedAt - 1, missingFinishedAt, missingJobId);
        assertThat(json("GET", "/check_status/" + missingJobId, null, access).status())
                .isEqualTo(HttpStatus.CONFLICT.value());
        assertThat(appJdbc.queryForObject("""
                SELECT r.status FROM processing_runs r JOIN books b ON b.id=r.book_id
                WHERE b.job_id=?
                """, String.class, missingJobId)).isEqualTo("RUNNING");
        assertThat(appJdbc.queryForObject(
                "SELECT status FROM books WHERE job_id=?", String.class, missingJobId))
                .isEqualTo("PROCESSING");
        ApiResponse availableBooks = json("POST", "/my_books", Map.of(), access);
        assertThat(availableBooks.status()).isEqualTo(HttpStatus.OK.value());
        assertThat((List<?>) availableBooks.body().get("books")).isNotEmpty();
        assertThat(appJdbc.queryForObject("""
                SELECT COUNT(*) FROM book_artifacts a
                JOIN processing_runs r ON r.id=a.processing_run_id
                JOIN books b ON b.id=r.book_id
                WHERE b.job_id=?
                """, Integer.class, missingJobId)).isEqualTo(1);

        assertMusicAuthorization(access, username, publicId, bookFolder);

        ApiResponse changed = json("POST", "/change_password", Map.of(
                "current_password", "password123", "new_password", "newpassword456"), access);
        assertThat(changed.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(json("POST", "/my_books", Map.of(), access).status())
                .isEqualTo(HttpStatus.UNAUTHORIZED.value());
        assertThat(json("POST", "/refresh", null, refresh).status())
                .isEqualTo(HttpStatus.BAD_REQUEST.value());

        Map changedLogin = json("POST", "/login", Map.of(
                "username", username, "password", "newpassword456"), null).body();
        String changedAccess = (String) changedLogin.get("access_token");
        String changedRefresh = (String) changedLogin.get("refresh_token");
        assertThat(json("POST", "/logout", Map.of("refresh_token", changedRefresh), changedAccess).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(json("POST", "/refresh", null, changedRefresh).status())
                .isEqualTo(HttpStatus.BAD_REQUEST.value());

        Map preResetLogin = json("POST", "/login", Map.of(
                "username", username, "password", "newpassword456"), null).body();
        String preResetAccess = (String) preResetLogin.get("access_token");
        String preResetRefresh = (String) preResetLogin.get("refresh_token");
        String recoveryCode = sendCode(email, "recovery");
        ApiResponse found = json("POST", "/find_id", Map.of("email", email, "code", recoveryCode), null);
        assertThat(found.status()).isEqualTo(HttpStatus.OK.value());
        assertThat(found.body().get("username")).isEqualTo(username);
        assertThat(json("POST", "/reset_password", Map.of(
                "email", email, "code", recoveryCode, "new_password", "recovered789"), null).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(json("POST", "/my_books", Map.of(), preResetAccess).status())
                .isEqualTo(HttpStatus.UNAUTHORIZED.value());
        assertThat(json("POST", "/refresh", null, preResetRefresh).status())
                .isEqualTo(HttpStatus.BAD_REQUEST.value());

        Map secondLogin = json("POST", "/login", Map.of(
                "username", username, "password", "recovered789"), null).body();
        String secondAccess = (String) secondLogin.get("access_token");
        assertThat(json("DELETE", "/account", null, secondAccess).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(json("POST", "/my_books", Map.of(), secondAccess).status())
                .isEqualTo(HttpStatus.UNAUTHORIZED.value());
        assertThat(ROOT.resolve("storage").resolve("users").resolve(publicId)).doesNotExist();
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM usage_events WHERE user_uuid=?",
                Integer.class, publicId)).isZero();
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM books WHERE owner_public_id=?",
                Integer.class, publicId)).isZero();
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM processing_runs", Integer.class)).isZero();
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM book_artifacts", Integer.class)).isZero();
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM book_music_bindings", Integer.class)).isZero();
        assertThat(appJdbc.queryForObject("SELECT COUNT(*) FROM music_assets", Integer.class)).isEqualTo(1);
        assertThat(queueJdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE user_uuid=?",
                Integer.class, publicId)).isZero();
        Path accountTrash = ROOT.resolve("storage").resolve(".trash").resolve("accounts");
        if (Files.exists(accountTrash)) {
            try (var entries = Files.list(accountTrash)) {
                assertThat(entries).isEmpty();
            }
        }
    }

    @Test
    void limitsRepeatedLoginAndVerificationRequests() throws Exception {
        String limitedEmail = "limited@example.com";
        assertThat(json("POST", "/send_code", Map.of(
                "email", limitedEmail, "purpose", "register"), null).status())
                .isEqualTo(HttpStatus.OK.value());
        assertThat(json("POST", "/send_code", Map.of(
                "email", limitedEmail, "purpose", "register"), null).status())
                .isEqualTo(HttpStatus.OK.value());
        ApiResponse limitedCode = json("POST", "/send_code", Map.of(
                "email", limitedEmail, "purpose", "register"), null);
        assertThat(limitedCode.status()).isEqualTo(HttpStatus.TOO_MANY_REQUESTS.value());
        assertThat(limitedCode.retryAfter()).isPresent();

        assertThat(json("POST", "/login", Map.of(
                "username", "missing-user", "password", "bad-password"), null).status())
                .isEqualTo(HttpStatus.BAD_REQUEST.value());
        assertThat(json("POST", "/login", Map.of(
                "username", "missing-user", "password", "bad-password"), null).status())
                .isEqualTo(HttpStatus.BAD_REQUEST.value());
        ApiResponse limitedLogin = json("POST", "/login", Map.of(
                "username", "missing-user", "password", "bad-password"), null);
        assertThat(limitedLogin.status()).isEqualTo(HttpStatus.TOO_MANY_REQUESTS.value());
        assertThat(limitedLogin.retryAfter()).isPresent();
    }

    private void assertMusicAuthorization(String access, String username, String publicId,
                                          String bookFolder) throws Exception {
        Path bookRoot = ROOT.resolve("storage").resolve("users").resolve(publicId).resolve(bookFolder);
        Files.createDirectories(bookRoot);
        Files.writeString(bookRoot.resolve("flow_full.json"),
                "{\"chapters\":[{\"segments\":[{\"music_filename\":\"referenced.wav\"}]}]}");
        Path musicRoot = ROOT.resolve("storage").resolve("defaults").resolve("music");
        Files.createDirectories(musicRoot);
        Files.writeString(musicRoot.resolve("referenced.wav"), "referenced");
        Files.writeString(musicRoot.resolve("unreferenced.wav"), "unreferenced");
        assertThat(rawStatus("/files/" + username + "/" + bookFolder
                + "/music/referenced.wav", access)).isEqualTo(HttpStatus.OK.value());
        assertThat(rawStatus("/files/" + username + "/" + bookFolder
                + "/music/unreferenced.wav", access)).isEqualTo(HttpStatus.NOT_FOUND.value());
    }

    private int rawStatus(String path, String bearer) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(URI.create(url(path)))
                .header("Authorization", "Bearer " + bearer).GET().build();
        return http.send(request, HttpResponse.BodyHandlers.discarding()).statusCode();
    }

    private String sendCode(String email, String purpose) throws Exception {
        Map body = json("POST", "/send_code", Map.of("email", email, "purpose", purpose), null).body();
        assertThat(body.get("development_code")).isInstanceOf(String.class);
        String code = (String) body.get("development_code");
        String stored = appJdbc.queryForObject(
                "SELECT code FROM verification_codes WHERE email = ? AND purpose = ?",
                String.class, email, purpose);
        assertThat(stored).isNotEqualTo(code).hasSize(64);
        return code;
    }

    private ApiResponse upload(String accessToken) throws Exception {
        return upload(accessToken, null);
    }

    private ApiResponse upload(String accessToken, String requestId) throws Exception {
        String boundary = "ebookstudio-test-boundary";
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        body.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"flow.pdf\"\r\n"
                + "Content-Type: application/pdf\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
        body.write("%PDF-1.4\n%%EOF".getBytes(StandardCharsets.US_ASCII));
        if (requestId != null) {
            body.write(("\r\n--" + boundary + "\r\nContent-Disposition: form-data; name=\"request_id\"\r\n\r\n"
                    + requestId).getBytes(StandardCharsets.US_ASCII));
        }
        body.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.US_ASCII));
        HttpRequest request = HttpRequest.newBuilder(URI.create(url("/upload_book")))
                .header("Authorization", "Bearer " + accessToken)
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofByteArray(body.toByteArray())).build();
        return response(http.send(request, HttpResponse.BodyHandlers.ofString()));
    }

    private ApiResponse json(String method, String path, Object body, String bearer) throws Exception {
        HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url(path)))
                .header("Accept", "application/json");
        if (bearer != null) builder.header("Authorization", "Bearer " + bearer);
        HttpRequest.BodyPublisher publisher = body == null
                ? HttpRequest.BodyPublishers.noBody()
                : HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body));
        if (body != null) builder.header("Content-Type", "application/json");
        builder.method(method, publisher);
        return response(http.send(builder.build(), HttpResponse.BodyHandlers.ofString()));
    }

    private ApiResponse response(HttpResponse<String> response) throws Exception {
        Map body = response.body() == null || response.body().isBlank()
                ? Map.of() : mapper.readValue(response.body(), Map.class);
        return new ApiResponse(response.statusCode(), body,
                response.headers().firstValue("Retry-After"));
    }

    private String url(String path) { return "http://127.0.0.1:" + port + path; }

    private static Path createRoot() {
        try { return Files.createTempDirectory("ebookstudio-spring-test-"); }
        catch (Exception error) { throw new IllegalStateException(error); }
    }

    private record ApiResponse(int status, Map body, java.util.Optional<String> retryAfter) { }
}