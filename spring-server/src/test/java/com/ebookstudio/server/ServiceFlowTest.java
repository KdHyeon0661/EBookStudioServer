package com.ebookstudio.server;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
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
    JdbcTemplate jdbc;

    private final HttpClient http = HttpClient.newHttpClient();

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry registry) {
        registry.add("spring.datasource.url", () -> "jdbc:sqlite:" + ROOT.resolve("test.db"));
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
        assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='music_prompt_cache'",
                Integer.class)).isEqualTo(1);
        assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='usage_events'",
                Integer.class)).isEqualTo(1);

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

        String uploadRequestId = java.util.UUID.randomUUID().toString();
        ApiResponse accepted = upload(access, uploadRequestId);
        assertThat(accepted.status()).isEqualTo(HttpStatus.ACCEPTED.value());
        ApiResponse duplicateAccepted = upload(access, uploadRequestId);
        assertThat(duplicateAccepted.body().get("job_id")).isEqualTo(accepted.body().get("job_id"));
        assertThat(duplicateAccepted.body().get("book_folder")).isEqualTo(accepted.body().get("book_folder"));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM jobs WHERE id=?", Integer.class,
                uploadRequestId)).isEqualTo(1);
        String jobId = (String) accepted.body().get("job_id");
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

        ApiResponse runningAccepted = upload(access);
        String runningJobId = (String) runningAccepted.body().get("job_id");
        jdbc.update("UPDATE jobs SET status='running', worker_id='test-worker', started_at=? WHERE id=?",
                System.currentTimeMillis() / 1000, runningJobId);
        ApiResponse requested = json("DELETE", "/jobs/" + runningJobId, null, access);
        assertThat(requested.body().get("status")).isEqualTo("cancel_requested");
        jdbc.update("UPDATE jobs SET status='cancelled', finished_at=?, worker_id=NULL WHERE id=?",
                System.currentTimeMillis() / 1000, runningJobId);

        String bookFolder = (String) accepted.body().get("book_folder");
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
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM usage_events WHERE user_uuid=?",
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
        String stored = jdbc.queryForObject(
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