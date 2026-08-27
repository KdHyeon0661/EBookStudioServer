package com.ebookstudio.server.book;

import com.ebookstudio.server.auth.JwtPrincipal;
import com.ebookstudio.server.config.EBookStudioProperties;
import com.ebookstudio.server.processing.ProcessingHistorySynchronizer;
import com.ebookstudio.server.processing.MusicJobProjectionService;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.*;
import java.text.Normalizer;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.*;
import java.util.stream.Stream;

@Service
public class BookService {
    private static final Logger log = LoggerFactory.getLogger(BookService.class);
    private final JdbcTemplate appJdbc;
    private final JdbcTemplate queueJdbc;
    private final ObjectMapper objectMapper;
    private final BookUploadCatalogService uploadCatalog;
    private final ProcessingHistorySynchronizer processingHistory;
    private final MusicJobProjectionService musicJobs;
    private final Path usersRoot;
    private final Path musicRoot;

    public BookService(JdbcTemplate appJdbc,
                       @Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc,
                       ObjectMapper objectMapper, EBookStudioProperties properties,
                       BookUploadCatalogService uploadCatalog,
                       ProcessingHistorySynchronizer processingHistory,
                       MusicJobProjectionService musicJobs) {
        this.appJdbc = appJdbc;
        this.queueJdbc = queueJdbc;
        this.objectMapper = objectMapper;
        this.uploadCatalog = uploadCatalog;
        this.processingHistory = processingHistory;
        this.musicJobs = musicJobs;
        Path storageRoot = Path.of(properties.storageRoot()).toAbsolutePath().normalize();
        this.usersRoot = storageRoot.resolve("users").normalize();
        this.musicRoot = storageRoot.resolve("defaults").resolve("music").normalize();
        try {
            Files.createDirectories(usersRoot);
            Files.createDirectories(musicRoot);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to initialize storage", e);
        }
    }

    public UploadAccepted upload(JwtPrincipal principal, MultipartFile file, String requestedJobId) {
        if (file == null || file.isEmpty()) throw new IllegalArgumentException("PDF file is required");
        String original = Optional.ofNullable(file.getOriginalFilename()).orElse("book.pdf");
        if (!original.toLowerCase(Locale.ROOT).endsWith(".pdf")) {
            throw new IllegalArgumentException("Only PDF files are allowed");
        }
        validatePdfHeader(file);

        String jobId = normalizedJobId(requestedJobId);
        UploadAccepted existing = queueJdbc.query("SELECT user_uuid, book_id, status FROM jobs WHERE id = ?",
                rs -> {
                    if (!rs.next()) return null;
                    if (!principal.publicId().equals(rs.getString("user_uuid")))
                        throw new SecurityException("Access denied");
                    return new UploadAccepted(jobId, rs.getString("status"), rs.getString("book_id"),
                            "Upload request already accepted.");
                }, jobId);
        if (existing != null) return existing;

        String base = safeBaseName(original.substring(0, original.length() - 4));
        String bookFolder = base + "_" + UUID.randomUUID().toString().replace("-", "").substring(0, 8);
        Path bookRoot = safeResolve(usersRoot.resolve(principal.publicId()), bookFolder);
        Path pdfPath = bookRoot.resolve(base + ".pdf").normalize();
        long createdAt = Instant.now().getEpochSecond();
        try {
            Files.createDirectories(bookRoot);
            file.transferTo(pdfPath);
            String sourceFileName = pdfPath.getFileName().toString();
            String sourceStorageKey = "users/" + principal.publicId() + "/"
                    + bookFolder + "/" + sourceFileName;
            uploadCatalog.register(principal.publicId(), bookFolder, jobId,
                    sourceFileName, sourceStorageKey, sha256(pdfPath),
                    Files.size(pdfPath), createdAt);
            queueJdbc.update("""
                    INSERT INTO jobs(id, type, user_uuid, book_id, status, created_at,
                                     json_path, music_folder, web_path_prefix, pdf_path, book_root_folder)
                    VALUES (?, 'analyze', ?, ?, 'queued', ?, NULL, ?, ?, ?, ?)
                    """, jobId, principal.publicId(), bookFolder, createdAt,
                    musicRoot.toString(), "/files/" + principal.username() + "/" + bookFolder,
                    pdfPath.toString(), bookRoot.toString());
        } catch (Exception e) {
            try {
                uploadCatalog.compensateFailedQueueRegistration(principal.publicId(), bookFolder);
            } catch (Exception compensationError) {
                e.addSuppressed(compensationError);
                log.error("Unable to compensate failed upload registration for {}", bookFolder,
                        compensationError);
            }
            deleteTreeQuietly(bookRoot);
            throw new IllegalStateException("Unable to accept upload", e);
        }

        return new UploadAccepted(jobId, "queued", bookFolder,
                "Upload accepted. Processing started.");
    }

    public JobStatus status(JwtPrincipal principal, String jobId) {
        JobStatus job = queueJdbc.query(jobSelect() + " WHERE id = ?", rs -> {
            if (!rs.next()) throw new NoSuchElementException("Job not found");
            if (!principal.publicId().equals(rs.getString("user_uuid"))) {
                throw new SecurityException("Access denied");
            }
            return readJob(rs);
        }, jobId);
        syncJobProjection(principal.publicId(), job);
        return job;
    }

    public JobStatus cancel(JwtPrincipal principal, String jobId) {
        JobTarget target = queueJdbc.query("SELECT type, user_uuid, book_id, status FROM jobs WHERE id = ?",
                rs -> rs.next() ? new JobTarget(rs.getString("type"), rs.getString("user_uuid"),
                        rs.getString("book_id"), rs.getString("status")) : null, jobId);
        if (target == null) throw new NoSuchElementException("Job not found");
        if (!principal.publicId().equals(target.userUuid())) throw new SecurityException("Access denied");

        long now = Instant.now().getEpochSecond();
        if ("queued".equals(target.status()) || "running".equals(target.status())) {
            int cancelled = queueJdbc.update("""
                    UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?,
                        available_at=NULL, error=NULL
                    WHERE id=? AND user_uuid=? AND status='queued'
                    """, now, now, jobId, principal.publicId());
            if (cancelled == 1 && "analyze".equals(target.type()) && target.bookId() != null) {
                deleteTreeQuietly(safeResolve(usersRoot.resolve(principal.publicId()), target.bookId()));
            }
            if (cancelled == 0) {
                queueJdbc.update("""
                        UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                        WHERE id=? AND user_uuid=? AND status='running'
                        """, now, jobId, principal.publicId());
            }
        }
        return status(principal, jobId);
    }

    public List<BookSummary> myBooks(JwtPrincipal principal) {
        reconcileUserBooks(principal.publicId());
        return appJdbc.query("""
                SELECT folder, title, author, cover_file, text_file
                FROM books
                WHERE owner_public_id=? AND status='READY'
                ORDER BY updated_at DESC
                """, (rs, rowNum) -> {
            String folder = rs.getString("folder");
            String coverFile = Optional.ofNullable(rs.getString("cover_file")).orElse("");
            String coverUrl = coverFile.isEmpty() ? ""
                    : "/files/" + principal.username() + "/" + folder + "/" + coverFile;
            return new BookSummary(
                    Optional.ofNullable(rs.getString("title")).orElse(stripFolderSuffix(folder)),
                    folder, coverUrl, coverFile,
                    Optional.ofNullable(rs.getString("text_file")).orElse(""),
                    Optional.ofNullable(rs.getString("author")).orElse("Unknown Author"));
        }, principal.publicId());
    }

    private void reconcileUserBooks(String userUuid) {
        List<JobStatus> jobs = queueJdbc.query(jobSelect()
                        + " WHERE user_uuid=? ORDER BY created_at",
                (rs, rowNum) -> readJob(rs), userUuid);
        for (JobStatus job : jobs) {
            try {
                syncJobProjection(userUuid, job);
            } catch (IllegalStateException error) {
                log.warn("Unable to reconcile job {}: {}", job.job_id(), error.getMessage());
            }
        }
    }

    private void syncJobProjection(String userUuid, JobStatus job) {
        if ("analyze".equals(job.type())) {
            syncBookProjection(userUuid, job);
        } else if ("music_generation".equals(job.type()) && job.book_id() != null) {
            musicJobs.synchronize(new MusicJobProjectionService.MusicQueueJob(
                    userUuid, job.book_id(), job.job_id(), job.status(), job.created_at(),
                    job.started_at(), job.finished_at(), job.error(),
                    job.attempt_count(), job.max_attempts()));
        }
    }

    private void syncBookProjection(String userUuid, JobStatus job) {
        if (!"analyze".equals(job.type()) || job.book_id() == null) return;
        JobResult result = job.result();
        Path bookRoot = safeResolve(usersRoot.resolve(userUuid), job.book_id());
        processingHistory.synchronize(new ProcessingHistorySynchronizer.QueueJobSnapshot(
                job.job_id(), job.status(), job.attempt_count(), job.started_at(),
                job.finished_at(), job.error(), bookRoot,
                result == null ? null : result.text(),
                result == null ? null : result.cover()));
        if (result != null && result.music_job_id() != null
                && !result.music_job_id().isBlank()) {
            queueJdbc.query("""
                    SELECT created_at, max_attempts FROM jobs
                    WHERE id=? AND type='music_generation'
                    """, rs -> {
                if (!rs.next()) throw new IllegalStateException("Music child job was not found");
                return musicJobs.ensureQueued(userUuid, job.book_id(), result.music_job_id(),
                        rs.getLong("created_at"), rs.getInt("max_attempts"));
            }, result.music_job_id());
        }

        String projectedStatus = switch (job.status()) {
            case "done" -> "READY";
            case "error" -> "FAILED";
            case "cancelled" -> "CANCELLED";
            case "cancel_requested" -> "CANCELLING";
            default -> "PROCESSING";
        };
        String title = result == null ? null : result.book_title();
        String author = result == null ? null : result.author();
        String cover = result == null ? null : result.cover();
        String text = result == null ? null : result.text();
        long updatedAt = job.finished_at() == null ? Instant.now().getEpochSecond() : job.finished_at();
        int updated = appJdbc.update("""
                UPDATE books SET status=?, title=COALESCE(?, title), author=COALESCE(?, author),
                    cover_file=COALESCE(?, cover_file), text_file=COALESCE(?, text_file), updated_at=?
                WHERE owner_public_id=? AND folder=?
                """, projectedStatus, title, author, cover, text, updatedAt, userUuid, job.book_id());
        if (updated == 0) {
            try {
                appJdbc.update("""
                        INSERT INTO books(owner_public_id, folder, job_id, status, title, author,
                                          cover_file, text_file, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, userUuid, job.book_id(), job.job_id(), projectedStatus, title, author,
                        cover, text, job.created_at(), updatedAt);
            } catch (DuplicateKeyException race) {
                appJdbc.update("UPDATE books SET status=?, updated_at=? WHERE owner_public_id=? AND folder=?",
                        projectedStatus, updatedAt, userUuid, job.book_id());
            }
        }
    }

    private static String jobSelect() {
        return """
                SELECT id, type, user_uuid, book_id, status, created_at, started_at, finished_at, error,
                       output_json, cover_file, book_title, author, music_job_id,
                       attempt_count, max_attempts, cancel_requested_at
                FROM jobs
                """;
    }

    private static JobStatus readJob(java.sql.ResultSet rs) throws java.sql.SQLException {
        JobResult result = null;
        if ("done".equals(rs.getString("status"))) {
            result = new JobResult(rs.getString("book_id"), rs.getString("book_title"),
                    rs.getString("output_json"), rs.getString("cover_file"), rs.getString("author"),
                    rs.getString("music_job_id"));
        }
        Long startedAt = rs.getObject("started_at") == null ? null : rs.getLong("started_at");
        Long finishedAt = rs.getObject("finished_at") == null ? null : rs.getLong("finished_at");
        Long cancelRequestedAt = rs.getObject("cancel_requested_at") == null
                ? null : rs.getLong("cancel_requested_at");
        return new JobStatus(rs.getString("id"), rs.getString("type"), rs.getString("book_id"),
                rs.getString("status"), rs.getLong("created_at"), startedAt, finishedAt,
                rs.getString("error"), rs.getInt("attempt_count"), rs.getInt("max_attempts"),
                cancelRequestedAt, result);
    }

    public void deleteBook(JwtPrincipal principal, String bookFolder) {
        requireSafeSegment(bookFolder);
        long now = Instant.now().getEpochSecond();
        queueJdbc.update("""
                UPDATE jobs SET status='cancelled', finished_at=?, cancel_requested_at=?, available_at=NULL
                WHERE user_uuid=? AND book_id=? AND status='queued'
                """, now, now, principal.publicId(), bookFolder);
        queueJdbc.update("""
                UPDATE jobs SET status='cancel_requested', cancel_requested_at=?
                WHERE user_uuid=? AND book_id=? AND status='running'
                """, now, principal.publicId(), bookFolder);
        Integer active = queueJdbc.queryForObject("""
                SELECT COUNT(*) FROM jobs
                WHERE user_uuid=? AND book_id=? AND status='cancel_requested'
                """, Integer.class, principal.publicId(), bookFolder);
        if (active != null && active > 0) {
            throw new IllegalStateException("Background processing is stopping; retry deletion shortly");
        }

        Path userRoot = usersRoot.resolve(principal.publicId()).normalize();
        Path target = safeResolve(userRoot, bookFolder);
        if (!Files.isDirectory(target)) throw new NoSuchElementException("Book not found");
        try {
            deleteTree(target);
            queueJdbc.update("DELETE FROM jobs WHERE user_uuid = ? AND book_id = ?",
                    principal.publicId(), bookFolder);
            appJdbc.update("DELETE FROM books WHERE owner_public_id = ? AND folder = ?",
                    principal.publicId(), bookFolder);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to delete book", e);
        }
    }

    public List<String> listMusic(JwtPrincipal principal, String username, String bookFolder) {
        requireUsername(principal, username);
        requireSafeSegment(bookFolder);
        Path bookRoot = safeResolve(usersRoot.resolve(principal.publicId()), bookFolder);
        if (!Files.isDirectory(bookRoot)) throw new NoSuchElementException("Book not found");
        return List.copyOf(referencedMusicFiles(bookRoot));
    }

    private Set<String> referencedMusicFiles(Path bookRoot) {
        Path jsonPath = findArtifact(bookRoot, "_full.json");
        if (jsonPath == null) return Set.of();
        try {
            JsonNode root = objectMapper.readTree(jsonPath.toFile());
            Set<String> files = new TreeSet<>();
            for (JsonNode chapter : root.path("chapters")) {
                for (JsonNode segment : chapter.path("segments")) {
                    String filename = segment.path("music_filename").asString("").trim();
                    if (isSafeSegment(filename)) files.add(filename);
                }
            }
            return files;
        } catch (Exception e) {
            throw new IllegalStateException("Unable to read book metadata", e);
        }
    }

    public Resource rootFile(JwtPrincipal principal, String username, String bookFolder, String filename) {
        requireUsername(principal, username);
        requireSafeSegment(bookFolder);
        requireSafeSegment(filename);
        Path base = safeResolve(usersRoot.resolve(principal.publicId()), bookFolder);
        return existingResource(safeResolve(base, filename));
    }

    public Resource musicFile(JwtPrincipal principal, String username, String bookFolder, String filename) {
        requireUsername(principal, username);
        requireSafeSegment(bookFolder);
        requireSafeSegment(filename);
        Path bookRoot = safeResolve(usersRoot.resolve(principal.publicId()), bookFolder);
        if (!Files.isDirectory(bookRoot) || !referencedMusicFiles(bookRoot).contains(filename)) {
            throw new NoSuchElementException("File not found");
        }
        Path direct = safeResolve(musicRoot, filename);
        if (Files.isRegularFile(direct)) return new FileSystemResource(direct);
        try (Stream<Path> stream = Files.walk(musicRoot, 2)) {
            Path found = stream.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().equals(filename))
                    .findFirst().orElseThrow(() -> new NoSuchElementException("File not found"));
            return new FileSystemResource(found);
        } catch (IOException e) {
            throw new NoSuchElementException("File not found");
        }
    }

    private BookSummary summarizeBook(JwtPrincipal principal, Path bookDir) {
        String folder = bookDir.getFileName().toString();
        Path json = findArtifact(bookDir, "_full.json");
        Path cover = findArtifact(bookDir, ".png");
        String title = stripFolderSuffix(folder);
        String author = "Unknown Author";
        if (json != null) {
            try {
                JsonNode info = objectMapper.readTree(json.toFile()).path("book_info");
                title = info.path("title").asString(title);
                author = info.path("author").asString(author);
            } catch (Exception ignored) {
            }
        }
        String coverFile = cover == null ? "" : cover.getFileName().toString();
        String textFile = json == null ? "" : json.getFileName().toString();
        String coverUrl = coverFile.isEmpty() ? ""
                : "/files/" + principal.username() + "/" + folder + "/" + coverFile;
        return new BookSummary(title, folder, coverUrl, coverFile, textFile, author);
    }

    private static String sha256(Path path) throws IOException {
        try (InputStream input = Files.newInputStream(path)) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[8192];
            for (int read; (read = input.read(buffer)) != -1; ) {
                digest.update(buffer, 0, read);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (java.security.NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private void validatePdfHeader(MultipartFile file) {
        try (InputStream input = file.getInputStream()) {
            byte[] header = input.readNBytes(4);
            if (!Arrays.equals(header, new byte[]{'%', 'P', 'D', 'F'})) {
                throw new IllegalArgumentException("Invalid PDF file");
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to read PDF file");
        }
    }

    private static String normalizedJobId(String requestedJobId) {
        if (requestedJobId == null || requestedJobId.isBlank()) return UUID.randomUUID().toString();
        try {
            return UUID.fromString(requestedJobId.trim()).toString();
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("Invalid request ID");
        }
    }

    private static String safeBaseName(String value) {
        String normalized = Normalizer.normalize(value, Normalizer.Form.NFKC)
                .replaceAll("[^\\p{L}\\p{N}._ -]", "_")
                .trim().replaceAll("\\s+", "_")
                .replaceAll("_+", "_");
        normalized = normalized.replaceAll("^[.]+|[.]+$", "");
        if (normalized.isBlank()) return "book";
        return normalized.substring(0, Math.min(normalized.length(), 100));
    }

    private static boolean isSafeSegment(String value) {
        return value != null && !value.isBlank() && !value.equals(".") && !value.equals("..")
                && !value.contains("/") && !value.contains("\\");
    }

    private static void requireSafeSegment(String value) {
        if (!isSafeSegment(value)) throw new IllegalArgumentException("Invalid path segment");
    }

    private static void requireUsername(JwtPrincipal principal, String username) {
        if (!principal.username().equals(username)) throw new SecurityException("Access denied");
    }

    private static Path safeResolve(Path base, String child) {
        requireSafeSegment(child);
        Path normalizedBase = base.toAbsolutePath().normalize();
        Path result = normalizedBase.resolve(child).normalize();
        if (!result.startsWith(normalizedBase)) throw new IllegalArgumentException("Unsafe path");
        return result;
    }

    private static Resource existingResource(Path path) {
        if (!Files.isRegularFile(path)) throw new NoSuchElementException("File not found");
        return new FileSystemResource(path);
    }

    private static Path findArtifact(Path directory, String suffix) {
        if (!Files.isDirectory(directory)) return null;
        try (Stream<Path> stream = Files.list(directory)) {
            return stream.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(suffix))
                    .findFirst().orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    private static String stripFolderSuffix(String folder) {
        return folder.replaceFirst("_[0-9a-fA-F]{8}$", "");
    }

    private long lastModified(Path path) {
        try { return Files.getLastModifiedTime(path).toMillis(); }
        catch (IOException e) { return 0; }
    }

    private static void deleteTree(Path root) throws IOException {
        try (Stream<Path> stream = Files.walk(root)) {
            for (Path path : stream.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(path);
        }
    }

    private static void deleteTreeQuietly(Path root) {
        try {
            if (Files.exists(root)) deleteTree(root);
        } catch (IOException error) {
            log.warn("Unable to remove failed upload directory {}", root, error);
        }
    }

    public record UploadAccepted(String job_id, String status, String book_folder, String message) { }
    private record JobTarget(String type, String userUuid, String bookId, String status) { }
    public record JobStatus(String job_id, String type, String book_id, String status, long created_at,
                            Long started_at, Long finished_at, String error,
                            int attempt_count, int max_attempts, Long cancel_requested_at,
                            JobResult result) { }
    public record JobResult(String book_folder, String book_title, String text,
                            String cover, String author, String music_job_id) { }
    public record BookSummary(String title, String folder, String cover_url, String cover_file,
                              String text_file, String author) { }
}
