package com.ebookstudio.server.processing;

import com.ebookstudio.server.auth.JwtPrincipal;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Locale;
import java.util.NoSuchElementException;

@Service
public class BookInsightsService {
    private final JdbcTemplate jdbc;
    private final ProcessingRunRepository runs;
    private final BookArtifactRepository artifacts;
    private final BookMusicBindingRepository bindings;

    public BookInsightsService(JdbcTemplate jdbc,
                               ProcessingRunRepository runs,
                               BookArtifactRepository artifacts,
                               BookMusicBindingRepository bindings) {
        this.jdbc = jdbc;
        this.runs = runs;
        this.artifacts = artifacts;
        this.bindings = bindings;
    }

    @Transactional(readOnly = true)
    public ProcessingHistoryResponse processingHistory(JwtPrincipal principal, String bookFolder) {
        OwnedBook book = findOwnedBook(principal, bookFolder);
        List<ProcessingRunResponse> history = runs.findByBookId(book.id()).stream()
                .map(run -> new ProcessingRunResponse(
                        run.id(), run.requestId().toString(), apiName(run.processType()),
                        apiName(run.status()), run.attemptCount(), run.maxAttempts(),
                        run.modelVersion(), run.startedAt(), run.finishedAt(),
                        run.errorCode(), run.createdAt(), run.updatedAt(),
                        artifacts.findByProcessingRunId(run.id()).stream()
                                .map(this::artifactResponse).toList()))
                .toList();
        return new ProcessingHistoryResponse(book.folder(), apiName(book.status()), history);
    }

    @Transactional(readOnly = true)
    public MusicTracksResponse musicTracks(JwtPrincipal principal, String bookFolder) {
        OwnedBook book = findOwnedBook(principal, bookFolder);
        List<BookMusicTrack> tracks = bindings.findTracksByBookId(book.id());
        List<MusicTrackResponse> responseTracks = tracks.stream().map(track -> {
            BookMusicBinding binding = track.binding();
            MusicAsset asset = track.asset();
            return new MusicTrackResponse(
                    binding.segmentKey(), apiName(binding.bindingType()), binding.processingRunId(),
                    asset.signature(), apiName(asset.assetSource()), asset.genre(), asset.bpm(),
                    asset.modelName(), asset.modelVersion(), fileName(asset.storageKey()),
                    asset.durationSeconds(), apiName(asset.status()), asset.reuseCount(),
                    asset.lastUsedAt());
        }).toList();
        long uniqueAssetCount = tracks.stream().map(track -> track.asset().id()).distinct().count();
        return new MusicTracksResponse(book.folder(), responseTracks.size(), uniqueAssetCount,
                responseTracks);
    }

    private OwnedBook findOwnedBook(JwtPrincipal principal, String bookFolder) {
        if (principal == null) throw new SecurityException("Authentication is required");
        if (bookFolder == null || bookFolder.isBlank() || bookFolder.length() > 160) {
            throw new IllegalArgumentException("Invalid book folder");
        }
        OwnedBook book = jdbc.query("""
                SELECT id, folder, status
                FROM books
                WHERE owner_public_id=? AND folder=?
                """, rs -> rs.next()
                ? new OwnedBook(rs.getLong("id"), rs.getString("folder"), rs.getString("status"))
                : null, principal.publicId(), bookFolder.trim());
        if (book == null) throw new NoSuchElementException("Book not found");
        return book;
    }

    private BookArtifactResponse artifactResponse(BookArtifact artifact) {
        return new BookArtifactResponse(apiName(artifact.artifactType()), artifact.fileName(),
                artifact.checksum(), artifact.fileSize(), artifact.version(), artifact.createdAt());
    }

    private static String apiName(Enum<?> value) {
        return value.name().toLowerCase(Locale.ROOT);
    }

    private static String apiName(String value) {
        return value.toLowerCase(Locale.ROOT);
    }

    private static String fileName(String storageKey) {
        if (storageKey == null || storageKey.isBlank()) return null;
        String normalized = storageKey.replace('\\', '/');
        int separator = normalized.lastIndexOf('/');
        return separator < 0 ? normalized : normalized.substring(separator + 1);
    }

    private record OwnedBook(long id, String folder, String status) { }

    public record ProcessingHistoryResponse(
            @JsonProperty("book_folder") String bookFolder,
            @JsonProperty("book_status") String bookStatus,
            @JsonProperty("runs") List<ProcessingRunResponse> runs) { }

    public record ProcessingRunResponse(
            @JsonProperty("run_id") long runId,
            @JsonProperty("request_id") String requestId,
            @JsonProperty("process_type") String processType,
            @JsonProperty("status") String status,
            @JsonProperty("attempt_count") int attemptCount,
            @JsonProperty("max_attempts") int maxAttempts,
            @JsonProperty("model_version") String modelVersion,
            @JsonProperty("started_at") Long startedAt,
            @JsonProperty("finished_at") Long finishedAt,
            @JsonProperty("error_code") String errorCode,
            @JsonProperty("created_at") long createdAt,
            @JsonProperty("updated_at") long updatedAt,
            @JsonProperty("artifacts") List<BookArtifactResponse> artifacts) { }

    public record BookArtifactResponse(
            @JsonProperty("artifact_type") String artifactType,
            @JsonProperty("file_name") String fileName,
            @JsonProperty("checksum") String checksum,
            @JsonProperty("file_size") Long fileSize,
            @JsonProperty("version") int version,
            @JsonProperty("created_at") long createdAt) { }

    public record MusicTracksResponse(
            @JsonProperty("book_folder") String bookFolder,
            @JsonProperty("track_count") int trackCount,
            @JsonProperty("unique_asset_count") long uniqueAssetCount,
            @JsonProperty("tracks") List<MusicTrackResponse> tracks) { }

    public record MusicTrackResponse(
            @JsonProperty("segment_key") String segmentKey,
            @JsonProperty("binding_type") String bindingType,
            @JsonProperty("processing_run_id") Long processingRunId,
            @JsonProperty("signature") String signature,
            @JsonProperty("asset_source") String assetSource,
            @JsonProperty("genre") String genre,
            @JsonProperty("bpm") int bpm,
            @JsonProperty("model_name") String modelName,
            @JsonProperty("model_version") String modelVersion,
            @JsonProperty("file_name") String fileName,
            @JsonProperty("duration_seconds") Integer durationSeconds,
            @JsonProperty("status") String status,
            @JsonProperty("reuse_count") int reuseCount,
            @JsonProperty("last_used_at") Long lastUsedAt) { }
}
