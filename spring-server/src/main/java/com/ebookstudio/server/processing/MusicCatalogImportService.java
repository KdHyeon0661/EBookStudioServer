package com.ebookstudio.server.processing;

import com.ebookstudio.server.config.EBookStudioProperties;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

@Service
public class MusicCatalogImportService {
    private static final Pattern GENERATED_FILE = Pattern.compile("^([0-9a-f]{64})\\.wav$");

    private final JdbcTemplate queueJdbc;
    private final ProcessingRunRepository runs;
    private final BookArtifactRepository artifacts;
    private final MusicAssetRepository musicAssets;
    private final BookMusicBindingRepository bindings;
    private final ObjectMapper objectMapper;
    private final Path storageRoot;
    private final Path musicRoot;

    public MusicCatalogImportService(
            @Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc,
            ProcessingRunRepository runs,
            BookArtifactRepository artifacts,
            MusicAssetRepository musicAssets,
            BookMusicBindingRepository bindings,
            ObjectMapper objectMapper,
            EBookStudioProperties properties) {
        this.queueJdbc = queueJdbc;
        this.runs = runs;
        this.artifacts = artifacts;
        this.musicAssets = musicAssets;
        this.bindings = bindings;
        this.objectMapper = objectMapper;
        this.storageRoot = Path.of(properties.storageRoot()).toAbsolutePath().normalize();
        this.musicRoot = storageRoot.resolve("defaults").resolve("music").normalize();
    }

    @Transactional
    public ProcessingRun completeMusicRun(long runId, long bookId,
                                          String queueJobId, long finishedAt) {
        ProcessingRun run = runs.findByIdForUpdate(runId)
                .orElseThrow(() -> new IllegalArgumentException("Music processing run not found"));
        if (run.processType() != ProcessingRun.ProcessType.MUSIC_GENERATION) {
            throw new IllegalArgumentException("Run is not a music generation job");
        }
        if (run.status() == ProcessingRun.Status.SUCCEEDED) return run;
        if (run.status() != ProcessingRun.Status.RUNNING) {
            throw new IllegalStateException("Music run is not ready for completion");
        }

        QueueMusicPaths paths = queueJdbc.query("""
                SELECT json_path, music_folder FROM jobs WHERE id=? AND type='music_generation'
                """, rs -> {
            if (!rs.next()) throw new IllegalStateException("Music queue job was not found");
            return new QueueMusicPaths(rs.getString("json_path"), rs.getString("music_folder"));
        }, queueJobId);
        Path jsonPath = requireStoredFile(paths.jsonPath(), "music book JSON");
        Path configuredMusicRoot = requireStoredDirectory(paths.musicFolder(), "music folder");
        if (!configuredMusicRoot.equals(musicRoot)) {
            throw new IllegalStateException("Music job uses an unexpected shared folder");
        }
        Path musicIndex = requireStoredFile(
                storageRoot.resolve("defaults").resolve("music_index.json").toString(),
                "music index");

        JsonNode book = readObject(jsonPath, "Music book JSON");
        JsonNode index = readObject(musicIndex, "Music index");
        List<SegmentMusic> segments = readSegments(book);
        if (segments.isEmpty()) {
            throw new IllegalStateException("Music book JSON has no mapped segments");
        }

        bindings.deleteByBookId(bookId);
        for (SegmentMusic segment : segments) {
            ResolvedAsset resolved = resolveAsset(segment, index, finishedAt);
            MusicAsset saved = musicAssets.saveSnapshot(resolved.asset());
            bindings.save(new BookMusicBinding(bookId, segment.segmentKey(), saved.id(),
                    runId, resolved.bindingType(), finishedAt, finishedAt));
        }
        saveArtifact(runId, BookArtifact.ArtifactType.BOOK_JSON, jsonPath, finishedAt);
        saveArtifact(runId, BookArtifact.ArtifactType.MUSIC_INDEX, musicIndex, finishedAt);

        if (!runs.markSucceeded(runId, finishedAt)) {
            throw new IllegalStateException("Music status changed concurrently");
        }
        return runs.findById(runId).orElseThrow();
    }

    private ResolvedAsset resolveAsset(SegmentMusic segment, JsonNode musicIndex, long now) {
        Matcher generated = GENERATED_FILE.matcher(segment.fileName().toLowerCase(Locale.ROOT));
        if (generated.matches()) {
            MusicCacheRow cache = findReadyCache(generated.group(1));
            Path assetPath = requireMusicFile(cache.relativePath());
            ModelIdentity model = modelIdentity(cache.generatorVersion());
            MusicAsset asset = new MusicAsset(null, cache.signature(),
                    MusicAsset.AssetSource.AI_GENERATED, cache.prompt(), cache.genre(),
                    normalizeBpm(cache.bpm()), cache.keywordsJson(), model.name(), model.version(),
                    "{\"target_duration_sec\":" + cache.targetDurationSec()
                            + ",\"segment_duration_sec\":" + cache.segmentDurationSec() + "}",
                    storageKey(assetPath), checksum(assetPath), cache.targetDurationSec(),
                    MusicAsset.Status.READY, cache.reuseCount(), cache.createdAt(),
                    Math.max(cache.updatedAt(), now), cache.lastUsedAt());
            BookMusicBinding.BindingType type = "ai_reused".equals(segment.source())
                    ? BookMusicBinding.BindingType.CACHE_REUSED
                    : BookMusicBinding.BindingType.GENERATED;
            return new ResolvedAsset(asset, type);
        }

        Path assetPath = findDefaultFile(segment.fileName());
        JsonNode metadata = findIndexEntry(musicIndex, segment.fileName());
        int bpm = normalizeBpm(metadata == null ? segment.bpm() : metadata.path("bpm").asInt(segment.bpm()));
        String genre = metadata == null ? "default" : metadata.path("genre").asString("default");
        String prompt = metadata == null ? "Bundled background music: " + segment.fileName()
                : metadata.path("prompt").asString("Bundled background music: " + segment.fileName());
        String key = storageKey(assetPath);
        String signature = sha256Text("default\u0000" + key);
        MusicAsset asset = new MusicAsset(null, signature, MusicAsset.AssetSource.DEFAULT,
                prompt, genre, bpm, "[]", "ebookstudio/default-library", "1", "{}",
                key, checksum(assetPath), null, MusicAsset.Status.READY, 0,
                now, now, now);
        return new ResolvedAsset(asset, BookMusicBinding.BindingType.DEFAULT);
    }

    private List<SegmentMusic> readSegments(JsonNode book) {
        JsonNode chapters = book.path("chapters");
        if (!chapters.isArray()) throw new IllegalStateException("Music book JSON has no chapters");
        List<SegmentMusic> result = new ArrayList<>();
        int chapterNumber = 0;
        for (JsonNode chapter : chapters) {
            chapterNumber++;
            int segmentNumber = 0;
            for (JsonNode segment : chapter.path("segments")) {
                segmentNumber++;
                String filename = segment.path("music_filename").asString("").trim();
                if (filename.isEmpty()) continue;
                requireSafeFileName(filename);
                result.add(new SegmentMusic("chapter-" + chapterNumber + ":segment-" + segmentNumber,
                        filename, segment.path("music_source").asString(""),
                        segment.path("bpm").asInt(80)));
            }
        }
        return result;
    }

    private MusicCacheRow findReadyCache(String signature) {
        return queueJdbc.query("""
                SELECT signature, prompt, genre, bpm, keywords_json,
                       target_duration_sec, segment_duration_sec, generator_version,
                       relative_path, reuse_count, created_at, updated_at, last_used_at
                FROM music_prompt_cache WHERE signature=? AND status='ready'
                """, rs -> {
            if (!rs.next()) throw new IllegalStateException("Ready prompt cache entry is missing");
            return mapCache(rs);
        }, signature);
    }

    private void saveArtifact(long runId, BookArtifact.ArtifactType type,
                              Path path, long createdAt) {
        if (artifacts.findLatest(runId, type).isPresent()) return;
        artifacts.insert(new BookArtifact(null, runId, type, storageKey(path),
                path.getFileName().toString(), checksum(path), fileSize(path), 1, createdAt));
    }

    private Path requireMusicFile(String relativePath) {
        if (relativePath == null || relativePath.isBlank()) {
            throw new IllegalStateException("Ready music cache has no file path");
        }
        Path resolved = musicRoot.resolve(relativePath).normalize();
        if (!resolved.startsWith(musicRoot) || !Files.isRegularFile(resolved)) {
            throw new IllegalStateException("Ready music cache file is missing");
        }
        return resolved;
    }

    private Path findDefaultFile(String filename) {
        requireSafeFileName(filename);
        try (Stream<Path> files = Files.walk(musicRoot, 3)) {
            return files.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().equals(filename))
                    .findFirst()
                    .orElseThrow(() -> new IllegalStateException("Default music file is missing"));
        } catch (IOException error) {
            throw new IllegalStateException("Unable to inspect default music folder", error);
        }
    }

    private static JsonNode findIndexEntry(JsonNode index, String filename) {
        for (JsonNode info : index) {
            if (filename.equals(info.path("filename").asString(""))) return info;
        }
        return null;
    }

    private JsonNode readObject(Path path, String label) {
        try {
            JsonNode root = objectMapper.readTree(path.toFile());
            if (root == null || !root.isObject()) {
                throw new IllegalStateException(label + " has an invalid structure");
            }
            return root;
        } catch (IllegalStateException validationError) {
            throw validationError;
        } catch (Exception error) {
            throw new IllegalStateException(label + " is not readable", error);
        }
    }

    private Path requireStoredFile(String value, String label) {
        Path path = requireStoredPath(value, label);
        if (!Files.isRegularFile(path)) throw new IllegalStateException(label + " is missing");
        return path;
    }

    private Path requireStoredDirectory(String value, String label) {
        Path path = requireStoredPath(value, label);
        if (!Files.isDirectory(path)) throw new IllegalStateException(label + " is missing");
        return path;
    }

    private Path requireStoredPath(String value, String label) {
        if (value == null || value.isBlank()) throw new IllegalStateException(label + " path is missing");
        Path path = Path.of(value).toAbsolutePath().normalize();
        if (!path.startsWith(storageRoot)) throw new IllegalStateException(label + " is outside storage");
        return path;
    }

    private String storageKey(Path path) {
        Path normalized = path.toAbsolutePath().normalize();
        if (!normalized.startsWith(storageRoot)) {
            throw new IllegalStateException("Music artifact is outside storage");
        }
        return storageRoot.relativize(normalized).toString().replace('\\', '/');
    }

    private static void requireSafeFileName(String filename) {
        Path path = Path.of(filename);
        if (path.isAbsolute() || path.getNameCount() != 1
                || filename.contains("/") || filename.contains("\\")) {
            throw new IllegalStateException("Music filename is unsafe");
        }
    }

    private static int normalizeBpm(int bpm) {
        return bpm < 20 || bpm > 300 ? 80 : bpm;
    }

    private static ModelIdentity modelIdentity(String generatorVersion) {
        String value = generatorVersion == null || generatorVersion.isBlank()
                ? "facebook/musicgen-small:unknown" : generatorVersion.trim();
        int separator = value.lastIndexOf(':');
        return separator > 0 && separator < value.length() - 1
                ? new ModelIdentity(value.substring(0, separator), value.substring(separator + 1))
                : new ModelIdentity(value, "unknown");
    }

    private static MusicCacheRow mapCache(ResultSet rs) throws SQLException {
        long lastUsed = rs.getLong("last_used_at");
        Long nullableLastUsed = rs.wasNull() ? null : lastUsed;
        return new MusicCacheRow(rs.getString("signature"), rs.getString("prompt"),
                rs.getString("genre"), rs.getInt("bpm"), rs.getString("keywords_json"),
                rs.getInt("target_duration_sec"), rs.getInt("segment_duration_sec"),
                rs.getString("generator_version"), rs.getString("relative_path"),
                rs.getInt("reuse_count"), rs.getLong("created_at"),
                rs.getLong("updated_at"), nullableLastUsed);
    }

    private static String checksum(Path path) {
        try (InputStream input = Files.newInputStream(path)) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[8192];
            for (int read; (read = input.read(buffer)) != -1; ) digest.update(buffer, 0, read);
            return HexFormat.of().formatHex(digest.digest());
        } catch (IOException error) {
            throw new IllegalStateException("Unable to checksum music artifact", error);
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static String sha256Text(String value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static long fileSize(Path path) {
        try {
            return Files.size(path);
        } catch (IOException error) {
            throw new IllegalStateException("Unable to read music artifact size", error);
        }
    }

    private record QueueMusicPaths(String jsonPath, String musicFolder) {
    }
    private record SegmentMusic(String segmentKey, String fileName, String source, int bpm) {
    }
    private record ResolvedAsset(MusicAsset asset, BookMusicBinding.BindingType bindingType) {
    }
    private record ModelIdentity(String name, String version) {
    }
    private record MusicCacheRow(
            String signature, String prompt, String genre, int bpm, String keywordsJson,
            int targetDurationSec, int segmentDurationSec, String generatorVersion,
            String relativePath, int reuseCount, long createdAt, long updatedAt,
            Long lastUsedAt) {
    }
}
