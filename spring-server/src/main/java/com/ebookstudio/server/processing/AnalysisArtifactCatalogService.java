package com.ebookstudio.server.processing;

import com.ebookstudio.server.config.EBookStudioProperties;
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
import java.util.HexFormat;
import java.util.List;

@Service
public class AnalysisArtifactCatalogService {
    private static final byte[] PNG_SIGNATURE = {
            (byte) 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a
    };

    private final ProcessingRunRepository runs;
    private final BookArtifactRepository artifacts;
    private final ObjectMapper objectMapper;
    private final Path storageRoot;

    public AnalysisArtifactCatalogService(ProcessingRunRepository runs,
                                          BookArtifactRepository artifacts,
                                          ObjectMapper objectMapper,
                                          EBookStudioProperties properties) {
        this.runs = runs;
        this.artifacts = artifacts;
        this.objectMapper = objectMapper;
        this.storageRoot = Path.of(properties.storageRoot()).toAbsolutePath().normalize();
    }

    @Transactional
    public ProcessingRun completeAnalysis(long runId, Path bookRoot,
                                          String outputJsonFile, String coverFile,
                                          long finishedAt) {
        ProcessingRun run = runs.findByIdForUpdate(runId)
                .orElseThrow(() -> new IllegalArgumentException("Processing run not found"));
        if (run.processType() != ProcessingRun.ProcessType.ANALYZE) {
            throw new IllegalArgumentException("Artifacts do not belong to an analysis run");
        }
        if (run.status() == ProcessingRun.Status.SUCCEEDED && hasCompleteArtifactSet(runId)) {
            return run;
        }
        if (run.status() != ProcessingRun.Status.RUNNING
                && run.status() != ProcessingRun.Status.SUCCEEDED) {
            throw new IllegalStateException("Analysis run is not ready for completion");
        }

        Path normalizedBookRoot = requireWithinStorage(bookRoot);
        Path bookJson = requireChild(normalizedBookRoot, outputJsonFile, "book JSON");
        Path coverImage = requireChild(normalizedBookRoot, coverFile, "cover image");
        Path musicIndex = requireRegularFile(
                storageRoot.resolve("defaults").resolve("music_index.json"), "music index");

        validateBookJson(bookJson);
        validateCoverImage(coverImage);
        validateMusicIndex(musicIndex);

        List<ArtifactFile> inspected = List.of(
                inspect(BookArtifact.ArtifactType.BOOK_JSON, bookJson),
                inspect(BookArtifact.ArtifactType.COVER_IMAGE, coverImage),
                inspect(BookArtifact.ArtifactType.MUSIC_INDEX, musicIndex));
        for (ArtifactFile file : inspected) saveVersion(runId, file, finishedAt);

        if (run.status() == ProcessingRun.Status.RUNNING
                && !runs.markSucceeded(runId, finishedAt)) {
            throw new IllegalStateException("Analysis status changed concurrently");
        }
        return runs.findById(runId).orElseThrow();
    }

    private boolean hasCompleteArtifactSet(long runId) {
        return artifacts.findLatest(runId, BookArtifact.ArtifactType.BOOK_JSON).isPresent()
                && artifacts.findLatest(runId, BookArtifact.ArtifactType.COVER_IMAGE).isPresent()
                && artifacts.findLatest(runId, BookArtifact.ArtifactType.MUSIC_INDEX).isPresent();
    }

    private void saveVersion(long runId, ArtifactFile file, long createdAt) {
        BookArtifact latest = artifacts.findLatest(runId, file.type()).orElse(null);
        if (latest != null
                && latest.storageKey().equals(file.storageKey())
                && java.util.Objects.equals(latest.checksum(), file.checksum())
                && java.util.Objects.equals(latest.fileSize(), file.fileSize())) {
            return;
        }
        int version = latest == null ? 1 : latest.version() + 1;
        artifacts.insert(new BookArtifact(null, runId, file.type(), file.storageKey(),
                file.fileName(), file.checksum(), file.fileSize(), version, createdAt));
    }

    private ArtifactFile inspect(BookArtifact.ArtifactType type, Path path) {
        try {
            return new ArtifactFile(type, storageKey(path), path.getFileName().toString(),
                    sha256(path), Files.size(path));
        } catch (IOException error) {
            throw new IllegalStateException("Unable to inspect analysis artifact", error);
        }
    }

    private void validateBookJson(Path path) {
        try {
            JsonNode root = objectMapper.readTree(path.toFile());
            if (root == null || !root.isObject()
                    || !root.path("book_info").isObject()
                    || !root.path("chapters").isArray()) {
                throw new IllegalStateException("Book JSON has an invalid structure");
            }
        } catch (IllegalStateException validationError) {
            throw validationError;
        } catch (Exception error) {
            throw new IllegalStateException("Book JSON is not readable", error);
        }
    }

    private void validateMusicIndex(Path path) {
        try {
            JsonNode root = objectMapper.readTree(path.toFile());
            if (root == null || !root.isObject()) {
                throw new IllegalStateException("Music index has an invalid structure");
            }
        } catch (IllegalStateException validationError) {
            throw validationError;
        } catch (Exception error) {
            throw new IllegalStateException("Music index is not readable", error);
        }
    }

    private static void validateCoverImage(Path path) {
        try (InputStream input = Files.newInputStream(path)) {
            byte[] header = input.readNBytes(PNG_SIGNATURE.length);
            boolean png = java.util.Arrays.equals(header, PNG_SIGNATURE);
            boolean jpeg = header.length >= 3 && (header[0] & 0xff) == 0xff
                    && (header[1] & 0xff) == 0xd8 && (header[2] & 0xff) == 0xff;
            if (!png && !jpeg) throw new IllegalStateException("Cover image has an invalid format");
        } catch (IOException error) {
            throw new IllegalStateException("Cover image is not readable", error);
        }
    }

    private Path requireChild(Path parent, String fileName, String label) {
        if (fileName == null || fileName.isBlank()) {
            throw new IllegalStateException("Analysis did not report the " + label);
        }
        Path name = Path.of(fileName);
        if (name.isAbsolute() || name.getNameCount() != 1
                || fileName.contains("/") || fileName.contains("\\")) {
            throw new IllegalStateException("Analysis reported an unsafe " + label + " path");
        }
        return requireRegularFile(parent.resolve(name), label);
    }

    private Path requireRegularFile(Path path, String label) {
        Path normalized = requireWithinStorage(path);
        if (!Files.isRegularFile(normalized)) {
            throw new IllegalStateException("Required " + label + " artifact is missing");
        }
        return normalized;
    }

    private Path requireWithinStorage(Path path) {
        if (path == null) throw new IllegalStateException("Artifact storage path is missing");
        Path normalized = path.toAbsolutePath().normalize();
        if (!normalized.startsWith(storageRoot)) {
            throw new IllegalStateException("Artifact path is outside the storage root");
        }
        return normalized;
    }

    private String storageKey(Path path) {
        return storageRoot.relativize(requireWithinStorage(path)).toString().replace('\\', '/');
    }

    private static String sha256(Path path) throws IOException {
        try (InputStream input = Files.newInputStream(path)) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[8192];
            for (int read; (read = input.read(buffer)) != -1; ) {
                digest.update(buffer, 0, read);
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private record ArtifactFile(BookArtifact.ArtifactType type, String storageKey,
                                String fileName, String checksum, long fileSize) {
    }
}
