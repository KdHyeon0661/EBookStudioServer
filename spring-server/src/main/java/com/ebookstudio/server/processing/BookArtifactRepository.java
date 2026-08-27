package com.ebookstudio.server.processing;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;

@Repository
public class BookArtifactRepository {
    private final JdbcTemplate jdbc;

    public BookArtifactRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public BookArtifact insert(BookArtifact artifact) {
        if (artifact.id() != null) throw new IllegalArgumentException("New artifact cannot have an id");
        jdbc.update("""
                INSERT INTO book_artifacts(processing_run_id, artifact_type, storage_key,
                                           file_name, checksum, file_size, version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, artifact.processingRunId(), artifact.artifactType().name(),
                artifact.storageKey(), artifact.fileName(), artifact.checksum(),
                artifact.fileSize(), artifact.version(), artifact.createdAt());
        return jdbc.query(selectSql() + " WHERE processing_run_id=? AND artifact_type=? AND version=?",
                rs -> {
                    if (!rs.next()) throw new IllegalStateException("Inserted artifact was not found");
                    return map(rs);
                }, artifact.processingRunId(), artifact.artifactType().name(), artifact.version());
    }

    public Optional<BookArtifact> findLatest(long processingRunId,
                                                    BookArtifact.ArtifactType artifactType) {
        return jdbc.query(selectSql() + """
                 WHERE processing_run_id=? AND artifact_type=?
                 ORDER BY version DESC LIMIT 1
                """, rs -> rs.next() ? Optional.of(map(rs)) : Optional.empty(),
                processingRunId, artifactType.name());
    }

    public List<BookArtifact> findByProcessingRunId(long processingRunId) {
        return jdbc.query(selectSql() +
                        " WHERE processing_run_id=? ORDER BY artifact_type, version",
                (rs, rowNum) -> map(rs), processingRunId);
    }

    public List<BookArtifact> findByBookId(long bookId) {
        return jdbc.query("""
                SELECT a.id, a.processing_run_id, a.artifact_type, a.storage_key,
                       a.file_name, a.checksum, a.file_size, a.version, a.created_at
                FROM book_artifacts a
                JOIN processing_runs r ON r.id=a.processing_run_id
                WHERE r.book_id=?
                ORDER BY r.created_at DESC, a.artifact_type, a.version
                """, (rs, rowNum) -> map(rs), bookId);
    }

    private static String selectSql() {
        return """
                SELECT id, processing_run_id, artifact_type, storage_key,
                       file_name, checksum, file_size, version, created_at
                FROM book_artifacts
                """;
    }

    private static BookArtifact map(ResultSet rs) throws SQLException {
        long fileSize = rs.getLong("file_size");
        Long nullableFileSize = rs.wasNull() ? null : fileSize;
        return new BookArtifact(rs.getLong("id"), rs.getLong("processing_run_id"),
                BookArtifact.ArtifactType.valueOf(rs.getString("artifact_type")),
                rs.getString("storage_key"), rs.getString("file_name"),
                rs.getString("checksum"), nullableFileSize,
                rs.getInt("version"), rs.getLong("created_at"));
    }
}
