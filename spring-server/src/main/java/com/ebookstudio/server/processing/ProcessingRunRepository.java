package com.ebookstudio.server.processing;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public class ProcessingRunRepository {
    private final JdbcTemplate jdbc;

    public ProcessingRunRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    public ProcessingRun insert(ProcessingRun run) {
        if (run.id() != null) throw new IllegalArgumentException("New processing run cannot have an id");
        jdbc.update("""
                INSERT INTO processing_runs(
                    book_id, request_id, queue_job_id, process_type, status,
                    attempt_count, max_attempts, model_version, started_at, finished_at,
                    error_code, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, run.bookId(), run.requestId().toString(), run.queueJobId(),
                run.processType().name(), run.status().name(), run.attemptCount(),
                run.maxAttempts(), run.modelVersion(), run.startedAt(), run.finishedAt(),
                run.errorCode(), run.errorMessage(), run.createdAt(), run.updatedAt());
        return findByRequestId(run.requestId()).orElseThrow();
    }

    public Optional<ProcessingRun> findById(long id) {
        return jdbc.query(selectSql() + " WHERE id=?", rs ->
                rs.next() ? Optional.of(map(rs)) : Optional.empty(), id);
    }

    public Optional<ProcessingRun> findByIdForUpdate(long id) {
        return jdbc.query(selectSql() + " WHERE id=? FOR UPDATE", rs ->
                rs.next() ? Optional.of(map(rs)) : Optional.empty(), id);
    }

    public Optional<ProcessingRun> findByRequestId(UUID requestId) {
        return jdbc.query(selectSql() + " WHERE request_id=?", rs ->
                rs.next() ? Optional.of(map(rs)) : Optional.empty(), requestId.toString());
    }

    public Optional<ProcessingRun> findByQueueJobId(String queueJobId) {
        return jdbc.query(selectSql() + " WHERE queue_job_id=?", rs ->
                rs.next() ? Optional.of(map(rs)) : Optional.empty(), queueJobId);
    }

    public List<ProcessingRun> findByBookId(long bookId) {
        return jdbc.query(selectSql() + " WHERE book_id=? ORDER BY created_at DESC, id DESC",
                (rs, rowNum) -> map(rs), bookId);
    }

    public boolean markRunning(long id, String modelVersion, long now) {
        return jdbc.update("""
                UPDATE processing_runs
                SET status='RUNNING', attempt_count=attempt_count+1,
                    started_at=COALESCE(started_at, ?), model_version=COALESCE(?, model_version),
                    error_code=NULL, error_message=NULL, updated_at=?
                WHERE id=? AND status='QUEUED' AND attempt_count < max_attempts
                """, now, modelVersion, now, id) == 1;
    }

    public boolean synchronizeAttemptCount(long id, int attemptCount, long now) {
        if (attemptCount < 0) throw new IllegalArgumentException("attemptCount cannot be negative");
        return jdbc.update("""
                UPDATE processing_runs SET attempt_count=?, updated_at=?
                WHERE id=? AND attempt_count < ? AND ? <= max_attempts
                  AND status IN ('QUEUED', 'RUNNING', 'CANCEL_REQUESTED')
                """, attemptCount, now, id, attemptCount, attemptCount) == 1;
    }

    public boolean requestCancellation(long id, long now) {
        return jdbc.update("""
                UPDATE processing_runs SET status='CANCEL_REQUESTED', updated_at=?
                WHERE id=? AND status IN ('QUEUED', 'RUNNING')
                """, now, id) == 1;
    }

    public boolean markSucceeded(long id, long now) {
        return jdbc.update("""
                UPDATE processing_runs
                SET status='SUCCEEDED', finished_at=?, error_code=NULL, error_message=NULL, updated_at=?
                WHERE id=? AND status='RUNNING'
                """, now, now, id) == 1;
    }

    public boolean markFailed(long id, String errorCode, String errorMessage, long now) {
        return jdbc.update("""
                UPDATE processing_runs
                SET status='FAILED', finished_at=?, error_code=?, error_message=?, updated_at=?
                WHERE id=? AND status IN ('RUNNING', 'CANCEL_REQUESTED')
                """, now, errorCode, errorMessage, now, id) == 1;
    }

    public boolean markCancelled(long id, long now) {
        return jdbc.update("""
                UPDATE processing_runs
                SET status='CANCELLED', finished_at=?, updated_at=?
                WHERE id=? AND status IN ('QUEUED', 'CANCEL_REQUESTED')
                """, now, now, id) == 1;
    }

    private static String selectSql() {
        return """
                SELECT id, book_id, request_id, queue_job_id, process_type, status,
                       attempt_count, max_attempts, model_version, started_at, finished_at,
                       error_code, error_message, created_at, updated_at
                FROM processing_runs
                """;
    }

    private static ProcessingRun map(ResultSet rs) throws SQLException {
        return new ProcessingRun(
                rs.getLong("id"), rs.getLong("book_id"),
                UUID.fromString(rs.getString("request_id")), rs.getString("queue_job_id"),
                ProcessingRun.ProcessType.valueOf(rs.getString("process_type")),
                ProcessingRun.Status.valueOf(rs.getString("status")),
                rs.getInt("attempt_count"), rs.getInt("max_attempts"),
                rs.getString("model_version"), nullableLong(rs, "started_at"),
                nullableLong(rs, "finished_at"), rs.getString("error_code"),
                rs.getString("error_message"), rs.getLong("created_at"),
                rs.getLong("updated_at"));
    }

    private static Long nullableLong(ResultSet rs, String column) throws SQLException {
        long value = rs.getLong(column);
        return rs.wasNull() ? null : value;
    }
}
