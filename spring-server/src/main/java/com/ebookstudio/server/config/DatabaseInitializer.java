package com.ebookstudio.server.config;

import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Set;

@Component
public class DatabaseInitializer {
    private final JdbcTemplate queueJdbc;

    public DatabaseInitializer(@Qualifier("queueJdbcTemplate") JdbcTemplate queueJdbc) {
        this.queueJdbc = queueJdbc;
    }

    @PostConstruct
    public void initialize() {
        queueJdbc.execute("PRAGMA journal_mode=WAL");
        queueJdbc.execute("PRAGMA busy_timeout=10000");
        queueJdbc.execute("PRAGMA foreign_keys=ON");
        queueJdbc.execute("""
                CREATE TABLE IF NOT EXISTS worker_nodes (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at INTEGER NOT NULL,
                    heartbeat_at INTEGER NOT NULL
                )
                """);
        queueJdbc.execute("""
                CREATE TABLE IF NOT EXISTS music_prompt_cache (
                    signature TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    bpm INTEGER NOT NULL,
                    keywords_json TEXT NOT NULL,
                    target_duration_sec INTEGER NOT NULL,
                    segment_duration_sec INTEGER NOT NULL,
                    generator_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    relative_path TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    generated_at INTEGER,
                    last_used_at INTEGER,
                    reuse_count INTEGER NOT NULL DEFAULT 0,
                    owner_job_id TEXT,
                    error TEXT
                )
                """);
        queueJdbc.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    user_uuid TEXT NOT NULL,
                    book_id TEXT,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at INTEGER NOT NULL,
                    started_at INTEGER,
                    finished_at INTEGER,
                    error TEXT,
                    json_path TEXT,
                    music_folder TEXT,
                    web_path_prefix TEXT,
                    pdf_path TEXT,
                    book_root_folder TEXT,
                    output_json TEXT,
                    cover_file TEXT,
                    book_title TEXT,
                    author TEXT,
                    parent_job_id TEXT,
                    music_job_id TEXT,
                    worker_id TEXT,
                    heartbeat_at INTEGER,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    available_at INTEGER,
                    cancel_requested_at INTEGER
                )
                """);

        addMissingColumn("jobs", "output_json", "TEXT");
        addMissingColumn("jobs", "cover_file", "TEXT");
        addMissingColumn("jobs", "book_title", "TEXT");
        addMissingColumn("jobs", "author", "TEXT");
        addMissingColumn("jobs", "parent_job_id", "TEXT");
        addMissingColumn("jobs", "music_job_id", "TEXT");
        addMissingColumn("jobs", "worker_id", "TEXT");
        addMissingColumn("jobs", "heartbeat_at", "INTEGER");
        addMissingColumn("jobs", "attempt_count", "INTEGER NOT NULL DEFAULT 0");
        addMissingColumn("jobs", "max_attempts", "INTEGER NOT NULL DEFAULT 3");
        addMissingColumn("jobs", "available_at", "INTEGER");
        addMissingColumn("jobs", "cancel_requested_at", "INTEGER");

        queueJdbc.execute("CREATE INDEX IF NOT EXISTS idx_music_prompt_cache_status_updated "
                + "ON music_prompt_cache(status, updated_at)");
        queueJdbc.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_type_created "
                + "ON jobs(status, type, created_at)");
        queueJdbc.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_uuid ON jobs(user_uuid)");
        queueJdbc.execute("CREATE INDEX IF NOT EXISTS idx_jobs_parent_job_id ON jobs(parent_job_id)");
    }

    private void addMissingColumn(String table, String name, String type) {
        Set<String> columns = new HashSet<>(queueJdbc.query("PRAGMA table_info(" + table + ")",
                (rs, rowNum) -> rs.getString("name")));
        if (!columns.contains(name)) {
            queueJdbc.execute("ALTER TABLE " + table + " ADD COLUMN " + name + " " + type);
        }
    }
}
