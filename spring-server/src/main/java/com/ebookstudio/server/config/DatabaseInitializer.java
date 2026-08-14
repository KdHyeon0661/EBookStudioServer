package com.ebookstudio.server.config;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Set;

@Component
public class DatabaseInitializer implements ApplicationRunner {
    private final JdbcTemplate jdbc;

    public DatabaseInitializer(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public void run(ApplicationArguments args) {
        jdbc.execute("PRAGMA journal_mode=WAL");
        jdbc.execute("PRAGMA busy_timeout=10000");

        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    public_id TEXT NOT NULL UNIQUE,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    auth_version INTEGER NOT NULL DEFAULT 0
                )
                """);
        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS token_blocklist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    jti TEXT NOT NULL UNIQUE,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER
                )
                """);
        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS verification_codes (
                    email TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    purpose TEXT NOT NULL DEFAULT 'register',
                    last_sent_at INTEGER NOT NULL DEFAULT 0,
                    failed_attempts INTEGER NOT NULL DEFAULT 0
                )
                """);
        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS request_rate_limits (
                    key_hash TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    window_started_at INTEGER NOT NULL,
                    request_count INTEGER NOT NULL
                )
                """);
        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS worker_nodes (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at INTEGER NOT NULL,
                    heartbeat_at INTEGER NOT NULL
                )
                """);
        jdbc.execute("""
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
        jdbc.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    user_uuid TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    book_id TEXT,
                    occurred_at INTEGER NOT NULL,
                    duration_seconds INTEGER NOT NULL,
                    page_turns INTEGER NOT NULL DEFAULT 0,
                    progress_percent INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY(user_uuid, event_id)
                )
                """);
        jdbc.execute("""
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

        addMissingColumn("token_blocklist", "expires_at", "INTEGER");
        addMissingColumn("users", "auth_version", "INTEGER NOT NULL DEFAULT 0");
        addMissingColumn("verification_codes", "purpose", "TEXT NOT NULL DEFAULT 'register'");
        addMissingColumn("verification_codes", "last_sent_at", "INTEGER NOT NULL DEFAULT 0");
        addMissingColumn("verification_codes", "failed_attempts", "INTEGER NOT NULL DEFAULT 0");
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

        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON request_rate_limits(window_started_at)");
        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_music_prompt_cache_status_updated "
                + "ON music_prompt_cache(status, updated_at)");
        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_user_occurred "
                + "ON usage_events(user_uuid, occurred_at)");
        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_type_occurred "
                + "ON usage_events(event_type, occurred_at)");
        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_type_created ON jobs(status, type, created_at)");
        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user_uuid ON jobs(user_uuid)");
        jdbc.execute("CREATE INDEX IF NOT EXISTS idx_jobs_parent_job_id ON jobs(parent_job_id)");
        migrateLegacyUsers();
    }

    private void addMissingColumn(String table, String name, String type) {
        Set<String> columns = new HashSet<>(jdbc.query("PRAGMA table_info(" + table + ")",
                (rs, rowNum) -> rs.getString("name")));
        if (!columns.contains(name)) {
            jdbc.execute("ALTER TABLE " + table + " ADD COLUMN " + name + " " + type);
        }
    }

    private void migrateLegacyUsers() {
        Integer legacyExists = jdbc.queryForObject(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='user'", Integer.class);
        if (legacyExists != null && legacyExists > 0) {
            jdbc.update("""
                    INSERT OR IGNORE INTO users(public_id, username, email, password_hash)
                    SELECT public_id, username, email, password_hash FROM user
                    """);
        }
    }
}