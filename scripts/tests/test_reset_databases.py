from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "reset-databases.py"
SPEC = importlib.util.spec_from_file_location("database_reset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RESET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESET)


def create_queue(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE jobs(
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            started_at INTEGER,
            heartbeat_at INTEGER
        );
        CREATE TABLE music_prompt_cache(signature TEXT PRIMARY KEY);
        CREATE TABLE worker_nodes(
            id TEXT PRIMARY KEY,
            heartbeat_at INTEGER NOT NULL
        );
        CREATE TABLE account_cleanup_tasks(
            user_uuid TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL
        );
        """
    )
    connection.commit()
    return connection


class FakeCursor:
    def __init__(self, owner: "FakePostgres") -> None:
        self.owner = owner

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def execute(self, query: str) -> None:
        self.owner.queries.append(query)
        if self.owner.fail:
            raise RuntimeError("postgres failure")


class FakePostgres:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class ResetSafetyTests(unittest.TestCase):
    def test_execute_requires_exact_confirmation_phrase(self) -> None:
        RESET.validate_execute_request(False, None)
        with self.assertRaises(RESET.SafetyError):
            RESET.validate_execute_request(True, "yes")
        RESET.validate_execute_request(True, RESET.CONFIRMATION_PHRASE)

    def test_other_postgres_sessions_block_execution(self) -> None:
        with self.assertRaises(RESET.SafetyError):
            RESET.require_idle_postgres({"other_client_sessions": 1})
        RESET.require_idle_postgres({"other_client_sessions": 0})

    def test_wrong_sqlite_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            connection = sqlite3.connect(Path(temporary_directory) / "wrong.db")
            try:
                connection.execute("CREATE TABLE unrelated(id INTEGER)")
                with self.assertRaises(RESET.SafetyError):
                    RESET.validate_queue_schema(connection)
            finally:
                connection.close()

    def test_live_and_stale_jobs_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            connection = create_queue(Path(temporary_directory) / "jobs.db")
            now = int(time.time())
            try:
                connection.executemany(
                    "INSERT INTO jobs(id, status, started_at, heartbeat_at) VALUES (?, ?, ?, ?)",
                    [
                        ("live", "running", now - 10, now - 5),
                        ("stale", "running", now - 1000, now - 900),
                        ("queued", "queued", None, None),
                    ],
                )
                connection.commit()
                summary = RESET.queue_summary(connection, now=now)
                self.assertEqual(summary["active_status_jobs"], 2)
                self.assertEqual(summary["live_jobs"], 1)
                with self.assertRaises(RESET.SafetyError):
                    RESET.require_idle_queue(summary)

                connection.execute("UPDATE jobs SET heartbeat_at=? WHERE id='live'", (now - 900,))
                connection.commit()
                RESET.require_idle_queue(RESET.queue_summary(connection, now=now))
            finally:
                connection.close()

    def test_sqlite_backup_is_complete_and_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            connection = create_queue(root / "jobs.db")
            backup_path = root / "backup" / "jobs.db"
            try:
                connection.execute(
                    "INSERT INTO jobs(id, status) VALUES ('job-1', 'queued')"
                )
                connection.commit()
                RESET.backup_sqlite(connection, backup_path)
                connection.execute("DELETE FROM jobs")
                connection.commit()
            finally:
                connection.close()

            backup = sqlite3.connect(backup_path)
            try:
                self.assertEqual(
                    backup.execute("SELECT id FROM jobs").fetchone()[0], "job-1"
                )
                self.assertEqual(backup.execute("PRAGMA quick_check").fetchone()[0], "ok")
            finally:
                backup.close()

    def test_failed_sqlite_backup_removes_partial_file(self) -> None:
        class FailingConnection:
            def backup(self, target) -> None:
                target.execute("CREATE TABLE partial(value TEXT)")
                raise RuntimeError("simulated backup failure")

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "jobs.db"
            with self.assertRaises(RuntimeError):
                RESET.backup_sqlite(FailingConnection(), destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + ".partial").exists())

    def test_postgres_failure_rolls_sqlite_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            connection = create_queue(Path(temporary_directory) / "jobs.db")
            try:
                connection.execute(
                    "INSERT INTO jobs(id, status) VALUES ('job-1', 'queued')"
                )
                connection.execute(
                    "INSERT INTO music_prompt_cache(signature) VALUES ('music-1')"
                )
                connection.commit()
                postgres = FakePostgres(fail=True)

                with self.assertRaises(RuntimeError):
                    RESET.reset_databases(postgres, connection)

                self.assertEqual(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM music_prompt_cache").fetchone()[0], 1
                )
                self.assertEqual(postgres.commits, 0)
                self.assertEqual(postgres.rollbacks, 1)
            finally:
                connection.close()

    def test_success_clears_only_service_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            connection = create_queue(Path(temporary_directory) / "jobs.db")
            try:
                connection.execute("CREATE TABLE retained(value TEXT)")
                connection.execute("INSERT INTO retained(value) VALUES ('keep')")
                connection.execute(
                    "INSERT INTO jobs(id, status) VALUES ('job-1', 'queued')"
                )
                connection.execute(
                    "INSERT INTO music_prompt_cache(signature) VALUES ('music-1')"
                )
                connection.execute(
                    "INSERT INTO worker_nodes(id, heartbeat_at) VALUES ('old-worker', 1)"
                )
                connection.commit()
                postgres = FakePostgres()

                RESET.reset_databases(postgres, connection)

                self.assertEqual(postgres.commits, 1)
                self.assertTrue(any("TRUNCATE TABLE" in query for query in postgres.queries))
                for table in RESET.QUEUE_TABLES:
                    self.assertEqual(
                        connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0],
                        0,
                    )
                self.assertEqual(
                    connection.execute("SELECT value FROM retained").fetchone()[0], "keep"
                )
            finally:
                connection.close()

    def test_failed_pg_dump_removes_partial_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "database.dump"

            def failing_runner(command, **_kwargs):
                output = next(item.split("=", 1)[1] for item in command if item.startswith("--file="))
                Path(output).write_bytes(b"partial")
                raise subprocess.CalledProcessError(1, command)

            with self.assertRaises(subprocess.CalledProcessError):
                RESET.backup_postgres(
                    "postgresql://example.invalid/database",
                    destination,
                    runner=failing_runner,
                    pg_dump_executable="pg_dump",
                )
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name(destination.name + ".partial").exists())

    def test_pg_dump_password_is_not_exposed_in_process_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "database.dump"
            observed = {}

            def successful_runner(command, **kwargs):
                observed["command"] = command
                observed["env"] = kwargs["env"]
                output = next(item.split("=", 1)[1] for item in command if item.startswith("--file="))
                Path(output).write_bytes(b"database backup")

            RESET.backup_postgres(
                "postgresql://db-user:s%40fe-secret@localhost:5432/ebookstudio?sslmode=require",
                destination,
                runner=successful_runner,
                pg_dump_executable="pg_dump",
            )

            command_text = " ".join(observed["command"])
            self.assertNotIn("s@fe-secret", command_text)
            self.assertNotIn("s%40fe-secret", command_text)
            self.assertNotIn("db-user", command_text)
            self.assertEqual(observed["env"]["PGPASSWORD"], "s@fe-secret")
            self.assertEqual(observed["env"]["PGUSER"], "db-user")
            self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
