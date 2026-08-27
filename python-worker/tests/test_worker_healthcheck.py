import sqlite3
import sys
import time

import worker_healthcheck


def create_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE worker_nodes(id TEXT, job_type TEXT, heartbeat_at INTEGER)"
        )
        connection.execute(
            "CREATE TABLE jobs(type TEXT, status TEXT, heartbeat_at INTEGER)"
        )


def run_check(monkeypatch, db_path, role="analyze", maximum_age=120):
    monkeypatch.setenv("EBOOK_QUEUE_DB_PATH", str(db_path))
    monkeypatch.setenv("WORKER_HEALTH_MAX_AGE_SECONDS", str(maximum_age))
    monkeypatch.setattr(sys, "argv", ["worker_healthcheck.py", role])
    return worker_healthcheck.main()


def test_idle_worker_node_heartbeat_is_healthy(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    create_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO worker_nodes VALUES ('worker-1', 'analyze', ?)",
            (int(time.time()),),
        )
    assert run_check(monkeypatch, db_path) == 0


def test_active_job_heartbeat_keeps_busy_worker_healthy(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    create_database(db_path)
    now = int(time.time())
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO worker_nodes VALUES ('worker-1', 'music_generation', ?)",
            (now - 1000,),
        )
        connection.execute(
            "INSERT INTO jobs VALUES ('music_generation', 'running', ?)", (now,)
        )
    assert run_check(monkeypatch, db_path, "music_generation") == 0


def test_stale_worker_is_unhealthy(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    create_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO worker_nodes VALUES ('worker-1', 'analyze', ?)",
            (int(time.time()) - 1000,),
        )
    assert run_check(monkeypatch, db_path, maximum_age=60) == 1
