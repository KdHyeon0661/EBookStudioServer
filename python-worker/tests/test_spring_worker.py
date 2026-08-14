import sqlite3
import time

import spring_worker


def configure_db(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(spring_worker, "DB_PATH", db_path)
    spring_worker.ensure_schema()
    return db_path


def test_claim_job_is_atomic(tmp_path, monkeypatch):
    db_path = configure_db(tmp_path, monkeypatch)
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at) "
            "VALUES ('job-1', 'analyze', 'user-1', 'queued', 1)"
        )
    claimed = spring_worker.claim_job("analyze", "worker-1")
    assert claimed is not None
    assert claimed["id"] == "job-1"
    assert claimed["worker_id"] == "worker-1"
    assert claimed["attempt_count"] == 1
    assert spring_worker.claim_job("analyze", "worker-2") is None
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT status FROM jobs WHERE id='job-1'").fetchone()[0] == "running"


def test_finish_job_persists_artifact_and_music_contract(tmp_path, monkeypatch):
    configure_db(tmp_path, monkeypatch)
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at, worker_id) "
            "VALUES ('job-2', 'analyze', 'user-1', 'running', 1, 'worker-1')"
        )
    spring_worker.finish_job(
        "job-2", status="done", worker_id="worker-1",
        result={
            "text_file": "book_full.json", "cover_image": "book_12345678.png",
            "title": "book", "real_author": "author", "music_job_id": "music-2",
        },
    )
    with spring_worker.connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id='job-2'").fetchone()
    assert row["status"] == "done"
    assert row["output_json"] == "book_full.json"
    assert row["cover_file"] == "book_12345678.png"
    assert row["music_job_id"] == "music-2"
    assert row["worker_id"] is None


def test_cancel_request_cannot_be_overwritten_by_late_worker_finish(tmp_path, monkeypatch):
    configure_db(tmp_path, monkeypatch)
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at, worker_id, cancel_requested_at) "
            "VALUES ('cancel-1', 'analyze', 'user-1', 'cancel_requested', 1, 'worker-1', 2)"
        )
    completed = spring_worker.finish_job(
        "cancel-1", status="done", worker_id="worker-1",
        result={"text_file": "should-not-win.json"},
    )
    assert completed is False
    with spring_worker.connect() as connection:
        row = connection.execute(
            "SELECT status, output_json, worker_id FROM jobs WHERE id='cancel-1'"
        ).fetchone()
    assert row["status"] == "cancelled"
    assert row["output_json"] is None
    assert row["worker_id"] is None


def test_recovery_finalizes_stale_cancel_request(tmp_path, monkeypatch):
    configure_db(tmp_path, monkeypatch)
    now = int(time.time())
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at, started_at, "
            "heartbeat_at, worker_id, cancel_requested_at) "
            "VALUES ('cancel-stale', 'analyze', 'user-1', 'cancel_requested', 1, ?, ?, 'gone', ?)",
            (now - 1000, now - 1000, now - 900),
        )
    assert spring_worker.recover_stuck_jobs(stale_seconds=300) == 1
    with spring_worker.connect() as connection:
        row = connection.execute(
            "SELECT status, worker_id FROM jobs WHERE id='cancel-stale'"
        ).fetchone()
    assert row["status"] == "cancelled"
    assert row["worker_id"] is None


def test_recovery_only_requeues_expired_heartbeats(tmp_path, monkeypatch):
    configure_db(tmp_path, monkeypatch)
    now = int(time.time())
    with spring_worker.connect() as connection:
        connection.executemany(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at, started_at, heartbeat_at, attempt_count) "
            "VALUES (?, 'analyze', 'user-1', 'running', 1, ?, ?, 1)",
            [("fresh", now, now), ("stale", now - 1000, now - 1000)],
        )
    assert spring_worker.recover_stuck_jobs(stale_seconds=300) == 1
    with spring_worker.connect() as connection:
        statuses = dict(connection.execute("SELECT id, status FROM jobs"))
    assert statuses == {"fresh": "running", "stale": "queued"}


def test_failed_job_retries_then_becomes_terminal(tmp_path, monkeypatch):
    configure_db(tmp_path, monkeypatch)
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at, worker_id, attempt_count, max_attempts) "
            "VALUES ('retry', 'analyze', 'user-1', 'running', 1, 'worker-1', 1, 2)"
        )
    spring_worker.fail_job({"id": "retry", "attempt_count": 1, "max_attempts": 2},
                           "worker-1", RuntimeError("temporary"))
    with spring_worker.connect() as connection:
        assert connection.execute("SELECT status FROM jobs WHERE id='retry'").fetchone()[0] == "queued"
        connection.execute(
            "UPDATE jobs SET status='running', worker_id='worker-2', attempt_count=2 WHERE id='retry'"
        )
    spring_worker.fail_job({"id": "retry", "attempt_count": 2, "max_attempts": 2},
                           "worker-2", RuntimeError("permanent"))
    with spring_worker.connect() as connection:
        row = connection.execute("SELECT status, error FROM jobs WHERE id='retry'").fetchone()
    assert row["status"] == "error"
    assert row["error"] == "permanent"


def test_music_enqueue_is_idempotent(tmp_path, monkeypatch):
    configure_db(tmp_path, monkeypatch)
    source = {
        "id": "analysis-1", "user_uuid": "user-1", "book_id": "book-1",
        "music_folder": "music", "web_path_prefix": "/files/user/book",
    }
    first = spring_worker.enqueue_music_generation(source, "book.json")
    second = spring_worker.enqueue_music_generation(source, "book.json")
    assert first == second
    with spring_worker.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE parent_job_id='analysis-1'"
        ).fetchone()[0] == 1