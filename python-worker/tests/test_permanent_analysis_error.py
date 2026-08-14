import spring_worker

from pdf_structure import PermanentAnalysisError


def test_permanent_analysis_error_is_not_retried(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(spring_worker, "DB_PATH", db_path)
    spring_worker.ensure_schema()
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, status, created_at, worker_id, "
            "attempt_count, max_attempts) "
            "VALUES ('unsupported', 'analyze', 'user-1', 'running', 1, 'worker-1', 1, 3)"
        )

    spring_worker.fail_job(
        {"id": "unsupported", "attempt_count": 1, "max_attempts": 3},
        "worker-1",
        PermanentAnalysisError("OCR is required"),
    )

    with spring_worker.connect() as connection:
        row = connection.execute(
            "SELECT status, attempt_count, error FROM jobs WHERE id='unsupported'"
        ).fetchone()
    assert row["status"] == "error"
    assert row["attempt_count"] == 1
    assert row["error"] == "OCR is required"
