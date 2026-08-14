import spring_worker


def test_analysis_completion_and_music_enqueue_commit_together(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setattr(spring_worker, "STORAGE_ROOT", storage)
    monkeypatch.setattr(spring_worker, "DB_PATH", storage / "jobs.db")
    spring_worker.ensure_schema()
    with spring_worker.connect() as connection:
        connection.execute(
            "INSERT INTO jobs(id, type, user_uuid, book_id, status, created_at, worker_id, "
            "music_folder, web_path_prefix) "
            "VALUES ('analysis-atomic', 'analyze', 'user-1', 'book-1', 'running', 1, "
            "'worker-1', 'music', '/files/user/book-1')"
        )
    job = {
        "id": "analysis-atomic",
        "type": "analyze",
        "user_uuid": "user-1",
        "book_id": "book-1",
        "music_folder": "music",
        "web_path_prefix": "/files/user/book-1",
    }
    result = {
        "text_file": "book_full.json",
        "cover_image": "book.png",
        "title": "book",
        "real_author": "author",
    }

    assert spring_worker.finish_analysis_with_music_job(
        job, "worker-1", "book_full.json", result
    ) is True

    with spring_worker.connect() as connection:
        parent = connection.execute(
            "SELECT status, output_json, music_job_id FROM jobs WHERE id='analysis-atomic'"
        ).fetchone()
        child = connection.execute(
            "SELECT status, parent_job_id, json_path FROM jobs WHERE parent_job_id='analysis-atomic'"
        ).fetchone()
    assert parent["status"] == "done"
    assert parent["output_json"] == "book_full.json"
    assert child["status"] == "queued"
    assert child["parent_job_id"] == "analysis-atomic"
    assert child["json_path"] == "book_full.json"
    assert parent["music_job_id"] == result["music_job_id"]
