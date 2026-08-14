import sqlite3

from music_prompt_catalog import MusicPromptCatalog, prompt_signature


def metadata():
    return {
        "prompt": "Genre: ambient. Mood: calm",
        "genre": "ambient",
        "bpm": 72,
        "keywords": ["calm", "night"],
        "target_duration_sec": 120,
        "segment_duration_sec": 30,
        "generator_version": "musicgen:test-v1",
    }


def signature_for(values=None):
    values = values or metadata()
    return prompt_signature(
        values["prompt"], values["genre"], values["bpm"], values["keywords"],
        values["target_duration_sec"], values["segment_duration_sec"],
        values["generator_version"],
    )


def test_signature_is_canonical_and_generator_versioned():
    base = metadata()
    spaced = {**base, "prompt": "  Genre:   ambient. Mood: calm  ", "genre": " Ambient "}
    assert signature_for(base) == signature_for(spaced)
    assert signature_for(base) != signature_for({**base, "generator_version": "musicgen:test-v2"})


def test_ready_asset_is_reused_and_counted(tmp_path):
    db_path = tmp_path / "jobs.db"
    music_root = tmp_path / "music"
    asset = music_root / "storage_001" / "track.wav"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"wave")
    catalog = MusicPromptCatalog(db_path, music_root)
    values = metadata()
    signature = signature_for(values)

    catalog.mark_ready(signature, values, asset.name, asset, "job-1")
    assert catalog.find_ready(signature) == str(asset.resolve())
    assert catalog.find_ready(signature) == str(asset.resolve())

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT status, relative_path, reuse_count FROM music_prompt_cache WHERE signature=?",
            (signature,),
        ).fetchone()
    assert row == ("ready", "storage_001/track.wav", 2)


def test_missing_file_invalidates_ready_catalog_entry(tmp_path):
    catalog = MusicPromptCatalog(tmp_path / "jobs.db", tmp_path / "music")
    values = metadata()
    signature = signature_for(values)
    asset = tmp_path / "music" / "track.wav"
    asset.write_bytes(b"wave")
    catalog.mark_ready(signature, values, asset.name, asset, "job-1")
    asset.unlink()

    assert catalog.find_ready(signature) is None
    with sqlite3.connect(tmp_path / "jobs.db") as connection:
        row = connection.execute(
            "SELECT status, error FROM music_prompt_cache WHERE signature=?", (signature,)
        ).fetchone()
    assert row == ("missing", "Catalog file is missing")


def test_generation_failure_is_persisted_for_diagnostics(tmp_path):
    catalog = MusicPromptCatalog(tmp_path / "jobs.db", tmp_path / "music")
    values = metadata()
    signature = signature_for(values)
    catalog.mark_generating(signature, values, f"{signature}.wav", "job-2")
    catalog.mark_failed(signature, "GPU out of memory")

    with sqlite3.connect(tmp_path / "jobs.db") as connection:
        row = connection.execute(
            "SELECT status, owner_job_id, error FROM music_prompt_cache WHERE signature=?",
            (signature,),
        ).fetchone()
    assert row == ("failed", None, "GPU out of memory")
