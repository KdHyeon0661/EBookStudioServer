import sys

import pytest

import spring_worker


@pytest.mark.parametrize(
    ("role", "idle_seconds"),
    [("analyze", 1.0), ("music_generation", 2.0)],
)
def test_main_runs_only_configured_worker_role(monkeypatch, role, idle_seconds):
    monkeypatch.setattr(sys, "argv", ["spring_worker.py"])
    monkeypatch.setenv("WORKER_ROLE", role)
    monkeypatch.setattr(spring_worker, "ensure_schema", lambda: None)
    monkeypatch.setattr(spring_worker, "ensure_storage_assets", lambda: None)
    monkeypatch.setattr(spring_worker, "recover_stuck_jobs", lambda: 0)
    calls = []
    monkeypatch.setattr(
        spring_worker, "run_loop", lambda selected, idle: calls.append((selected, idle))
    )

    spring_worker.main()

    assert calls == [(role, idle_seconds)]


def test_command_line_role_overrides_environment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spring_worker.py", "--role", "analyze"])
    monkeypatch.setenv("WORKER_ROLE", "music_generation")
    monkeypatch.setattr(spring_worker, "ensure_schema", lambda: None)
    monkeypatch.setattr(spring_worker, "ensure_storage_assets", lambda: None)
    monkeypatch.setattr(spring_worker, "recover_stuck_jobs", lambda: 0)
    calls = []
    monkeypatch.setattr(
        spring_worker, "run_loop", lambda selected, idle: calls.append((selected, idle))
    )

    spring_worker.main()

    assert calls == [("analyze", 1.0)]


def test_invalid_environment_role_fails_before_initialization(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spring_worker.py"])
    monkeypatch.setenv("WORKER_ROLE", "unexpected")
    initialized = []
    monkeypatch.setattr(spring_worker, "ensure_schema", lambda: initialized.append(True))

    with pytest.raises(SystemExit):
        spring_worker.main()

    assert initialized == []
