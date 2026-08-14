"""Container health check for one SQLite-backed worker role."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"analyze", "music_generation"}:
        print("usage: worker_healthcheck.py analyze|music_generation", file=sys.stderr)
        return 2
    role = sys.argv[1]
    db_path = Path(os.environ.get("EBOOK_DB_PATH", "/data/users.db"))
    maximum_age = int(os.environ.get("WORKER_HEALTH_MAX_AGE_SECONDS", "120"))
    if not db_path.is_file():
        print(f"database does not exist: {db_path}", file=sys.stderr)
        return 1
    try:
        connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
        node = connection.execute(
            "SELECT MAX(heartbeat_at) FROM worker_nodes WHERE job_type=?", (role,)
        ).fetchone()[0]
        job = connection.execute(
            "SELECT MAX(heartbeat_at) FROM jobs "
            "WHERE type=? AND status IN ('running', 'cancel_requested')", (role,)
        ).fetchone()[0]
        connection.close()
    except (sqlite3.Error, OSError) as error:
        print(f"health query failed: {error}", file=sys.stderr)
        return 1
    heartbeat = max(value for value in (node, job) if value is not None) \
        if node is not None or job is not None else 0
    age = int(time.time()) - int(heartbeat)
    if heartbeat <= 0 or age > maximum_age:
        print(f"{role} heartbeat is stale: age={age}s", file=sys.stderr)
        return 1
    print(f"{role} worker healthy: heartbeat_age={age}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
