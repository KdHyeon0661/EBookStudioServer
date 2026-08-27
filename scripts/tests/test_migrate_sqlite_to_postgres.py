from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "migrate-sqlite-to-postgres.py"
SPEC = importlib.util.spec_from_file_location("legacy_migration", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


class QueueCopyTests(unittest.TestCase):
    def test_queue_copy_removes_every_postgresql_owned_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "legacy.db"
            output_path = root / "jobs.db"
            source = sqlite3.connect(source_path)
            try:
                source.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY)")
                source.execute("INSERT INTO jobs(id) VALUES ('job-1')")
                for table in MIGRATION.APP_ONLY_TABLES:
                    source.execute(f'CREATE TABLE "{table}"(id INTEGER)')
                source.commit()

                temporary = MIGRATION.create_queue_copy(source, output_path)
            finally:
                source.close()

            queue = sqlite3.connect(temporary)
            try:
                tables = {
                    row[0]
                    for row in queue.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertIn("jobs", tables)
                self.assertTrue(set(MIGRATION.APP_ONLY_TABLES).isdisjoint(tables))
                self.assertEqual(queue.execute("SELECT id FROM jobs").fetchone()[0], "job-1")
            finally:
                queue.close()

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = sqlite3.connect(root / "legacy.db")
            destination = root / "jobs.db"
            destination.write_text("keep", encoding="utf-8")
            try:
                with self.assertRaises(FileExistsError):
                    MIGRATION.create_queue_copy(source, destination)
            finally:
                source.close()
            self.assertEqual(destination.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
