from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from anima_tagger.domain.models import ImageAsset, JobStatus, RunConfig, Stage, TagRow, TagSource


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


class Repository:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            self._local.connection = connection
        return connection

    def _initialize(self) -> None:
        self._connection().executescript("""
        CREATE TABLE IF NOT EXISTS folders(folder_id TEXT PRIMARY KEY, root_path TEXT UNIQUE NOT NULL, last_scan_at TEXT);
        CREATE TABLE IF NOT EXISTS images(
          image_id TEXT PRIMARY KEY, folder_id TEXT NOT NULL REFERENCES folders(folder_id), absolute_path TEXT NOT NULL,
          relative_path TEXT NOT NULL, stem TEXT NOT NULL, extension TEXT NOT NULL, size_bytes INTEGER NOT NULL,
          modified_ns INTEGER NOT NULL, content_fingerprint TEXT NOT NULL, UNIQUE(folder_id, relative_path));
        CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, folder_id TEXT NOT NULL, config_json TEXT NOT NULL,
          start_mode TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT);
        CREATE TABLE IF NOT EXISTS stage_jobs(
          job_id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(image_id), stage TEXT NOT NULL,
          status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
          run_id TEXT, started_at TEXT, finished_at TEXT, error_code TEXT, error_message TEXT,
          UNIQUE(image_id, stage));
        CREATE TABLE IF NOT EXISTS review_results(
          image_id TEXT PRIMARY KEY REFERENCES images(image_id), score INTEGER, comment TEXT, final_caption TEXT,
          status TEXT, raw_json TEXT, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tag_rows(
          tag_id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(image_id), source TEXT NOT NULL,
          en TEXT NOT NULL, cn TEXT, suggestion TEXT, enabled INTEGER NOT NULL, position INTEGER NOT NULL,
          is_natural_language INTEGER NOT NULL, manually_overridden INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          event_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, level TEXT NOT NULL, run_id TEXT, image_id TEXT,
          stage TEXT, event_code TEXT NOT NULL, message TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ui_preferences(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)

    def register_folder(self, root: Path) -> str:
        normalized = str(root.resolve())
        folder_id = uuid.uuid5(uuid.NAMESPACE_URL, normalized.casefold()).hex
        self._connection().execute(
            "INSERT INTO folders(folder_id,root_path,last_scan_at) VALUES(?,?,?) ON CONFLICT(folder_id) DO UPDATE SET last_scan_at=excluded.last_scan_at",
            (folder_id, normalized, now()),
        )
        return folder_id

    def upsert_images(self, assets: list[ImageAsset]) -> None:
        with self._connection() as db:
            db.executemany("""INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET
              absolute_path=excluded.absolute_path, relative_path=excluded.relative_path, stem=excluded.stem,
              extension=excluded.extension, size_bytes=excluded.size_bytes, modified_ns=excluded.modified_ns,
              content_fingerprint=excluded.content_fingerprint""", [tuple(asset.__dict__.values()) for asset in assets])

    def create_run(self, config: RunConfig, start_mode: str) -> str:
        run_id = uuid.uuid4().hex
        folder_id = self.register_folder(config.folder)
        self._connection().execute("INSERT INTO runs VALUES(?,?,?,?,?,NULL)", (run_id, folder_id, json.dumps(config.safe_snapshot(), ensure_ascii=False), start_mode, now()))
        return run_id

    def recover_interrupted_jobs(self, folder_id: str) -> int:
        cursor = self._connection().execute("""UPDATE stage_jobs SET status='pending',started_at=NULL,
          error_code='UNEXPECTED_SHUTDOWN',error_message='检测到上次运行异常退出，已恢复等待调度'
          WHERE status='running' AND image_id IN (SELECT image_id FROM images WHERE folder_id=?)""", (folder_id,))
        return cursor.rowcount

    def ensure_job(
        self,
        asset: ImageAsset,
        stage: Stage,
        run_id: str,
        status: JobStatus,
        max_attempts: int | None = None,
    ) -> None:
        attempts = max_attempts or (1 if stage is Stage.FIRST_REVIEW else 5)
        self._connection().execute("""INSERT INTO stage_jobs(job_id,image_id,stage,status,max_attempts,run_id)
          VALUES(?,?,?,?,?,?) ON CONFLICT(image_id,stage) DO UPDATE SET status=excluded.status, run_id=excluded.run_id,
          max_attempts=excluded.max_attempts,error_code=NULL,error_message=NULL
          WHERE stage_jobs.status NOT IN ('running')""",
          (uuid.uuid4().hex, asset.image_id, stage.value, status.value, attempts, run_id))

    def set_job(self, image_id: str, stage: Stage, status: JobStatus, *, error_code: str | None = None, error_message: str | None = None, increment: bool = False) -> None:
        self._connection().execute("""UPDATE stage_jobs SET status=?, attempt_count=attempt_count+?,
          started_at=CASE WHEN ?='running' THEN ? ELSE started_at END,
          finished_at=CASE WHEN ? IN ('succeeded','failed','skipped','blocked') THEN ? ELSE NULL END,
          error_code=?,error_message=? WHERE image_id=? AND stage=?""",
          (status.value, int(increment), status.value, now(), status.value, now(), error_code, (error_message or "")[:1000] or None, image_id, stage.value))

    def event(self, code: str, message: str, *, run_id: str | None = None, image_id: str | None = None, stage: Stage | None = None, level: str = "INFO") -> None:
        self._connection().execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, now(), level, run_id, image_id, stage.value if stage else None, code, message[:2000]))

    def recent_events(self, limit: int = 200) -> list[sqlite3.Row]:
        return self._connection().execute("""SELECT created_at,level,image_id,stage,event_code,message
          FROM events ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()[::-1]

    def sync_first_tags(self, asset: ImageAsset, caption: str, trigger_word: str) -> None:
        tags = [tag.strip() for tag in caption.split(",") if tag.strip()]
        with self._connection() as db:
            db.execute("DELETE FROM tag_rows WHERE image_id=? AND source=? AND manually_overridden=0", (asset.image_id, TagSource.FIRST_REVIEW.value))
            for position, tag in enumerate(tags):
                db.execute("INSERT INTO tag_rows VALUES(?,?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, asset.image_id, TagSource.FIRST_REVIEW.value, tag, None, None, 1, 1000 + position, 0, 0))

    def sync_second_tags(self, asset: ImageAsset, result: dict) -> None:
        validity = {row["tag"].casefold(): row for row in result["first_caption_validity"]}
        translations = {
            row["tag"].casefold(): row["cn"].strip()
            for row in result.get("tag_translations", [])
            if isinstance(row, dict) and isinstance(row.get("tag"), str) and isinstance(row.get("cn"), str)
        }
        with self._connection() as db:
            db.execute("""INSERT INTO review_results VALUES(?,?,?,?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET
              score=excluded.score,comment=excluded.comment,final_caption=excluded.final_caption,status=excluded.status,
              raw_json=excluded.raw_json,updated_at=excluded.updated_at""", (asset.image_id, result["score"], result["comment"], result["final_caption"], result["first_review_status"], json.dumps(result, ensure_ascii=False), now()))
            for tag, cn in translations.items():
                db.execute(
                    """UPDATE tag_rows SET cn=? WHERE image_id=? AND source=?
                    AND lower(en)=? AND manually_overridden=0""",
                    (cn, asset.image_id, TagSource.FIRST_REVIEW.value, tag),
                )
            for tag, row in validity.items():
                db.execute(
                    """UPDATE tag_rows SET enabled=?,suggestion=?
                    WHERE image_id=? AND source=? AND lower(en)=? AND manually_overridden=0""",
                    (
                        int(row["valid"]), row.get("suggestion"), asset.image_id,
                        TagSource.FIRST_REVIEW.value, tag,
                    ),
                )
            db.execute("DELETE FROM tag_rows WHERE image_id=? AND source=? AND manually_overridden=0", (asset.image_id, TagSource.SECOND_REVIEW.value))
            caption = result["final_caption"]
            tag_part, sep, natural = caption.partition(". ")
            for position, tag in enumerate(item.strip() for item in tag_part.split(",") if item.strip()):
                db.execute("INSERT INTO tag_rows VALUES(?,?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, asset.image_id, TagSource.SECOND_REVIEW.value, tag, translations.get(tag.casefold()), None, 1, position, 0, 0))
            if sep:
                db.execute("INSERT INTO tag_rows VALUES(?,?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, asset.image_id, TagSource.SECOND_REVIEW.value, natural, None, None, 1, 900, 1, 0))

    def review(self, image_id: str) -> sqlite3.Row | None:
        return self._connection().execute("SELECT * FROM review_results WHERE image_id=?", (image_id,)).fetchone()

    def update_review_comment(self, image_id: str, comment: str) -> None:
        self._connection().execute(
            "UPDATE review_results SET comment=?,updated_at=? WHERE image_id=?",
            (comment.strip(), now(), image_id),
        )

    def tags(self, image_id: str) -> list[sqlite3.Row]:
        return self._connection().execute("SELECT * FROM tag_rows WHERE image_id=? ORDER BY position,source", (image_id,)).fetchall()

    def update_tag(self, tag_id: str, en: str, cn: str, suggestion: str, enabled: bool) -> None:
        self._connection().execute("UPDATE tag_rows SET en=?,cn=?,suggestion=?,enabled=?,manually_overridden=1 WHERE tag_id=?", (en.strip(), cn.strip() or None, suggestion.strip() or None, int(enabled), tag_id))

    def update_missing_tag_translations(self, image_id: str, translations: dict[str, str]) -> int:
        """Fill only blank translations so editor changes are never overwritten."""
        updated = 0
        with self._connection() as db:
            for tag, cn in translations.items():
                cursor = db.execute(
                    """UPDATE tag_rows SET cn=? WHERE image_id=? AND lower(en)=?
                    AND trim(COALESCE(cn, ''))=''""",
                    (cn.strip(), image_id, tag.casefold()),
                )
                updated += cursor.rowcount
        return updated

    def add_tag(self, image_id: str, en: str) -> None:
        position = self._connection().execute("SELECT COALESCE(MAX(position),0)+1 FROM tag_rows WHERE image_id=?", (image_id,)).fetchone()[0]
        self._connection().execute("INSERT INTO tag_rows VALUES(?,?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, image_id, TagSource.MANUAL.value, en, None, None, 1, position, 0, 1))

    def image_states(self, folder_id: str) -> dict[str, dict[str, object]]:
        rows = self._connection().execute(
            """SELECT i.image_id, r.score,
              MAX(CASE WHEN j.stage='first_review' THEN j.status END) AS first_status,
              MAX(CASE WHEN j.stage='second_review' THEN j.status END) AS second_status
              FROM images i
              LEFT JOIN stage_jobs j ON j.image_id=i.image_id
              LEFT JOIN review_results r ON r.image_id=i.image_id
              WHERE i.folder_id=? GROUP BY i.image_id,r.score""",
            (folder_id,),
        ).fetchall()
        return {row["image_id"]: dict(row) for row in rows}

    def stage_summary(self, folder_id: str, stage: Stage) -> dict[str, int]:
        rows = self._connection().execute(
            """SELECT status,COUNT(*) AS count FROM stage_jobs
              WHERE stage=? AND image_id IN (SELECT image_id FROM images WHERE folder_id=?)
              GROUP BY status""",
            (stage.value, folder_id),
        ).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        done = counts.get("succeeded", 0) + counts.get("skipped", 0)
        failed = counts.get("failed", 0) + counts.get("blocked", 0)
        return {"done": done, "failed": failed, "total": sum(counts.values())}

    def failed_jobs(self, folder_id: str, limit: int = 100) -> list[sqlite3.Row]:
        return self._connection().execute(
            """SELECT j.image_id,j.stage,j.error_message,i.absolute_path
              FROM stage_jobs j JOIN images i ON i.image_id=j.image_id
              WHERE i.folder_id=? AND j.status IN ('failed','blocked')
              ORDER BY j.finished_at DESC LIMIT ?""",
            (folder_id, limit),
        ).fetchall()

    def retry_job(self, image_id: str, stage: Stage) -> None:
        self._connection().execute(
            """UPDATE stage_jobs SET status='pending',attempt_count=0,error_code=NULL,
              error_message=NULL,started_at=NULL,finished_at=NULL WHERE image_id=? AND stage=?""",
            (image_id, stage.value),
        )

    def failed_stage_image_ids(self, folder_id: str, stage: Stage) -> list[str]:
        rows = self._connection().execute(
            """SELECT j.image_id FROM stage_jobs j JOIN images i ON i.image_id=j.image_id
            WHERE i.folder_id=? AND j.stage=? AND j.status IN ('failed', 'blocked')""",
            (folder_id, stage.value),
        ).fetchall()
        return [row["image_id"] for row in rows]

    def clear_events(self, keep_days: int = 30) -> int:
        """Delete old events older than `keep_days`; pass 0 to delete all. Returns deleted count."""
        if keep_days > 0:
            cursor = self._connection().execute(
                "DELETE FROM events WHERE created_at < datetime('now', ?)",
                (f"-{keep_days} days",),
            )
        else:
            cursor = self._connection().execute("DELETE FROM events")
        return cursor.rowcount

    def preference(self, key: str, default: str) -> str:
        row = self._connection().execute("SELECT value FROM ui_preferences WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_preference(self, key: str, value: str) -> None:
        self._connection().execute("INSERT INTO ui_preferences VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
