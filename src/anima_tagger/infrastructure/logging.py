from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo


class BeijingFormatter(logging.Formatter):
    """Keep persisted log-file timestamps aligned with the application's UI."""

    def formatTime(self, record, datefmt=None):
        value = datetime.fromtimestamp(record.created, ZoneInfo("Asia/Shanghai"))
        return value.strftime(datefmt) if datefmt else value.isoformat(timespec="seconds")


def configure_logging(data_dir: Path) -> None:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(log_dir / "anima_tagger.log", when="midnight", backupCount=14, encoding="utf-8")
    handler.setFormatter(BeijingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("anima_tagger")
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(handler)
