from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from anima_tagger.infrastructure.repository import Repository
from anima_tagger.infrastructure.logging import configure_logging
from anima_tagger.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parent / "resources" / "genetic-data-svgrepo-com.ico"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    data = Path(os.getenv("LOCALAPPDATA", Path.home())) / "AnimaTagger"
    configure_logging(data)
    window = MainWindow(Repository(data / "anima_tagger.sqlite3"))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
