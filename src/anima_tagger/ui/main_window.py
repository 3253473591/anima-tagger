from __future__ import annotations

import html
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PyQt6.QtCore import QObject, QPoint, QRect, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from anima_tagger.application.controller import RunController
from anima_tagger.application.review import ReviewService
from anima_tagger.domain.models import LoraMode, RunConfig, Stage
from anima_tagger.infrastructure.cloud_review import CloudReviewAdapter
from anima_tagger.infrastructure.credentials import get_api_key, set_api_key
from anima_tagger.infrastructure.files import final_path, is_nonempty, scan_images, stage_path
from anima_tagger.infrastructure.hardware import sample_hardware
from anima_tagger.infrastructure.local_inference import LocalInferenceAdapter
from anima_tagger.infrastructure.repository import Repository


class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.fn())
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class PreviewCanvas(QLabel):
    """Render a fresh high-DPI preview from the original file for every view change."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 320)
        self.setStyleSheet("background:#171a20;")
        self.setScaledContents(False)
        self._path = ""
        self._rendered = QPixmap()
        self._fit = True
        self._scale = 1.0
        self._offset = QPoint()
        self._press = QPoint()
        self._dragged = False

    def set_image(self, path: str) -> None:
        self._path = path
        self._fit = True
        self._scale = 1.0
        self._offset = QPoint()
        self.reload_from_original()

    def reload_from_original(self) -> None:
        """Read the source again; thumbnail or prior preview pixmaps are never reused."""
        self._rendered = QPixmap()
        if not self._path:
            self.update()
            return
        source = QImage(self._path)
        if source.isNull():
            self.update()
            return
        dpr = max(1.0, self.devicePixelRatioF())
        if self._fit:
            logical_target = self.size()
        else:
            logical_target = QSize(
                max(1, int(source.width() * self._scale)),
                max(1, int(source.height() * self._scale)),
            )
        physical_target = QSize(
            max(1, int(logical_target.width() * dpr)),
            max(1, int(logical_target.height() * dpr)),
        )
        scaled = source.scaled(
            physical_target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._rendered = QPixmap.fromImage(scaled)
        self._rendered.setDevicePixelRatio(dpr)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position().toPoint()
            self._dragged = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            delta = point - self._press
            if delta.manhattanLength() > 2:
                self._dragged = True
                self._fit = False
                self._offset += delta
                self._press = point
                self.update()

    def mouseReleaseEvent(self, event):
        self.unsetCursor()
        if event.button() == Qt.MouseButton.LeftButton and not self._dragged:
            self._fit = not self._fit
            self._scale = 1.0
            self._offset = QPoint()
            self.reload_from_original()

    def wheelEvent(self, event):
        self._fit = False
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._scale = max(0.1, min(8.0, self._scale * factor))
        self.reload_from_original()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reload_from_original()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#171a20"))
        if self._rendered.isNull():
            return
        shown_size = self._rendered.deviceIndependentSize().toSize()
        x = (self.width() - shown_size.width()) // 2 + self._offset.x()
        y = (self.height() - shown_size.height()) // 2 + self._offset.y()
        painter.drawPixmap(x, y, self._rendered)


class EditableComment(QPlainTextEdit):
    edit_finished = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setPlaceholderText("暂无二审评语")

    def mouseDoubleClickEvent(self, event):
        self.setReadOnly(False)
        self.setFocus()
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event):
        if not self.isReadOnly():
            self.setReadOnly(True)
            self.edit_finished.emit(self.toPlainText())
        super().focusOutEvent(event)


class ClickLabel(QLabel):
    clicked = pyqtSignal()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class MetricRing(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.value = 0
        self.display = "--"
        self.setMinimumSize(105, 105)

    def set_metric(self, value: int, display: str, alert: bool = False) -> None:
        self.value = max(0, min(100, value))
        self.display = display
        self.alert = alert
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRect(16, 8, min(self.width() - 32, 70), min(self.height() - 34, 70))
        rect.moveLeft((self.width() - rect.width()) // 2)
        pen = QPen(QColor("#e5e7eb"), 7)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)
        color = QColor("#ef4444" if getattr(self, "alert", False) else "#3b82f6")
        painter.setPen(QPen(color, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, -int(360 * 16 * self.value / 100))
        painter.setPen(QColor("#111827"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.display)
        painter.setPen(QColor("#6b7280"))
        painter.drawText(QRect(0, rect.bottom() + 6, self.width(), 22), Qt.AlignmentFlag.AlignCenter, self.title)


class MainWindow(QMainWindow):
    progress_signal = pyqtSignal(str, int, int, str)
    log_signal = pyqtSignal(str)
    STATE_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    LOADED_ROLE = int(Qt.ItemDataRole.UserRole) + 2

    def __init__(self, repository: Repository):
        super().__init__()
        self.repo = repository
        self.controller = RunController(repository, self.progress_signal.emit, self.log_signal.emit)
        self.review_service = ReviewService(repository)
        self.settings = QSettings("AnimaTagger", "AnimaTagger")
        self.pool = QThreadPool.globalInstance()
        self.assets = []
        self.filtered_assets = []
        self.current = -1
        self.folder_id = ""
        self._run_active = False
        self._gemma_warmup_active = False
        self._restoring = True
        self._log_lines: list[tuple[str, str]] = []
        self._grid_scroll = 0
        self.setWindowTitle("Anima / Danbooru LoRA 自动打标")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 680)
        self._build_ui()
        self._restore()
        self._wire()
        self._restoring = False
        self._set_dirty(False)
        self._install_shortcuts()
        self.hardware_timer = QTimer(self)
        self.hardware_timer.timeout.connect(self._refresh_runtime)
        self.hardware_timer.start(1000)
        QTimer.singleShot(0, self._initial_scan)
        QTimer.singleShot(0, self._warm_gemma4)

    def _build_ui(self):
        self.root_stack = QStackedWidget()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_config_page(), "参数配置")
        self.tabs.addTab(self._build_console_page(), "运行控制台")
        self.tabs.addTab(self._build_grid_page(), "数据集浏览")
        self.root_stack.addWidget(self.tabs)
        self.editor_page = self._build_editor_page()
        self.root_stack.addWidget(self.editor_page)
        self.setCentralWidget(self.root_stack)
        self.setStyleSheet(self._style_sheet())

    def _build_config_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(16, 16, 16, 16)
        columns = QHBoxLayout()
        columns.setSpacing(16)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        self.folder = QLineEdit()
        self.model = QLineEdit()
        self.mmproj = QLineEdit()
        self.mtp = QLineEdit()
        for field in (self.folder, self.model, self.mmproj, self.mtp):
            field.setFont(QFont("Consolas", 9))
        self.mode = QComboBox()
        self.mode.addItem("角色 LoRA", "character")
        self.mode.addItem("画风 LoRA", "style")
        self.trigger = QLineEdit("anima_character")
        self.columns = QSpinBox()
        self.columns.setRange(2, 20)
        self.columns.setValue(10)

        base_form = QFormLayout()
        base_form.addRow("图片文件夹", self._path_field(self.folder, True))
        base_form.addRow("LoRA 类型", self.mode)
        base_form.addRow("Trigger word", self.trigger)
        base_form.addRow("缩略图每行数量", self.columns)
        left.addWidget(self._card("▣  基础任务配置", base_form))

        model_form = QFormLayout()
        model_form.addRow("Gemma4 GGUF", self._path_field(self.model, False))
        model_form.addRow(self._collapsible_path("mmproj GGUF", self.mmproj))
        model_form.addRow(self._collapsible_path("MTP GGUF（可选）", self.mtp))
        left.addWidget(self._card("◇  本地模型配置", model_form))
        left.addStretch()

        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("保存在系统凭据管理器中")
        self.toggle_key_button = QToolButton()
        self.toggle_key_button.setText("显示")
        self.toggle_key_button.clicked.connect(self.toggle_api_key_visibility)
        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self.key)
        key_layout.addWidget(self.toggle_key_button)
        self.proxy = QLineEdit()
        self.cloud_model = QComboBox()
        self.cloud_model.setEditable(True)
        self.cloud_model.addItem("gemini-3.5-flash")
        self.refresh_models_button = QPushButton("刷新模型")
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.addWidget(self.cloud_model)
        model_layout.addWidget(self.refresh_models_button)
        self.base_url = QLineEdit("https://api.example.com/v1")
        cloud_form = QFormLayout()
        cloud_form.addRow("Gemini Key", key_row)
        cloud_form.addRow("代理地址", self.proxy)
        cloud_form.addRow("云端模型", model_row)
        cloud_form.addRow("OpenAI 兼容地址", self.base_url)
        right.addWidget(self._card("☁  云端 API 配置", cloud_form))

        self.first_concurrency = QSpinBox()
        self.first_concurrency.setRange(1, 8)
        self.second_concurrency = QSpinBox()
        self.second_concurrency.setRange(1, 32)
        self.retry_count = QSpinBox()
        self.retry_count.setRange(1, 20)
        self.retry_count.setValue(5)
        self.resume_enabled = QCheckBox("启用断点续跑")
        self.resume_enabled.setChecked(True)
        self.first_rule_mode = QComboBox()
        self.first_rule_mode.addItem("程序过滤后写入一审", "filter")
        self.first_rule_mode.addItem("原样写入，交二审修复", "second_review")
        advanced_form = QFormLayout()
        advanced_form.addRow("一审并发数", self.first_concurrency)
        advanced_form.addRow("二审并发数", self.second_concurrency)
        advanced_form.addRow("API 失败重试次数", self.retry_count)
        advanced_form.addRow("一审格式处理", self.first_rule_mode)
        advanced_form.addRow("断点续跑", self.resume_enabled)
        right.addWidget(self._card("⚙  并发与高级规则", advanced_form))
        right.addStretch()
        outer.addLayout(columns, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        self.dirty_dot = QLabel("●  有未保存修改")
        self.dirty_dot.setObjectName("dirtyDot")
        self.save_config_button = QPushButton("保存配置")
        self.save_config_button.setObjectName("primaryButton")
        self.save_config_button.clicked.connect(self.save_config)
        footer.addWidget(self.dirty_dot)
        footer.addWidget(self.save_config_button)
        outer.addLayout(footer)
        return page

    def _build_console_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(16, 16, 16, 16)
        top = QHBoxLayout()
        controls = QGridLayout()
        self.start_button = QPushButton("一键启动")
        self.start_button.setProperty("kind", "start")
        self.global_pause = self._toggle_button("全局暂停", self._toggle_all)
        self.first_pause = self._toggle_button("一审暂停", self._toggle_first)
        self.second_pause = self._toggle_button("二审暂停", self._toggle_second)
        self.retry_first_button = QPushButton("一审重试")
        self.retry_second_button = QPushButton("二审重试")
        self.retry_first_button.setProperty("kind", "retry")
        self.retry_second_button.setProperty("kind", "retry")
        controls.addWidget(self.start_button, 0, 0)
        controls.addWidget(self.global_pause, 0, 1)
        controls.addWidget(self.first_pause, 1, 0)
        controls.addWidget(self.second_pause, 1, 1)
        controls.addWidget(self.retry_first_button, 2, 0)
        controls.addWidget(self.retry_second_button, 2, 1)
        control_box = QWidget()
        control_box.setLayout(controls)
        top.addWidget(self._card("▶  任务控制", control_box), 1)

        self.first_progress, first_card = self._progress_card("一审进度")
        self.second_progress, second_card = self._progress_card("二审进度")
        top.addWidget(first_card, 2)
        top.addWidget(second_card, 2)
        outer.addLayout(top)

        bottom = QSplitter()
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_tools = QHBoxLayout()
        self.log_filter = QComboBox()
        self.log_filter.addItems(["全部", "警告", "错误"])
        self.clear_log_button = QToolButton()
        self.clear_log_button.setText("清空")
        self.export_log_button = QToolButton()
        self.export_log_button.setText("导出")
        self.auto_scroll = QCheckBox("自动滚动")
        self.auto_scroll.setChecked(True)
        log_tools.addWidget(QLabel("日志级别"))
        log_tools.addWidget(self.log_filter)
        log_tools.addStretch()
        log_tools.addWidget(self.auto_scroll)
        log_tools.addWidget(self.clear_log_button)
        log_tools.addWidget(self.export_log_button)
        log_layout.addLayout(log_tools)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(1000)
        log_layout.addWidget(self.log_view)
        bottom.addWidget(self._card("运行日志", log_panel))

        side = QSplitter(Qt.Orientation.Vertical)
        metric_panel = QWidget()
        metric_layout = QGridLayout(metric_panel)
        self.metrics = {
            "gpu": MetricRing("GPU 利用率"),
            "vram": MetricRing("显存占用"),
            "temperature": MetricRing("GPU 温度"),
            "memory": MetricRing("系统内存"),
        }
        for i, ring in enumerate(self.metrics.values()):
            metric_layout.addWidget(ring, i // 2, i % 2)
        side.addWidget(self._card("硬件监控", metric_panel))

        failed_panel = QWidget()
        failed_layout = QVBoxLayout(failed_panel)
        failed_layout.setContentsMargins(0, 0, 0, 0)
        self.failed_table = QTableWidget(0, 4)
        self.failed_table.setHorizontalHeaderLabels(["文件名", "阶段", "失败原因", "操作"])
        self.failed_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.failed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.failed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        failed_layout.addWidget(self.failed_table)
        side.addWidget(self._card("失败任务", failed_panel))
        bottom.addWidget(side)
        bottom.setStretchFactor(0, 65)
        bottom.setStretchFactor(1, 35)
        outer.addWidget(bottom, 1)
        return page

    def _build_grid_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        toolbar = QHBoxLayout()
        self.grid_path = QPushButton("尚未选择图片文件夹")
        self.grid_path.setObjectName("pathButton")
        self.grid_path.setToolTip("点击前往参数配置")
        self.refresh_grid_button = QPushButton("刷新")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["按文件名", "按处理状态", "按二审评分"])
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["全部", "仅一审完成", "仅二审完成", "仅失败"])
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["批量操作", "批量重跑一审", "批量重跑二审", "批量导出"])
        toolbar.addWidget(self.grid_path, 1)
        toolbar.addWidget(self.refresh_grid_button)
        toolbar.addWidget(self.sort_combo)
        toolbar.addWidget(self.filter_combo)
        toolbar.addWidget(self.batch_combo)
        layout.addLayout(toolbar)
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.setSpacing(8)
        layout.addWidget(self.grid, 1)
        return page

    def _build_editor_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("editorPage")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        self.editor_splitter = QSplitter()
        self.editor_splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_grid = QGridLayout(left)
        left_grid.setContentsMargins(0, 0, 0, 0)
        self.preview = PreviewCanvas()
        left_grid.addWidget(self.preview, 0, 0, 3, 3)
        self.back_button = QPushButton("← 返回网格")
        self.back_button.setObjectName("overlayButton")
        left_grid.addWidget(self.back_button, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.filename = QLabel()
        self.filename.setObjectName("imageInfo")
        self.filename.setAlignment(Qt.AlignmentFlag.AlignRight)
        left_grid.addWidget(self.filename, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.prev_button = QPushButton("‹ 上一张")
        self.next_button = QPushButton("下一张 ›")
        self.prev_button.setObjectName("overlayButton")
        self.next_button.setObjectName("overlayButton")
        nav = QHBoxLayout()
        nav.addStretch()
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        nav.addStretch()
        left_grid.addLayout(nav, 2, 0, 1, 3, Qt.AlignmentFlag.AlignBottom)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        review_widget = QWidget()
        review_layout = QVBoxLayout(review_widget)
        review_head = QHBoxLayout()
        review_head.addWidget(QLabel("二审评估"))
        review_head.addStretch()
        self.review_status = QLabel("未完成")
        self.review_status.setObjectName("statusPill")
        review_head.addWidget(self.review_status)
        review_layout.addLayout(review_head)
        review_body = QHBoxLayout()
        self.score_label = QLabel("--")
        self.score_label.setObjectName("scoreLabel")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        review_body.addWidget(self.score_label, 1)
        self.review_comment = EditableComment()
        review_body.addWidget(self.review_comment, 3)
        review_layout.addLayout(review_body, 1)
        self.review_stats = QLabel("一审标签 0 · 二审保留 0 · 二审新增 0")
        review_layout.addWidget(self.review_stats)
        right_splitter.addWidget(self._card("", review_widget))

        tag_widget = QWidget()
        tag_layout = QVBoxLayout(tag_widget)
        tag_layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["标签名 (en)", "标签名 (cn)", "建议", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        tag_layout.addWidget(self.table, 1)
        tag_actions = QHBoxLayout()
        self.add_tag_button = QPushButton("新增标签")
        self.translate_missing_button = QPushButton("翻译缺失标签")
        self.export_current_button = QPushButton("导出当前标签")
        self.export_current_button.setObjectName("primaryButton")
        tag_actions.addWidget(self.add_tag_button)
        tag_actions.addWidget(self.translate_missing_button)
        tag_actions.addStretch()
        tag_actions.addWidget(self.export_current_button)
        tag_layout.addLayout(tag_actions)
        right_splitter.addWidget(self._card("标签编辑", tag_widget))
        right_splitter.setSizes([300, 700])
        self.editor_splitter.addWidget(left)
        self.editor_splitter.addWidget(right_splitter)
        outer.addWidget(self.editor_splitter)
        return page

    def _card(self, title: str, content) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            layout.addWidget(label)
        if isinstance(content, QWidget):
            layout.addWidget(content)
        else:
            layout.addLayout(content)
        return card

    def _path_field(self, field: QLineEdit, directory: bool) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QToolButton()
        button.setText("浏览")
        button.clicked.connect(lambda: self._choose(field, directory))
        layout.addWidget(field)
        layout.addWidget(button)
        return holder

    def _collapsible_path(self, title: str, field: QLineEdit) -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        toggle = QToolButton()
        toggle.setText(f"▸ {title}")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        path = self._path_field(field, False)
        path.setVisible(False)

        def changed(checked):
            toggle.setText(("▾ " if checked else "▸ ") + title)
            path.setVisible(checked)

        toggle.toggled.connect(changed)
        layout.addWidget(toggle)
        layout.addWidget(path)
        return holder

    def _progress_card(self, title: str):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        light = QLabel("●  等待")
        light.setObjectName("stageState")
        counts = ClickLabel("0 / 0   失败 0")
        counts.setObjectName("stageCounts")
        counts.setCursor(Qt.CursorShape.PointingHandCursor)
        counts.setToolTip("点击查看失败任务")
        counts.clicked.connect(self._focus_failed_jobs)
        header.addWidget(light)
        header.addStretch()
        header.addWidget(counts)
        bar = QProgressBar()
        bar.setTextVisible(False)
        bar.setMaximum(1)
        layout.addLayout(header)
        layout.addWidget(bar)
        card = self._card(title, content)
        bar.state_label = light
        bar.counts_label = counts
        return bar, card

    def _toggle_button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("paused", False)
        button.clicked.connect(lambda: callback(button))
        return button

    def _wire(self):
        self.start_button.clicked.connect(lambda: self.start("all"))
        self.retry_first_button.clicked.connect(lambda: self._retry_stage(Stage.FIRST_REVIEW))
        self.retry_second_button.clicked.connect(lambda: self._retry_stage(Stage.SECOND_REVIEW))
        self.refresh_models_button.clicked.connect(self.refresh_cloud_models)
        self.refresh_grid_button.clicked.connect(self.scan)
        self.grid_path.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.grid.itemDoubleClicked.connect(lambda item: self.open_image(item.data(Qt.ItemDataRole.UserRole)))
        self.grid.customContextMenuRequested.connect(self._grid_menu)
        self.grid.verticalScrollBar().valueChanged.connect(lambda _: QTimer.singleShot(30, self._load_visible_thumbnails))
        self.sort_combo.currentIndexChanged.connect(self._apply_grid_view)
        self.filter_combo.currentIndexChanged.connect(self._apply_grid_view)
        self.batch_combo.activated.connect(self._batch_action)
        self.progress_signal.connect(self._progress)
        self.log_signal.connect(self.append_log)
        self.log_filter.currentIndexChanged.connect(self._render_logs)
        self.clear_log_button.clicked.connect(self._clear_logs)
        self.export_log_button.clicked.connect(self._export_logs)
        self.failed_table.cellClicked.connect(self._failed_open)
        self.back_button.clicked.connect(self.return_to_grid)
        self.prev_button.clicked.connect(lambda: self.open_image(self.current - 1))
        self.next_button.clicked.connect(lambda: self.open_image(self.current + 1))
        self.table.itemChanged.connect(self._save_cell)
        self.table.cellClicked.connect(self._table_clicked)
        self.add_tag_button.clicked.connect(self.add_tag)
        self.translate_missing_button.clicked.connect(self.translate_missing_tags)
        self.export_current_button.clicked.connect(self.export_current)
        self.review_comment.edit_finished.connect(self._save_review_comment)
        self.editor_splitter.splitterMoved.connect(self._splitter_moved)
        for widget in self._config_widgets():
            signal = getattr(widget, "textChanged", None) or getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None) or getattr(widget, "toggled", None)
            if signal is not None:
                signal.connect(lambda *_: self._set_dirty(True))

    def _config_widgets(self):
        return (
            self.folder, self.model, self.mmproj, self.mtp, self.trigger, self.mode,
            self.first_concurrency, self.second_concurrency, self.key, self.proxy,
            self.cloud_model, self.base_url, self.columns, self.retry_count, self.resume_enabled, self.first_rule_mode,
        )

    def _install_shortcuts(self):
        self.shortcuts = []
        for sequence, parent, callback in (
            ("Esc", self, self.return_to_grid),
            ("F5", self, self.scan),
            (Qt.Key.Key_Left, self.editor_page, lambda: self.open_image(self.current - 1)),
            (Qt.Key.Key_Right, self.editor_page, lambda: self.open_image(self.current + 1)),
        ):
            shortcut = QShortcut(QKeySequence(sequence), parent)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def _restore(self):
        defaults = {
            "folder": "", "model": "", "mmproj": "", "mtp": "",
            "trigger": "anima_character", "proxy": "", "cloud_model": "gemini-3.5-flash",
            "base_url": "https://api.example.com/v1",
        }
        for name, default in defaults.items():
            value = str(self.settings.value(name, default))
            if name == "cloud_model":
                self.cloud_model.setCurrentText(value)
            else:
                getattr(self, name).setText(value)
        saved_key = get_api_key("default")
        if saved_key:
            self.key.setText(saved_key)
        self.columns.setValue(int(self.settings.value("columns", 10)))
        self.first_concurrency.setValue(int(self.settings.value("first_concurrency", 1)))
        self.second_concurrency.setValue(int(self.settings.value("second_concurrency", 2)))
        self.retry_count.setValue(int(self.settings.value("retry_count", 5)))
        self.resume_enabled.setChecked(str(self.settings.value("resume_enabled", "true")).lower() == "true")
        rule_mode = self.first_rule_mode.findData(self.settings.value("first_caption_rule_mode", "filter"))
        self.first_rule_mode.setCurrentIndex(max(0, rule_mode))
        index = self.mode.findData(self.settings.value("mode", "character"))
        self.mode.setCurrentIndex(max(0, index))

    def _initial_scan(self):
        self.repo.clear_events(30)
        if self.folder.text().strip() and Path(self.folder.text().strip()).is_dir():
            self.scan(quiet=True)

    def _set_dirty(self, dirty: bool):
        if self._restoring:
            return
        self.dirty_dot.setVisible(dirty)

    def _choose(self, target: QLineEdit, directory: bool):
        if directory:
            value = QFileDialog.getExistingDirectory(self, "选择图片文件夹", target.text())
        else:
            value = QFileDialog.getOpenFileName(self, "选择 GGUF", target.text(), "GGUF (*.gguf)")[0]
        if value:
            target.setText(value)

    def save_config(self):
        try:
            self._config(persist=True)
            self.scan(quiet=True)
        except Exception as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return
        self._set_dirty(False)
        self.statusBar().showMessage("配置已保存并同步到所有页面", 3000)

    def _config(self, persist: bool = False) -> RunConfig:
        if persist:
            for name in ("folder", "model", "mmproj", "mtp", "trigger", "proxy", "base_url"):
                self.settings.setValue(name, getattr(self, name).text().strip())
            self.settings.setValue("cloud_model", self.cloud_model.currentText().strip())
            self.settings.setValue("columns", self.columns.value())
            self.settings.setValue("first_concurrency", self.first_concurrency.value())
            self.settings.setValue("second_concurrency", self.second_concurrency.value())
            self.settings.setValue("retry_count", self.retry_count.value())
            self.settings.setValue("resume_enabled", self.resume_enabled.isChecked())
            self.settings.setValue("first_caption_rule_mode", self.first_rule_mode.currentData())
            self.settings.setValue("mode", self.mode.currentData())
            if self.key.text().strip():
                set_api_key("default", self.key.text().strip())
        return RunConfig(
            folder=Path(self.folder.text().strip()),
            lora_mode=LoraMode(self.mode.currentData()),
            trigger_word=self.trigger.text().strip(),
            gemma4_gguf_path=Path(self.model.text().strip()),
            mmproj_path=Path(self.mmproj.text().strip()),
            mtp_path=Path(self.mtp.text().strip()) if self.mtp.text().strip() else None,
            first_review_concurrency=self.first_concurrency.value(),
            second_review_concurrency=self.second_concurrency.value(),
            proxy_url=self.proxy.text().strip() or None,
            cloud_model=self.cloud_model.currentText().strip(),
            cloud_base_url=self.base_url.text().strip(),
            thumbnail_columns=self.columns.value(),
            api_retry_count=self.retry_count.value(),
            resume_enabled=self.resume_enabled.isChecked(),
            first_caption_rule_mode=self.first_rule_mode.currentData(),
        )

    def _warm_gemma4(self):
        """Load the configured vision model in the background while the UI becomes usable."""
        if self._gemma_warmup_active:
            return
        model_path = Path(self.model.text().strip())
        mmproj_path = Path(self.mmproj.text().strip())
        if not model_path.is_file() or not mmproj_path.is_file():
            return
        self._gemma_warmup_active = True
        adapter = LocalInferenceAdapter(model_path, mmproj_path)
        task = Task(lambda: (adapter.load(), adapter)[1])
        task.signals.finished.connect(self._gemma_warmed)
        task.signals.failed.connect(self._gemma_warmup_failed)
        self.pool.start(task)

    def _gemma_warmed(self, adapter: LocalInferenceAdapter):
        self._gemma_warmup_active = False
        self.controller.set_warm_local(adapter)
        self.statusBar().showMessage("Gemma4 已在后台加载完成", 4000)

    def _gemma_warmup_failed(self, text: str):
        self._gemma_warmup_active = False
        self.statusBar().showMessage(f"Gemma4 后台加载失败：{text}", 8000)

    def toggle_api_key_visibility(self):
        visible = self.key.echoMode() == QLineEdit.EchoMode.Password
        self.key.setEchoMode(QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password)
        self.toggle_key_button.setText("隐藏" if visible else "显示")

    def refresh_cloud_models(self):
        api_key = self.key.text().strip() or get_api_key("default")
        if not api_key:
            QMessageBox.warning(self, "缺少 API Key", "请先填写 Gemini Key。")
            return
        self.refresh_models_button.setEnabled(False)
        self.refresh_models_button.setText("正在获取…")
        task = Task(
            lambda: CloudReviewAdapter.list_models(
                api_key, self.base_url.text().strip(), self.proxy.text().strip() or None
            )
        )
        task.signals.finished.connect(self._models_loaded)
        task.signals.failed.connect(self._models_load_failed)
        self.pool.start(task)

    def _models_loaded(self, models):
        selected = self.cloud_model.currentText().strip()
        self.cloud_model.clear()
        self.cloud_model.addItems(models)
        self.cloud_model.setCurrentText(selected if selected in models else (models[0] if models else selected))
        self.refresh_models_button.setEnabled(True)
        self.refresh_models_button.setText("刷新模型")
        self.statusBar().showMessage(f"已获取 {len(models)} 个模型", 3000)

    def _models_load_failed(self, text):
        self.refresh_models_button.setEnabled(True)
        self.refresh_models_button.setText("刷新模型")
        QMessageBox.warning(self, "获取模型失败", text)

    def scan(self, quiet: bool = False):
        root = Path(self.folder.text().strip())
        if not root.is_dir():
            if not quiet:
                QMessageBox.warning(self, "无法刷新", "请先在参数配置页选择有效的图片文件夹并保存。")
            return False
        try:
            self.folder_id = self.repo.register_folder(root)
            self.assets = scan_images(root, self.folder_id)
            self.repo.upsert_images(self.assets)
            self.grid_path.setText(str(root))
            self._apply_grid_view()
            self._refresh_progress_from_repo()
            self._refresh_failed_jobs()
            return True
        except Exception as exc:
            if not quiet:
                QMessageBox.critical(self, "扫描失败", str(exc))
            return False

    def _asset_state(self, asset, states):
        state = states.get(asset.image_id, {})
        first = state.get("first_status")
        second = state.get("second_status")
        if is_nonempty(final_path(asset.path)):
            return "reviewed"
        if first in {"failed", "blocked"} or second in {"failed", "blocked"}:
            return "failed"
        if second in {"succeeded", "skipped"} or is_nonempty(stage_path(asset.path, 2)):
            return "second"
        if first in {"succeeded", "skipped"} or is_nonempty(stage_path(asset.path, 1)):
            return "first"
        return "pending"

    def _apply_grid_view(self):
        if not hasattr(self, "grid"):
            return
        states = self.repo.image_states(self.folder_id) if self.folder_id else {}
        assets = list(self.assets)
        filter_name = self.filter_combo.currentText()
        if filter_name == "仅一审完成":
            assets = [a for a in assets if self._asset_state(a, states) in {"first", "second", "reviewed"}]
        elif filter_name == "仅二审完成":
            assets = [a for a in assets if self._asset_state(a, states) in {"second", "reviewed"}]
        elif filter_name == "仅失败":
            assets = [a for a in assets if self._asset_state(a, states) == "failed"]
        if self.sort_combo.currentText() == "按处理状态":
            order = {"failed": 0, "pending": 1, "first": 2, "second": 3, "reviewed": 4}
            assets.sort(key=lambda a: (order[self._asset_state(a, states)], a.path.name.casefold()))
        elif self.sort_combo.currentText() == "按二审评分":
            assets.sort(key=lambda a: (-(states.get(a.image_id, {}).get("score") or -1), a.path.name.casefold()))
        else:
            assets.sort(key=lambda a: a.path.name.casefold())
        self.filtered_assets = assets
        size = max(88, min(190, int(max(600, self.grid.width() - 30) / self.columns.value()) - 16))
        self.grid.setIconSize(QSize(size, size))
        self.grid.clear()
        labels = {"pending": "●", "first": "✓", "second": "✓✓", "reviewed": "已复核", "failed": "!"}
        for asset in assets:
            state = self._asset_state(asset, states)
            item = QListWidgetItem(f"{asset.path.name}\n{labels[state]}")
            item.setData(Qt.ItemDataRole.UserRole, self.assets.index(asset))
            item.setData(self.STATE_ROLE, state)
            item.setData(self.LOADED_ROLE, False)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            item.setSizeHint(QSize(size + 18, size + 52))
            item.setToolTip(f"{asset.relative_path}\n状态：{state}")
            self.grid.addItem(item)
        QTimer.singleShot(0, self._load_visible_thumbnails)

    def _load_visible_thumbnails(self):
        if not self.grid.count():
            return
        viewport = self.grid.viewport().rect().adjusted(0, -250, 0, 250)
        size = self.grid.iconSize()
        colors = {"pending": "#9ca3af", "first": "#22c55e", "second": "#16a34a", "reviewed": "#2563eb", "failed": "#ef4444"}
        for i in range(self.grid.count()):
            item = self.grid.item(i)
            if item.data(self.LOADED_ROLE) or not self.grid.visualItemRect(item).intersects(viewport):
                continue
            asset = self.assets[item.data(Qt.ItemDataRole.UserRole)]
            source = QPixmap(asset.absolute_path)
            canvas = QPixmap(size)
            canvas.fill(QColor("#e5e7eb"))
            painter = QPainter(canvas)
            shown = source.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap((size.width() - shown.width()) // 2, (size.height() - shown.height()) // 2, shown)
            painter.setBrush(QColor(colors[item.data(self.STATE_ROLE)]))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(size.width() - 24, size.height() - 24, 16, 16)
            painter.end()
            item.setIcon(QIcon(canvas))
            item.setData(self.LOADED_ROLE, True)

    def _grid_menu(self, position):
        item = self.grid.itemAt(position)
        if not item:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        first = menu.addAction("重跑一审")
        second = menu.addAction("重跑二审")
        menu.addSeparator()
        folder = menu.addAction("打开所在文件夹")
        chosen = menu.exec(self.grid.mapToGlobal(position))
        if chosen == first:
            self._retry_asset(index, Stage.FIRST_REVIEW)
        elif chosen == second:
            self._retry_asset(index, Stage.SECOND_REVIEW)
        elif chosen == folder:
            os.startfile(str(self.assets[index].path.parent))

    def _batch_action(self, index: int):
        if index == 0:
            return
        if index == 3:
            self._batch_export()
            self.batch_combo.setCurrentIndex(0)
            return
        stage = Stage.FIRST_REVIEW if index == 1 else Stage.SECOND_REVIEW
        selected = self.grid.selectedItems() or [self.grid.item(i) for i in range(self.grid.count())]
        for item in selected:
            self._prepare_rerun(self.assets[item.data(Qt.ItemDataRole.UserRole)], stage)
        self.batch_combo.setCurrentIndex(0)
        self.start("first" if stage is Stage.FIRST_REVIEW else "second")

    def _batch_export(self):
        selected = self.grid.selectedItems()
        if selected:
            assets = [self.assets[item.data(Qt.ItemDataRole.UserRole)] for item in selected]
        else:
            assets = list(self.filtered_assets)
        trigger = self.trigger.text().strip()
        mode = LoraMode(self.mode.currentData())
        exported, skipped = 0, 0
        failures: list[tuple[str, str]] = []
        for asset in assets:
            try:
                self.review_service.export(asset, trigger, mode, overwrite=True, skip_validation=True)
                exported += 1
            except Exception as exc:
                skipped += 1
                failures.append((asset.path.name, str(exc)))
        self._apply_grid_view()
        msg = f"批量导出完成：{exported} 张成功，{skipped} 张跳过"
        if failures:
            detail = "；".join(f"{name}: {err[:60]}" for name, err in failures[:5])
            if len(failures) > 5:
                detail += f" …等 {len(failures)} 项"
            msg += f"（{detail}）"
        self.statusBar().showMessage(msg, 10000)

    def _retry_asset(self, index: int, stage: Stage):
        self._prepare_rerun(self.assets[index], stage)
        self.start("first" if stage is Stage.FIRST_REVIEW else "second")

    def _retry_stage(self, stage: Stage):
        if self._run_active:
            QMessageBox.information(self, "任务运行中", "请先暂停或等待当前任务完成。")
            return
        if not self.assets and not self.scan():
            return
        image_ids = self.repo.failed_stage_image_ids(self.folder_id, stage)
        if not image_ids:
            self.statusBar().showMessage(f"没有可重试的{('一审' if stage is Stage.FIRST_REVIEW else '二审')}失败任务", 3000)
            return
        assets = {asset.image_id: asset for asset in self.assets}
        for image_id in image_ids:
            asset = assets.get(image_id)
            if asset:
                self._prepare_rerun(asset, stage)
            else:
                self.repo.retry_job(image_id, stage)
        self.start("first" if stage is Stage.FIRST_REVIEW else "second")

    def _prepare_rerun(self, asset, stage: Stage):
        output = stage_path(asset.path, 1 if stage is Stage.FIRST_REVIEW else 2)
        if output.exists():
            output.unlink()
        self.repo.retry_job(asset.image_id, stage)

    def start(self, mode: str):
        if self._run_active:
            QMessageBox.information(self, "任务运行中", "当前已有任务运行，请使用暂停或恢复按钮。")
            return
        try:
            config = self._config()
        except Exception as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return
        if not self.assets and not self.scan():
            return
        self._run_active = True
        self.tabs.setCurrentIndex(1)
        task = Task(lambda: self.controller.run(config, mode))
        task.signals.finished.connect(self._run_finished)
        task.signals.failed.connect(self._run_failed)
        self.pool.start(task)

    def _run_finished(self, result):
        self._run_active = False
        self.scan(quiet=True)
        self.load_run_log()
        self.statusBar().showMessage(
            f"运行完成：一审 {result['first_done']}，二审 {result['second_done']}，失败 {result['failed']}", 6000
        )

    def _run_failed(self, text):
        self._run_active = False
        self.load_run_log()
        self._refresh_runtime()
        QMessageBox.critical(self, "运行失败", text)

    def _toggle_all(self, button):
        paused = not bool(button.property("paused"))
        (self.controller.pause_all if paused else self.controller.resume_all)()
        self._set_pause_button(button, paused, "全局")
        self._set_pause_button(self.first_pause, paused, "一审")
        self._set_pause_button(self.second_pause, paused, "二审")

    def _toggle_first(self, button):
        paused = not bool(button.property("paused"))
        (self.controller.pause_first if paused else self.controller.resume_first)()
        self._set_pause_button(button, paused, "一审")

    def _toggle_second(self, button):
        paused = not bool(button.property("paused"))
        (self.controller.pause_second if paused else self.controller.resume_second)()
        self._set_pause_button(button, paused, "二审")

    def _set_pause_button(self, button, paused: bool, name: str):
        button.setProperty("paused", paused)
        button.setText(name + ("恢复" if paused else "暂停"))
        button.style().unpolish(button)
        button.style().polish(button)

    def _progress(self, stage, done, total, name):
        bar = self.first_progress if stage == "first" else self.second_progress
        bar.setMaximum(max(1, total))
        bar.setValue(done)
        bar.state_label.setText("●  运行中")
        bar.counts_label.setText(f"{done} / {total}   {name}")
        self._refresh_failed_jobs()
        self._apply_grid_view()

    def _refresh_runtime(self):
        self._hardware()
        self._refresh_progress_from_repo()
        self._refresh_failed_jobs()

    def _refresh_progress_from_repo(self):
        if not self.folder_id:
            return
        for stage, bar in ((Stage.FIRST_REVIEW, self.first_progress), (Stage.SECOND_REVIEW, self.second_progress)):
            summary = self.repo.stage_summary(self.folder_id, stage)
            bar.setMaximum(max(1, summary["total"]))
            bar.setValue(summary["done"])
            state = "运行中" if self._run_active else ("已完成" if summary["total"] and summary["done"] == summary["total"] else "等待")
            bar.state_label.setText(f"●  {state}")
            bar.counts_label.setText(
                f"{summary['done']} / {summary['total']}   <span style='color:#ef4444'>失败 {summary['failed']}</span>"
            )

    def _hardware(self):
        data = sample_hardware()
        self.metrics["gpu"].set_metric(data["gpu_percent"], data["gpu"], data["gpu_percent"] >= 90)
        self.metrics["vram"].set_metric(data["vram_percent"], data["vram"], data["vram_percent"] >= 90)
        self.metrics["temperature"].set_metric(
            min(100, data["temperature_value"]), data["temperature"], data["temperature_value"] >= 85
        )
        self.metrics["memory"].set_metric(data["memory_percent"], data["memory"], data["memory_percent"] >= 90)

    def append_log(self, line: str):
        level = "ERROR" if line.startswith("ERROR") else "WARNING" if line.startswith("WARNING") else "INFO"
        self._log_lines.append((level, line))
        self._render_logs()

    def load_run_log(self):
        self._log_lines = []
        for event in self.repo.recent_events(500):
            image = f" [{event['image_id'][:8]}]" if event["image_id"] else ""
            stage = f" [{event['stage']}]" if event["stage"] else ""
            line = f"{self._beijing_time(event['created_at'])} {event['level']} {event['event_code']}{stage}{image}: {event['message']}"
            self._log_lines.append((event["level"], line))
        self._render_logs()

    @staticmethod
    def _beijing_time(value: str) -> str:
        try:
            return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Shanghai")).isoformat()
        except ValueError:
            return value

    def _render_logs(self):
        selected = self.log_filter.currentText()
        accepted = {"全部": {"INFO", "WARNING", "ERROR"}, "警告": {"WARNING"}, "错误": {"ERROR"}}[selected]
        colors = {"INFO": "#6b7280", "WARNING": "#d97706", "ERROR": "#dc2626"}
        icons = {"INFO": "●", "WARNING": "▲", "ERROR": "✖"}
        blocks = [
            f"<div style='color:{colors[level]}'>{icons[level]} {html.escape(line)}</div>"
            for level, line in self._log_lines if level in accepted
        ]
        self.log_view.setHtml("".join(blocks))
        if self.auto_scroll.isChecked():
            scroll = self.log_view.verticalScrollBar()
            scroll.setValue(scroll.maximum())

    def _clear_logs(self):
        deleted = self.repo.clear_events(0)
        self._log_lines.clear()
        self.log_view.clear()
        self.statusBar().showMessage(f"已清空 {deleted} 条日志", 3000)

    def _export_logs(self):
        path = QFileDialog.getSaveFileName(self, "导出日志", "anima-tagger.log", "Log (*.log);;Text (*.txt)")[0]
        if path:
            Path(path).write_text("\n".join(line for _, line in self._log_lines), encoding="utf-8")

    def _refresh_failed_jobs(self):
        if not self.folder_id:
            return
        rows = self.repo.failed_jobs(self.folder_id)
        self.failed_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            name = Path(row["absolute_path"]).name
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, row["image_id"])
            name_item.setData(self.STATE_ROLE, row["absolute_path"])
            self.failed_table.setItem(row_index, 0, name_item)
            self.failed_table.setItem(row_index, 1, QTableWidgetItem("一审" if row["stage"] == Stage.FIRST_REVIEW.value else "二审"))
            self.failed_table.setItem(row_index, 2, QTableWidgetItem(row["error_message"] or "未知错误"))
            retry = QPushButton("重试")
            retry.clicked.connect(lambda _, image_id=row["image_id"], stage=row["stage"]: self._retry_failed(image_id, stage))
            self.failed_table.setCellWidget(row_index, 3, retry)

    def _retry_failed(self, image_id: str, stage_name: str):
        stage = Stage(stage_name)
        asset = next((item for item in self.assets if item.image_id == image_id), None)
        if asset:
            self._prepare_rerun(asset, stage)
        else:
            self.repo.retry_job(image_id, stage)
        self.start("first" if stage is Stage.FIRST_REVIEW else "second")

    def _failed_open(self, row: int, column: int):
        if column != 0:
            return
        image_id = self.failed_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        for index, asset in enumerate(self.assets):
            if asset.image_id == image_id:
                self.open_image(index)
                return

    def _focus_failed_jobs(self):
        self.failed_table.setFocus()
        if self.failed_table.rowCount():
            self.failed_table.selectRow(0)
            self.failed_table.scrollToItem(self.failed_table.item(0, 0))

    def open_image(self, index: int):
        if not self.assets or index < 0 or index >= len(self.assets):
            return
        self.current = index
        asset = self.assets[index]
        image = QImage(asset.absolute_path)
        self.filename.setText(f"{asset.path.name}\n{image.width()} × {image.height()}")
        self.preview.set_image(asset.absolute_path)
        self.load_tags()
        if self.root_stack.currentIndex() == 0:
            self._grid_scroll = self.grid.verticalScrollBar().value()
        self.root_stack.setCurrentIndex(1)

    def return_to_grid(self):
        if self.root_stack.currentIndex() != 1:
            return
        self.root_stack.setCurrentIndex(0)
        self.tabs.setCurrentIndex(2)
        self._apply_grid_view()
        QTimer.singleShot(0, lambda: self.grid.verticalScrollBar().setValue(self._grid_scroll))

    def load_tags(self):
        asset = self.assets[self.current]
        review = self.repo.review(asset.image_id)
        rows = self.repo.tags(asset.image_id)
        if review:
            score = review["score"]
            color = "#16a34a" if score >= 8 else "#d97706" if score >= 5 else "#dc2626"
            self.score_label.setText(f"{score}\n/ 10")
            self.score_label.setStyleSheet(f"color:{color};")
            self.review_comment.setPlainText(review["comment"] or "")
            self.review_status.setText(review["status"] or "已完成")
        else:
            self.score_label.setText("--")
            self.score_label.setStyleSheet("color:#9ca3af;")
            self.review_comment.clear()
            self.review_status.setText("未完成")
        first = sum(row["source"] == "first_review" for row in rows)
        kept = sum(row["source"] == "first_review" and row["enabled"] for row in rows)
        added = sum(row["source"] in {"second_review", "manual"} for row in rows)
        self.review_stats.setText(f"一审标签 {first} · 二审保留 {kept} · 二审新增 {added}")
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [row["en"], row["cn"] or "", row["suggestion"] or ""]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row["tag_id"])
                self.table.setItem(r, c, item)
            status = QTableWidgetItem("启用" if row["enabled"] else "停用")
            status.setData(Qt.ItemDataRole.UserRole, row["tag_id"])
            status.setData(self.STATE_ROLE, bool(row["enabled"]))
            status.setFlags(status.flags() & ~Qt.ItemFlag.ItemIsEditable)
            status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status.setForeground(QColor("#16a34a" if row["enabled"] else "#9ca3af"))
            self.table.setItem(r, 3, status)
        self.table.blockSignals(False)

    def _table_clicked(self, row: int, column: int):
        if column != 3:
            return
        item = self.table.item(row, column)
        enabled = not bool(item.data(self.STATE_ROLE))
        item.setData(self.STATE_ROLE, enabled)
        item.setText("启用" if enabled else "停用")
        item.setForeground(QColor("#16a34a" if enabled else "#9ca3af"))
        self._save_table_row(row)

    def _save_cell(self, item):
        if item.column() < 3:
            self._save_table_row(item.row())

    def _save_table_row(self, row: int):
        items = [self.table.item(row, c) for c in range(4)]
        if any(item is None for item in items):
            return
        self.repo.update_tag(
            items[0].data(Qt.ItemDataRole.UserRole),
            items[0].text(), items[1].text(), items[2].text(),
            bool(items[3].data(self.STATE_ROLE)),
        )

    def add_tag(self):
        if self.current >= 0:
            self.repo.add_tag(self.assets[self.current].image_id, "new tag")
            self.load_tags()

    def translate_missing_tags(self):
        if self.current < 0:
            return
        asset = self.assets[self.current]
        tags = [
            row["en"]
            for row in self.repo.tags(asset.image_id)
            if not row["is_natural_language"] and not (row["cn"] or "").strip()
        ]
        if not tags:
            self.statusBar().showMessage("当前图片没有缺失翻译的标签", 3000)
            return
        model_path = Path(self.model.text().strip())
        if not model_path.is_file():
            QMessageBox.warning(self, "模型路径无效", "请先配置有效的 Gemma4 GGUF 模型路径。")
            return
        self.translate_missing_button.setEnabled(False)
        self.translate_missing_button.setText("正在翻译...")
        task = Task(lambda: LocalInferenceAdapter(model_path, Path(), n_ctx=4096, n_gpu_layers=-1).translate_tags(tags))
        task.signals.finished.connect(lambda translations: self._translations_ready(asset.image_id, translations))
        task.signals.failed.connect(self._translations_failed)
        self.pool.start(task)

    def _translations_ready(self, image_id: str, translations: dict[str, str]):
        updated = self.repo.update_missing_tag_translations(image_id, translations)
        self.translate_missing_button.setEnabled(True)
        self.translate_missing_button.setText("翻译缺失标签")
        if self.current >= 0 and self.assets[self.current].image_id == image_id:
            self.load_tags()
        self.statusBar().showMessage(f"已补齐 {updated} 个标签翻译", 4000)

    def _translations_failed(self, text: str):
        self.translate_missing_button.setEnabled(True)
        self.translate_missing_button.setText("翻译缺失标签")
        QMessageBox.warning(self, "标签翻译失败", text)

    def _save_review_comment(self, text: str):
        if self.current >= 0 and self.repo.review(self.assets[self.current].image_id):
            self.repo.update_review_comment(self.assets[self.current].image_id, text)

    def export_current(self):
        if self.current < 0:
            return
        overwrite = final_path(self.assets[self.current].path).exists()
        if overwrite and QMessageBox.question(self, "覆盖确认", "最终标签文件已存在，是否覆盖？") != QMessageBox.StandardButton.Yes:
            return
        try:
            destination = self.review_service.export(
                self.assets[self.current], self.trigger.text().strip(), LoraMode(self.mode.currentData()), overwrite=overwrite
            )
            self.statusBar().showMessage(f"已导出 {destination.name}", 4000)
            self._apply_grid_view()
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _splitter_moved(self, position, index):
        total = sum(self.editor_splitter.sizes())
        if not total:
            return
        left = max(int(total * 0.3), min(position, int(total * 0.7)))
        if left != position:
            self.editor_splitter.setSizes([left, total - left])
        self.repo.set_preference("splitter_ratio", str(left / total))
        self.preview.reload_from_original()

    def showEvent(self, event):
        super().showEvent(event)
        ratio = float(self.repo.preference("splitter_ratio", "0.5"))
        self.editor_splitter.setSizes([int(1000 * ratio), int(1000 * (1 - ratio))])
        self.load_run_log()

    @staticmethod
    def _style_sheet() -> str:
        return """
        QMainWindow, QWidget { background:#f5f7fa; color:#1f2937; font-size:13px; }
        QTabWidget::pane { border:0; }
        QTabBar::tab { padding:12px 24px; border-bottom:2px solid transparent; }
        QTabBar::tab:selected { color:#2563eb; border-bottom:2px solid #2563eb; }
        QFrame#card { background:white; border:1px solid #e5e7eb; border-radius:8px; }
        QLabel#cardTitle { font-size:15px; font-weight:600; padding-bottom:6px; }
        QLabel#dirtyDot { color:#f59e0b; }
        QLabel#stageState { color:#16a34a; }
        QLabel#stageCounts { color:#4b5563; }
        QLabel#statusPill { background:#eef2ff; color:#4f46e5; padding:4px 8px; border-radius:8px; }
        QLabel#scoreLabel { font-size:28px; font-weight:700; }
        QLabel#imageInfo { background:rgba(17,24,39,190); color:white; padding:8px 12px; border-radius:6px; }
        QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit, QTableWidget {
            background:white; border:1px solid #d1d5db; border-radius:5px; padding:5px;
        }
        QSpinBox { padding:5px 24px 5px 5px; }
        QSpinBox::up-button, QSpinBox::down-button {
            subcontrol-origin:border; width:18px; border-left:1px solid #d1d5db;
        }
        QSpinBox::up-button { subcontrol-position:top right; border-bottom:1px solid #d1d5db; }
        QSpinBox::down-button { subcontrol-position:bottom right; }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover { background:#eff6ff; }
        QPushButton, QToolButton { background:#fff; border:1px solid #d1d5db; border-radius:5px; padding:7px 12px; }
        QPushButton:hover, QToolButton:hover { border-color:#60a5fa; background:#eff6ff; }
        QPushButton#primaryButton { background:#2563eb; color:white; border:0; padding:9px 18px; }
        QPushButton[kind="start"] { background:#16a34a; }
        QPushButton[kind="retry"] { background:#dc2626; color:white; border:0; }
        QPushButton[paused="true"] { background:#16a34a; color:white; }
        QPushButton[paused="false"] { background:#f59e0b; color:white; }
        QPushButton#overlayButton { background:rgba(17,24,39,190); color:white; border:0; margin:12px; }
        QPushButton#pathButton { text-align:left; color:#4b5563; background:transparent; border:0; }
        QProgressBar { background:#e5e7eb; border:0; border-radius:6px; height:12px; }
        QProgressBar::chunk { border-radius:6px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #60a5fa,stop:1 #2563eb); }
        QListWidget { border:0; background:#f5f7fa; }
        QListWidget::item { background:white; border:1px solid #e5e7eb; border-radius:6px; padding:4px; }
        QListWidget::item:hover { border-color:#93c5fd; background:#eff6ff; }
        QListWidget::item:selected { border:2px solid #3b82f6; background:#dbeafe; }
        QTableWidget::item:hover { background:#f3f4f6; }
        QHeaderView::section { background:#f9fafb; border:0; border-bottom:1px solid #e5e7eb; padding:7px; }
        """
