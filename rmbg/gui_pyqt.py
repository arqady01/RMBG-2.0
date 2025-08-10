#!/usr/bin/env python3
import os
import sys
import platform
import pathlib
from typing import List, Optional

from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt
from PyQt6 import QtCore, QtGui, QtWidgets
import torch

# Import processing utilities from the CLI module
try:
    from rmbg.rmbg_cli import get_device, load_model, process_one
except Exception:
    # Fallback: allow running the script directly when package import is not available
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from rmbg_cli import get_device, load_model, process_one  # type: ignore


# Watchdog for directory monitoring (workflow integration)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except Exception:
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore
    WATCHDOG_AVAILABLE = False


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".jfif", ".avif"}


def get_icon(name: str) -> QtGui.QIcon:
    # Icon loading is kept for potential future use; returns empty icon if not found/unused
    base = pathlib.Path(__file__).resolve().parent / "assets" / "icons" / f"{name}.svg"
    if base.exists():
        return QtGui.QIcon(str(base))
    return QtGui.QIcon()


class _ImageFileHandler(FileSystemEventHandler):  # type: ignore
    def __init__(self, on_created_cb, on_deleted_cb, on_moved_cb):
        super().__init__()
        self._on_created_cb = on_created_cb
        self._on_deleted_cb = on_deleted_cb
        self._on_moved_cb = on_moved_cb

    def on_created(self, event):  # type: ignore
        if getattr(event, "is_directory", False):
            return
        self._maybe_emit_created(event.src_path)

    def on_moved(self, event):  # type: ignore
        if getattr(event, "is_directory", False):
            return
        src = getattr(event, "src_path", "")
        dest = getattr(event, "dest_path", "")
        self._maybe_emit_moved(src, dest)

    def on_deleted(self, event):  # type: ignore
        if getattr(event, "is_directory", False):
            return
        self._maybe_emit_deleted(event.src_path)
    def _maybe_emit_created(self, path: str) -> None:
        try:
            p = pathlib.Path(path)
            if p.suffix.lower() in IMAGE_EXTS:
                self._on_created_cb([str(p)])
        except Exception:
            pass

    def _maybe_emit_deleted(self, path: str) -> None:
        try:
            p = pathlib.Path(path)
            if p.suffix.lower() in IMAGE_EXTS:
                self._on_deleted_cb([str(p)])
        except Exception:
            pass

    def _maybe_emit_moved(self, src: str, dest: str) -> None:
        try:
            sp = pathlib.Path(src)
            dp = pathlib.Path(dest) if dest else None
            # Treat as move when both sides present; otherwise fallback to create/delete
            if dp and (sp.suffix.lower() in IMAGE_EXTS or dp.suffix.lower() in IMAGE_EXTS):
                self._on_moved_cb([(str(sp), str(dp))])
            else:
                # If destination missing or extension changed away from images, emit delete
                if sp.suffix.lower() in IMAGE_EXTS:
                    self._on_deleted_cb([str(sp)])
                if dp and dp.suffix.lower() in IMAGE_EXTS:
                    self._on_created_cb([str(dp)])
        except Exception:
            pass


class DirectoryWatcher(QtCore.QThread):
    files_detected = QtCore.pyqtSignal(list)
    files_deleted = QtCore.pyqtSignal(list)
    files_moved = QtCore.pyqtSignal(list)  # list of (old, new)
    status_changed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, watch_dir: pathlib.Path, recursive: bool = True, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.watch_dir = watch_dir
        self.recursive = recursive
        self._should_stop = False
        self._observer: Optional[Observer] = None  # type: ignore

    def stop(self) -> None:
        self._should_stop = True

    def _on_new_files(self, paths: List[str]) -> None:
        self.files_detected.emit(paths)

    def _on_deleted_files(self, paths: List[str]) -> None:
        self.files_deleted.emit(paths)

    def _on_moved_files(self, pairs: List[tuple]) -> None:
        self.files_moved.emit(pairs)

    def run(self) -> None:
        if not WATCHDOG_AVAILABLE:
            self.failed.emit("watchdog 未安装，无法启用目录监控。请先 pip install watchdog。")
            return
        try:
            handler = _ImageFileHandler(self._on_new_files, self._on_deleted_files, self._on_moved_files)
            observer = Observer()
            observer.schedule(handler, str(self.watch_dir), recursive=self.recursive)
            observer.start()
            self._observer = observer
            self.status_changed.emit(f"已开始监控: {self.watch_dir}")
            while not self._should_stop:
                self.msleep(200)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            try:
                if self._observer is not None:
                    self._observer.stop()
                    self._observer.join(timeout=3)
            except Exception:
                pass
            self.status_changed.emit("已停止监控")


class BackgroundRemoverWorker(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    status_changed = QtCore.pyqtSignal(str)
    image_done = QtCore.pyqtSignal(str, str)  # (input_path, output_path)
    finished_success = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)
    file_error = QtCore.pyqtSignal(str, str)  # (input_path, error_message)

    def __init__(
        self,
        input_files: List[pathlib.Path],
        output_dir: Optional[pathlib.Path],
        device_choice: str,
        input_size: int,
        output_size: Optional[int],
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.input_files = input_files
        self.output_dir = output_dir
        self.device_choice = device_choice
        self.input_size = input_size
        self.output_size = output_size

    def run(self) -> None:
        try:
            # Enable MPS fallback on macOS to survive unsupported ops
            if platform.system() == "Darwin":
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            self.status_changed.emit("Loading model...")
            device = get_device(self.device_choice)
            model = load_model(device)

            total = len(self.input_files)
            for idx, in_path in enumerate(self.input_files, start=1):
                try:
                    if self.output_dir:
                        out_dir = self.output_dir
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{in_path.stem}_rmbg.png"
                    else:
                        out_path = in_path.with_name(f"{in_path.stem}_rmbg.png")

                    self.status_changed.emit(f"[{idx}/{total}] 正在处理: {in_path}")
                    saved_path = process_one(
                        model,
                        device,
                        str(in_path),
                        str(out_path),
                        self.input_size,
                        output_size=self.output_size,
                    )
                    self.status_changed.emit(f"[{idx}/{total}] 完成: {in_path} -> {saved_path}")
                    self.image_done.emit(str(in_path), saved_path)
                except Exception as e:  # continue on error per file
                    self.status_changed.emit(f"[{idx}/{total}] 失败: {in_path} | 错误: {e}")
                    self.file_error.emit(str(in_path), str(e))
                finally:
                    self.progress_changed.emit(int(idx * 100 / total))

            self.status_changed.emit("All done.")
            self.finished_success.emit()
        except Exception as e:
            self.failed.emit(str(e))


class ImagePreview(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(240, 240)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setScaledContents(False)
        self._pil_image: Optional[Image.Image] = None
        self._zoom_factor: float = 1.0
        self._fit_to_window: bool = True
        self._checkerboard: bool = False

        # Prebuild checkerboard tile
        self._checker_tile = self._build_checker_tile(10)

    def _build_checker_tile(self, size: int) -> QtGui.QPixmap:
        pix = QtGui.QPixmap(size * 2, size * 2)
        pix.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(pix)
        color = QtGui.QColor(220, 220, 220)
        painter.fillRect(0, 0, size, size, color)
        painter.fillRect(size, size, size, size, color)
        painter.end()
        return pix

    def set_checkerboard(self, enabled: bool) -> None:
        self._checkerboard = enabled
        self.update()

    def set_fit_to_window(self, enabled: bool) -> None:
        self._fit_to_window = enabled
        self.update_display()

    def set_zoom_factor(self, factor: float) -> None:
        self._zoom_factor = max(0.1, min(4.0, factor))
        if not self._fit_to_window:
            self.update_display()

    def set_pil_image(self, im: Image.Image) -> None:
        self._pil_image = im
        self.update_display()

    def update_display(self) -> None:
        if self._pil_image is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText("No preview")
            return
        qim = ImageQt(self._pil_image)
        pix = QtGui.QPixmap.fromImage(qim)
        if self._fit_to_window:
            scaled = pix.scaled(
                self.width(),
                self.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        else:
            w = int(pix.width() * self._zoom_factor)
            h = int(pix.height() * self._zoom_factor)
            scaled = pix.scaled(w, h, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")

    def clear_preview(self) -> None:
        self._pil_image = None
        self.setPixmap(QtGui.QPixmap())
        self.setText("No preview")

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._fit_to_window or self._pil_image is None:
            event.ignore()
            return
        delta = event.angleDelta().y() / 120.0
        self.set_zoom_factor(self._zoom_factor * (1.0 + 0.1 * delta))
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._fit_to_window:
            self.update_display()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self._checkerboard:
            painter = QtGui.QPainter(self)
            brush = QtGui.QBrush(self._checker_tile)
            painter.fillRect(self.rect(), brush)
            painter.end()
        super().paintEvent(event)


class CompareSlider(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pix_in: Optional[QtGui.QPixmap] = None
        self._pix_out: Optional[QtGui.QPixmap] = None
        self._ratio: float = 0.5
        self._fit_to_window: bool = True
        self._checkerboard: bool = False
        self._checker_tile = self._build_checker_tile(10)
        self.setMinimumHeight(240)

        # Optional HUD slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(self._ratio * 100))
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def _build_checker_tile(self, size: int) -> QtGui.QPixmap:
        pix = QtGui.QPixmap(size * 2, size * 2)
        pix.fill(QtCore.Qt.GlobalColor.white)
        painter = QtGui.QPainter(pix)
        color = QtGui.QColor(220, 220, 220)
        painter.fillRect(0, 0, size, size, color)
        painter.fillRect(size, size, size, size, color)
        painter.end()
        return pix

    def set_fit_to_window(self, enabled: bool) -> None:
        self._fit_to_window = enabled
        self.update()

    def set_checkerboard(self, enabled: bool) -> None:
        self._checkerboard = enabled
        self.update()

    def set_images(self, im_in: Optional[Image.Image], im_out: Optional[Image.Image]) -> None:
        def to_pix(img: Optional[Image.Image]) -> Optional[QtGui.QPixmap]:
            if img is None:
                return None
            qim = ImageQt(img)
            return QtGui.QPixmap.fromImage(qim)

        self._pix_in = to_pix(im_in)
        self._pix_out = to_pix(im_out)
        self.update()

    def _on_slider(self, value: int) -> None:
        self._ratio = max(0.0, min(1.0, value / 100.0))
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # position slider at bottom
        self.slider.setGeometry(10, self.height() - 28, self.width() - 20, 18)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        if self._checkerboard:
            brush = QtGui.QBrush(self._checker_tile)
            painter.fillRect(self.rect(), brush)

        def scaled(pix: QtGui.QPixmap) -> QtGui.QPixmap:
            if self._fit_to_window:
                return pix.scaled(self.width(), self.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            return pix

        if self._pix_in is not None:
            pix_in = scaled(self._pix_in)
            x_in = (self.width() - pix_in.width()) // 2
            y_in = (self.height() - pix_in.height()) // 2
            painter.drawPixmap(x_in, y_in, pix_in)

        if self._pix_out is not None:
            pix_out = scaled(self._pix_out)
            x_out = (self.width() - pix_out.width()) // 2
            y_out = (self.height() - pix_out.height()) // 2
            # Clip to ratio
            clip_w = int(pix_out.width() * self._ratio)
            painter.save()
            painter.setClipRect(x_out, y_out, clip_w, pix_out.height())
            painter.drawPixmap(x_out, y_out, pix_out)
            painter.restore()

        # Draw divider line
        if self._pix_out is not None:
            pix_out = scaled(self._pix_out)
            x_out = (self.width() - pix_out.width()) // 2
            y_out = (self.height() - pix_out.height()) // 2
            x_div = x_out + int(pix_out.width() * self._ratio)
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 200))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(x_div, y_out, x_div, y_out + pix_out.height())
        painter.end()


class Card(QtWidgets.QFrame):
    def __init__(self, title: str, content: QtWidgets.QWidget, icon: Optional[QtGui.QIcon] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setObjectName("card")
        v = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        if icon is not None:
            lab_icon = QtWidgets.QLabel()
            lab_icon.setPixmap(icon.pixmap(16, 16))
            header.addWidget(lab_icon)
        lab_title = QtWidgets.QLabel(title)
        lab_title.setObjectName("cardTitle")
        header.addWidget(lab_title)
        header.addStretch(1)
        v.addLayout(header)
        v.addWidget(content)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RMBG-2.0 Background Remover (PyQt6)")
        self.resize(1200, 760)
        # Apply button-only black/white styling, keep overall design light
        self.setStyleSheet(
            """
            QPushButton { background: #FFFFFF; color: #000000; border: 1px solid #000000; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background: #000000; color: #FFFFFF; }
            QPushButton:pressed { background: #000000; color: #FFFFFF; }
            QPushButton:disabled { background: #F2F2F2; color: #B0B0B0; border: 1px solid #D0D0D0; }
            #cardTitle { font-weight: 600; }
            """
        )

        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # Top controls
        controls = QtWidgets.QHBoxLayout()
        self.btn_add_files = QtWidgets.QPushButton("选择图片…")
        self.btn_add_dir = QtWidgets.QPushButton("选择文件夹…")
        self.btn_clear = QtWidgets.QPushButton("清空")

        controls.addWidget(self.btn_add_files)
        controls.addWidget(self.btn_add_dir)
        controls.addWidget(self.btn_clear)
        controls.addStretch(1)

        self.combo_device = QtWidgets.QComboBox()
        self.combo_device.addItems(["auto", "cpu", "mps", "cuda"])
        self.combo_device.setCurrentText("auto")
        device_tooltip = (
            "设备：选择推理设备\n"
            "- auto：自动选择（优先 Apple Silicon 的 MPS，其次 CUDA，否则 CPU）。在 macOS 上如遇不支持的算子会自动回退到 CPU。\n"
            "- cpu：最稳定、兼容性最好，速度较慢。\n"
            "- mps：Apple Silicon 的 Metal 加速，个别算子可能不支持，程序会回退到 CPU。\n"
            "- cuda：NVIDIA GPU + CUDA 环境可用。"
        )
        self.combo_device.setToolTip(device_tooltip)
        self.lbl_device = QtWidgets.QLabel("设备:")
        self.lbl_device.setToolTip(device_tooltip)
        controls.addWidget(self.lbl_device)
        controls.addWidget(self.combo_device)

        # 输入尺寸下拉与说明
        controls.addSpacing(12)
        self.lbl_input_size = QtWidgets.QLabel("输入尺寸:")
        size_tooltip = (
            "输入尺寸：将图片在送入模型前缩放到该大小的正方形进行推理。\n"
            "尺寸越大，细节更好但速度更慢、占用更高；尺寸越小，速度更快但细节减少。\n"
            "导出的结果尺寸仍与原图一致（掩码会缩放回原图大小）。"
        )
        self.lbl_input_size.setToolTip(size_tooltip)

        self.combo_size = QtWidgets.QComboBox()
        self.combo_size.setToolTip(size_tooltip)
        size_options = [512, 640, 768, 896, 1024, 1152, 1280, 1536]
        for s in size_options:
            self.combo_size.addItem(str(s), userData=s)
        # 默认 1024
        default_index = size_options.index(1024) if 1024 in size_options else 0
        self.combo_size.setCurrentIndex(default_index)

        controls.addWidget(self.lbl_input_size)
        controls.addWidget(self.combo_size)

        self.btn_output_dir = QtWidgets.QPushButton("输出目录…")
        self.lbl_output_dir = QtWidgets.QLabel("未设置（默认与输入同目录）")
        self.lbl_output_dir.setStyleSheet("color: #666;")
        controls.addSpacing(12)
        controls.addWidget(self.btn_output_dir)
        controls.addWidget(self.lbl_output_dir, 1)

        # 输出尺寸（可选）
        controls.addSpacing(12)
        self.lbl_output_size = QtWidgets.QLabel("输出尺寸:")
        out_size_tooltip = (
            "输出尺寸：将最终导出的 PNG 调整为指定的正方形大小，并用透明边距居中填充。\n"
            "选择“原图”则保持与原始尺寸一致。"
        )
        self.lbl_output_size.setToolTip(out_size_tooltip)
        self.combo_output_size = QtWidgets.QComboBox()
        self.combo_output_size.setToolTip(out_size_tooltip)
        self.combo_output_size.addItem("原图", userData=None)
        for s in [512, 640, 768, 896, 1024, 1152, 1280, 1536]:
            self.combo_output_size.addItem(str(s), userData=s)
        self.combo_output_size.setCurrentIndex(0)
        controls.addWidget(self.lbl_output_size)
        controls.addWidget(self.combo_output_size)

        # 监控文件夹（工作流集成）
        controls.addSpacing(12)
        self.btn_watch_dir = QtWidgets.QPushButton("监控当前列表目录")
        self.btn_stop_watch = QtWidgets.QPushButton("停止监控")
        self.btn_stop_watch.setEnabled(False)
        watch_tip = (
            "监控当前列表中首个文件所在的文件夹（含子目录），新图片将自动加入待处理队列。\n"
            "建议：先添加一个该目录下的文件以确定监控目标。也可在没有文件时手动添加后再启动。\n"
            "需安装 watchdog：pip install watchdog"
        )
        self.btn_watch_dir.setToolTip(watch_tip)
        self.btn_stop_watch.setToolTip(watch_tip)
        controls.addWidget(self.btn_watch_dir)
        controls.addWidget(self.btn_stop_watch)

        main_layout.addLayout(controls)

        # Middle: list + previews
        middle = QtWidgets.QHBoxLayout()

        left = QtWidgets.QVBoxLayout()
        self.list_files = QtWidgets.QListWidget()
        self.list_files.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left.addWidget(QtWidgets.QLabel("待处理文件"))
        # 快速筛选 + 删除所选
        filter_row = QtWidgets.QHBoxLayout()
        self.edit_filter = QtWidgets.QLineEdit()
        self.edit_filter.setPlaceholderText("快速筛选（按文件名或完整路径匹配，模糊包含）")
        self.edit_filter.setClearButtonEnabled(True)
        self.edit_filter.setToolTip("输入关键字筛选列表：按文件名或完整路径匹配；支持模糊包含。\n例如输入 'cat' 将匹配文件名或路径中包含 'cat' 的条目。")
        self.btn_delete_selected = QtWidgets.QPushButton("删除所选")
        self.btn_delete_selected.setToolTip("删除当前列表中已选中的文件项（仅从列表移除，不会删除磁盘文件）")
        filter_row.addWidget(self.edit_filter, 1)
        filter_row.addWidget(self.btn_delete_selected)
        left.addLayout(filter_row)
        left.addWidget(self.list_files, 1)

        # Completed list
        left.addWidget(QtWidgets.QLabel("已完成"))
        self.list_done = QtWidgets.QListWidget()
        self.list_done.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        left.addWidget(self.list_done, 1)
        middle.addLayout(left, 3)

        # Center: tabbed previews (输入 / 输出 / 对比滑块)
        center = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()
        # 输入
        tab_in = QtWidgets.QWidget()
        v_in = QtWidgets.QVBoxLayout(tab_in)
        self.preview_input = ImagePreview()
        v_in.addWidget(self.preview_input, 1)
        # 输出
        tab_out = QtWidgets.QWidget()
        v_out = QtWidgets.QVBoxLayout(tab_out)
        self.preview_output = ImagePreview()
        v_out.addWidget(self.preview_output, 1)
        # 对比滑块
        tab_cmp = QtWidgets.QWidget()
        v_cmp = QtWidgets.QVBoxLayout(tab_cmp)
        self.compare = CompareSlider()
        v_cmp.addWidget(self.compare, 1)

        self.tabs.addTab(tab_in, "输入")
        self.tabs.addTab(tab_out, "输出")
        self.tabs.addTab(tab_cmp, "对比")

        center.addWidget(self.tabs, 1)

        # Preview controls: fit/zoom toggle and checkerboard toggle
        controls_row = QtWidgets.QHBoxLayout()
        self.chk_fit = QtWidgets.QCheckBox("适应窗口")
        self.chk_fit.setChecked(True)
        self.chk_checker = QtWidgets.QCheckBox("棋盘格背景")
        controls_row.addWidget(self.chk_fit)
        controls_row.addWidget(self.chk_checker)
        controls_row.addStretch(1)
        center.addLayout(controls_row)

        # Wire preview toggles
        def _apply_fit(val: bool) -> None:
            for w in (self.preview_input, self.preview_output):
                w.set_fit_to_window(val)
            self.compare.set_fit_to_window(val)
        def _apply_checker(val: bool) -> None:
            for w in (self.preview_input, self.preview_output):
                w.set_checkerboard(val)
            self.compare.set_checkerboard(val)
        self.chk_fit.toggled.connect(_apply_fit)
        self.chk_checker.toggled.connect(_apply_checker)

        middle.addLayout(center, 5)
        main_layout.addLayout(middle, 1)

        # Bottom: status bar + collapsible log panel
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        # Collapsible log
        self.grp_log = QtWidgets.QGroupBox("进度日志")
        self.grp_log.setCheckable(True)
        self.grp_log.setChecked(False)
        v_log = QtWidgets.QVBoxLayout(self.grp_log)
        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(120)
        v_log.addWidget(self.txt_log)
        main_layout.addWidget(self.grp_log)

        # Right-side controls: parameters as cards (with icons)
        right = QtWidgets.QVBoxLayout()
        def make_card(title: str, widget: QtWidgets.QWidget) -> Card:
            return Card(title, widget)

        # Device + sizes card
        card1_widget = QtWidgets.QWidget()
        c1 = QtWidgets.QFormLayout(card1_widget)
        c1.addRow(self.lbl_device, self.combo_device)
        c1.addRow(self.lbl_input_size, self.combo_size)
        c1.addRow(self.lbl_output_size, self.combo_output_size)
        card1 = make_card("设备与尺寸", card1_widget)

        # Output directory card
        card2_widget = QtWidgets.QWidget()
        c2 = QtWidgets.QHBoxLayout(card2_widget)
        c2.addWidget(self.btn_output_dir)
        c2.addWidget(self.lbl_output_dir, 1)
        card2 = make_card("输出目录", card2_widget)

        # Watch card
        card3_widget = QtWidgets.QWidget()
        c3 = QtWidgets.QHBoxLayout(card3_widget)
        c3.addWidget(self.btn_watch_dir)
        c3.addWidget(self.btn_stop_watch)
        card3 = make_card("目录监控", card3_widget)

        # Run card
        card4_widget = QtWidgets.QWidget()
        c4 = QtWidgets.QHBoxLayout(card4_widget)
        self.btn_run = QtWidgets.QPushButton("开始处理")
        c4.addWidget(self.btn_run)
        card4 = make_card("运行", card4_widget)

        right.addWidget(card1)
        right.addWidget(card2)
        right.addWidget(card3)
        right.addWidget(card4)
        right.addStretch(1)
        middle.addLayout(right, 3)

        # State
        self.output_dir: Optional[pathlib.Path] = None
        self.worker: Optional[BackgroundRemoverWorker] = None
        self.watcher: Optional[DirectoryWatcher] = None
        self.watch_dir: Optional[pathlib.Path] = None

        # Connections
        self.btn_add_files.clicked.connect(self.on_add_files)
        self.btn_add_dir.clicked.connect(self.on_add_dir)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_output_dir.clicked.connect(self.on_choose_output_dir)
        self.btn_run.clicked.connect(self.on_run)
        self.list_files.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_done.itemSelectionChanged.connect(self.on_selection_changed)
        self.edit_filter.textChanged.connect(self.on_filter_changed)
        self.btn_delete_selected.clicked.connect(self.on_delete_selected)
        self.btn_watch_dir.clicked.connect(self.on_watch_dir)
        self.btn_stop_watch.clicked.connect(self.on_stop_watch)

    # UI helpers
    def add_files(self, files: List[pathlib.Path]) -> None:
        existing = {self.list_files.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list_files.count())}
        existing_done = {self.list_done.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list_done.count())}
        for f in files:
            fp = str(f.resolve())
            if fp in existing or fp in existing_done:
                continue
            # Skip output files to avoid re-adding
            if f.name.endswith("_rmbg.png"):
                continue
            item = QtWidgets.QListWidgetItem(f.name)
            item.setToolTip(fp)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, fp)
            self.list_files.addItem(item)

        if self.list_files.count() > 0 and len(self.list_files.selectedItems()) == 0:
            self.list_files.setCurrentRow(0)

    def selected_file_path(self) -> Optional[pathlib.Path]:
        # Prefer selection from pending list; fallback to done list
        items = self.list_files.selectedItems()
        if not items:
            items = self.list_done.selectedItems()
            if not items:
                return None
        return pathlib.Path(items[0].data(QtCore.Qt.ItemDataRole.UserRole))

    def load_preview(self, input_path: pathlib.Path, output_path: Optional[pathlib.Path]) -> None:
        # Load input preview
        try:
            img = Image.open(input_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            self.preview_input.set_pil_image(img)
        except Exception:
            self.preview_input.clear_preview()

        # Load output preview
        if output_path and output_path.exists():
            try:
                out_img = Image.open(output_path)
                out_img = ImageOps.exif_transpose(out_img)
                self.preview_output.set_pil_image(out_img)
            except Exception:
                self.preview_output.clear_preview()
        else:
            self.preview_output.clear_preview()

    # Slots
    def on_add_files(self) -> None:
        exts = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.jfif *.avif)"
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择图片", str(pathlib.Path.home()), exts)
        if files:
            self.add_files([pathlib.Path(f) for f in files])

    def on_add_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", str(pathlib.Path.home()))
        if d:
            folder = pathlib.Path(d)
            imgs: List[pathlib.Path] = []
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff", "*.jfif", "*.avif"):
                imgs.extend(folder.rglob(pattern))
            self.add_files(sorted(imgs))

    def on_clear(self) -> None:
        self.list_files.clear()
        self.preview_input.clear_preview()
        self.preview_output.clear_preview()
        self.txt_log.clear()
        try:
            self.statusBar().showMessage("就绪")
        except Exception:
            pass
        if hasattr(self, "edit_filter"):
            self.edit_filter.clear()

    def on_choose_output_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", str(pathlib.Path.home()))
        if d:
            self.output_dir = pathlib.Path(d)
            self.lbl_output_dir.setText(str(self.output_dir))
            self.lbl_output_dir.setStyleSheet("")
        else:
            self.output_dir = None
            self.lbl_output_dir.setText("未设置（默认与输入同目录）")
            self.lbl_output_dir.setStyleSheet("color: #666;")

    def on_selection_changed(self) -> None:
        p = self.selected_file_path()
        if not p:
            return
        out = (self.output_dir / f"{p.stem}_rmbg.png") if self.output_dir else p.with_name(f"{p.stem}_rmbg.png")
        self.load_preview(p, out)

    def toggle_ui(self, enabled: bool) -> None:
        for w in [self.btn_add_files, self.btn_add_dir, self.btn_clear, self.btn_output_dir, self.btn_run, self.list_files, self.combo_device, self.combo_size, self.combo_output_size, self.edit_filter, self.btn_delete_selected, self.btn_watch_dir, self.btn_stop_watch]:
            w.setEnabled(enabled)

    def on_run(self) -> None:
        files = [pathlib.Path(self.list_files.item(i).data(QtCore.Qt.ItemDataRole.UserRole)) for i in range(self.list_files.count())]
        if not files:
            QtWidgets.QMessageBox.information(self, "提示", "请先添加图片或目录")
            return
        device_choice = self.combo_device.currentText()
        size = int(self.combo_size.currentData())
        out_size = self.combo_output_size.currentData()

        # Pre-check device availability and warn + adjust if needed
        if device_choice == "cuda" and not torch.cuda.is_available():
            QtWidgets.QMessageBox.warning(self, "设备不可用", "未检测到可用的 CUDA 设备，将改用 CPU 运行。")
            self.txt_log.append("警告：CUDA 不可用，已改用 CPU。")
            device_choice = "cpu"
        if device_choice == "mps" and not torch.backends.mps.is_available():
            QtWidgets.QMessageBox.warning(self, "设备不可用", "未检测到可用的 MPS（Apple Silicon）设备，将改用 CPU 运行。")
            self.txt_log.append("警告：MPS 不可用，已改用 CPU。")
            device_choice = "cpu"

        # Reset summary counters for this run
        self.run_total_count = len(files)
        self.run_success_count = 0
        self.run_failure_count = 0
        self.run_failed_paths = []

        try:
            self.statusBar().showMessage("启动中…")
        except Exception:
            pass
        self.toggle_ui(False)

        self.worker = BackgroundRemoverWorker(files, self.output_dir, device_choice, size, out_size)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.image_done.connect(self.on_image_done)
        self.worker.finished_success.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.file_error.connect(self.on_file_error)
        self.worker.start()

    @QtCore.pyqtSlot(str, str)
    def on_image_done(self, input_path: str, output_path: str) -> None:
        # Refresh preview if the current selection matches
        current = self.selected_file_path()
        if current and str(current.resolve()) == str(pathlib.Path(input_path).resolve()):
            self.load_preview(pathlib.Path(input_path), pathlib.Path(output_path))
        # Update counters
        self.run_success_count += 1
        # Move item from pending to done
        self._move_to_done(input_path, output_path)

    def on_finished(self) -> None:
        # Show summary
        summary = f"处理完成：成功 {self.run_success_count}，失败 {self.run_failure_count}，总计 {self.run_total_count}。"
        try:
            self.statusBar().showMessage("完成")
        except Exception:
            pass
        self.txt_log.append(summary)
        if self.run_failure_count > 0:
            failed_preview = "\n".join(self.run_failed_paths[:10])
            more = "" if self.run_failure_count <= 10 else "\n..."
            QtWidgets.QMessageBox.information(self, "完成（包含失败）", summary + ("\n失败文件（前10）：\n" + failed_preview + more if failed_preview else ""))
        else:
            QtWidgets.QMessageBox.information(self, "完成", summary)
        self.toggle_ui(True)
        self._reset_run_counters()

    def on_failed(self, msg: str) -> None:
        try:
            self.statusBar().showMessage(f"失败: {msg}")
        except Exception:
            pass
        partial = f"已完成：成功 {self.run_success_count}，失败 {self.run_failure_count}，总计 {self.run_total_count}。"
        QtWidgets.QMessageBox.critical(self, "运行中出错", msg + "\n" + partial)
        self.toggle_ui(True)
        self._reset_run_counters()

    @QtCore.pyqtSlot(str)
    def on_status_changed(self, msg: str) -> None:
        # Update one-line status and append to the log
        try:
            self.statusBar().showMessage(msg)
        except Exception:
            pass
        self.txt_log.append(msg)

    @QtCore.pyqtSlot(str, str)
    def on_file_error(self, input_path: str, error_msg: str) -> None:
        self.txt_log.append(f"ERROR - {input_path}: {error_msg}")
        self.run_failure_count += 1
        self.run_failed_paths.append(input_path)

    # Workflow: directory watch
    def on_watch_dir(self) -> None:
        if not WATCHDOG_AVAILABLE:
            QtWidgets.QMessageBox.information(self, "提示", "未安装 watchdog，无法启用目录监控。\n请先：pip install watchdog")
            return
        # Prefer using the directory of the first listed file; fallback to manual choose
        if self.list_files.count() > 0:
            first_item = self.list_files.item(0)
            first_path = pathlib.Path(first_item.data(QtCore.Qt.ItemDataRole.UserRole))
            self.watch_dir = first_path.parent
        else:
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择监控的文件夹", str(pathlib.Path.home()))
            if not d:
                return
            self.watch_dir = pathlib.Path(d)
        if self.watcher is not None:
            try:
                self.watcher.stop()
            except Exception:
                pass
            self.watcher = None

        self.watcher = DirectoryWatcher(self.watch_dir, recursive=True)
        self.watcher.files_detected.connect(self.on_files_detected)
        self.watcher.files_deleted.connect(self.on_files_deleted)
        self.watcher.files_moved.connect(self.on_files_moved)
        self.watcher.status_changed.connect(self.on_status_changed)
        self.watcher.failed.connect(lambda m: QtWidgets.QMessageBox.critical(self, "监控出错", m))
        self.watcher.start()
        self.btn_watch_dir.setEnabled(False)
        self.btn_stop_watch.setEnabled(True)

    def on_stop_watch(self) -> None:
        if self.watcher is not None:
            try:
                self.watcher.stop()
                self.watcher.wait(1000)
            except Exception:
                pass
            self.watcher = None
        self.btn_watch_dir.setEnabled(True)
        self.btn_stop_watch.setEnabled(False)
        self.txt_log.append("已停止监控")

    @QtCore.pyqtSlot(list)
    def on_files_detected(self, paths: list) -> None:
        # Deduplicate with existing list entries
        existing = {self.list_files.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list_files.count())}
        existing_done = {self.list_done.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.list_done.count())}
        new_paths = []
        for p in paths:
            # Skip outputs and duplicates
            if p not in existing and p not in existing_done and not str(p).endswith("_rmbg.png"):
                new_paths.append(pathlib.Path(p))
        if not new_paths:
            return
        self.add_files(new_paths)
        self.txt_log.append(f"监控新增 {len(new_paths)} 个文件，已加入待处理列表。")

    @QtCore.pyqtSlot(list)
    def on_files_deleted(self, paths: list) -> None:
        # Remove entries whose paths were deleted on disk
        path_set = set(paths)
        to_remove = []
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            p = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if p in path_set:
                to_remove.append(i)
        for r in reversed(to_remove):
            self.list_files.takeItem(r)
        if to_remove:
            self.txt_log.append(f"监控删除 {len(to_remove)} 个文件，已从列表移除。")

    @QtCore.pyqtSlot(list)
    def on_files_moved(self, pairs: list) -> None:
        # Update entries moved/renamed; pairs: [(old, new), ...]
        mapping = {old: new for (old, new) in pairs}
        updated = 0
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            p = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if p in mapping:
                new_path = mapping[p]
                item.setText(pathlib.Path(new_path).name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, new_path)
                item.setToolTip(new_path)
                updated += 1
        if updated:
            self.txt_log.append(f"监控重命名/移动 {updated} 个文件，列表已同步更新。")

    # Filtering and deletion
    def on_filter_changed(self, text: str) -> None:
        query = (text or "").strip().lower()
        first_visible = None
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            name = item.text().lower()
            path = str(item.data(QtCore.Qt.ItemDataRole.UserRole)).lower()
            matched = (query in name) or (query in path)
            item.setHidden(not matched)
            if matched and first_visible is None:
                first_visible = i
        # Maintain sensible selection
        current = self.list_files.currentRow()
        if current < 0 or (current >= 0 and self.list_files.item(current).isHidden()):
            if first_visible is not None:
                self.list_files.setCurrentRow(first_visible)
            else:
                self.list_files.clearSelection()

    def on_delete_selected(self) -> None:
        # Remove from pending or done
        selected_pending = self.list_files.selectedItems()
        selected_done = self.list_done.selectedItems()
        if not selected_pending and not selected_done:
            return
        if selected_pending:
            rows = sorted([self.list_files.row(it) for it in selected_pending], reverse=True)
            for r in rows:
                self.list_files.takeItem(r)
        if selected_done:
            rows = sorted([self.list_done.row(it) for it in selected_done], reverse=True)
            for r in rows:
                self.list_done.takeItem(r)
        # Refresh preview and selection after deletion
        self.on_selection_changed()

    def _move_to_done(self, input_path: str, output_path: str) -> None:
        # Find item in pending matching input_path
        for i in range(self.list_files.count()):
            item = self.list_files.item(i)
            p = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if p == input_path:
                self.list_files.takeItem(i)
                done_item = QtWidgets.QListWidgetItem(pathlib.Path(output_path).name)
                done_item.setToolTip(output_path)
                done_item.setData(QtCore.Qt.ItemDataRole.UserRole, output_path)
                self.list_done.addItem(done_item)
                break

    def _reset_run_counters(self) -> None:
        self.run_total_count = 0
        self.run_success_count = 0
        self.run_failure_count = 0
        self.run_failed_paths = []


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
