"""PySide6 operator UI for CNC vision segmentation workflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vision.detect.detector_api import DetectionResult, Detector, create_detector, load_config
from vision.ui.camera_discovery import CameraInfo, enumerate_cameras, open_camera_capture
from vision.ui.image_overlay import compose_overlay
from vision.ui.view_model import CLASS_LABELS, SOURCE_LABELS, detection_to_view_state


LOGGER = logging.getLogger("vision.ui")


class VisionMainWindow(QMainWindow):
    """Main desktop interface for running and reviewing segmentation inference."""

    def __init__(self, config_path: str = "config/vision_config.json") -> None:
        super().__init__()
        self.setWindowTitle("CNC Vision Ovládací Pult")
        self.resize(1480, 920)

        self._config_path = config_path
        self._config = load_config(config_path)
        self._detector: Detector = create_detector(self._config)
        self._camera: Optional[Any] = None
        self._active_camera_info: Optional[CameraInfo] = None
        self._camera_infos: list[CameraInfo] = []
        self._frame: Optional[Any] = None
        self._last_presented_frame: Optional[Any] = None
        self._last_result: Optional[DetectionResult] = None
        self._dataset_window = None
        self._model_window = None
        self._input_source = "žiadny"
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._apply_styles()
        self._reload_detector(initial=True)
        self._refresh_camera_list()
        self._log("Rozhranie je pripravené")
        self.statusBar().showMessage("Pripravené")

    def _build_ui(self) -> None:
        """Create and connect all visual controls."""

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(18, 18, 18, 18)
        hero_layout.setSpacing(10)

        title = QLabel("CNC Vision Pult")
        title.setObjectName("heroTitle")
        subtitle = QLabel("Operátorské rozhranie pre výber kamery, živú inferenciu a kontrolu výsledkov")
        subtitle.setObjectName("heroSubtitle")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        chip_row = QHBoxLayout()
        chip_row.setSpacing(10)
        self.chip_camera = QLabel("KAMERA: OFFLINE")
        self.chip_detector = QLabel("DETEKTOR: PRIPRAVENÝ")
        self.chip_fail = QLabel("STAV: ČAKÁ")
        for chip in [self.chip_camera, self.chip_detector, self.chip_fail]:
            chip.setProperty("chip", True)
            chip_row.addWidget(chip)
        chip_row.addStretch(1)
        hero_layout.addLayout(chip_row)

        self.preview = QLabel("Nie je načítaný žiadny snímok")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(980, 720)
        self.preview.setObjectName("preview")

        left_layout.addWidget(hero)
        left_layout.addWidget(self.preview, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        source_box = QGroupBox("Zdroj")
        source_layout = QVBoxLayout(source_box)
        self.cmb_camera = QComboBox()
        self.cmb_camera.setMinimumContentsLength(24)
        self.btn_refresh_cams = QPushButton("Obnov kamery")
        camera_row = QHBoxLayout()
        camera_row.addWidget(self.cmb_camera, 1)
        camera_row.addWidget(self.btn_refresh_cams)
        source_layout.addLayout(camera_row)

        button_row_1 = QHBoxLayout()
        self.btn_open_cam = QPushButton("Otvor vybranú")
        self.btn_stop_cam = QPushButton("Zastav kameru")
        button_row_1.addWidget(self.btn_open_cam)
        button_row_1.addWidget(self.btn_stop_cam)
        source_layout.addLayout(button_row_1)

        button_row_2 = QHBoxLayout()
        self.btn_load_image = QPushButton("Načítaj obrázok")
        self.btn_detect_once = QPushButton("Deteguj raz")
        button_row_2.addWidget(self.btn_load_image)
        button_row_2.addWidget(self.btn_detect_once)
        source_layout.addLayout(button_row_2)

        button_row_3 = QHBoxLayout()
        self.btn_reload = QPushButton("Načítaj detektor znova")
        self.btn_snapshot = QPushButton("Ulož snímku")
        button_row_3.addWidget(self.btn_reload)
        button_row_3.addWidget(self.btn_snapshot)
        source_layout.addLayout(button_row_3)

        button_row_4 = QHBoxLayout()
        self.btn_dataset = QPushButton("Dataset a súradnice")
        self.btn_models = QPushButton("Modely a tréning")
        button_row_4.addWidget(self.btn_dataset)
        button_row_4.addWidget(self.btn_models)
        source_layout.addLayout(button_row_4)

        live_box = QGroupBox("Živé spracovanie")
        live_form = QFormLayout(live_box)
        live_form.setContentsMargins(14, 18, 14, 14)
        live_form.setSpacing(10)
        self.chk_live_detect = QCheckBox("Spúšťaj detekciu na každom snímku kamery")
        self.chk_live_detect.setChecked(True)
        self.chk_show_overlay = QCheckBox("Zobraz overlay segmentácie")
        self.chk_show_overlay.setChecked(True)
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(40, 2000)
        self.spin_interval.setSingleStep(20)
        self.spin_interval.setSuffix(" ms")
        self.spin_interval.setValue(120)
        alpha_wrap = QWidget()
        alpha_layout = QHBoxLayout(alpha_wrap)
        alpha_layout.setContentsMargins(0, 0, 0, 0)
        alpha_layout.setSpacing(10)
        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(0, 100)
        self.sld_alpha.setValue(35)
        self.lbl_alpha = QLabel("35%")
        alpha_layout.addWidget(self.sld_alpha, 1)
        alpha_layout.addWidget(self.lbl_alpha)
        live_form.addRow(self.chk_live_detect)
        live_form.addRow(self.chk_show_overlay)
        live_form.addRow("Interval snímok", self.spin_interval)
        live_form.addRow("Krytie overlayu", alpha_wrap)

        tabs = QTabWidget()
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(10)

        self.lbl_status_head = QLabel("Zatiaľ bez inferencie")
        self.lbl_status_head.setObjectName("statusHead")
        self.lbl_status = QLabel("zdroj=žiadny | stav=čaká")
        self.lbl_latency = QLabel("inferencia: - ms")
        self.lbl_runtime = QLabel("kamera: nečinná")
        self.lbl_image_size = QLabel("obraz: -")
        self.lbl_input = QLabel("vstup: žiadny")
        self.lbl_model = QLabel("model: nenastavený")
        self.lbl_config = QLabel(f"konfigurácia: {self._config_path}")
        self.lbl_snapshot = QLabel("snímka: žiadna")
        for widget in [
            self.lbl_status_head,
            self.lbl_status,
            self.lbl_latency,
            self.lbl_runtime,
            self.lbl_image_size,
            self.lbl_input,
            self.lbl_model,
            self.lbl_config,
            self.lbl_snapshot,
        ]:
            widget.setWordWrap(True)
            status_layout.addWidget(widget)

        self.bar_workpiece = self._make_confidence_bar("OBROBOK", "#4dd08b")
        self.bar_clamp = self._make_confidence_bar("UPÍNKA", "#ff6b57")
        self.bar_hand = self._make_confidence_bar("RUKA", "#f7c66a")
        self.bar_tool = self._make_confidence_bar("NÁSTROJ", "#58b9ff")
        for bar in [self.bar_workpiece, self.bar_clamp, self.bar_hand, self.bar_tool]:
            status_layout.addWidget(bar)
        status_layout.addStretch(1)

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(12, 12, 12, 12)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Záznam relácie")
        log_layout.addWidget(self.txt_log)

        tabs.addTab(status_tab, "Stav")
        tabs.addTab(log_tab, "Záznam relácie")

        right_layout.addWidget(source_box)
        right_layout.addWidget(live_box)
        right_layout.addWidget(tabs, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([1030, 420])

        self.btn_refresh_cams.clicked.connect(self._refresh_camera_list)
        self.btn_open_cam.clicked.connect(self._open_selected_camera)
        self.btn_stop_cam.clicked.connect(self._stop_camera)
        self.btn_load_image.clicked.connect(self._load_image)
        self.btn_detect_once.clicked.connect(self._detect_once)
        self.btn_reload.clicked.connect(self._reload_detector)
        self.btn_snapshot.clicked.connect(self._save_snapshot)
        self.btn_dataset.clicked.connect(self._open_dataset_window)
        self.btn_models.clicked.connect(self._open_model_training_window)
        self.chk_show_overlay.toggled.connect(self._rerender_last_frame)
        self.chk_live_detect.toggled.connect(self._on_live_mode_changed)
        self.spin_interval.valueChanged.connect(self._update_timer_interval)
        self.sld_alpha.valueChanged.connect(self._on_alpha_changed)

    def _apply_styles(self) -> None:
        """Apply a denser industrial UI style so the desk feels more like a tool."""

        self.setStyleSheet(
            """
            QMainWindow {
                background: #14161b;
                color: #f2ede5;
            }
            QFrame#hero {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1d2128, stop:1 #242b33);
                border: 1px solid #3a414d;
                border-radius: 18px;
            }
            QLabel#heroTitle {
                font-size: 28px;
                font-weight: 700;
                color: #f6f0e4;
            }
            QLabel#heroSubtitle {
                font-size: 13px;
                color: #c6bcab;
            }
            QLabel[chip="true"] {
                background: #232932;
                color: #f7efe1;
                border: 1px solid #4b5564;
                border-radius: 999px;
                padding: 7px 12px;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#statusHead {
                font-size: 18px;
                font-weight: 700;
                color: #f7c66a;
            }
            QGroupBox {
                border: 1px solid #39404b;
                border-radius: 16px;
                margin-top: 16px;
                padding-top: 18px;
                background: #1a1e25;
                color: #f1eadc;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #f7c66a;
            }
            QPushButton {
                background: #252b34;
                color: #f6efe3;
                border: 1px solid #4a5565;
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2e3540;
                border-color: #ff9248;
            }
            QPushButton:pressed {
                background: #1d2229;
            }
            QComboBox, QSpinBox, QPlainTextEdit {
                background: #101318;
                color: #f5eee1;
                border: 1px solid #495261;
                border-radius: 10px;
                padding: 8px 10px;
            }
            QCheckBox {
                color: #efe5d6;
                spacing: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #39404b;
                background: #171b21;
                border-radius: 14px;
            }
            QTabBar::tab {
                background: #222730;
                color: #dbcdb8;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QTabBar::tab:selected {
                background: #2d3440;
                color: #f7c66a;
            }
            QLabel#preview {
                background: #06080b;
                color: #c9c0b2;
                border: 1px solid #333947;
                border-radius: 20px;
            }
            QSlider::groove:horizontal {
                border: 0;
                height: 6px;
                background: #2a313b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f7c66a;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            """
        )

    def _make_confidence_bar(self, label: str, color: str) -> QProgressBar:
        """Create a color-coded confidence bar."""

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat(f"{label}  %p%")
        bar.setTextVisible(True)
        bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: 1px solid #434b57;
                border-radius: 8px;
                background: #0f1216;
                color: #f7efe1;
                text-align: center;
                min-height: 22px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 7px;
            }}
            """
        )
        return bar

    def _import_cv2(self):
        """Import OpenCV lazily so UI import stays lightweight."""

        try:
            import cv2  # type: ignore

            return cv2
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Chýbajúca závislosť",
                "OpenCV nie je nainštalované. Spustite `pip install -r requirements-runtime.txt`.",
            )
            raise RuntimeError("OpenCV nie je dostupné") from exc

    def _log(self, message: str) -> None:
        """Append a message to the in-app session log and to the runtime logger."""

        stamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.appendPlainText(f"[{stamp}] {message}")
        LOGGER.info(message)

    def _set_chip_state(self, label: QLabel, text: str, tone: str) -> None:
        """Update chip text and tint."""

        colors = {
            "neutral": ("#232932", "#4b5564", "#f7efe1"),
            "ok": ("#183127", "#3c8b67", "#d7ffe8"),
            "warn": ("#372913", "#b9852f", "#ffe7bd"),
            "bad": ("#3a1d1d", "#c05c54", "#ffe2de"),
            "info": ("#172b38", "#4b90b4", "#d7efff"),
        }
        bg, border, fg = colors.get(tone, colors["neutral"])
        label.setText(text)
        label.setStyleSheet(
            f"background:{bg}; border:1px solid {border}; color:{fg}; border-radius:999px; padding:7px 12px; font-size:12px; font-weight:600;"
        )

    def _refresh_runtime_labels(self) -> None:
        """Refresh config/runtime labels shown in the status tab."""

        model_path = self._config.get("yolo", {}).get("model_path", "nenastavený")
        camera_text = self._active_camera_info.label if self._active_camera_info else "kamera: offline"
        self.lbl_model.setText(f"model: {model_path}")
        self.lbl_config.setText(f"konfigurácia: {self._config_path}")
        self.lbl_input.setText(f"vstup: {self._input_source}")
        self.lbl_runtime.setText(camera_text)

    def _preferred_camera_row(self, cameras: list[CameraInfo]) -> int:
        """Prefer external cameras over integrated when selecting a default row."""

        for idx, camera in enumerate(cameras):
            if "integrated" not in camera.label.lower():
                return idx
        return 0

    def _refresh_camera_list(self) -> None:
        """Reload the connected camera inventory into the picker."""

        active_path = self._active_camera_info.device_path if self._active_camera_info else None
        self._camera_infos = enumerate_cameras()
        self.cmb_camera.clear()

        if not self._camera_infos:
            self.cmb_camera.addItem("Nenašla sa žiadna kamera", None)
            self.btn_open_cam.setEnabled(False)
            self._log("Neboli nájdené žiadne kamerové zariadenia")
            return

        self.btn_open_cam.setEnabled(True)
        for camera in self._camera_infos:
            self.cmb_camera.addItem(camera.label, camera)

        if active_path:
            for idx, camera in enumerate(self._camera_infos):
                if camera.device_path == active_path:
                    self.cmb_camera.setCurrentIndex(idx)
                    break
        else:
            self.cmb_camera.setCurrentIndex(self._preferred_camera_row(self._camera_infos))

        self._log(f"Zoznam kamier obnovený: nájdené {len(self._camera_infos)} zariadenia")

    def _selected_camera(self) -> Optional[CameraInfo]:
        """Return currently selected camera descriptor."""

        data = self.cmb_camera.currentData()
        return data if isinstance(data, CameraInfo) else None

    def _open_dataset_window(self) -> None:
        """Open or focus the dataset capture and explanation window."""

        if self._dataset_window is None:
            from vision.ui.dataset_window import DatasetWindow

            self._dataset_window = DatasetWindow(config_path=self._config_path, parent=self)
        self._dataset_window.show()
        self._dataset_window.raise_()
        self._dataset_window.activateWindow()

    def _open_model_training_window(self) -> None:
        """Open or focus the dedicated model/training window."""

        if self._model_window is None:
            from vision.ui.model_training_window import ModelTrainingWindow

            self._model_window = ModelTrainingWindow(
                config_path=self._config_path,
                on_config_changed=self._handle_model_config_changed,
                parent=self,
            )
        self._model_window.show()
        self._model_window.raise_()
        self._model_window.activateWindow()

    def _handle_model_config_changed(self) -> None:
        """React to model configuration changes from the child window."""

        self._reload_detector()
        self._log("Konfigurácia modelu bola aktualizovaná v okne Modely a tréning")

    def _reload_detector(self, initial: bool = False) -> None:
        """Reload configuration and detector pipeline from disk."""

        self._config = load_config(self._config_path)
        self._detector = create_detector(self._config)
        self._refresh_runtime_labels()
        self._set_chip_state(self.chip_detector, "DETEKTOR: PRIPRAVENÝ", "info")
        if not initial:
            self._log("Detektor bol znovu načítaný z konfigurácie")
            self.statusBar().showMessage("Detektor znovu načítaný", 3000)

    def _open_selected_camera(self) -> None:
        """Open the currently selected camera device."""

        camera = self._selected_camera()
        if camera is None:
            QMessageBox.information(self, "Bez kamery", "Nie je vybraná platná kamera.")
            return

        cv2 = self._import_cv2()
        self._stop_camera(log_message=False)
        cap = open_camera_capture(cv2, camera)
        if cap is None:
            QMessageBox.warning(self, "Chyba kamery", f"Nepodarilo sa otvoriť {camera.label}")
            self._log(f"Nepodarilo sa otvoriť kameru {camera.label}")
            return

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            QMessageBox.warning(self, "Chyba kamery", f"{camera.label} sa otvorila, ale nedá sa z nej čítať obraz.")
            self._log(f"Kamera sa otvorila, ale nevrátila žiadny snímok: {camera.label}")
            return

        self._camera = cap
        self._active_camera_info = camera
        self._frame = frame
        self._input_source = camera.label
        self._refresh_runtime_labels()
        self._set_chip_state(self.chip_camera, f"KAMERA: {camera.device_path}", "ok")
        self._timer.start(self.spin_interval.value())
        self._log(f"Kamera otvorená: {camera.label}")

        if self.chk_live_detect.isChecked():
            self._run_detection(log_result=True)
        else:
            self._last_result = None
            self._present_frame(frame)

    def _stop_camera(self, log_message: bool = True) -> None:
        """Stop camera stream and release resources safely."""

        self._timer.stop()
        if self._camera is not None:
            self._camera.release()
        self._camera = None
        self._active_camera_info = None
        self._refresh_runtime_labels()
        self._set_chip_state(self.chip_camera, "KAMERA: OFFLINE", "neutral")
        if log_message:
            self._log("Kamera zastavená")

    def _load_image(self) -> None:
        """Load a single image for offline detection and review."""

        cv2 = self._import_cv2()
        path, _ = QFileDialog.getOpenFileName(self, "Vyber obrázok", "", "Obrázky (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Chyba obrázka", f"Nepodarilo sa načítať: {path}")
            self._log(f"Nepodarilo sa načítať obrázok {path}")
            return

        self._stop_camera(log_message=False)
        self._frame = frame
        self._input_source = path
        self._refresh_runtime_labels()
        self._set_chip_state(self.chip_camera, "KAMERA: REŽIM OBRÁZKA", "info")
        self._log(f"Obrázok načítaný: {path}")
        self._run_detection(log_result=True)

    def _tick(self) -> None:
        """Periodic camera callback: grab frame and run one inference step."""

        if self._camera is None:
            return

        ok, frame = self._camera.read()
        if not ok or frame is None:
            self.statusBar().showMessage("Čítanie z kamery zlyhalo", 2000)
            return

        self._frame = frame
        if self.chk_live_detect.isChecked():
            self._run_detection(log_result=False)
        else:
            self._last_result = None
            self._present_frame(frame)
            self.lbl_status_head.setText("Len živý náhľad")
            self.lbl_image_size.setText(f"obraz: {frame.shape[1]}x{frame.shape[0]}")
            self.statusBar().showMessage("Beží živý náhľad", 1200)

    def _detect_once(self) -> None:
        """Run detector on the latest frame and refresh visualization + status widgets."""

        if self._frame is None:
            QMessageBox.information(self, "Bez snímky", "Najprv načítajte obrázok alebo otvorte kameru.")
            return

        self._run_detection(log_result=True)

    def _run_detection(self, log_result: bool) -> None:
        """Execute detector on the current frame and update the desk state."""

        assert self._frame is not None
        result = self._detector.detect(self._frame)
        self._last_result = result
        self._present_frame(self._frame, result)
        self._update_detection_panel(result)

        if log_result:
            state = detection_to_view_state(result)
            self._log(
                "Detekcia dokončená "
                f"(zdroj={SOURCE_LABELS.get(result.source, result.source)}, "
                f"stav={state.fail_reason}, "
                f"{CLASS_LABELS['workpiece']}={result.confidences.get('workpiece', 0.0):.2f})"
            )

    def _present_frame(self, frame: Any, result: Optional[DetectionResult] = None) -> None:
        """Render either the raw frame or the overlay-enhanced view."""

        output = frame
        if result is not None and self.chk_show_overlay.isChecked():
            output = compose_overlay(frame, result, alpha=self.sld_alpha.value() / 100.0)

        self._last_presented_frame = output
        self._set_preview(output)
        self.lbl_image_size.setText(f"obraz: {frame.shape[1]}x{frame.shape[0]}")

    def _update_detection_panel(self, result: DetectionResult) -> None:
        """Update status labels, chips, and confidence bars from a detection result."""

        state = detection_to_view_state(result)
        self.lbl_status_head.setText(f"Zdroj detektora: {SOURCE_LABELS.get(result.source, result.source)}")
        self.lbl_status.setText(state.status_line)
        self.lbl_latency.setText(f"inferencia: {state.inference_ms:.1f} ms")
        self._set_bar_value(self.bar_workpiece, state.confidences.get("workpiece", 0.0))
        self._set_bar_value(self.bar_clamp, state.confidences.get("clamp", 0.0))
        self._set_bar_value(self.bar_hand, state.confidences.get("hand", 0.0))
        self._set_bar_value(self.bar_tool, state.confidences.get("tool", 0.0))

        fail_reason = state.fail_reason
        if fail_reason == "v poriadku":
            self._set_chip_state(self.chip_fail, "STAV: V PORIADKU", "ok")
        elif fail_reason == "detegovaná ruka":
            self._set_chip_state(self.chip_fail, "STAV: DETEGOVANÁ RUKA", "warn")
        else:
            self._set_chip_state(self.chip_fail, f"STAV: {fail_reason.upper()}", "bad")
        self._set_chip_state(self.chip_detector, f"DETEKTOR: {SOURCE_LABELS.get(result.source, result.source).upper()}", "info")
        self.statusBar().showMessage(f"Detekcia zdroj={SOURCE_LABELS.get(result.source, result.source)} stav={fail_reason}", 2500)

    def _set_bar_value(self, bar: QProgressBar, confidence: float) -> None:
        """Convert detector confidence into percent for UI bars."""

        value = max(0, min(int(round(confidence * 100.0)), 100))
        bar.setValue(value)

    def _set_preview(self, image_bgr: Any) -> None:
        """Render a BGR frame into the Qt label with aspect-preserving scaling."""

        cv2 = self._import_cv2()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimg = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def _rerender_last_frame(self) -> None:
        """Repaint current frame after overlay settings change."""

        if self._frame is None:
            return
        self._present_frame(self._frame, self._last_result)

    def _on_live_mode_changed(self, checked: bool) -> None:
        """Update live mode state and status messaging."""

        mode = "živá detekcia" if checked else "len náhľad"
        self._log(f"Živý režim zmenený na {mode}")
        if checked and self._frame is not None:
            self._run_detection(log_result=True)
        elif not checked and self._frame is not None:
            self._last_result = None
            self._present_frame(self._frame)
        if self._camera is not None:
            self._timer.start(self.spin_interval.value())

    def _update_timer_interval(self, value: int) -> None:
        """Apply new timer interval while camera stream is running."""

        if self._camera is not None and self._timer.isActive():
            self._timer.start(value)
        self.statusBar().showMessage(f"Interval snímok nastavený na {value} ms", 1500)

    def _on_alpha_changed(self, value: int) -> None:
        """Update alpha label and rerender overlay preview."""

        self.lbl_alpha.setText(f"{value}%")
        self._rerender_last_frame()

    def _save_snapshot(self) -> None:
        """Save the current presented frame and metadata into runs/ui."""

        if self._last_presented_frame is None:
            QMessageBox.information(self, "Bez dát", "Zatiaľ nie je čo uložiť.")
            return

        cv2 = self._import_cv2()
        out_dir = Path("runs/ui") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._frame is not None:
            cv2.imwrite(str(out_dir / "input.png"), self._frame)
        cv2.imwrite(str(out_dir / "preview.png"), self._last_presented_frame)

        payload = {
            "saved_at": datetime.now().isoformat(),
            "config_path": self._config_path,
            "input_source": self._input_source,
            "active_camera": self._active_camera_info.label if self._active_camera_info else None,
            "overlay_enabled": self.chk_show_overlay.isChecked(),
            "overlay_alpha": self.sld_alpha.value(),
            "live_detect": self.chk_live_detect.isChecked(),
        }
        if self._last_result is not None:
            payload.update(
                {
                    "source": self._last_result.source,
                    "fail_reason": self._last_result.fail_reason,
                    "confidences": self._last_result.confidences,
                    "inference_ms": self._last_result.inference_ms,
                    "image_size": self._last_result.image_size,
                    "debug": self._last_result.debug,
                }
            )

        (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.lbl_snapshot.setText(f"snímka: {out_dir}")
        self._log(f"Snímka uložená do {out_dir}")
        self.statusBar().showMessage(f"Snímka uložená do {out_dir}", 3000)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Repaint preview on resize so scaling stays sharp."""

        super().resizeEvent(event)
        if self._last_presented_frame is not None:
            self._set_preview(self._last_presented_frame)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Qt close callback to release camera and timers cleanly."""

        self._stop_camera(log_message=False)
        super().closeEvent(event)
