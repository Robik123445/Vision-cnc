"""PySide6 operator UI for CNC vision segmentation workflow."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vision.detect.detector_api import Detector, create_detector, load_config
from vision.ui.image_overlay import compose_overlay
from vision.ui.view_model import detection_to_view_state


LOGGER = logging.getLogger("vision.ui")


class VisionMainWindow(QMainWindow):
    """Main desktop interface for running and reviewing segmentation inference."""

    def __init__(self, config_path: str = "config/vision_config.json") -> None:
        super().__init__()
        self.setWindowTitle("CNC Vision - YOLO Segmentation Console")
        self.resize(1280, 860)

        self._config = load_config(config_path)
        self._detector: Detector = create_detector(self._config)
        self._camera: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._last_overlay: Optional[np.ndarray] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self) -> None:
        """Create and connect all visual controls."""

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        left = QVBoxLayout()
        right = QVBoxLayout()

        self.preview = QLabel("No frame")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(920, 700)
        self.preview.setStyleSheet("background:#111; color:#bbb; border:1px solid #333;")
        left.addWidget(self.preview)

        ctrl = QGroupBox("Controls")
        ctrl_grid = QGridLayout(ctrl)

        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 15)
        self.camera_index.setValue(0)

        self.btn_open_cam = QPushButton("Open Camera")
        self.btn_stop_cam = QPushButton("Stop Camera")
        self.btn_load_image = QPushButton("Load Image")
        self.btn_detect_once = QPushButton("Detect Once")
        self.btn_snapshot = QPushButton("Save Snapshot")

        ctrl_grid.addWidget(QLabel("Camera index"), 0, 0)
        ctrl_grid.addWidget(self.camera_index, 0, 1)
        ctrl_grid.addWidget(self.btn_open_cam, 1, 0)
        ctrl_grid.addWidget(self.btn_stop_cam, 1, 1)
        ctrl_grid.addWidget(self.btn_load_image, 2, 0)
        ctrl_grid.addWidget(self.btn_detect_once, 2, 1)
        ctrl_grid.addWidget(self.btn_snapshot, 3, 0, 1, 2)

        self.lbl_status = QLabel("source=none | fail=idle")
        self.lbl_latency = QLabel("inference: - ms")
        self.lbl_conf_workpiece = QLabel("workpiece: 0.00")
        self.lbl_conf_clamp = QLabel("clamp: 0.00")
        self.lbl_conf_hand = QLabel("hand: 0.00")
        self.lbl_conf_tool = QLabel("tool: 0.00")

        status_box = QGroupBox("Detection status")
        status_layout = QVBoxLayout(status_box)
        for w in [
            self.lbl_status,
            self.lbl_latency,
            self.lbl_conf_workpiece,
            self.lbl_conf_clamp,
            self.lbl_conf_hand,
            self.lbl_conf_tool,
        ]:
            status_layout.addWidget(w)

        right.addWidget(ctrl)
        right.addWidget(status_box)
        right.addStretch(1)

        layout.addLayout(left, 3)
        layout.addLayout(right, 1)

        self.btn_open_cam.clicked.connect(self._open_camera)
        self.btn_stop_cam.clicked.connect(self._stop_camera)
        self.btn_load_image.clicked.connect(self._load_image)
        self.btn_detect_once.clicked.connect(self._detect_once)
        self.btn_snapshot.clicked.connect(self._save_snapshot)

    def _open_camera(self) -> None:
        """Open camera stream and start periodic detection loop."""

        self._stop_camera()
        idx = int(self.camera_index.value())
        cam = cv2.VideoCapture(idx)
        ok, _ = cam.read()
        if not ok:
            cam.release()
            QMessageBox.warning(self, "Camera error", f"Unable to open camera {idx}")
            return
        self._camera = cam
        self._timer.start(80)

    def _stop_camera(self) -> None:
        """Stop camera stream and release resources safely."""

        self._timer.stop()
        if self._camera is not None:
            self._camera.release()
        self._camera = None

    def _load_image(self) -> None:
        """Load single image for offline detection and review."""

        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Image error", f"Unable to load: {path}")
            return
        self._frame = frame
        self._detect_once()

    def _tick(self) -> None:
        """Periodic camera callback: grab frame and run one inference step."""

        if self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok:
            return
        self._frame = frame
        self._detect_once()

    def _detect_once(self) -> None:
        """Run detector on latest frame and refresh visualization + status widgets."""

        if self._frame is None:
            return
        result = self._detector.detect(self._frame)
        overlay = compose_overlay(self._frame, result)
        self._last_overlay = overlay
        self._set_preview(overlay)

        state = detection_to_view_state(result)
        self.lbl_status.setText(state.status_line)
        self.lbl_latency.setText(f"inference: {state.inference_ms:.1f} ms")
        self.lbl_conf_workpiece.setText(f"workpiece: {state.confidences.get('workpiece', 0.0):.2f}")
        self.lbl_conf_clamp.setText(f"clamp: {state.confidences.get('clamp', 0.0):.2f}")
        self.lbl_conf_hand.setText(f"hand: {state.confidences.get('hand', 0.0):.2f}")
        self.lbl_conf_tool.setText(f"tool: {state.confidences.get('tool', 0.0):.2f}")

    def _set_preview(self, image_bgr: np.ndarray) -> None:
        """Render BGR frame into Qt label with aspect-preserving scaling."""

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _save_snapshot(self) -> None:
        """Save current overlay frame and metadata into runs/ui directory."""

        if self._last_overlay is None:
            QMessageBox.information(self, "No data", "No overlay to save yet.")
            return
        out_dir = Path("runs/ui") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "overlay.png"), self._last_overlay)

        payload = {
            "status": self.lbl_status.text(),
            "inference": self.lbl_latency.text(),
            "saved_at": datetime.now().isoformat(),
        }
        (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("UI snapshot saved: %s", out_dir)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Qt close callback to release camera and timers cleanly."""

        self._stop_camera()
        super().closeEvent(event)
