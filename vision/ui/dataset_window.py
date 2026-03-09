"""Dataset capture and calibration explanation window for the CNC Vision UI."""

from __future__ import annotations

import json
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from vision.detect.detector_api import load_config, save_config
from vision.ui.camera_discovery import CameraInfo, enumerate_cameras, open_camera_capture
from vision.ui.dataset_helpers import build_capture_metadata, prepare_session_dir, sanitize_session_name
from vision.ui.training_helpers import PROJECT_ROOT, to_project_relative


class DatasetWindow(QDialog):
    """Dedicated window for dataset collection and coordinate explanations."""

    def __init__(self, config_path: str, parent=None) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._project_root = PROJECT_ROOT
        self._camera: Optional[Any] = None
        self._active_camera: Optional[CameraInfo] = None
        self._camera_infos: list[CameraInfo] = []
        self._frame: Optional[Any] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._frame_index = 0
        self._saved_count = 0
        self._session_dir: Optional[Path] = None

        self.setWindowTitle("Dataset a súradnice")
        self.resize(1220, 860)
        self._build_ui()
        self._load_defaults_from_config()
        self._refresh_camera_list()
        self._refresh_help()

    def _build_ui(self) -> None:
        """Create the tabs and widgets."""

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        capture_tab = QWidget()
        capture_layout = QHBoxLayout(capture_tab)
        capture_layout.setContentsMargins(8, 8, 8, 8)
        capture_layout.setSpacing(12)

        left = QVBoxLayout()
        self.preview = QLabel("Najprv otvorte kameru pre zber datasetu")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(760, 620)
        self.preview.setStyleSheet(
            "background:#080b10; color:#d7cdbd; border:1px solid #39414d; border-radius:18px;"
        )
        left.addWidget(self.preview, 1)

        counters = QHBoxLayout()
        self.lbl_frames = QLabel("Zachytené snímky: 0")
        self.lbl_saved = QLabel("Uložené vzorky: 0")
        self.lbl_session_dir = QLabel("Relácia: -")
        self.lbl_session_dir.setWordWrap(True)
        counters.addWidget(self.lbl_frames)
        counters.addWidget(self.lbl_saved)
        counters.addWidget(self.lbl_session_dir, 1)
        left.addLayout(counters)

        right = QVBoxLayout()

        source_box = QGroupBox("Zdroj kamery")
        source_layout = QVBoxLayout(source_box)
        self.cmb_camera = QComboBox()
        self.btn_refresh_cams = QPushButton("Obnov kamery")
        row = QHBoxLayout()
        row.addWidget(self.cmb_camera, 1)
        row.addWidget(self.btn_refresh_cams)
        source_layout.addLayout(row)
        row = QHBoxLayout()
        self.btn_open_camera = QPushButton("Otvor vybranú kameru")
        self.btn_stop_camera = QPushButton("Zastav kameru")
        row.addWidget(self.btn_open_camera)
        row.addWidget(self.btn_stop_camera)
        source_layout.addLayout(row)

        dataset_box = QGroupBox("Nastavenie zberu")
        dataset_form = QFormLayout(dataset_box)
        self.edit_output_dir = QLineEdit("dataset/raw")
        self.edit_output_dir.setPlaceholderText("Koreňový priečinok pre raw dataset")
        self.edit_session_name = QLineEdit("nova_relacia")
        self.edit_session_name.setPlaceholderText("Názov relácie, napr. doska_800x500")
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("Poznámka k sérii")
        self.cmb_machine_state = QComboBox()
        self.cmb_machine_state.setEditable(True)
        self.cmb_machine_state.addItems(["idle", "setup", "loading", "cutting", "unknown"])
        self.chk_auto_capture = QCheckBox("Automaticky ukladať každý N-ty snímok")
        self.spin_every_n = QSpinBox()
        self.spin_every_n.setRange(1, 10000)
        self.spin_every_n.setValue(10)
        self.spin_every_n.setSuffix(" snímok")
        self.spin_interval_ms = QSpinBox()
        self.spin_interval_ms.setRange(30, 2000)
        self.spin_interval_ms.setValue(100)
        self.spin_interval_ms.setSuffix(" ms")
        self.chk_store_defaults = QCheckBox("Uložiť tieto nastavenia ako predvolené")
        self.chk_store_defaults.setChecked(True)

        output_row = QWidget()
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.setSpacing(8)
        self.btn_pick_output = QPushButton("Priečinok")
        output_row_layout.addWidget(self.edit_output_dir, 1)
        output_row_layout.addWidget(self.btn_pick_output)

        dataset_form.addRow("Výstupný priečinok", output_row)
        dataset_form.addRow("Názov relácie", self.edit_session_name)
        dataset_form.addRow("Poznámka", self.edit_note)
        dataset_form.addRow("Stav stroja", self.cmb_machine_state)
        dataset_form.addRow(self.chk_auto_capture)
        dataset_form.addRow("Každý", self.spin_every_n)
        dataset_form.addRow("Interval čítania", self.spin_interval_ms)
        dataset_form.addRow(self.chk_store_defaults)

        actions_box = QGroupBox("Akcie")
        actions_layout = QVBoxLayout(actions_box)
        row = QHBoxLayout()
        self.btn_new_session = QPushButton("Nová relácia")
        self.btn_save_frame = QPushButton("Ulož aktuálny snímok")
        row.addWidget(self.btn_new_session)
        row.addWidget(self.btn_save_frame)
        actions_layout.addLayout(row)
        self.lbl_hint = QLabel(
            "Tip: pre ostrý dataset striedajte uhly, svetlo, typ materiálu a stav stroja. "
            "Každý uložený obrázok dostane aj JSON s metadátami."
        )
        self.lbl_hint.setWordWrap(True)
        actions_layout.addWidget(self.lbl_hint)

        right.addWidget(source_box)
        right.addWidget(dataset_box)
        right.addWidget(actions_box)
        right.addStretch(1)

        capture_layout.addLayout(left, 3)
        capture_layout.addLayout(right, 2)

        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        self.lbl_calibration_status = QLabel("Stav kalibrácie: -")
        self.lbl_calibration_status.setWordWrap(True)
        self.btn_refresh_help = QPushButton("Obnov vysvetlivky a stav")
        help_row = QHBoxLayout()
        help_row.addWidget(self.lbl_calibration_status, 1)
        help_row.addWidget(self.btn_refresh_help)
        self.txt_help = QTextBrowser()
        self.txt_help.setOpenExternalLinks(False)
        help_layout.addLayout(help_row)
        help_layout.addWidget(self.txt_help, 1)

        tabs.addTab(capture_tab, "Tvorba datasetu")
        tabs.addTab(help_tab, "Vysvetlivky a súradnice")

        self.btn_refresh_cams.clicked.connect(self._refresh_camera_list)
        self.btn_open_camera.clicked.connect(self._open_selected_camera)
        self.btn_stop_camera.clicked.connect(self._stop_camera)
        self.btn_pick_output.clicked.connect(self._pick_output_dir)
        self.btn_save_frame.clicked.connect(self._save_current_frame)
        self.btn_new_session.clicked.connect(self._reset_session)
        self.btn_refresh_help.clicked.connect(self._refresh_help)
        self.spin_interval_ms.valueChanged.connect(self._update_timer_interval)

    def _import_cv2(self):
        """Import OpenCV lazily."""

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

    def _load_defaults_from_config(self) -> None:
        """Load optional dataset defaults from the config."""

        config = load_config(self._config_path)
        dataset_cfg = config.get("dataset", {})
        self.edit_output_dir.setText(dataset_cfg.get("raw_dir", "dataset/raw"))
        self.cmb_machine_state.setCurrentText(dataset_cfg.get("default_machine_state", "unknown"))
        self.edit_note.setText(dataset_cfg.get("default_note", ""))

    def _save_defaults_to_config(self) -> None:
        """Persist optional dataset defaults when requested by the user."""

        if not self.chk_store_defaults.isChecked():
            return

        config = load_config(self._config_path)
        dataset_cfg = config.setdefault("dataset", {})
        dataset_cfg["raw_dir"] = self.edit_output_dir.text().strip() or "dataset/raw"
        dataset_cfg["default_machine_state"] = self.cmb_machine_state.currentText().strip() or "unknown"
        dataset_cfg["default_note"] = self.edit_note.text().strip()
        save_config(config, self._config_path)

    def _preferred_camera_row(self) -> int:
        """Prefer external cameras by default."""

        for idx, camera in enumerate(self._camera_infos):
            if "integrated" not in camera.label.lower():
                return idx
        return 0

    def _refresh_camera_list(self) -> None:
        """Reload available camera devices."""

        self._camera_infos = enumerate_cameras()
        self.cmb_camera.clear()
        if not self._camera_infos:
            self.cmb_camera.addItem("Nenašla sa žiadna kamera", None)
            self.btn_open_camera.setEnabled(False)
            return

        self.btn_open_camera.setEnabled(True)
        for camera in self._camera_infos:
            self.cmb_camera.addItem(camera.label, camera)
        self.cmb_camera.setCurrentIndex(self._preferred_camera_row())

    def _selected_camera(self) -> Optional[CameraInfo]:
        """Return selected camera descriptor."""

        data = self.cmb_camera.currentData()
        return data if isinstance(data, CameraInfo) else None

    def _open_selected_camera(self) -> None:
        """Open selected camera and start the preview loop."""

        camera = self._selected_camera()
        if camera is None:
            QMessageBox.information(self, "Bez kamery", "Nie je vybraná žiadna platná kamera.")
            return

        cv2 = self._import_cv2()
        self._stop_camera(silent=True)
        cap = open_camera_capture(cv2, camera)
        if cap is None:
            QMessageBox.warning(self, "Chyba kamery", f"Nepodarilo sa otvoriť {camera.label}")
            return

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            QMessageBox.warning(self, "Chyba kamery", f"{camera.label} je otvorená, ale nevracia obraz.")
            return

        self._camera = cap
        self._active_camera = camera
        self._frame = frame
        self._frame_index = 0
        self._set_preview(frame)
        self._timer.start(self.spin_interval_ms.value())
        self._update_status_labels()

    def _stop_camera(self, silent: bool = False) -> None:
        """Stop active camera stream."""

        self._timer.stop()
        if self._camera is not None:
            self._camera.release()
        self._camera = None
        self._active_camera = None
        if not silent:
            self._update_status_labels()

    def _pick_output_dir(self) -> None:
        """Choose dataset output directory."""

        path = QFileDialog.getExistingDirectory(self, "Vyber výstupný priečinok", str(self._project_root / "dataset"))
        if path:
            self.edit_output_dir.setText(to_project_relative(path, project_root=self._project_root))

    def _resolve_output_root(self) -> Path:
        """Resolve output root relative to the project."""

        text = self.edit_output_dir.text().strip() or "dataset/raw"
        path = Path(text)
        return path if path.is_absolute() else self._project_root / path

    def _ensure_session_dir(self) -> Path:
        """Create the active session directory if needed."""

        if self._session_dir is None:
            session_name = sanitize_session_name(self.edit_session_name.text().strip() or "nova_relacia")
            self._session_dir = prepare_session_dir(self._resolve_output_root(), session_name)
            self._update_status_labels()
        return self._session_dir

    def _reset_session(self) -> None:
        """Start a fresh dataset capture session."""

        self._session_dir = None
        self._saved_count = 0
        self._frame_index = 0
        self._update_status_labels()
        self._save_defaults_to_config()
        QMessageBox.information(self, "Nová relácia", "Dataset relácia bola vynulovaná. Ďalší save vytvorí nový priečinok.")

    def _tick(self) -> None:
        """Read one frame and optionally save it into the dataset."""

        if self._camera is None:
            return

        ok, frame = self._camera.read()
        if not ok or frame is None:
            return

        self._frame = frame
        self._frame_index += 1
        self._set_preview(frame)
        if self.chk_auto_capture.isChecked():
            every = max(self.spin_every_n.value(), 1)
            if self._frame_index % every == 0:
                self._save_current_frame(show_message=False)
        self._update_status_labels()

    def _save_current_frame(self, show_message: bool = True) -> None:
        """Save current frame and metadata to the active dataset session."""

        if self._frame is None:
            QMessageBox.information(self, "Bez snímky", "Najprv otvorte kameru alebo načítajte obraz.")
            return

        cv2 = self._import_cv2()
        session_dir = self._ensure_session_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = session_dir / f"{timestamp}.jpg"
        meta_path = session_dir / f"{timestamp}.json"

        cv2.imwrite(str(image_path), self._frame)
        camera_label = self._active_camera.label if self._active_camera is not None else "neznáma kamera"
        image_size = (int(self._frame.shape[1]), int(self._frame.shape[0])) if hasattr(self._frame, "shape") else None
        payload = build_capture_metadata(
            timestamp=timestamp,
            note=self.edit_note.text().strip(),
            machine_state=self.cmb_machine_state.currentText().strip() or "unknown",
            camera_label=camera_label,
            frame_index=self._frame_index,
            saved_index=self._saved_count + 1,
            session_name=sanitize_session_name(self.edit_session_name.text().strip() or "nova_relacia"),
            image_size=image_size,
        )
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self._saved_count += 1
        self._save_defaults_to_config()
        self._update_status_labels()
        if show_message:
            QMessageBox.information(self, "Snímka uložená", f"Uložené do:\n{image_path}")

    def _set_preview(self, image_bgr: Any) -> None:
        """Render preview into the dataset window."""

        cv2 = self._import_cv2()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimg = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)

    def _update_timer_interval(self, value: int) -> None:
        """Apply timer interval changes live."""

        if self._timer.isActive():
            self._timer.start(value)

    def _update_status_labels(self) -> None:
        """Refresh capture counters and session labels."""

        self.lbl_frames.setText(f"Zachytené snímky: {self._frame_index}")
        self.lbl_saved.setText(f"Uložené vzorky: {self._saved_count}")
        session_text = str(self._session_dir) if self._session_dir is not None else "-"
        self.lbl_session_dir.setText(f"Relácia: {session_text}")

    def _refresh_help(self) -> None:
        """Refresh coordinate explanations and project calibration status."""

        config = load_config(self._config_path)
        calibration_cfg = config.get("calibration", {})
        intrinsics_path = calibration_cfg.get("intrinsics_path", "calibration/intrinsics.json")
        plane_path = calibration_cfg.get("plane_path", "calibration/plane.json")
        workspace = calibration_cfg.get("workspace_mm", {})
        origin = calibration_cfg.get("origin", "bottom_left")
        model_path = config.get("yolo", {}).get("model_path", "nenastavený")

        intrinsics_exists = self._resolve_path_for_info(intrinsics_path).exists()
        plane_exists = self._resolve_path_for_info(plane_path).exists()
        self.lbl_calibration_status.setText(
            "Kalibrácia: "
            f"intrinsics={'ANO' if intrinsics_exists else 'NIE'}, "
            f"plane={'ANO' if plane_exists else 'NIE'}, "
            f"workspace={workspace.get('width', '?')}x{workspace.get('height', '?')} mm, "
            f"origin={origin}, model={model_path}"
        )

        html = f"""
        <h2>Ako pracovať s datasetom</h2>
        <ol>
          <li>Vyberte kameru a otvorte ju v karte <b>Tvorba datasetu</b>.</li>
          <li>Nastavte výstupný priečinok, názov relácie, poznámku a stav stroja.</li>
          <li>Pre jednotlivé vzorky používajte <b>Ulož aktuálny snímok</b>.</li>
          <li>Pre sériový zber zapnite <b>Automaticky ukladať každý N-ty snímok</b>.</li>
          <li>Po zbere dáta anotujte do YOLO segmentation formátu a až potom spúšťajte tréning.</li>
        </ol>

        <h2>Čo sa ukladá</h2>
        <p>Každý obrázok sa uloží ako <code>.jpg</code> a vedľa neho vznikne <code>.json</code> s metadátami:
        poznámka, stav stroja, kamera, poradové číslo snímky a veľkosť obrazu.</p>

        <h2>Súradnice a kalibrácia</h2>
        <p><b>Pixelové súradnice</b> sú pozície v obraze z kamery. Napríklad bod <code>(x_px, y_px)</code>
        je miesto v snímke v pixeloch.</p>
        <p><b>Milimetrové súradnice</b> sú pozície na stole CNC. Po kalibrácii vieme pixely prepočítať
        na reálne mm súradnice <code>(x_mm, y_mm)</code>.</p>

        <h3>1. Intrinsics</h3>
        <p>Súbor <code>{intrinsics_path}</code> odstraňuje skreslenie objektívu. Bez neho môže byť obraz
        mierne zakrivený a prepočet bodov bude nepresný.</p>

        <h3>2. Plane kalibrácia</h3>
        <p>Súbor <code>{plane_path}</code> definuje prevod pixelov na rovinu stola. V projekte sa používa
        homografia pixel -&gt; mm. To je základ pre CNC-safe geometriu.</p>

        <h3>3. Pôvod súradníc</h3>
        <p>Aktuálny pôvod je <b>{origin}</b>. To znamená, že bod <code>(0, 0)</code> je orientovaný podľa tejto
        voľby a rozmery pracovnej plochy sú <b>{workspace.get('width', '?')} x {workspace.get('height', '?')} mm</b>.</p>

        <h2>Odporúčaný pracovný postup</h2>
        <ol>
          <li>Nasnímajte raw dataset pre rôzne svetlo, materiály a polohy obrobku.</li>
          <li>Anotujte masky tried: obrobok, upínka, ruka, nástroj.</li>
          <li>Skontrolujte kalibráciu kamery a roviny.</li>
          <li>Spustite tréning v okne <b>Modely a tréning</b>.</li>
          <li>Po tréningu nastavte nový model ako aktívny a otestujte ho v hlavnom pulte.</li>
        </ol>

        <h2>Keď si nie ste istý</h2>
        <p>Ak overlay alebo mm mriežka nesedia s hranami stola, problém býva v kalibrácii <code>plane.json</code>.
        Ak sú hrany obrazu zakrivené, problém býva v <code>intrinsics.json</code>.</p>
        """
        self.txt_help.setHtml(html)

    def _resolve_path_for_info(self, text: str) -> Path:
        """Resolve config paths relative to the project."""

        path = Path(text)
        return path if path.is_absolute() else self._project_root / path

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep preview sharp after resize."""

        super().resizeEvent(event)
        if self._frame is not None:
            self._set_preview(self._frame)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Release camera on close."""

        self._stop_camera(silent=True)
        super().closeEvent(event)
