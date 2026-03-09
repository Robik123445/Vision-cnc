"""Model management and training window for the CNC Vision UI."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QProcess
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vision.detect.detector_api import load_config, save_config
from vision.ui.training_helpers import PROJECT_ROOT, build_training_command, discover_model_candidates, to_project_relative


class ModelTrainingWindow(QDialog):
    """Separate operator window for model selection and training."""

    def __init__(
        self,
        config_path: str,
        on_config_changed: Optional[Callable[[], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._project_root = PROJECT_ROOT
        self._on_config_changed = on_config_changed
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._append_process_output)
        self._process.finished.connect(self._training_finished)

        self.setWindowTitle("Modely a tréning")
        self.resize(900, 720)
        self._build_ui()
        self._load_from_config()
        self._refresh_models()

    def _build_ui(self) -> None:
        """Build window layout with model and training tabs."""

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_box = QGroupBox("Aktívny model")
        model_form = QFormLayout(model_box)
        self.lbl_active_model = QLabel("Aktívny model: -")
        self.cmb_models = QComboBox()
        self.cmb_models.currentTextChanged.connect(self._sync_model_from_combo)
        self.edit_model_path = QLineEdit()
        self.edit_model_path.setPlaceholderText("Cesta k .pt modelu")
        self.lbl_model_exists = QLabel("Stav súboru: -")
        model_form.addRow(self.lbl_active_model)
        model_form.addRow("Nájdené modely", self.cmb_models)
        model_form.addRow("Vybraný model", self.edit_model_path)
        model_form.addRow(self.lbl_model_exists)

        model_buttons_top = QHBoxLayout()
        self.btn_refresh_models = QPushButton("Obnov zoznam modelov")
        self.btn_pick_model = QPushButton("Vybrať .pt súbor")
        model_buttons_top.addWidget(self.btn_refresh_models)
        model_buttons_top.addWidget(self.btn_pick_model)

        model_buttons_bottom = QHBoxLayout()
        self.btn_import_model = QPushButton("Skopírovať do models/")
        self.btn_activate_model = QPushButton("Nastaviť ako aktívny")
        model_buttons_bottom.addWidget(self.btn_import_model)
        model_buttons_bottom.addWidget(self.btn_activate_model)

        model_layout.addWidget(model_box)
        model_layout.addLayout(model_buttons_top)
        model_layout.addLayout(model_buttons_bottom)
        model_layout.addStretch(1)

        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        train_box = QGroupBox("Parametre tréningu")
        train_form = QFormLayout(train_box)

        self.edit_data_path = QLineEdit("dataset/data.yaml")
        self.edit_base_model = QLineEdit("yolov8n-seg.pt")
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(64, 2048)
        self.spin_imgsz.setValue(640)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 5000)
        self.spin_epochs.setValue(80)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(8)
        self.cmb_device = QComboBox()
        self.cmb_device.setEditable(True)
        self.cmb_device.addItems(["cpu", "cuda:0", "cuda:1", "mps"])
        self.edit_project_dir = QLineEdit("runs/train")
        self.edit_run_name = QLineEdit("workpiece_seg")
        self.chk_auto_activate = QCheckBox("Po úspešnom tréningu aktivovať models/workpiece-seg.pt")
        self.chk_auto_activate.setChecked(True)

        data_row = self._path_row(self.edit_data_path, self._pick_data_yaml)
        base_model_row = self._path_row(self.edit_base_model, self._pick_base_model)
        project_row = self._path_row(self.edit_project_dir, self._pick_project_dir, folder=True)

        train_form.addRow("Dataset YAML", data_row)
        train_form.addRow("Štartovací model", base_model_row)
        train_form.addRow("Veľkosť obrázka", self.spin_imgsz)
        train_form.addRow("Epochy", self.spin_epochs)
        train_form.addRow("Batch", self.spin_batch)
        train_form.addRow("Zariadenie", self.cmb_device)
        train_form.addRow("Výstupný priečinok", project_row)
        train_form.addRow("Názov behu", self.edit_run_name)
        train_form.addRow(self.chk_auto_activate)

        training_buttons = QHBoxLayout()
        self.btn_start_training = QPushButton("Spustiť tréning")
        self.btn_stop_training = QPushButton("Zastaviť tréning")
        self.btn_stop_training.setEnabled(False)
        training_buttons.addWidget(self.btn_start_training)
        training_buttons.addWidget(self.btn_stop_training)

        self.lbl_training_status = QLabel("Tréning nebeží")
        self.txt_training_log = QPlainTextEdit()
        self.txt_training_log.setReadOnly(True)
        self.txt_training_log.setPlaceholderText("Sem sa bude vypisovať priebeh tréningu")

        training_layout.addWidget(train_box)
        training_layout.addLayout(training_buttons)
        training_layout.addWidget(self.lbl_training_status)
        training_layout.addWidget(self.txt_training_log, 1)

        tabs.addTab(model_tab, "Model")
        tabs.addTab(training_tab, "Tréning")

        self.btn_refresh_models.clicked.connect(self._refresh_models)
        self.btn_pick_model.clicked.connect(self._pick_model_file)
        self.btn_import_model.clicked.connect(self._import_model_into_project)
        self.btn_activate_model.clicked.connect(self._activate_model)
        self.edit_model_path.textChanged.connect(self._update_model_path_status)
        self.btn_start_training.clicked.connect(self._start_training)
        self.btn_stop_training.clicked.connect(self._stop_training)

    def _path_row(self, line_edit: QLineEdit, handler: Callable[[], None], folder: bool = False) -> QWidget:
        """Build a line edit row with a browse button."""

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        button = QPushButton("Vybrať priečinok" if folder else "Prehľadať")
        button.clicked.connect(handler)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return row

    def _load_from_config(self) -> None:
        """Load model settings from the current configuration."""

        config = load_config(self._config_path)
        model_path = config.get("yolo", {}).get("model_path", "")
        self.edit_model_path.setText(model_path)
        self.lbl_active_model.setText(f"Aktívny model: {model_path or '-'}")
        if model_path:
            self.edit_base_model.setText(model_path)
        self._update_model_path_status()

    def _refresh_models(self) -> None:
        """Refresh local model inventory in the combo box."""

        current = self.edit_model_path.text().strip()
        discovered = discover_model_candidates(project_root=self._project_root)
        if current and current not in discovered:
            discovered.insert(0, current)

        self.cmb_models.blockSignals(True)
        self.cmb_models.clear()
        if discovered:
            self.cmb_models.addItems(discovered)
            if current:
                idx = self.cmb_models.findText(current)
                if idx >= 0:
                    self.cmb_models.setCurrentIndex(idx)
        else:
            self.cmb_models.addItem("Neboli nájdené žiadne .pt modely")
        self.cmb_models.blockSignals(False)
        self._append_log(f"Nájdených modelov: {max(len(discovered), 0)}")

    def _sync_model_from_combo(self, text: str) -> None:
        """Mirror combo selection into the active model line edit."""

        if text and not text.startswith("Neboli nájdené"):
            self.edit_model_path.setText(text)

    def _pick_model_file(self) -> None:
        """Pick a YOLO weight file from disk."""

        path, _ = QFileDialog.getOpenFileName(self, "Vyber model", "", "YOLO model (*.pt)")
        if not path:
            return
        self.edit_model_path.setText(to_project_relative(path, project_root=self._project_root))

    def _pick_data_yaml(self) -> None:
        """Pick dataset YAML for training."""

        path, _ = QFileDialog.getOpenFileName(self, "Vyber data.yaml", "", "YAML (*.yaml *.yml)")
        if path:
            self.edit_data_path.setText(to_project_relative(path, project_root=self._project_root))

    def _pick_base_model(self) -> None:
        """Pick training base model."""

        path, _ = QFileDialog.getOpenFileName(self, "Vyber štartovací model", "", "YOLO model (*.pt)")
        if path:
            self.edit_base_model.setText(to_project_relative(path, project_root=self._project_root))

    def _pick_project_dir(self) -> None:
        """Pick training output directory."""

        path = QFileDialog.getExistingDirectory(self, "Vyber výstupný priečinok", str(self._project_root / "runs"))
        if path:
            self.edit_project_dir.setText(to_project_relative(path, project_root=self._project_root))

    def _resolve_path(self, path_text: str) -> Path:
        """Resolve project-local paths relative to the repo root."""

        path = Path(path_text)
        return path if path.is_absolute() else self._project_root / path

    def _update_model_path_status(self) -> None:
        """Show whether the selected model file exists on disk."""

        text = self.edit_model_path.text().strip()
        if not text:
            self.lbl_model_exists.setText("Stav súboru: model nie je vybraný")
            return
        resolved = self._resolve_path(text)
        if resolved.exists():
            self.lbl_model_exists.setText(f"Stav súboru: nájdený ({resolved})")
        else:
            self.lbl_model_exists.setText(f"Stav súboru: nenájdený ({resolved})")

    def _write_active_model_to_config(self, model_path: str) -> None:
        """Persist selected model path into the detector config."""

        config = load_config(self._config_path)
        config.setdefault("detector", {})
        config.setdefault("yolo", {})
        config["detector"]["type"] = "yolo"
        config["yolo"]["model_path"] = model_path
        save_config(config, self._config_path)
        self.lbl_active_model.setText(f"Aktívny model: {model_path}")
        self.edit_base_model.setText(model_path)
        self._append_log(f"Aktívny model uložený do konfigurácie: {model_path}")
        if self._on_config_changed is not None:
            self._on_config_changed()

    def _activate_model(self) -> None:
        """Activate selected model path in the config."""

        model_path = self.edit_model_path.text().strip()
        if not model_path:
            QMessageBox.information(self, "Bez modelu", "Najprv vyberte model.")
            return

        self._write_active_model_to_config(model_path)
        QMessageBox.information(self, "Model uložený", "Aktívny model bol uložený do konfigurácie.")

    def _import_model_into_project(self) -> None:
        """Copy an external model into `models/` and select it."""

        model_path = self.edit_model_path.text().strip()
        if not model_path:
            QMessageBox.information(self, "Bez modelu", "Najprv vyberte modelový súbor.")
            return

        source = self._resolve_path(model_path)
        if not source.exists():
            QMessageBox.warning(self, "Model neexistuje", f"Súbor sa nenašiel: {source}")
            return

        target_dir = self._project_root / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        relative_target = to_project_relative(target, project_root=self._project_root)
        self.edit_model_path.setText(relative_target)
        self._refresh_models()
        self._append_log(f"Model skopírovaný do projektu: {relative_target}")

    def _append_log(self, text: str) -> None:
        """Append one line into the training log view."""

        self.txt_training_log.appendPlainText(text)

    def _start_training(self) -> None:
        """Launch the training script inside a QProcess."""

        if self._process.state() != QProcess.ProcessState.NotRunning:
            return

        data_path = self.edit_data_path.text().strip()
        if not data_path:
            QMessageBox.information(self, "Bez datasetu", "Vyberte dataset YAML pre tréning.")
            return

        command = build_training_command(
            sys.executable,
            data_path=data_path,
            imgsz=self.spin_imgsz.value(),
            epochs=self.spin_epochs.value(),
            batch=self.spin_batch.value(),
            device=self.cmb_device.currentText().strip() or "cpu",
            project_dir=self.edit_project_dir.text().strip() or "runs/train",
            run_name=self.edit_run_name.text().strip() or "workpiece_seg",
            model_path=self.edit_base_model.text().strip() or "yolov8n-seg.pt",
        )

        self.txt_training_log.clear()
        self._append_log("Spúšťam tréning:")
        self._append_log(" ".join(command))

        self._process.setWorkingDirectory(str(self._project_root))
        self._process.start(command[0], command[1:])
        if not self._process.waitForStarted(3000):
            QMessageBox.warning(self, "Tréning sa nespustil", "Nepodarilo sa spustiť tréningový proces.")
            return

        self.lbl_training_status.setText("Tréning beží...")
        self.btn_start_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)

    def _append_process_output(self) -> None:
        """Append process output into the log pane."""

        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.txt_training_log.moveCursor(QTextCursor.MoveOperation.End)
            self.txt_training_log.insertPlainText(data)
            self.txt_training_log.ensureCursorVisible()

    def _stop_training(self) -> None:
        """Stop the running training process."""

        if self._process.state() == QProcess.ProcessState.NotRunning:
            return

        self._append_log("\nZastavujem tréning...\n")
        self._process.terminate()
        if not self._process.waitForFinished(2000):
            self._process.kill()
            self._process.waitForFinished(2000)

    def _training_finished(self, exit_code: int, exit_status) -> None:
        """Handle training process completion."""

        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        if exit_code == 0:
            self.lbl_training_status.setText("Tréning dokončený úspešne")
            self._append_log("\nTréning dokončený úspešne.")
            trained_model = self._project_root / "models" / "workpiece-seg.pt"
            if self.chk_auto_activate.isChecked() and trained_model.exists():
                relative = to_project_relative(trained_model, project_root=self._project_root)
                self.edit_model_path.setText(relative)
                self._write_active_model_to_config(relative)
                self._append_log(f"Aktívny model prepnutý na {relative}")
            self._refresh_models()
        else:
            self.lbl_training_status.setText(f"Tréning skončil s chybou (kód {exit_code})")
            self._append_log(f"\nTréning skončil s chybou. Návratový kód: {exit_code}")
