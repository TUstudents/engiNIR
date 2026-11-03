"""Adding data files to a project, with their calibration (attachment contract).

:class:`AddDataFileDialog` asks how a data file is calibrated:

- dark/white references, each from a file or from a frame range of the data
  file itself (a white tile, shutter-closed frames);
- the references of another data file of the project (one set per session);
- the detector levels of the file (or the EVK defaults);
- raw counts.

It shows the mean spectrum and the saturated share of each reference before
it is accepted. The answer is a :class:`DataFileRequest`, whose
:meth:`~DataFileRequest.add_dataset_kwargs` go straight to
:meth:`enginir.Project.add_dataset`, the same rule scripts use.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from .._workflow import DEFAULT_BLACK_LEVEL, DEFAULT_WHITE_LEVEL, read_frames
from ..io import load_spectra
from ..project import CalibrationSpec, DatasetRef
from ._theme import panel_colors, secondary_text_style
from ._ui_helpers import style_axes

logger = get_logger(__name__)

_FILE_FILTER = "Hyperspectral files (*.eiff *.hdr *.h5 *.spc *.spa *.jdx *.csv);;All files (*)"
#: Calibration choices of the dialog.
REFERENCES, SHARED, LEVELS, RAW = "references", "shared", "levels", "raw"


@dataclass(frozen=True)
class DataFileRequest:
    """How to add one data file: its path, signal unit and reference sources."""

    path: str
    unit: str
    dark: str | None = None
    white: str | None = None
    dark_frames: tuple[int, int] | None = None
    white_frames: tuple[int, int] | None = None
    references_from: str | None = None

    def add_dataset_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`enginir.Project.add_dataset` (besides the path)."""
        return {"unit": self.unit, "dark": self.dark, "white": self.white,
                "dark_frames": self.dark_frames, "white_frames": self.white_frames,
                "references_from": self.references_from}


def saturation_level(meta: Mapping[str, Any], dtype: np.dtype) -> float | None:
    """The detector's saturation value: from the bit depth, else the integer dtype's maximum."""
    bits = meta.get("bits_per_value")
    if isinstance(bits, (int, np.integer)) and 0 < int(bits) < 32:
        return float(2 ** int(bits) - 1)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def reference_summary(raw: np.ndarray, saturation: float | None) -> tuple[np.ndarray, float]:
    """Mean spectrum over all pixels of a reference and its share of saturated values (%)."""
    values = np.asarray(raw)
    mean = values.reshape(-1, values.shape[-1]).mean(axis=0)
    share = 0.0 if saturation is None else float(np.mean(values >= saturation) * 100.0)
    return mean, share


class _ReferenceSource(QWidget):
    """Where one reference comes from: a file, or a frame range of the data file."""

    def __init__(self, label: str, n_frames: int, default_frames: tuple[int, int],
                 on_change: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self._from_file = QRadioButton("File")
        self._from_frames = QRadioButton("Frames of this file")
        group = QButtonGroup(self)
        group.addButton(self._from_file)
        group.addButton(self._from_frames)
        self._path = QLineEdit()
        self._path.setPlaceholderText(f"{label} reference file")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        self._first = QSpinBox()
        self._last = QSpinBox()
        for spin, value in ((self._first, default_frames[0]), (self._last, default_frames[1])):
            spin.setRange(0, max(0, n_frames - 1))
            spin.setValue(value)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._from_file, 0, 0)
        layout.addWidget(self._path, 0, 1, 1, 3)
        layout.addWidget(browse, 0, 4)
        layout.addWidget(self._from_frames, 1, 0)
        layout.addWidget(self._first, 1, 1)
        layout.addWidget(QLabel("to"), 1, 2)
        layout.addWidget(self._last, 1, 3)
        self._from_frames.setChecked(True)
        for signal in (self._from_file.toggled, self._path.textChanged,
                       self._first.valueChanged, self._last.valueChanged):
            signal.connect(on_change)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"{self.label} reference", "", _FILE_FILTER)
        if path:
            self._path.setText(path)
            self._from_file.setChecked(True)

    def use_file(self, path: str) -> None:
        self._path.setText(path)
        self._from_file.setChecked(True)

    def use_frames(self, first: int, last: int) -> None:
        self._first.setValue(first)
        self._last.setValue(last)
        self._from_frames.setChecked(True)

    def source(self) -> tuple[str | None, tuple[int, int] | None]:
        """``(file, None)`` or ``(None, (first, last))``."""
        if self._from_file.isChecked():
            return (self._path.text().strip() or None), None
        return None, (self._first.value(), self._last.value())


class AddDataFileDialog(QDialog):
    """Choose the calibration of a data file before it is added to the project."""

    def __init__(
        self,
        path: str,
        header: Any,
        datasets: Mapping[str, DatasetRef],
        active_calibrations: Mapping[str, CalibrationSpec],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add data file")
        self._path = os.path.abspath(path)
        self._header = header
        n_frames, height, n_bands = (int(n) for n in header.cube_shape)
        self._precalibrated = np.dtype(header.X.dtype).kind == "f"
        self._saturation = saturation_level(header.meta, np.dtype(header.X.dtype))
        self.request: DataFileRequest | None = None
        # Created first: choosing the default option below already schedules a check.
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(250)
        self._check_timer.timeout.connect(self.check_references)

        info = QLabel(f"<b>{os.path.basename(path)}</b><br>{n_frames} frames × {height} rows × "
                      f"{n_bands} bands, {np.dtype(header.X.dtype).name}")
        layout = QVBoxLayout(self)
        layout.addWidget(info)

        self._unit = QComboBox()
        self._choice: dict[str, QRadioButton] = {}
        calibration = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration)
        if self._precalibrated:
            calibration_layout.addWidget(QLabel(
                "The file holds floating-point values: they are taken as already calibrated."))
            self._unit.addItems(["Reflectance", "Absorbance"])
        else:
            group = QButtonGroup(self)
            for key, text in ((REFERENCES, "Dark and white references"),
                              (SHARED, "Same references as another data file"),
                              (LEVELS, "Detector levels, no references"),
                              (RAW, "Raw counts, no calibration")):
                button = QRadioButton(text)
                group.addButton(button)
                self._choice[key] = button
                button.toggled.connect(self._on_choice_changed)
            self._dark = _ReferenceSource("Dark", n_frames, (0, min(2, n_frames - 1)),
                                          self._schedule_check)
            self._white = _ReferenceSource("White", n_frames,
                                           (max(0, n_frames - 3), n_frames - 1),
                                           self._schedule_check)
            references = QWidget()
            references_layout = QFormLayout(references)
            references_layout.setContentsMargins(24, 0, 0, 0)
            references_layout.addRow("Dark:", self._dark)
            references_layout.addRow("White:", self._white)
            self._references = references
            self._shared = QComboBox()
            for dataset_id, dataset in datasets.items():
                active = active_calibrations.get(dataset_id)
                if active is not None and active.dark_ref_data is not None:
                    self._shared.addItem(os.path.basename(dataset.source_path),
                                         dataset.source_path)
            self._shared.currentIndexChanged.connect(self._schedule_check)
            black = header.meta.get("black_level", DEFAULT_BLACK_LEVEL)
            white = header.meta.get("white_level", DEFAULT_WHITE_LEVEL)
            self._choice[LEVELS].setText(f"Detector levels, no references "
                                         f"(black {black:g}, white {white:g})")
            calibration_layout.addWidget(self._choice[REFERENCES])
            calibration_layout.addWidget(references)
            calibration_layout.addWidget(self._choice[SHARED])
            shared_row = QHBoxLayout()
            shared_row.setContentsMargins(24, 0, 0, 0)
            shared_row.addWidget(self._shared, 1)
            calibration_layout.addLayout(shared_row)
            calibration_layout.addWidget(self._choice[LEVELS])
            calibration_layout.addWidget(self._choice[RAW])
            self._unit.addItems(["Absorbance", "Reflectance"])
            if self._shared.count() == 0:
                self._choice[SHARED].setEnabled(False)
                self._shared.setEnabled(False)
            self._choice[REFERENCES].setChecked(True)
        layout.addWidget(calibration)
        unit_row = QFormLayout()
        unit_row.addRow("Signal unit:", self._unit)
        layout.addLayout(unit_row)

        self._check = QLabel("")
        self._check.setWordWrap(True)
        self._check.setStyleSheet(secondary_text_style())
        self._figure = Figure(figsize=(4.5, 1.8), facecolor=panel_colors()["bg"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(140)
        layout.addWidget(self._canvas)
        layout.addWidget(self._check)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                         | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._on_choice_changed()

    # -- state ----------------------------------------------------------

    def choice(self) -> str:
        if self._precalibrated:
            return LEVELS  # no reference context; the unit decides the calibration
        return next(key for key, button in self._choice.items() if button.isChecked())

    def set_choice(self, key: str) -> None:
        self._choice[key].setChecked(True)

    def dark_source(self) -> _ReferenceSource:
        return self._dark

    def white_source(self) -> _ReferenceSource:
        return self._white

    def set_shared(self, path: str) -> None:
        self._shared.setCurrentIndex(self._shared.findData(path))

    def set_unit(self, unit: str) -> None:
        self._unit.setCurrentText(unit)

    def _on_choice_changed(self, *_args) -> None:
        if self._precalibrated:
            self.check_references()
            return
        choice = self.choice()
        self._references.setEnabled(choice == REFERENCES)
        self._shared.setEnabled(choice == SHARED and self._shared.count() > 0)
        self._unit.setEnabled(choice != RAW)
        self._schedule_check()

    def _schedule_check(self, *_args) -> None:
        self._check_timer.start()

    # -- reference check ------------------------------------------------

    def _raw_reference(self, source: _ReferenceSource) -> np.ndarray:
        file, frames = source.source()
        if frames is not None:
            first, last = frames
            if last < first:
                raise ValueError(f"{source.label}: the last frame comes before the first")
            return read_frames(self._path, first, last)
        if file is None:
            raise ValueError(f"{source.label}: choose a reference file")
        X = np.asarray(load_spectra(file).X)
        return X[:, np.newaxis, :] if X.ndim == 2 else X

    def check_references(self) -> bool:
        """Show the references' mean spectra and saturated shares; ``False`` if one is invalid."""
        colors = panel_colors()
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        style_axes(ax, colors)
        lines: list[str] = []
        ok = True
        choice = self.choice()
        if choice == REFERENCES:
            for source, color in ((self._dark, "#7f7f7f"), (self._white, colors["accent"])):
                try:
                    raw = self._raw_reference(source)
                except (OSError, ValueError, KeyError) as exc:
                    lines.append(f"{source.label}: {exc}")
                    ok = False
                    continue
                mean, saturated = reference_summary(raw, self._saturation)
                ax.plot(np.asarray(self._header.wavelengths), mean, color=color,
                        label=source.label)
                lines.append(f"{source.label}: mean {float(mean.mean()):.4g} counts, "
                             f"{saturated:.2f} % saturated")
            ax.set_xlabel("Wavelength (nm)", color=colors["fg"], fontsize=8)
            ax.set_ylabel("Counts", color=colors["fg"], fontsize=8)
        elif choice == SHARED:
            if self._shared.count() == 0:
                lines.append("No data file of the project has reference spectra.")
                ok = False
            else:
                lines.append(f"References copied from {self._shared.currentText()}.")
        elif choice == LEVELS and not self._precalibrated:
            lines.append("Reflectance from the detector levels; no reference spectra.")
        elif choice == RAW:
            lines.append("Raw counts; ROIs keep no reference spectra.")
        self._figure.tight_layout()
        self._canvas.draw_idle()
        self._check.setText("\n".join(lines))
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(ok)
        return ok

    def check_text(self) -> str:
        return self._check.text()

    # -- result ---------------------------------------------------------

    def build_request(self) -> DataFileRequest:
        """The request for the current choices (without checking the references)."""
        choice = self.choice()
        if choice == RAW:
            return DataFileRequest(self._path, unit="DN")
        unit = self._unit.currentText()
        if choice == SHARED:
            return DataFileRequest(self._path, unit=unit,
                                   references_from=str(self._shared.currentData()))
        if choice == REFERENCES:
            dark_file, dark_frames = self._dark.source()
            white_file, white_frames = self._white.source()
            return DataFileRequest(self._path, unit=unit, dark=dark_file, white=white_file,
                                   dark_frames=dark_frames, white_frames=white_frames)
        return DataFileRequest(self._path, unit=unit)

    def _accept(self) -> None:
        self._check_timer.stop()
        if not self.check_references():
            return
        self.request = self.build_request()
        self.accept()


__all__ = ["AddDataFileDialog", "DataFileRequest", "reference_summary", "saturation_level"]
