"""Adding data files to a project, with their reference correction (attachment contract).

:class:`AddDataFileDialog` asks how a data file is calibrated:

- dark/white references, each from a file or from a frame range of the data
  file itself (a white tile, shutter-closed frames);
- the references of another data file of the project (one set per session);
- the detector levels of the file (or the EVK defaults);
- raw counts.

Its plot shows the mean spectrum of the file (a few middle frames) in counts,
with what the chosen reference correction uses: the mean reference spectra (with their
saturated share), the references of the other data file, or the detector
levels. The answer is a :class:`DataFileRequest`, whose
:meth:`~DataFileRequest.add_dataset_kwargs` go straight to
:meth:`enginir.Project.add_dataset`, the same rule scripts use.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from .._workflow import data_file_labels, read_frames
from ..io import load_spectra
from ..io import is_band_numbers, user_table_folder, wavelength_tables
from ..project import ReferenceCorrection, DatasetRecord
from ._preferences import load as load_preferences
from ._preferences import start_folder
from ._theme import GUIDE_COLOR, panel_colors, secondary_text_style
from ._ui_helpers import spectral_axis_label, style_axes

logger = get_logger(__name__)

_FILE_FILTER = "Hyperspectral files (*.eiff *.hdr *.h5 *.spc *.spa *.jdx *.csv);;All files (*)"
#: Reference correction choices of the dialog.
REFERENCES, SHARED, LEVELS, RAW = "references", "shared", "levels", "raw"
#: Working signal stored with calibrated counts; Explore and Fit model choose the shown one.
WORKING_SIGNAL = "Absorbance"


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
    black_level: float | None = None
    white_level: float | None = None
    #: Name of the wavelength table for a file that stores no wavelengths, or ``None``.
    wavelength_table: str | None = None
    #: The dataset's name ("Layer 7"; ``""``: none, its file name is shown).
    name: str | None = None

    def add_dataset_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`enginir.Project.add_dataset` (besides the path)."""
        return {"unit": self.unit, "dark": self.dark, "white": self.white,
                "dark_frames": self.dark_frames, "white_frames": self.white_frames,
                "references_from": self.references_from, "black_level": self.black_level,
                "white_level": self.white_level, "wavelength_table": self.wavelength_table,
                "name": self.name}


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
        path, _ = QFileDialog.getOpenFileName(self, f"{self.label} reference",
                                              start_folder("dataset"), _FILE_FILTER)
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
    """Choose the reference correction of a data file before it is added to the project."""

    def __init__(
        self,
        path: str,
        header: Any,
        datasets: Mapping[str, DatasetRecord],
        active_reference_corrections: Mapping[str, ReferenceCorrection],
        parent: QWidget | None = None,
        *,
        trained_with: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add dataset")
        self._path = os.path.abspath(path)
        self._header = header
        n_frames, height, n_bands = (int(n) for n in header.cube_shape)
        self._already_corrected = np.dtype(header.X.dtype).kind == "f"
        self._saturation = saturation_level(header.meta, np.dtype(header.X.dtype))
        self.request: DataFileRequest | None = None
        self._data_mean: np.ndarray | None = None  # read on the first check
        #: Mean dark and white spectra of each data file that has references, by path.
        self._shared_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Created first: choosing the default option below already schedules a check.
        self._check_timer = QTimer(self)
        self._check_timer.setSingleShot(True)
        self._check_timer.setInterval(250)
        self._check_timer.timeout.connect(self.check_references)

        info = QLabel(f"<b>{os.path.basename(path)}</b><br>{n_frames} frames × {height} rows × "
                      f"{n_bands} bands, {np.dtype(header.X.dtype).name}")
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        names = QFormLayout()
        self._name = QLineEdit()  # a new dataset: the dialog opens for unregistered files only
        self._name.setPlaceholderText(os.path.basename(path))
        self._name.setToolTip("A short name shown instead of the file name, and {name} in "
                              "ROI grid names (e.g. 'Layer 7'). Empty: the file name.")
        names.addRow("Name:", self._name)
        layout.addLayout(names)
        self._wavelengths = self._wavelength_choice(header, n_bands)
        if self._wavelengths is not None:
            row = QFormLayout()
            row.addRow("Wavelengths:", self._wavelengths)
            layout.addLayout(row)
        if trained_with:
            # Opened from Predict: say what the model used, never reuse it silently
            # (dataset-reference-correction contract).
            trained = QLabel(f"The model was fitted with: {trained_with}. Choose this scan's "
                             "own reference correction (references of this session, or the "
                             "levels).")
            trained.setWordWrap(True)
            trained.setStyleSheet(secondary_text_style())
            layout.addWidget(trained)

        self._unit = QComboBox()
        self._choice: dict[str, QRadioButton] = {}
        reference_correction = QGroupBox("Reference correction")
        reference_correction_layout = QVBoxLayout(reference_correction)
        if self._already_corrected:
            reference_correction_layout.addWidget(QLabel(
                "The file holds floating-point values: they are taken as already corrected."))
            self._unit.addItems(["Reflectance", "Absorbance"])
        else:
            group = QButtonGroup(self)
            for key, text in ((REFERENCES, "Dark and white references"),
                              (SHARED, "Same references as another dataset"),
                              (LEVELS, "Detector calibrated with dark and white reference levels"),
                              (RAW, "Raw counts (DN), no reference correction")):
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
            labels = data_file_labels(datasets)
            for dataset_id, dataset in datasets.items():
                active = active_reference_corrections.get(dataset_id)
                if active is not None and active.dark_ref_data is not None \
                        and active.white_ref_data is not None:
                    self._shared.addItem(labels[dataset_id], dataset.source_path)
                    self._shared_means[dataset.source_path] = (
                        np.asarray(active.dark_ref_data, dtype=np.float64).mean(axis=0),
                        np.asarray(active.white_ref_data, dtype=np.float64).mean(axis=0))
            self._shared.currentIndexChanged.connect(self._schedule_check)
            limit = float(np.iinfo(header.X.dtype).max) if np.issubdtype(
                header.X.dtype, np.integer) else 1e9
            self._black = QDoubleSpinBox()
            self._white_level = QDoubleSpinBox()
            # The file's own levels, else those of Tools > Preferences (default 128 / 3100).
            preferences = load_preferences()
            for spin, value in ((self._black, header.meta.get("black_level",
                                                              preferences.black_level)),
                                (self._white_level,
                                 header.meta.get("white_level", preferences.white_level))):
                spin.setRange(0.0, limit)
                spin.setDecimals(1)
                spin.setValue(float(value))
                spin.valueChanged.connect(self._schedule_check)
            self._choice[LEVELS].setText("Detector calibrated with dark and white reference levels")
            reference_correction_layout.addWidget(self._choice[REFERENCES])
            reference_correction_layout.addWidget(references)
            reference_correction_layout.addWidget(self._choice[SHARED])
            shared_row = QHBoxLayout()
            shared_row.setContentsMargins(24, 0, 0, 0)
            shared_row.addWidget(self._shared, 1)
            reference_correction_layout.addLayout(shared_row)
            reference_correction_layout.addWidget(self._choice[LEVELS])
            levels_row = QHBoxLayout()
            levels_row.setContentsMargins(24, 0, 0, 0)
            levels_row.addWidget(QLabel("Black:"))
            levels_row.addWidget(self._black)
            levels_row.addWidget(QLabel("White:"))
            levels_row.addWidget(self._white_level)
            levels_row.addStretch(1)
            reference_correction_layout.addLayout(levels_row)
            reference_correction_layout.addWidget(self._choice[RAW])
            if self._shared.count() == 0:
                self._choice[SHARED].setEnabled(False)
                self._shared.setEnabled(False)
            # The instrument's own levels need nothing else; references are a deliberate step.
            self._choice[LEVELS].setChecked(True)
        layout.addWidget(reference_correction)
        if self._already_corrected:
            # The only case where the unit is a fact about the file rather than a view of it.
            unit_row = QFormLayout()
            unit_row.addRow("The file holds:", self._unit)
            layout.addLayout(unit_row)
        else:
            note = QLabel("The project keeps the raw counts (DN) with these references or "
                          "levels; the signal is chosen in Explore and in Fit Model.")
            note.setWordWrap(True)
            note.setStyleSheet(secondary_text_style())
            layout.addWidget(note)

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
        if self._already_corrected:
            return LEVELS  # no reference context; the unit decides the reference correction
        return next(key for key, button in self._choice.items() if button.isChecked())

    def set_choice(self, key: str) -> None:
        self._choice[key].setChecked(True)

    def dark_source(self) -> _ReferenceSource:
        return self._dark

    def white_source(self) -> _ReferenceSource:
        return self._white

    def set_levels(self, black: float, white: float) -> None:
        """Set the detector levels used without references."""
        self._black.setValue(black)
        self._white_level.setValue(white)

    def set_unit(self, unit: str) -> None:
        self._unit.setCurrentText(unit)

    def _on_choice_changed(self, *_args) -> None:
        if self._already_corrected:
            self.check_references()
            return
        choice = self.choice()
        self._references.setEnabled(choice == REFERENCES)
        self._shared.setEnabled(choice == SHARED and self._shared.count() > 0)
        self._black.setEnabled(choice == LEVELS)
        self._white_level.setEnabled(choice == LEVELS)
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

    def data_mean(self) -> np.ndarray | None:
        """Mean spectrum of up to five middle frames of the file, as recorded."""
        if self._data_mean is None:
            n_frames = int(self._header.cube_shape[0])
            first = max(0, n_frames // 2 - 2)
            try:
                frames = read_frames(self._path, first, min(n_frames - 1, first + 4))
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("No data preview for {}: {}", self._path, exc)
                return None
            self._data_mean = frames.reshape(-1, frames.shape[-1]).mean(axis=0)
        return self._data_mean

    def check_references(self) -> bool:
        """Plot the file and what the chosen reference correction uses; ``False`` if a reference is invalid.

        The file's mean spectrum is always shown in counts (or in its own
        values when precalibrated), with the mean dark and white references,
        the other data file's references, or the detector levels.
        """
        colors = panel_colors()
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        style_axes(ax, colors)
        wavelengths = np.asarray(self._header.wavelengths)
        lines: list[str] = []
        ok = True
        choice = self.choice()
        data = self.data_mean()
        if data is not None:
            ax.plot(wavelengths, data, color=colors["fg"], linewidth=1.0, label="Data (mean)")
        if choice == REFERENCES:
            for source, color in ((self._dark, GUIDE_COLOR), (self._white, colors["accent"])):
                try:
                    raw = self._raw_reference(source)
                except (OSError, ValueError, KeyError) as exc:
                    lines.append(f"{source.label}: {exc}")
                    ok = False
                    continue
                mean, saturated = reference_summary(raw, self._saturation)
                ax.plot(wavelengths, mean, color=color, label=source.label)
                lines.append(f"{source.label}: mean {float(mean.mean()):.4g} counts, "
                             f"{saturated:.2f} % saturated")
        elif choice == SHARED:
            if self._shared.count() == 0:
                lines.append("No dataset of the project has reference spectra.")
                ok = False
            else:
                lines.append(f"References copied from {self._shared.currentText()}.")
                means = self._shared_means.get(str(self._shared.currentData()))
                if means is not None:
                    ax.plot(wavelengths, means[0], color=GUIDE_COLOR, label="Dark")
                    ax.plot(wavelengths, means[1], color=colors["accent"], label="White")
        elif choice == LEVELS and not self._already_corrected:
            black, white = self._black.value(), self._white_level.value()
            ax.axhline(black, color=GUIDE_COLOR, linestyle="--", label=f"Black level {black:g}")
            ax.axhline(white, color=colors["accent"], linestyle="--",
                       label=f"White level {white:g}")
            if white > black:
                lines.append("Reflectance from the detector levels; no reference spectra.")
            else:
                lines.append("The white level must be above the black level.")
                ok = False
        elif choice == RAW:
            lines.append("Raw counts; ROIs keep no reference spectra.")
        ax.set_xlabel(spectral_axis_label(self._header.wavelengths)[0], color=colors["fg"],
                      fontsize=8)
        ax.set_ylabel("Values" if self._already_corrected else "Counts", color=colors["fg"],
                      fontsize=8)
        if ax.get_lines():
            ax.legend(fontsize=7, loc="best", frameon=False, labelcolor=colors["fg"])
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

    def _reference_request(self) -> DataFileRequest:
        """The request for the current choices (without checking the references)."""
        choice = self.choice()
        if choice == RAW:
            return DataFileRequest(self._path, unit="DN")
        # Counts are calibrated to reflectance; absorbance is the working default (as in
        # Project.add_dataset). Floating-point files hold the unit chosen for them.
        unit = self._unit.currentText() if self._already_corrected else WORKING_SIGNAL
        if choice == SHARED:
            return DataFileRequest(self._path, unit=unit,
                                   references_from=str(self._shared.currentData()))
        if choice == REFERENCES:
            dark_file, dark_frames = self._dark.source()
            white_file, white_frames = self._white.source()
            return DataFileRequest(self._path, unit=unit, dark=dark_file, white=white_file,
                                   dark_frames=dark_frames, white_frames=white_frames)
        if not self._already_corrected:
            return DataFileRequest(self._path, unit=unit, black_level=self._black.value(),
                                   white_level=self._white_level.value())
        return DataFileRequest(self._path, unit=unit)

    def _wavelength_choice(self, header: Any, n_bands: int) -> QComboBox | None:
        """For a file that stores no wavelengths: band numbers or a table of its band count.

        The only table of that band count is preselected (plan wavelength-tables, D2);
        ``None`` for a file with its own wavelengths.
        """
        if header.meta.get("x_units") != "band_index" and not is_band_numbers(header.wavelengths):
            return None
        combo = QComboBox()
        combo.addItem("Band numbers (the file has no wavelengths)", None)
        tables = wavelength_tables(n_bands)
        for table in tables:
            combo.addItem(table.describe(), table.name)
            combo.setItemData(combo.count() - 1, table.path, Qt.ItemDataRole.ToolTipRole)
        if len(tables) == 1:
            combo.setCurrentIndex(1)
        combo.setToolTip(f"Tables of {n_bands} bands: those of engiNIR and the CSV files in "
                         f"{user_table_folder()}")
        return combo

    def build_request(self) -> DataFileRequest:
        """The request for the current choices, with the chosen wavelength table."""
        request = replace(self._reference_request(), name=self._name.text().strip())
        table = None if self._wavelengths is None else self._wavelengths.currentData()
        return replace(request, wavelength_table=table) if table else request

    def _accept(self) -> None:
        self._check_timer.stop()
        if not self.check_references():
            return
        self.request = self.build_request()
        self.accept()


class SetWavelengthsDialog(QDialog):
    """Project > Set Wavelengths: a wavelength table for datasets added with band numbers.

    Lists the datasets whose axis is band numbers (checked: ``preselected``,
    else all) and the tables of their band counts; :attr:`choice` is
    ``(table name, dataset ids of its band count)`` after OK. Their ROIs get
    the wavelengths too (plan wavelength-tables, D1).
    """

    def __init__(self, datasets: Mapping[str, DatasetRecord], labels: Mapping[str, str], *,
                 preselected: tuple[str, ...] = (), table: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set wavelengths")
        self.choice: tuple[str, tuple[str, ...]] | None = None
        self._datasets = {dataset_id: ref for dataset_id, ref in datasets.items()
                          if is_band_numbers(ref.wavelengths)}
        layout = QVBoxLayout(self)
        intro = QLabel("These datasets have band numbers: their files store no wavelengths. "
                       "A wavelength table gives them and their ROIs wavelengths in nm. Models "
                       "keep the band numbers they were fitted with and predict as before.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self._list = QListWidget()
        for dataset_id, ref in self._datasets.items():
            item = QListWidgetItem(f"{labels.get(dataset_id, dataset_id)}  ({ref.n_bands} bands)")
            item.setData(Qt.ItemDataRole.UserRole, dataset_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = not preselected or dataset_id in preselected
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        layout.addWidget(self._list)
        self._table = QComboBox()
        counts = sorted({ref.n_bands for ref in self._datasets.values()})
        for found in (t for n in counts for t in wavelength_tables(n)):
            self._table.addItem(found.describe(), found.name)
            self._table.setItemData(self._table.count() - 1, found.n_bands,
                                    Qt.ItemDataRole.UserRole + 1)
        if table is not None and self._table.findData(table) >= 0:
            self._table.setCurrentIndex(self._table.findData(table))
        form = QFormLayout()
        form.addRow("Table:", self._table)
        layout.addLayout(form)
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet(secondary_text_style())
        layout.addWidget(self._note)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                         | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._table.currentIndexChanged.connect(self._update)
        self._list.itemChanged.connect(self._update)
        self._update()

    def _selected(self) -> tuple[str, ...]:
        """The checked datasets of the chosen table's band count."""
        n_bands = self._table.currentData(Qt.ItemDataRole.UserRole + 1)
        return tuple(str(item.data(Qt.ItemDataRole.UserRole))
                     for item in (self._list.item(i) for i in range(self._list.count()))
                     if item is not None and item.checkState() == Qt.CheckState.Checked
                     and self._datasets[str(item.data(Qt.ItemDataRole.UserRole))].n_bands
                     == n_bands)

    def _update(self, *_args: Any) -> None:
        selected = self._selected()
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(bool(selected))
        if self._table.count() == 0:
            self._note.setText(f"No wavelength table has these band counts. Put one in "
                               f"{user_table_folder()} (a CSV file 'band,wavelength_nm' with a "
                               "'# name: ...' line).")
        else:
            self._note.setText(f"{len(selected)} dataset(s) of the table's band count are "
                               "chosen.")

    def _accept(self) -> None:
        selected = self._selected()
        if selected and self._table.currentData():
            self.choice = (str(self._table.currentData()), selected)
            self.accept()


__all__ = ["AddDataFileDialog", "DataFileRequest", "SetWavelengthsDialog", "reference_summary",
           "saturation_level"]
