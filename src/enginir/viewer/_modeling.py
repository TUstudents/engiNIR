"""Modeling view widgets for target editing and model workflows."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from PyQt6.QtCore import QRegularExpression, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QMenu,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .._checks import log_and_raise
from .._logging import get_logger
from ..io.project_store import load_model_run_artifact_refs
from .._workflow import recipe_id_for, resolve_sample_set, resolve_signal_domain
from ..modeling import fold_warning_text
from ..preprocessing import PreprocessingRecipe
from ..preprocessing.recipe import format_ranges, parse_ranges
from ..project import check_signal_domain_compatibility
from ..project import (
    FitSpec,
    ModelRun,
    SampleSet,
    SignalDomainContract,
    TargetDefinition,
)
from ._band_ranges import (
    WATER_BANDS,
    BandRangeDialog,
    BandSource,
    ProposalRun,
    ProposeBandsDialog,
)
from ._jobs import BusyStatusWidget, BackgroundWorker, start_background_job
from ._state import ProjectState, signal_label, ViewerState
from ._theme import secondary_text_style
from ._ui_helpers import spectral_axis_label
from ..project._ids import now_iso

logger = get_logger(__name__)


def _finite_float_from_text(text: str) -> float | None:
    """Return one finite float from dialog text, or ``None`` when invalid."""
    try:
        value = float(text)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


class NewROIDialog(QDialog):
    """Dialog for naming a new training ROI and optionally assigning one target value."""

    def __init__(
        self,
        f0: int,
        f1: int,
        r0: int,
        r1: int,
        viewer_state: ViewerState,
        selected_target: TargetDefinition | None,
        parent: QWidget | None = None,
        *,
        prediction: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New ROI")
        self._selected_target = selected_target

        n_pixels = (f1 - f0 + 1) * (r1 - r0 + 1)
        layout = QFormLayout(self)

        self._name_edit = QLineEdit()
        layout.addRow("Name:", self._name_edit)

        self._target_value_edit: QLineEdit | None = None
        if selected_target is not None:
            self._target_value_edit = QLineEdit()
            self._target_value_edit.setPlaceholderText("leave blank = unlabelled")
            layout.addRow(f"{selected_target.name}:", self._target_value_edit)
        else:
            layout.addRow("Target:", QLabel("none selected"))

        layout.addRow("Pixels:", QLabel(str(n_pixels)))

        s = viewer_state
        layout.addRow("Signal:", QLabel(signal_label(s.y_unit)))

        filt = s.preprocess_filter
        if filt != "none":
            filt = f"{filt}  w={s.sg_window}  p={s.sg_polyorder}"
        layout.addRow("Smoothing and derivatives:", QLabel(filt))

        norm_parts = []
        if s.preprocess_snv:
            norm_parts.append("SNV")
        if s.preprocess_msc:
            norm_parts.append("MSC")
        if s.preprocess_emsc:
            norm_parts.append("EMSC")
        if s.preprocess_detrend:
            norm_parts.append(f"Detrend(order={s.detrend_order})")
        layout.addRow("Scatter correction:", QLabel(", ".join(norm_parts) or "none"))
        from ._roi_grid import prediction_only_check

        self.prediction_check = prediction_only_check(prediction)
        layout.addRow(self.prediction_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Name cannot be empty.")
            return
        target_text = ""
        if self._target_value_edit is not None:
            target_text = self._target_value_edit.text().strip()
        if target_text:
            if _finite_float_from_text(target_text) is None:
                label = self._selected_target.name if self._selected_target is not None else "Value"
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"'{target_text}' is not a finite numeric value for {label}.",
                )
                return
        self.accept()

    @property
    def roi_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def job(self) -> str:
        """``"prediction"`` when Prediction only is ticked, else ``"calibration"``."""
        return "prediction" if self.prediction_check.isChecked() else "calibration"

    @property
    def target_value(self) -> float | None:
        """Return the optional finite target value entered for the new ROI."""
        if self._target_value_edit is None:
            return None
        text = self._target_value_edit.text().strip()
        if not text:
            return None
        value = _finite_float_from_text(text)
        if value is None:
            log_and_raise("NewROIDialog target_value requires finite numeric text")
        return value


class TargetSelectionDialog(QDialog):
    """Dialog for selecting an active target or defining a new target."""

    def __init__(
        self,
        targets: list[TargetDefinition],
        current_target_id: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select target")
        self._targets_by_id = {target.target_id: target for target in targets}
        name_counts: dict[str, int] = {}
        for target in targets:
            name_counts[target.name] = name_counts.get(target.name, 0) + 1

        layout = QFormLayout(self)
        self._target_combo = QComboBox()
        self._target_combo.setEditable(True)
        self._target_combo.addItem("", None)
        for target in targets:
            label = target.name
            if name_counts[target.name] > 1:
                label = f"{target.name} [{target.target_id}]"
            self._target_combo.addItem(label, target.target_id)
        layout.addRow("Target:", self._target_combo)

        self._unit_edit = QLineEdit()
        self._unit_edit.setPlaceholderText("required for a new target")
        layout.addRow("Unit:", self._unit_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._target_combo.currentTextChanged.connect(self._sync_target_fields)
        current_target = (
            None if current_target_id is None else self._targets_by_id.get(current_target_id)
        )
        if current_target is not None:
            current_index = self._target_combo.findData(current_target.target_id)
            if current_index >= 0:
                self._target_combo.setCurrentIndex(current_index)
        self._sync_target_fields(self._target_combo.currentText())

    def _sync_target_fields(self, _text: str) -> None:
        target = self._selected_target()
        if target is None:
            if not self._unit_edit.isEnabled():
                self._unit_edit.clear()
            self._unit_edit.setEnabled(bool(self._target_combo.currentText().strip()))
            return
        self._unit_edit.setText(target.unit)
        self._unit_edit.setEnabled(False)

    def _on_accept(self) -> None:
        target_name = self.target_name
        if not target_name:
            self.accept()
            return
        if self.selected_target_id is not None:
            self.accept()
            return
        if not self.target_unit:
            QMessageBox.warning(self, "Validation", "Unit is required for a new target.")
            return
        self.accept()

    @property
    def target_name(self) -> str:
        """Return the selected or newly entered target name."""
        target = self._selected_target()
        if target is not None:
            return target.name
        return self._target_combo.currentText().strip()

    @property
    def target_unit(self) -> str:
        """Return the unit entered for a new target, or the selected target unit."""
        return self._unit_edit.text().strip()

    @property
    def selected_target_id(self) -> str | None:
        """Return the selected existing target id, or ``None`` for a new target."""
        current_index = self._target_combo.currentIndex()
        if current_index <= 0:
            return None
        current_text = self._target_combo.currentText().strip()
        if current_text != self._target_combo.itemText(current_index).strip():
            return None
        target_id = self._target_combo.currentData()
        return None if target_id is None else str(target_id)

    def _selected_target(self) -> TargetDefinition | None:
        selected_target_id = self.selected_target_id
        if selected_target_id is None:
            return None
        return self._targets_by_id.get(selected_target_id)


class _SeedEdit(QLineEdit):
    """The pixel-subsample seed, 0 <= seed < 2**63 (fit specification §7).

    A spin box is 32-bit; a stored seed above 2**31 - 1 would be clamped, and a
    refit would draw other pixels.
    """

    def __init__(self) -> None:
        super().__init__("0")
        self.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{1,19}"), self))
        self.setToolTip("Seed of the pixel subsample: an integer from 0 to 2⁶³ − 1")

    def value(self) -> int:
        """The seed; an empty field is 0. A value of 2**63 or more is left to the fit's check."""
        return int(self.text() or 0)

    def setValue(self, value: int) -> None:  # noqa: N802 (QSpinBox-like API)
        self.setText(str(int(value)))


class FitModelRunDialog(QDialog):
    """Everything a fit needs in one place (model diagnostics contract).

    Target, signal unit, preprocessing (starting from the Explore preview,
    but shown and chosen here), model, components and cross-validation. The
    sample set is the set of all ROIs with a value for the target, from all
    data files; ``readiness(target_id)`` describes it before the fit.
    """

    def __init__(
        self,
        targets: list[tuple[str, str]],
        *,
        default_target_id: str | None,
        recipe: PreprocessingRecipe,
        default_signal_unit: str,
        readiness: Callable[[str], tuple[str, bool]],
        smallest_roi: Callable[[str], int | None] = lambda _target_id: None,
        band_source: Callable[[str, str, PreprocessingRecipe], BandSource | None] | None = None,
        band_proposer: Callable[..., ProposalRun | None] | None = None,
        fold_warner: Callable[[str, Any, Any], list[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit model")
        self._readiness = readiness
        self._fold_warner = fold_warner
        self._smallest_roi = smallest_roi
        self._band_source = band_source
        self._band_proposer = band_proposer
        layout = QFormLayout(self)

        self._target_combo = QComboBox()
        for target_id, label in targets:
            self._target_combo.addItem(label, target_id)
        if default_target_id is not None:
            self._target_combo.setCurrentIndex(max(0, self._target_combo.findData(
                default_target_id)))
        self._target_combo.currentIndexChanged.connect(self._update_readiness)
        layout.addRow("Target:", self._target_combo)

        self._signal_unit_combo = QComboBox()
        for label, unit in (("Raw counts (DN)", "DN"), ("Reflectance", "Reflectance"),
                            ("Absorbance", "Absorbance")):
            self._signal_unit_combo.addItem(label, unit)
        current_signal_index = self._signal_unit_combo.findData(default_signal_unit)
        if current_signal_index >= 0:
            self._signal_unit_combo.setCurrentIndex(current_signal_index)
        layout.addRow("Signal unit:", self._signal_unit_combo)

        self._filter_combo = QComboBox()
        for label, value in (("None", "none"), ("SG smooth", "sg_smooth"),
                             ("SG 1st derivative", "sg_d1"), ("SG 2nd derivative", "sg_d2")):
            self._filter_combo.addItem(label, value)
        self._filter_combo.setCurrentIndex(max(0, self._filter_combo.findData(recipe.filter)))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(3, 101)
        self._window_spin.setSingleStep(2)
        self._window_spin.setValue(recipe.sg_window)
        self._poly_spin = QSpinBox()
        self._poly_spin.setRange(1, 10)
        self._poly_spin.setValue(recipe.sg_polyorder)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._filter_combo, 1)
        filter_row.addWidget(QLabel("window"))
        filter_row.addWidget(self._window_spin)
        filter_row.addWidget(QLabel("order"))
        filter_row.addWidget(self._poly_spin)
        layout.addRow("Smoothing and derivatives:", filter_row)
        self._snv = QCheckBox("SNV")
        self._msc = QCheckBox("MSC")
        self._emsc = QCheckBox("EMSC")
        self._detrend = QCheckBox("Detrend")
        self._detrend_order = QSpinBox()
        self._detrend_order.setRange(1, 3)
        self._detrend_order.setValue(recipe.detrend_order)
        for box, value in ((self._snv, recipe.snv), (self._msc, recipe.msc),
                           (self._emsc, recipe.emsc), (self._detrend, recipe.detrend)):
            box.setChecked(value)
        scatter_row = QHBoxLayout()
        for widget in (self._snv, self._msc, self._emsc, self._detrend, self._detrend_order):
            scatter_row.addWidget(widget)
        layout.addRow("Scatter correction:", scatter_row)
        # AsLS and the Savitzky-Golay edge handling have no controls here (scripts
        # only); a preset recipe keeps them (viewer contract, Fit again).
        self._kept_recipe = recipe
        self._asls_label = QLabel("")
        self._asls_label.setStyleSheet(secondary_text_style())
        layout.addRow("", self._asls_label)
        self._show_kept_recipe()
        self._filter_combo.currentIndexChanged.connect(self._update_enabled)
        self._detrend.toggled.connect(self._update_enabled)
        self._update_enabled()
        layout.addRow("Bands:", self._bands_row(recipe))
        hint = QLabel("Starts from the Explore preview; the model is fitted with this.")
        hint.setStyleSheet(secondary_text_style())
        layout.addRow("", hint)

        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(["PLS", "PCR"])
        layout.addRow("Model:", self._model_type_combo)

        self._components_spin = QSpinBox()
        self._components_spin.setRange(1, 128)
        self._components_spin.setValue(6)
        self._auto_check = QCheckBox("Auto")
        self._auto_check.setToolTip(
            "Choose the number from the component curve (RMSECV (per ROI) and Q² against the "
             "number "
            "of components) by the rule; the curve is shown on the Components page.")
        self._rule_combo = QComboBox()
        for label, value, tip in (
                ("one SE (recommended)", "one_se",
                 "The fewest components whose RMSECV (per ROI) is within one standard "
                 "error of the "
                 "lowest."),
                ("lowest RMSECV (per ROI)", "min_rmsecv", "The number with the lowest RMSECV "
                 "(per ROI); tends to "
                 "choose too many."),
                ("Q² limit", "q2_limit", "SIMCA's rule R1: add components while each one's Q² "
                 "exceeds 0.05; stops at a scatter component that predicts nothing alone."),
                ("F-test", "f_test", "Haaland and Thomas: the fewest components whose PRESS "
                 "is not significantly larger than the lowest (F test over the samples)."),
                ("Q² with look-ahead", "q2_lookahead", "As Q² limit, but a component that "
                 "fails is kept when the next one passes.")):
            self._rule_combo.addItem(label, value)
            self._rule_combo.setItemData(self._rule_combo.count() - 1, tip,
                                         Qt.ItemDataRole.ToolTipRole)
        self._guard_check = QCheckBox("drop noise")
        self._guard_check.setChecked(True)
        self._guard_check.setToolTip(
            "Noise guard: at most one component more than those explaining at least 0.5 % of "
            "the spectra, and no trailing component that explains under 1 % of the spectra "
            "and does not improve the prediction (Q²ₐ ≤ 0.05).")
        self._max_spin = QSpinBox()
        self._max_spin.setRange(0, 128)
        self._max_spin.setSpecialValueText("default")
        self._max_spin.setPrefix("max ")
        components_row = QHBoxLayout()
        for widget in (self._components_spin, self._auto_check, self._rule_combo,
                       self._guard_check, self._max_spin):
            components_row.addWidget(widget)
        layout.addRow("Components:", components_row)
        self._auto_check.toggled.connect(self._update_components)
        self._update_components()

        self._cv_combo = QComboBox()
        self._cv_combo.addItem("Auto (7 round-robin folds, or leave one out)", "auto")
        self._cv_combo.addItem("Leave one out", "logo")
        self._cv_combo.addItem("7 folds", 7)
        layout.addRow("Validation:", self._cv_combo)

        self._group_by_combo = QComboBox()
        self._group_by_combo.addItem("Each ROI", "roi")
        self._group_by_combo.addItem("ROIs with the same Sample label", "sample")
        self._group_by_combo.addItem("ROIs with the same target value", "target")
        self._group_by_combo.setToolTip(
            "Which ROIs a validation fold holds out together. Choose the Sample label when "
            "several ROIs come from one physical sample (set it in the ROI table), so that "
            "no sample is on both sides of a fold.")
        layout.addRow("Hold out together:", self._group_by_combo)
        layout.addRow(self._fitting_group())
        # The fold warnings follow the grouping and the folds (contract §7.4).
        for combo in (self._group_by_combo, self._cv_combo):
            combo.currentIndexChanged.connect(self._update_readiness)

        self._readiness_label = QLabel("")
        self._readiness_label.setWordWrap(True)
        layout.addRow("Calibration set:", self._readiness_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addRow(self._buttons)
        self._update_readiness()

    def _bands_row(self, recipe: PreprocessingRecipe) -> QWidget:
        """Keep and exclude ranges (project-schema contract §4; viewer contract)."""
        unit = "nm"
        self._keep_edit = QLineEdit(format_ranges(recipe.bands_keep))
        self._keep_edit.setPlaceholderText("all bands")
        self._keep_edit.setToolTip(f"Ranges to keep, e.g. 1100-1350, 1500-2300 ({unit}; band "
                                   "numbers for data without wavelengths). Empty: every band.")
        self._exclude_edit = QLineEdit(format_ranges(recipe.bands_exclude))
        self._exclude_edit.setPlaceholderText("none")
        self._exclude_edit.setToolTip("Ranges to leave out, e.g. 1400-1500. Applied after "
                                      "every other step, so normalization sees all bands.")
        water = QToolButton()
        water.setText("Water bands")
        water.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(water)
        for low, high in WATER_BANDS:
            action = menu.addAction(f"Exclude {low:.0f}-{high:.0f} nm")
            if action is not None:
                action.triggered.connect(
                    lambda _checked=False, r=(low, high): self._add_exclude(r))
        water.setMenu(menu)
        water.setToolTip("Common NIR water bands; added to Exclude only when chosen.")
        self._select_bands = QPushButton("Select on Plot...")
        self._select_bands.clicked.connect(self._select_bands_on_plot)
        self._select_bands.setEnabled(self._band_source is not None)
        self._propose_bands = QPushButton("Propose...")
        self._propose_bands.setToolTip(
            "Propose keep ranges by forward interval PLS with nested validation, using this "
            "dialog's target, preprocessing, model and fitting choices.")
        self._propose_bands.clicked.connect(self._propose_bands_clicked)
        self._propose_bands.setEnabled(self._band_proposer is not None)
        row = QWidget()
        grid = QGridLayout(row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Keep"), 0, 0)
        grid.addWidget(self._keep_edit, 0, 1)
        grid.addWidget(QLabel("Exclude"), 1, 0)
        grid.addWidget(self._exclude_edit, 1, 1)
        grid.addWidget(water, 0, 2)
        grid.addWidget(self._select_bands, 1, 2)
        grid.addWidget(self._propose_bands, 0, 3)
        return row

    def _propose_bands_clicked(self) -> None:
        if self._band_proposer is None:
            return
        try:
            base = replace(self.recipe, bands_keep=(), bands_exclude=())
            run = self._band_proposer(self.target_id, base, self.model_type,
                                      self.hyperparameters, self.fit_spec)
        except ValueError as exc:
            QMessageBox.warning(self, "Propose bands", str(exc))
            return
        if run is None:
            QMessageBox.information(self, "Propose bands", "Open a project first.")
            return
        dialog = ProposeBandsDialog(run, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.proposal is not None:
            self._keep_edit.setText(format_ranges(dialog.proposal.ranges))
            self._exclude_edit.setText("")
            if dialog.proposal.n_components:
                self._auto_check.setChecked(False)
                self._components_spin.setValue(dialog.proposal.n_components)

    def _add_exclude(self, bounds: tuple[float, float]) -> None:
        current = parse_ranges(self._exclude_edit.text(), "Exclude")
        self._exclude_edit.setText(format_ranges(current + (bounds,)))

    def _select_bands_on_plot(self) -> None:
        if self._band_source is None:
            return
        try:
            keep = parse_ranges(self._keep_edit.text(), "Keep")
            exclude = parse_ranges(self._exclude_edit.text(), "Exclude")
            base = replace(self.recipe, bands_keep=(), bands_exclude=())
        except ValueError as exc:
            QMessageBox.warning(self, "Bands", str(exc))
            return
        source = self._band_source(self.target_id, self.signal_unit, base)
        if source is None:
            QMessageBox.information(self, "Bands", "No ROI has a value for this target.")
            return
        dialog = BandRangeDialog(source, keep=keep, exclude=exclude, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._keep_edit.setText(format_ranges(dialog.ranges("keep")))
            self._exclude_edit.setText(format_ranges(dialog.ranges("exclude")))

    def _update_components(self, *_args: Any) -> None:
        auto = self._auto_check.isChecked()
        self._components_spin.setEnabled(not auto)
        self._rule_combo.setEnabled(auto)
        self._guard_check.setEnabled(auto)
        self._max_spin.setEnabled(auto)

    def apply_model_run(self, model_run: ModelRun, *, signal_unit: str | None = None) -> None:
        """Preset every choice from a stored model run (for "Fit again...").

        ``signal_unit`` is the run's input contract's signal unit: the fit
        repeats it, whatever Explore shows (viewer-workspace-ui contract).
        """
        self._model_type_combo.setCurrentText(model_run.model_type)
        if signal_unit is not None:
            index = self._signal_unit_combo.findData(signal_unit)
            if index >= 0:
                self._signal_unit_combo.setCurrentIndex(index)
        hyper = model_run.hyperparameters
        self._components_spin.setValue(int(hyper.get("n_components", 6)))
        self._auto_check.setChecked(False)
        for combo, key in ((self._x_scaling_combo, "x_scaling"),
                           (self._y_scaling_combo, "y_scaling")):
            if key in hyper and combo.findData(hyper[key]) >= 0:
                combo.setCurrentIndex(combo.findData(hyper[key]))
        spec = model_run.fit_specification
        cv: int | str = ("logo" if spec.fold_rule == "leave_one_group_out"
                         else "auto" if spec.n_folds == 7 else spec.n_folds)
        if self._cv_combo.findData(cv) < 0:
            self._cv_combo.addItem(f"{cv} folds", cv)
        self._cv_combo.setCurrentIndex(self._cv_combo.findData(cv))
        self._group_by_combo.setCurrentIndex(max(0, self._group_by_combo.findData(spec.group_by)))
        self._draw_check.setChecked(spec.pixels_per_roi is not None)
        if spec.pixels_per_roi is not None:
            self._pixels_spin.setValue(spec.pixels_per_roi)
            self._seed_spin.setValue(spec.seed)
        self._exclude_check.setChecked(spec.exclusion == "dmodx")
        self._level_spin.setValue(spec.exclusion_level)
        self._rounds_spin.setValue(spec.exclusion_rounds)
        self._share_spin.setValue(round(spec.exclusion_max_fraction * 100))
        self._formula_combo.setCurrentIndex(max(0, self._formula_combo.findData(
            spec.exclusion_formula)))
        self._update_fitting()

    def _fitting_group(self) -> QGroupBox:
        """Pixel subsample, scaling and DModX exclusion (sample-set-model-run contract §7)."""
        group = QGroupBox("Fitting")
        form = QFormLayout(group)

        self._draw_check = QCheckBox("Same number from every ROI")
        self._draw_check.setToolTip("Draw this many pixels from each ROI, so every ROI has "
                                    "the same weight; at most the smallest ROI's pixel count.")
        self._pixels_spin = QSpinBox()
        self._pixels_spin.setRange(1, 1_000_000)
        self._pixels_spin.setValue(50)
        self._seed_spin = _SeedEdit()
        self._smallest_label = QLabel("")
        self._smallest_label.setStyleSheet(secondary_text_style())
        pixels_row = QHBoxLayout()
        for widget in (self._draw_check, self._pixels_spin, QLabel("seed"), self._seed_spin,
                       self._smallest_label):
            pixels_row.addWidget(widget)
        form.addRow("Pixels per ROI:", pixels_row)

        self._x_scaling_combo = QComboBox()
        self._y_scaling_combo = QComboBox()
        for combo in (self._x_scaling_combo, self._y_scaling_combo):
            for label, value in (("Center", "center"), ("Unit variance", "uv"),
                                 ("Pareto", "pareto")):
                combo.addItem(label, value)
        self._y_scaling_combo.setCurrentIndex(self._y_scaling_combo.findData("uv"))
        scaling_row = QHBoxLayout()
        for widget in (QLabel("X"), self._x_scaling_combo, QLabel("Y"), self._y_scaling_combo):
            scaling_row.addWidget(widget)
        scaling_row.addStretch(1)
        form.addRow("Scaling:", scaling_row)

        self._exclude_check = QCheckBox("Exclude pixels above DCrit (DModX)")
        self._exclude_check.setToolTip(
            "Leave out pixels above the critical distance to the model, inside every "
            "validation fold from its calibration pixels only; held-out pixels are never "
            "excluded and ROIs are never left out.")
        self._level_spin = QDoubleSpinBox()
        self._level_spin.setRange(0.9, 0.999)
        self._level_spin.setDecimals(3)
        self._level_spin.setSingleStep(0.005)
        self._level_spin.setValue(0.99)
        self._rounds_spin = QSpinBox()
        self._rounds_spin.setRange(1, 2)
        self._share_spin = QSpinBox()
        self._share_spin.setRange(1, 50)
        self._share_spin.setValue(20)
        self._share_spin.setSuffix(" %")
        self._formula_combo = QComboBox()
        self._formula_combo.addItem("Unweighted", "dmodx-1")
        self._formula_combo.addItem("Weighted by modeling power (SIMCA)", "dmodx-2")
        form.addRow("", self._exclude_check)
        exclusion_row = QHBoxLayout()
        for widget in (QLabel("level"), self._level_spin, QLabel("rounds"), self._rounds_spin,
                       QLabel("at most"), self._share_spin, QLabel("per ROI")):
            exclusion_row.addWidget(widget)
        form.addRow("DModX:", exclusion_row)
        form.addRow("", self._formula_combo)

        for signal in (self._draw_check.toggled, self._exclude_check.toggled,
                       self._model_type_combo.currentIndexChanged,
                       self._target_combo.currentIndexChanged):
            signal.connect(self._update_fitting)
        self._update_fitting()
        return group

    def _update_fitting(self, *_args: Any) -> None:
        target_id = self._target_combo.currentData()
        smallest = None if target_id is None else self._smallest_roi(str(target_id))
        if smallest is not None:
            self._pixels_spin.setMaximum(max(1, smallest))
            self._smallest_label.setText(f"(smallest ROI: {smallest})")
        else:
            self._smallest_label.setText("")
        for widget in (self._pixels_spin, self._seed_spin):
            widget.setEnabled(self._draw_check.isChecked())
        self._y_scaling_combo.setEnabled(self._model_type_combo.currentText() == "PLS")
        for widget in (self._level_spin, self._rounds_spin, self._share_spin,
                       self._formula_combo):
            widget.setEnabled(self._exclude_check.isChecked())

    def _update_enabled(self, *_args: Any) -> None:
        smoothing = self._filter_combo.currentData() != "none"
        self._window_spin.setEnabled(smoothing)
        self._poly_spin.setEnabled(smoothing)
        self._detrend_order.setEnabled(self._detrend.isChecked())

    def _update_readiness(self, *_args: Any) -> None:
        target_id = self._target_combo.currentData()
        text, ready = ("Define a target first.", False) if target_id is None \
            else self._readiness(str(target_id))
        if target_id is not None and self._fold_warner is not None \
                and hasattr(self, "_group_by_combo"):
            try:
                spec, cv = self.fit_spec, self.cv
            except ValueError:
                spec, cv = None, None
            warnings = self._fold_warner(str(target_id), spec, None if spec else cv)
            if warnings:
                text += "\n⚠ Validation likely optimistic: " + "; ".join(warnings) + "."
        self._readiness_label.setText(text)
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(ready)

    @property
    def readiness_text(self) -> str:
        return self._readiness_label.text()

    def _on_accept(self) -> None:
        try:
            self.recipe  # noqa: B018 (validates the preprocessing)
        except ValueError as exc:
            QMessageBox.warning(self, "Preprocessing", str(exc))
            return
        self.accept()

    @property
    def target_id(self) -> str:
        """Return the selected target id."""
        return str(self._target_combo.currentData())

    @property
    def recipe(self) -> PreprocessingRecipe:
        """The chosen preprocessing (raises ``ValueError`` if it is not valid)."""
        kept = self._kept_recipe
        return PreprocessingRecipe(
            filter=str(self._filter_combo.currentData()), sg_window=self._window_spin.value(),
            sg_polyorder=self._poly_spin.value(), snv=self._snv.isChecked(),
            msc=self._msc.isChecked(), emsc=self._emsc.isChecked(),
            detrend=self._detrend.isChecked(), detrend_order=self._detrend_order.value(),
            sg_edges=kept.sg_edges, asls=kept.asls, asls_lambda=kept.asls_lambda,
            asls_p=kept.asls_p, asls_iterations=kept.asls_iterations,
            bands_keep=parse_ranges(self._keep_edit.text(), "Keep"),
            bands_exclude=parse_ranges(self._exclude_edit.text(), "Exclude"))

    def _show_kept_recipe(self) -> None:
        kept = self._kept_recipe
        notes = []
        if kept.asls:
            notes.append(f"AsLS baseline correction first (λ={kept.asls_lambda:g}, "
                         f"p={kept.asls_p:g}), kept from the recipe")
        if kept.filter != "none" and kept.sg_edges == "interp":
            notes.append("Savitzky-Golay edges kept (extrapolated), as in the recipe")
        self._asls_label.setText("; ".join(notes))
        self._asls_label.setVisible(bool(notes))

    def set_recipe(self, recipe: PreprocessingRecipe) -> None:
        """Choose a preprocessing (for scripts and tests)."""
        self._filter_combo.setCurrentIndex(max(0, self._filter_combo.findData(recipe.filter)))
        self._window_spin.setValue(recipe.sg_window)
        self._poly_spin.setValue(recipe.sg_polyorder)
        for box, value in ((self._snv, recipe.snv), (self._msc, recipe.msc),
                           (self._emsc, recipe.emsc), (self._detrend, recipe.detrend)):
            box.setChecked(value)
        self._detrend_order.setValue(recipe.detrend_order)
        self._keep_edit.setText(format_ranges(recipe.bands_keep))
        self._exclude_edit.setText(format_ranges(recipe.bands_exclude))
        self._kept_recipe = recipe
        self._show_kept_recipe()

    @property
    def model_type(self) -> str:
        """Return the selected model family."""
        return self._model_type_combo.currentText()

    @property
    def n_components(self) -> int | str:
        """The number of components, or ``"auto"`` to choose it from the component curve."""
        return "auto" if self._auto_check.isChecked() else int(self._components_spin.value())

    @property
    def signal_unit(self) -> str:
        """Return the signal unit requested for fitting."""
        return str(self._signal_unit_combo.currentData())

    @property
    def cv(self) -> int | str:
        """Return the selected grouped-validation policy."""
        return self._cv_combo.currentData()

    @property
    def group_by(self) -> str:
        """Return which ROIs are held out together: ``"roi"``, ``"sample"`` or ``"target"``."""
        return str(self._group_by_combo.currentData())

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Components and the X (and, for PLS, Y) scaling of the model."""
        values: dict[str, Any] = {"n_components": self.n_components,
                                  "x_scaling": str(self._x_scaling_combo.currentData())}
        if self.model_type == "PLS":
            values["y_scaling"] = str(self._y_scaling_combo.currentData())
        return values

    @property
    def fit_spec(self) -> FitSpec | None:
        """The fit specification of the choices, or ``None`` for the defaults (then ``cv``)."""
        draw = self._draw_check.isChecked()
        exclude = self._exclude_check.isChecked()
        auto = self._auto_check.isChecked()
        if self.group_by == "roi" and not draw and not exclude and not auto:
            return None
        return replace(
            FitSpec.from_cv(self.cv),
            components_rule=str(self._rule_combo.currentData()) if auto else "fixed",
            components_max=(self._max_spin.value() or None) if auto else None,
            components_noise_guard=("on" if self._guard_check.isChecked() else "off")
            if auto else "default",
            group_by=self.group_by,
            pixels_per_roi=self._pixels_spin.value() if draw else None,
            seed=self._seed_spin.value() if draw else 0,
            exclusion="dmodx" if exclude else "none",
            exclusion_level=round(self._level_spin.value(), 3),
            exclusion_rounds=self._rounds_spin.value(),
            exclusion_max_fraction=self._share_spin.value() / 100.0,
            exclusion_formula=str(self._formula_combo.currentData()),
        )


def _component_payload(model_run: ModelRun, artifact: dict[str, Any]) -> dict[str, Any]:
    """Curve, loadings of the first components, CoeffCS and training distances (§7.4, §7.5).

    Presentation only: every value comes from the stored run.
    """
    payload: dict[str, Any] = {"n_components": model_run.hyperparameters.get("n_components"),
                               # LV for PLS, PC for PCR (terminology contract, LV, PC)
                               "component_prefix": "PC" if model_run.model_type == "PCR"
                               else "LV",
                               "fit_statistics": {key: model_run.metrics[key] for key in
                                                  ("rmsee", "r2_adj", "pixel_rmsee")
                                                  if key in model_run.metrics}}
    curve = artifact.get("component_curve")
    if isinstance(curve, dict) and "rmsecv" in curve:
        payload["component_curve"] = curve
        spec = model_run.fit_specification
        selected = model_run.hyperparameters.get("n_components")
        payload["components_choice"] = {
            "rule": spec.components_rule, "noise_guard": spec.components_noise_guard,
            "rule_choice": spec.components_rule_choice or selected,
            "n_groups": model_run.cv_spec.get("n_groups")}
    loadings = artifact.get("x_loadings_", artifact.get("loadings_"))
    if loadings is not None:
        matrix = np.asarray(loadings, dtype=np.float32)
        if matrix.ndim == 2:
            payload["loadings_matrix"] = matrix[:, :3]
    coef_cs = artifact.get("coef_cs")
    if coef_cs is not None:
        values = np.asarray(coef_cs, dtype=np.float32).squeeze()
        if values.ndim == 1:
            payload["coef_cs"] = values
    for key in ("selectivity_ratio", "mean_spectrum"):
        values = artifact.get(key)
        if values is not None and np.ndim(np.squeeze(values)) == 1:
            payload[key] = np.asarray(values, dtype=np.float32).reshape(-1)
    vip = artifact.get("vip")
    if vip is not None and np.ndim(np.squeeze(vip)) == 1:
        payload["vip"] = np.asarray(vip, dtype=np.float32).reshape(-1)
    scores = artifact.get("x_scores_", artifact.get("scores_"))
    if scores is not None:
        matrix = np.asarray(scores, dtype=np.float64)
        if matrix.ndim == 2 and matrix.shape[0] > 1 and matrix.shape[1] >= 2:
            payload["score_variances"] = matrix[:, :2].var(axis=0, ddof=1)
    for key in ("t2", "dmodx"):
        if artifact.get(key) is not None:
            payload[f"{key}_pixels"] = np.asarray(artifact[key], dtype=np.float32)
    for key in ("t2_crit_099", "t2_crit_2_095", "t2_crit_2_099", "dcrit_099"):
        if artifact.get(key) is not None:
            payload[key] = float(np.asarray(artifact[key]))
    for key in ("roi_share_above_dcrit", "roi_share_above_t2crit"):  # per ROI, as stored
        if artifact.get(key) is not None:
            payload[key] = np.asarray(artifact[key], dtype=np.float32).reshape(-1)
    return payload


def _roi_score_payload(artifact: dict[str, Any] | None) -> dict[str, Any]:
    """The ROI scores the fit stored (model-workspace-diagnostics contract §2).

    The viewer never rebuilds them from pixel counts: a run without stored
    ROI scores shows them as not available.
    """
    scores = None if artifact is None else artifact.get("roi_scores")
    if scores is None:
        return {"scores_message": "ROI scores are not available for this model."}
    matrix = np.asarray(scores, dtype=np.float32)
    if matrix.ndim != 2:
        return {"scores_message": "ROI scores of this model cannot be read (not shown)."}
    return {"scores_roi": matrix,
            "scores_without_pixels": int(np.sum(~np.isfinite(matrix).all(axis=1)))}


#: Readable names of the stored cross-validation splitters.
_CV_KINDS = {"RoundRobinGroups": "round-robin", "GroupKFold": "group k-fold",
             "KFold": "k-fold", "LeaveOneGroupOut": "leave one out",
             "LeaveOneOut": "leave one out"}


def describe_cv(cv_spec: Mapping[str, Any]) -> str:
    """How a model was cross-validated, e.g. ``"7 folds, round-robin over 20 samples"``.

    Reads the stored splitter (``type``, ``n_splits``, ``n_groups``,
    ``grouping_unit``) and the older one-key form (``{"cv": 7}``); ``"-"``
    when nothing is recorded.
    """
    if not cv_spec:
        return "-"
    if "type" not in cv_spec:
        value = cv_spec.get("cv", next(iter(cv_spec.values())) if len(cv_spec) == 1 else None)
        return "-" if value is None else str(value)
    kind = _CV_KINDS.get(str(cv_spec["type"]), str(cv_spec["type"]))
    n_splits = cv_spec.get("n_splits")
    text = kind if n_splits is None else f"{n_splits} folds, {kind}"
    n_groups = cv_spec.get("n_groups")
    if n_groups is not None:
        # Fold groups are samples: ROIs with one sample label, or an ROI without one.
        text += f" over {n_groups} sample{'' if int(n_groups) == 1 else 's'}"
    return text


def model_label(model_run: ModelRun) -> str:
    """The model's name (e.g. ``"PLS 7 LV, SNV"``), else its type; never its id."""
    return model_run.name or f"{model_run.model_type} model"


class ModelingView(QWidget):
    """Modeling controls for target editing, sample-set builds, and model fitting."""

    view_state_changed = pyqtSignal()
    #: Show the ROI table (target values of all ROIs).
    roi_table_requested = pyqtSignal()

    def __init__(
        self,
        viewer_state: ViewerState,
        project_state: ProjectState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = viewer_state
        self.project_state = project_state
        self._selected_target_id: str | None = None
        self._selected_model_run_id: str | None = None
        self._last_sample_set_id: str | None = None
        self._job_thread: QThread | None = None
        self._job_worker: BackgroundWorker | None = None
        # Hold the relay strongly so Qt callbacks remain connected for the full job lifetime.
        self._job_relay: object | None = None
        self._cached_diagnostics_model_key: tuple[str, str] | None = None
        self._cached_diagnostics_model_artifact: dict[str, Any] | None = None
        # Model runs are immutable, so their CV outputs are read once per run.
        self._cached_cv_outputs_key: tuple[str, str] | None = None
        self._cached_cv_outputs: dict[str, Any] | None = None
        self._cached_workspace_diagnostics_token: tuple[Any, ...] | None = None
        self._cached_workspace_diagnostics_payload: dict[str, Any] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_group())

        self.project_state.project_loaded.connect(self._invalidate_diagnostics_cache)
        self.project_state.project_loaded.connect(self._sync_selected_target)
        self.project_state.datasets_changed.connect(self._update_readiness_status)
        self.project_state.calibrations_changed.connect(self._update_readiness_status)
        self.project_state.targets_changed.connect(self._sync_selected_target)
        self.project_state.rois_changed.connect(self._invalidate_diagnostics_cache)
        self.project_state.rois_changed.connect(self._update_readiness_status)
        self.project_state.target_assignments_changed.connect(self._update_readiness_status)
        self.project_state.sample_sets_changed.connect(self._invalidate_diagnostics_cache)
        self.project_state.sample_sets_changed.connect(self._update_model_status)
        self.project_state.model_runs_changed.connect(self._invalidate_diagnostics_cache)
        self.project_state.model_runs_changed.connect(self._update_model_status)
        self.project_state.active_model_changed.connect(self._on_active_model_changed)
        self.state.units_changed.connect(self._update_readiness_status)
        self._sync_selected_target()

    def _build_group(self) -> QGroupBox:
        box = QGroupBox("Modeling")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        action_panel = QWidget(box)
        action_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        action_grid = QGridLayout(action_panel)
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(8)
        action_grid.setVerticalSpacing(4)

        def configure_status(label: QLabel) -> None:
            label.setStyleSheet(secondary_text_style())
            label.setWordWrap(True)
            label.setMaximumWidth(132)

        def configure_action(button: QToolButton) -> None:
            button.setMinimumWidth(132)
            button.setMaximumWidth(132)

        self.target_status_label = QLabel("Target: none")
        configure_status(self.target_status_label)
        action_grid.addWidget(self.target_status_label, 0, 0)

        self._sample_set_status = QLabel("Calibration set: none")
        configure_status(self._sample_set_status)
        action_grid.addWidget(self._sample_set_status, 0, 1)

        self._readiness_status = QLabel("Ready: define or select a target.")
        configure_status(self._readiness_status)
        action_grid.addWidget(self._readiness_status, 2, 0)

        self._model_status = QLabel("Model: none")
        configure_status(self._model_status)
        action_grid.addWidget(self._model_status, 2, 1)

        self._btn_target = QToolButton()
        self._btn_target.setText("Target...")
        configure_action(self._btn_target)
        self._btn_target.clicked.connect(self.select_target_btn_clicked)
        action_grid.addWidget(self._btn_target, 1, 0)

        self._btn_assign_target = QToolButton()
        self._btn_assign_target.setText("Target Values...")
        self._btn_assign_target.setToolTip("Type the target values of all ROIs in the ROI table")
        configure_action(self._btn_assign_target)
        self._btn_assign_target.clicked.connect(self.target_values_btn_clicked)
        action_grid.addWidget(self._btn_assign_target, 1, 1)

        self._btn_fit_model = QToolButton()
        self._btn_fit_model.setText("Fit Model")
        configure_action(self._btn_fit_model)
        self._btn_fit_model.clicked.connect(lambda: self.fit_model_btn_clicked())
        action_grid.addWidget(self._btn_fit_model, 3, 1)

        self._btn_fit_again = QToolButton()
        self._btn_fit_again.setText("Fit Again...")
        self._btn_fit_again.setToolTip("Open the fit dialog with the selected model's settings, "
                                       "e.g. to fit with another number of components")
        configure_action(self._btn_fit_again)
        self._btn_fit_again.clicked.connect(self.fit_again_btn_clicked)
        action_grid.addWidget(self._btn_fit_again, 3, 0)

        self._service_status = BusyStatusWidget(idle_text="Service: idle", parent=self)
        self._service_status.setMaximumWidth(272)
        self._service_status.label.setWordWrap(True)
        action_grid.addWidget(self._service_status, 4, 0, 1, 2)

        layout.addWidget(action_panel)
        layout.addStretch(1)

        return box

    def visible_targets(self) -> list[TargetDefinition]:
        return sorted(
            self.project_state.targets.values(),
            key=lambda target: target.name,
        )

    def current_target(self) -> TargetDefinition | None:
        if self._selected_target_id is None:
            return None
        target = self.project_state.targets.get(self._selected_target_id)
        if target is None:
            return None
        return target

    @property
    def selected_target_id(self) -> str | None:
        return self._selected_target_id

    @selected_target_id.setter
    def selected_target_id(self, value: str | None) -> None:
        self._selected_target_id = None if value is None else str(value)

    def select_target_btn_clicked(self) -> None:
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self,
                "No project open",
                "No project open. Use Project > New or Project > Open first.",
            )
            return
        dlg = TargetSelectionDialog(self.visible_targets(), self._selected_target_id, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not dlg.target_name:
            self._selected_target_id = None
            self._update_target_status()
            self._update_model_status()
            return
        if dlg.selected_target_id is not None:
            self._selected_target_id = dlg.selected_target_id
            self._update_target_status()
            self._update_model_status()
            return

        target = TargetDefinition(
            target_id=str(uuid.uuid4()),
            name=dlg.target_name,
            unit=dlg.target_unit,
            value_type="float",
            missing_value_policy="allowed",
            created_at=now_iso(),
        )
        self.project_state.define_target(target)
        self._selected_target_id = target.target_id
        self._update_target_status()
        self._update_model_status()

    def target_values_btn_clicked(self) -> None:
        """Open the ROI table on the selected target (ROI/target contract)."""
        if self.project_state.project_path is None:
            QMessageBox.warning(
                self,
                "No project open",
                "No project open. Use Project > New or Project > Open first.",
            )
            return
        self.roi_table_requested.emit()

    def fit_again_btn_clicked(self) -> None:
        """Open the fit dialog with the selected model's settings (e.g. other components)."""
        model_run = self._resolved_workspace_model_run()
        if model_run is None:
            QMessageBox.information(self, "No model", "Select a fitted model first.")
            return
        self.fit_model_btn_clicked(preset=model_run)

    def _show_bands_in_explore(self, recipe: PreprocessingRecipe) -> None:
        """The Explore preview shows the fit's band selection (unselected bands blank)."""
        state = self.state
        if (state.bands_keep, state.bands_exclude) == (recipe.bands_keep, recipe.bands_exclude):
            return
        state.bands_keep, state.bands_exclude = recipe.bands_keep, recipe.bands_exclude
        state._image_cache = None
        state.units_changed.emit()

    def _band_proposer(self, target_id: str, recipe: PreprocessingRecipe, model_type: str,
                       hyperparameters: dict[str, Any], fit_spec: FitSpec | None
                       ) -> ProposalRun | None:
        """A band proposal through the facade (headless contract, Band proposal)."""
        project = self.project_state.project
        target = self.project_state.targets.get(target_id)
        if project is None or target is None:
            return None

        def run(progress: Callable[[int, int], None], cancel: Callable[[], bool]) -> Any:
            return project.propose_bands(target, recipe=recipe, model=model_type,
                                         fit_spec=fit_spec, progress=progress, cancel=cancel,
                                         **hyperparameters)

        return run

    def _band_source(self, target_id: str, signal_unit: str,
                     recipe: PreprocessingRecipe) -> BandSource | None:
        """Mean spectrum of the target's ROIs and the selected model's VIP or coefficients."""
        state = self.project_state
        rois = [roi for roi_id, roi in state.rois.items()
                if (assignment := state.target_assignments.get((roi_id, target_id))) is not None
                and assignment.value is not None]
        path = state.project_path
        if not rois or path is None:
            return None
        datasets, calibrations = dict(state.datasets), dict(state.calibrations)
        from ..modeling import roi_spectrum_summaries

        def load() -> tuple[np.ndarray, np.ndarray]:
            summaries, skipped = roi_spectrum_summaries(
                str(path), rois, datasets=datasets, calibrations=calibrations,
                signal_unit=signal_unit, recipe=recipe)
            if not summaries:
                log_and_raise("No ROI gives {} spectra: {}", signal_unit,
                              "; ".join(sorted(set(skipped.values()))) or "no spectra")
            means = np.stack([np.asarray(s.mean, dtype=np.float64)
                              for s in summaries.values()])
            return recipe.output_wavelengths(np.asarray(rois[0].wavelengths)), means.mean(axis=0)

        guide = None
        payload = self.workspace_diagnostics_payload() or {}
        axis = payload.get("spectral_axis")
        for key, label in (("vip", "VIP"), ("coefficients", "Coefficient")):
            values = payload.get(key)
            if values is not None and axis is not None and np.size(values) == np.size(axis):
                guide = (np.asarray(axis, dtype=float), np.asarray(values, dtype=float), label)
                break
        return BandSource(load=load, guide=guide,
                          axis_label=spectral_axis_label(rois[0].wavelengths)[0])

    def fit_model_btn_clicked(self, preset: ModelRun | None = None) -> None:
        if self._job_thread is not None:
            return
        state = self.project_state
        if state.project_path is None:
            QMessageBox.warning(self, "No project open", "Open a project before fitting a model.")
            return
        if not state.targets:
            QMessageBox.information(self, "No target", "Define a target and its values first.")
            return
        current = self.current_target()
        preset_sample_set = (None if preset is None
                             else state.sample_sets.get(preset.sample_set_id))
        preset_recipe = (None if preset_sample_set is None else state.recipes.get(
            preset_sample_set.preprocessing_recipe_id))
        dlg = FitModelRunDialog(
            [(target_id, target.name if not target.unit else f"{target.name} ({target.unit})")
             for target_id, target in sorted(state.targets.items(),
                                             key=lambda item: item[1].name.lower())],
            default_target_id=(preset_sample_set.target_id if preset_sample_set is not None
                               else None if current is None else current.target_id),
            recipe=(PreprocessingRecipe.from_dict(preset_recipe) if preset_recipe is not None
                    else self.state.recipe),
            default_signal_unit=self.state.y_unit,
            readiness=self.fit_readiness,
            smallest_roi=self._smallest_labelled_roi,
            band_source=self._band_source,
            band_proposer=self._band_proposer,
            fold_warner=self._dialog_fold_warnings,
            parent=self,
        )
        if preset is not None:
            domain = state.signal_domains.get(preset.signal_domain_id)
            dlg.apply_model_run(
                preset, signal_unit=None if domain is None else domain.required_signal_unit)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._selected_target_id = dlg.target_id
        self._show_bands_in_explore(dlg.recipe)
        planned = self._plan_fit_sample_set(dlg.target_id, dlg.recipe)
        if planned is None:
            return
        sample_set, is_new, recipe = planned
        # Like Project.fit: check, fit, and store the recipe and sample set only with the model.
        preflight_error = self._validate_fit_preflight(
            sample_set=sample_set,
            cv=dlg.cv,
            required_signal_unit=dlg.signal_unit,
        )
        if preflight_error is not None:
            QMessageBox.warning(self, "Fit model failed", preflight_error)
            return
        signal_domain = resolve_signal_domain(self._build_signal_domain_contract(
            sample_set=sample_set,
            required_signal_unit=dlg.signal_unit,
        ), self.project_state.signal_domains)
        project_path = self._require_project_path()
        target = self.project_state.targets.get(sample_set.target_id)
        target_name = None if target is None else target.name
        resolved_signal_domains = dict(self.project_state.signal_domains)
        resolved_signal_domains[signal_domain.signal_domain_id] = signal_domain
        datasets = dict(self.project_state.datasets)
        calibrations = dict(self.project_state.calibrations)
        rois = dict(self.project_state.rois)
        target_assignments = dict(self.project_state.target_assignments)
        recipes = dict(self.project_state.recipes)
        recipes.setdefault(sample_set.preprocessing_recipe_id, recipe.to_dict())

        from ..modeling import fit_model_run as fit_project_model_run

        # A fit specification, when the choices need one, also carries the
        # fold rule (cv and a fit_spec that sets the folds are not combined).
        fit_spec = dlg.fit_spec
        cv: int | str | None = dlg.cv if fit_spec is None else None
        hyperparameters = dlg.hyperparameters

        def job():
            return fit_project_model_run(
                project_path=project_path,
                sample_set=sample_set,
                datasets=datasets,
                calibrations=calibrations,
                rois=rois,
                target_assignments=target_assignments,
                recipes=recipes,
                signal_domains=resolved_signal_domains,
                signal_domain_id=signal_domain.signal_domain_id,
                model_type=dlg.model_type,
                hyperparameters=hyperparameters,
                cv=cv,
                fit_spec=fit_spec,
                target_name=target_name,
            )

        self._set_busy(True, "Fitting model...")
        self._job_thread, self._job_worker, self._job_relay = start_background_job(
            owner=self,
            fn=job,
            on_success=lambda model_run_result: self._on_fit_model_success(
                model_run_result=model_run_result,
                signal_domain=signal_domain,
                project_path=project_path,
                sample_set=sample_set if is_new else None,
                recipe=recipe,
            ),
            on_error=lambda msg: self._on_job_error("Fit model failed", msg),
            on_finished=self._clear_job,
        )

    def _smallest_labelled_roi(self, target_id: str) -> int | None:
        """Pixel count of the smallest ROI with a value for ``target_id`` (the draw limit)."""
        state = self.project_state
        counts = [roi.n_pixels for roi in state.rois.values()
                  if (roi.roi_id, target_id) in state.target_assignments
                  and roi.job != "prediction"]
        return min(counts) if counts else None

    def fit_readiness(self, target_id: str) -> tuple[str, bool]:
        """What a fit of ``target_id`` would use, per data file; and whether it can run."""
        state = self.project_state
        per_file: dict[str, list[int]] = {}
        for roi in state.rois.values():
            counts = per_file.setdefault(roi.dataset_id, [0, 0, 0])
            if roi.job == "prediction":
                counts[2] += 1
                continue
            counts[1] += 1
            counts[0] += (roi.roi_id, target_id) in state.target_assignments
        lines = []
        for dataset_id, (labelled, total, predicted) in sorted(
                per_file.items(), key=lambda item: self._data_file_name(item[0]).lower()):
            text = (f"{self._data_file_name(dataset_id)}: {labelled} of {total} ROIs "
                    "have a value")
            if predicted:
                text += f" ({predicted} prediction ROI(s) not used)"
            lines.append(text)
        try:
            sample_set, _new = resolve_sample_set(
                rois=state.rois, target_assignments=state.target_assignments,
                target_id=target_id, recipe_id="recipe_preview", sample_sets={},
                target_name=self._target_name(target_id))
        except ValueError as exc:
            lines.append(f"Cannot fit: {exc}")
            return "\n".join(lines), False
        lines.append(f"The model uses {len(sample_set.roi_ids)} ROIs from "
                     f"{len(sample_set.dataset_ids)} dataset(s).")
        return "\n".join(lines), True

    def _target_name(self, target_id: str) -> str | None:
        target = self.project_state.targets.get(target_id)
        return target.name if target is not None else None

    def _data_file_name(self, dataset_id: str) -> str:
        return self.project_state.data_file_label(dataset_id)

    def _plan_fit_sample_set(self, target_id: str, recipe: PreprocessingRecipe
                             ) -> tuple[SampleSet, bool, PreprocessingRecipe] | None:
        """The stored sample set of the target's labelled ROIs with ``recipe``, or a new one."""
        state = self.project_state
        try:
            sample_set, is_new = resolve_sample_set(
                rois=state.rois, target_assignments=state.target_assignments,
                target_id=target_id, recipe_id=recipe_id_for(recipe),
                sample_sets=state.sample_sets, target_name=self._target_name(target_id))
        except ValueError as exc:
            logger.error("No calibration set for the fit: {}", exc)
            QMessageBox.warning(self, "Fit model failed", str(exc))
            return None
        return sample_set, is_new, recipe

    def _validate_fit_preflight(
        self,
        *,
        sample_set: SampleSet,
        cv: int | str,
        required_signal_unit: str,
    ) -> str | None:
        stale_message = self._stale_message(sample_set)
        if stale_message is not None:
            return stale_message
        n_roi_groups = self._sample_set_group_count(sample_set)
        cv_error = self._validate_cv_preflight(n_roi_groups=n_roi_groups, cv=cv)
        if cv_error is not None:
            return cv_error
        signal_domain_error = self._validate_signal_domain_preflight(
            sample_set=sample_set,
            required_signal_unit=required_signal_unit,
        )
        if signal_domain_error is not None:
            return signal_domain_error
        return None

    def _stale_message(self, sample_set: SampleSet) -> str | None:
        """Why a stored sample set cannot be refit (a planned one is never stale)."""
        if sample_set.sample_set_id not in self.project_state.sample_sets:
            return None
        return self.project_state.stale_sample_set_message(sample_set.sample_set_id)

    def _default_sample_set_id(self) -> str | None:
        current_target = self.current_target()
        if self._last_sample_set_id in self.project_state.sample_sets and (
            current_target is None
            or self.project_state.sample_sets[self._last_sample_set_id].target_id
            == current_target.target_id
        ):
            return self._last_sample_set_id
        if current_target is not None:
            for sample_set in self.project_state.sample_sets.values():
                if sample_set.target_id == current_target.target_id:
                    return sample_set.sample_set_id
            return None
        return next(iter(self.project_state.sample_sets), None)

    def _build_signal_domain_contract(
        self,
        *,
        sample_set: SampleSet,
        required_signal_unit: str,
    ) -> SignalDomainContract:
        stale_message = self._stale_message(sample_set)
        if stale_message is not None:
            log_and_raise("{}", stale_message)
        # Opening the project validated that every ROI's dataset and calibration exist.
        roi = self.project_state.rois[sample_set.roi_ids[0]]
        return SignalDomainContract.for_training_data(
            dataset=self.project_state.datasets[roi.dataset_id],
            calibration=self.project_state.calibrations[roi.calibration_id],
            required_signal_unit=required_signal_unit,
        )

    def _signal_unit_mismatch(self, sample_set: SampleSet, required_signal_unit: str
                              ) -> str | None:
        """Why the ROIs' calibration cannot be modelled in ``required_signal_unit``, if so.

        A model in raw counts (DN) needs raw-count data; data with references,
        detector levels or calibrated values are modelled as reflectance or
        absorbance (dataset-calibration contract). Asked before a signal domain
        is built, so a readiness check does not log a refused domain as an error.
        """
        roi = self.project_state.rois.get(sample_set.roi_ids[0]) if sample_set.roi_ids else None
        calibration = None if roi is None else self.project_state.calibrations.get(
            roi.calibration_id)
        if calibration is None:
            return None
        if (required_signal_unit == "DN") == (calibration.mode == "raw_dn"):
            return None
        if required_signal_unit == "DN":
            return ("models on raw counts (DN) need datasets without reference correction; "
                    "these ROIs have references or detector levels, so choose Reflectance "
                    "or Absorbance")
        return (f"these ROIs hold raw counts without references or detector levels, so a "
                f"model in {required_signal_unit} is not possible; choose Raw counts (DN)")

    def fold_warnings_for(self, sample_set: SampleSet, *, fit_spec: Any) -> list[str]:
        """The fold warnings (in words) a fit of ``sample_set`` with ``fit_spec`` would record."""
        return self._predicted_warnings(list(sample_set.roi_ids), sample_set.target_id,
                                        fit_spec, None)

    def _predicted_warnings(self, roi_ids: list[str], target_id: str, fit_spec: Any,
                            cv: Any) -> list[str]:
        from ..modeling import predicted_fold_warnings

        try:
            codes = predicted_fold_warnings(
                roi_ids=roi_ids, target_id=target_id, rois=self.project_state.rois,
                target_assignments=self.project_state.target_assignments, fit_spec=fit_spec,
                cv=cv)
        except ValueError:  # an impossible request: the fit itself explains it
            return []
        return fold_warning_text(codes)

    def _dialog_fold_warnings(self, target_id: str, fit_spec: Any, cv: Any) -> list[str]:
        """Fold warnings for the fit dialog: all ROIs with a value for the target."""
        roi_ids = [roi_id for roi_id, roi in self.project_state.rois.items()
                   if (roi_id, target_id) in self.project_state.target_assignments
                   and roi.job != "prediction"]
        return self._predicted_warnings(roi_ids, target_id, fit_spec, cv)

    def _sample_set_label(self, sample_set: SampleSet) -> str:
        """``"20 ROIs, SNV"``: the ROI count and the recipe (ids stay in tooltips and exports)."""
        n = len(sample_set.roi_ids)
        recipe = self.project_state.recipes.get(sample_set.preprocessing_recipe_id)
        described = "none" if recipe is None else PreprocessingRecipe.from_dict(recipe).describe()
        rois = f"{n} ROI" if n == 1 else f"{n} ROIs"
        return rois if described == "none" else f"{rois}, {described}"

    def _sync_selected_target(self) -> None:
        visible_targets = self.visible_targets()
        visible_target_ids = {target.target_id for target in visible_targets}
        if self._selected_target_id not in visible_target_ids:
            self._selected_target_id = None if not visible_targets else visible_targets[0].target_id
        self._update_target_status()
        self._update_model_status()
        self._update_readiness_status()

    def select_model_run(self, model_run_id: str | None) -> None:
        selected_model_run_id = None if model_run_id is None else str(model_run_id)
        if selected_model_run_id not in self.project_state.model_runs:
            selected_model_run_id = None
        if selected_model_run_id == self._selected_model_run_id:
            return
        self._selected_model_run_id = selected_model_run_id
        self._invalidate_diagnostics_cache()
        self._update_model_status()

    def workspace_summary(self) -> dict[str, Any]:
        model_run = self._resolved_workspace_model_run()
        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        target = self._resolved_workspace_target(model_run=model_run, sample_set=sample_set)
        target_text = "none"
        if target is not None:
            target_text = target.name if not target.unit else f"{target.name} ({target.unit})"

        roi_count = 0
        pixel_count = 0
        if sample_set is not None:
            roi_count = len(sample_set.roi_ids)
            pixel_count = sum(
                int(self.project_state.rois[roi_id].n_pixels)
                for roi_id in sample_set.roi_ids
                if roi_id in self.project_state.rois
            )

        model_text = "none"
        components_text = "-"
        cv_text = "-"
        r2_text = "-"
        rmsecv_text = "-"
        bias_text = "-"
        if model_run is not None:
            model_text = model_label(model_run)
            n_components = model_run.hyperparameters.get("n_components")
            if n_components is not None:
                components_text = str(n_components)
            cv_text = describe_cv(model_run.cv_spec)
            if "r2" in model_run.metrics:
                r2_text = f"{float(model_run.metrics['r2']):.4g}"
            if "rmsecv" in model_run.metrics:
                rmsecv_text = f"{float(model_run.metrics['rmsecv']):.4g}"
            if "bias" in model_run.metrics:
                bias_text = f"{float(model_run.metrics['bias']):.4g}"
        roi_metrics, pixel_metrics, warnings = self._level_metrics(model_run)

        return {
            "title": "Model diagnostics",
            "target": target_text,
            "sample_set": "none" if sample_set is None else self._sample_set_label(sample_set),
            "model": model_text,
            "components": components_text,
            "cv": cv_text,
            "roi_count": str(roi_count),
            "pixel_count": str(pixel_count),
            "r2": r2_text,
            "rmsecv": rmsecv_text,
            "bias": bias_text,
            "roi_metrics": roi_metrics,
            "pixel_metrics": pixel_metrics,
            "warnings": warnings,
            "readiness": self._readiness_status.text(),
        }

    def _level_metrics(self, model_run: ModelRun | None
                       ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
        """(label, value) rows *per ROI* and *per pixel*, and the fold warnings in words.

        Per ROI: over ROI means (the model's quality); per pixel: over single
        spectra (model-workspace-diagnostics contract, Levels).
        """
        if model_run is None:
            return [], [], ""
        metrics = model_run.metrics

        def value(key: str) -> str | None:
            number = metrics.get(key)
            return None if number is None else f"{float(number):.4g}"

        r2y = None
        payload = self.workspace_diagnostics_payload() or {}  # cached per run
        curve = payload.get("component_curve")
        n = model_run.hyperparameters.get("n_components")
        if isinstance(curve, dict) and curve.get("r2y_cum_roi") is not None \
                and isinstance(n, (int, np.integer)) \
                and 1 <= int(n) <= np.size(curve["r2y_cum_roi"]):
            r2y = f"{float(np.asarray(curve['r2y_cum_roi'])[int(n) - 1]):.4g}"
        roi = [(label, text) for label, text in (
            ("R²Y (fit)", r2y), ("Q² (CV R²)", value("r2")), ("RMSEE (per ROI)", value("rmsee")),
            ("RMSECV (per ROI)", value("rmsecv")), ("Bias", value("bias")), ("RPD", value("rpd")))
            if text is not None]
        pixel = [(label, text) for label, text in (
            ("R² (CV)", value("pixel_r2")), ("RMSEE (per pixel)", value("pixel_rmsee")),
            ("RMSECV (per pixel)", value("pixel_rmsecv")), ("RPD", value("pixel_rpd")),
            ("Pixels", None if metrics.get("n_pixels") is None
             else str(int(metrics["n_pixels"]))))
            if text is not None]
        codes = model_run.cv_spec.get("warnings")
        warnings = "; ".join(fold_warning_text(codes)) if codes else ""
        return roi, pixel, warnings

    def workspace_diagnostics_provenance(self) -> dict[str, str]:
        """Provenance lines for exported diagnostics (Plot panel policy)."""
        provenance = {"project_id": str(self.project_state.project_meta.get("project_id", ""))}
        model_run = self._resolved_workspace_model_run()
        if model_run is None:
            return provenance
        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        target = self._resolved_workspace_target(model_run=model_run, sample_set=sample_set)
        provenance["model_run_id"] = model_run.model_run_id
        provenance["model_type"] = model_run.model_type
        provenance["sample_set_id"] = model_run.sample_set_id
        if sample_set is not None:
            provenance["dataset_ids"] = ", ".join(sample_set.dataset_ids)
            provenance["recipe_id"] = sample_set.preprocessing_recipe_id
        signal_domain = self.project_state.signal_domains.get(model_run.signal_domain_id)
        provenance["signal_domain_id"] = model_run.signal_domain_id
        if signal_domain is not None:
            provenance["signal_unit"] = signal_domain.required_signal_unit
        if target is not None:
            provenance["target"] = target.name
            provenance["target_unit"] = target.unit
        n_components = model_run.hyperparameters.get("n_components")
        if n_components is not None:
            provenance["components"] = str(n_components)
        provenance["cv"] = ", ".join(f"{key}={value}"
                                     for key, value in model_run.cv_spec.items())
        return provenance

    def workspace_diagnostics_payload(self) -> dict[str, Any] | None:
        """Return one stable diagnostics payload for the selected Model workspace run.

        The payload object is cached by the selected run and loaded artifact
        identities so repeated shell refreshes can treat unchanged diagnostics as
        a no-op redraw.
        """
        model_run = self._resolved_workspace_model_run()
        if model_run is None:
            self._cached_workspace_diagnostics_token = None
            self._cached_workspace_diagnostics_payload = None
            return None

        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        target = self._resolved_workspace_target(model_run=model_run, sample_set=sample_set)
        target_unit = "" if target is None else target.unit

        cv_outputs = model_run.artifact_refs.get("cv_outputs")
        if (
            not isinstance(cv_outputs, dict)
            or cv_outputs.get("y_true_roi") is None
            or cv_outputs.get("y_pred_roi") is None
        ):
            cv_outputs = self._load_model_cv_outputs(model_run)
        if not isinstance(cv_outputs, dict):
            self._cached_workspace_diagnostics_token = None
            self._cached_workspace_diagnostics_payload = None
            return None
        y_true_roi = cv_outputs.get("y_true_roi")
        y_pred_roi = cv_outputs.get("y_pred_roi")
        if y_true_roi is None or y_pred_roi is None:
            self._cached_workspace_diagnostics_token = None
            self._cached_workspace_diagnostics_payload = None
            return None

        artifact = self._load_model_artifact(model_run)
        roi_ids = cv_outputs.get("roi_ids")
        roi_names = None if roi_ids is None else tuple(
            self.project_state.rois[str(roi_id)].name
            if str(roi_id) in self.project_state.rois else str(roi_id)
            for roi_id in np.asarray(roi_ids, dtype=object).reshape(-1)
        )
        token = (
            model_run.model_run_id,
            target_unit,
            id(cv_outputs),
            id(y_true_roi),
            id(y_pred_roi),
            id(roi_ids),
            roi_names,  # a renamed ROI changes the exported names
            id(artifact),
        )
        if self._cached_workspace_diagnostics_token == token:
            return self._cached_workspace_diagnostics_payload

        payload: dict[str, Any] = {
            "target_unit": target_unit,
            "y_true_roi": np.asarray(y_true_roi, dtype=np.float32),
            "y_pred_roi": np.asarray(y_pred_roi, dtype=np.float32),
        }
        if roi_ids is not None:
            payload["roi_ids"] = np.asarray(roi_ids, dtype=object).reshape(-1)
            payload["roi_names"] = roi_names
        payload.update(_roi_score_payload(artifact))
        if artifact is not None:
            coefficients = artifact.get("coef_")
            if coefficients is not None:
                coef_arr = np.asarray(coefficients, dtype=np.float32).squeeze()
                if coef_arr.ndim == 2:
                    coef_arr = coef_arr[:, 0]
                if coef_arr.ndim == 1:
                    payload["coefficients"] = coef_arr
            loadings = artifact.get("x_loadings_")
            if loadings is None:
                loadings = artifact.get("loadings_")
            if loadings is not None:
                loadings_arr = np.asarray(loadings, dtype=np.float32).squeeze()
                if loadings_arr.ndim == 2:
                    loadings_arr = loadings_arr[:, 0]
                if loadings_arr.ndim == 1:
                    payload["loadings"] = loadings_arr
            wavelengths = artifact.get("wavelengths")
            if wavelengths is not None:
                spectral_axis = np.asarray(wavelengths, dtype=np.float32).squeeze()
                if spectral_axis.ndim == 1:
                    payload["spectral_axis"] = spectral_axis
            payload.update(_component_payload(model_run, artifact))
        self._cached_workspace_diagnostics_token = token
        self._cached_workspace_diagnostics_payload = payload
        return payload

    def _load_model_cv_outputs(self, model_run: ModelRun) -> dict[str, Any] | None:
        project_path = self.project_state.project_path
        if not project_path:
            return None
        cache_key = (project_path, model_run.model_run_id)
        if self._cached_cv_outputs_key == cache_key:
            return self._cached_cv_outputs
        try:
            artifact_refs = load_model_run_artifact_refs(
                project_path,
                model_run.model_run_id,
                keys=("cv_outputs",),
            )
        except Exception as exc:
            logger.warning(
                "Could not load model CV outputs '{}': {}",
                model_run.model_run_id,
                exc,
            )
            return None
        cv_outputs = artifact_refs.get("cv_outputs")
        self._cached_cv_outputs_key = cache_key
        self._cached_cv_outputs = cv_outputs if isinstance(cv_outputs, dict) else None
        return self._cached_cv_outputs

    def _load_model_artifact(self, model_run) -> dict[str, Any] | None:
        project_path = self.project_state.project_path
        if not project_path:
            return None
        cache_key = (project_path, model_run.model_run_id)
        if self._cached_diagnostics_model_key == cache_key:
            return self._cached_diagnostics_model_artifact
        try:
            artifact_refs = load_model_run_artifact_refs(
                project_path,
                model_run.model_run_id,
                keys=("prediction_model", "diagnostics", "component_curve"),
            )
        except Exception as exc:
            logger.warning(
                "Could not load model diagnostics artifact '{}': {}",
                model_run.model_run_id,
                exc,
            )
            return None
        artifact = {}
        prediction_model = artifact_refs.get("prediction_model")
        diagnostics = artifact_refs.get("diagnostics")
        if isinstance(prediction_model, dict):
            artifact.update(prediction_model)
        if isinstance(diagnostics, dict):
            artifact.update(diagnostics)
        curve = artifact_refs.get("component_curve")
        if isinstance(curve, dict) and curve:
            artifact["component_curve"] = curve
        if not artifact:
            return None
        self._cached_diagnostics_model_key = cache_key
        self._cached_diagnostics_model_artifact = artifact
        return artifact

    def _update_target_status(self) -> None:
        target = self.current_target()
        if target is None:
            self.target_status_label.setText("Target: none")
            self.view_state_changed.emit()
            return
        label = target.name if not target.unit else f"{target.name} ({target.unit})"
        self.target_status_label.setText(f"Target: {label}")
        self.view_state_changed.emit()

    def _update_model_status(self) -> None:
        model_run = self._resolved_workspace_model_run()
        sample_set = self._resolved_workspace_sample_set(model_run=model_run)
        if sample_set is None:
            self._sample_set_status.setText("Calibration set: none")
        else:
            suffix = (
                " (stale: deleted ROI)"
                if self.project_state.missing_roi_ids_for_sample_set(sample_set.sample_set_id)
                else ""
            )
            self._sample_set_status.setText(
                f"Calibration set: {self._sample_set_label(sample_set)}{suffix}")

        if model_run is None:
            self._model_status.setText("Model: none")
        else:
            metric = model_run.metrics.get("rmsecv")
            suffix = "" if metric is None else f" | RMSECV (per ROI)={float(metric):.4g}"
            self._model_status.setText(f"Model: {model_label(model_run)}{suffix}")
        self._update_readiness_status()
        self.view_state_changed.emit()

    def _on_active_model_changed(self, model_run_id: str) -> None:
        logger.info("Active model -> {}", model_run_id)
        self._invalidate_diagnostics_cache()
        self._update_model_status()

    def _invalidate_diagnostics_cache(self) -> None:
        self._cached_diagnostics_model_key = None
        self._cached_diagnostics_model_artifact = None
        self._cached_cv_outputs_key = None
        self._cached_cv_outputs = None
        self._cached_workspace_diagnostics_token = None
        self._cached_workspace_diagnostics_payload = None

    def _require_project_path(self) -> str:
        project_path = self.project_state.project_path
        if project_path is not None:
            return project_path
        log_and_raise(
            "ModelingView requires an open project before starting a background job.", error=RuntimeError,
        )

    def _update_readiness_status(self) -> None:
        model_run = self._resolved_workspace_model_run()
        if model_run is not None:
            message = f"Ready: model {model_label(model_run)} is available for prediction."
            stale_message = self.project_state.stale_sample_set_message(model_run.sample_set_id)
            if stale_message is not None:
                message += f" {stale_message}"
            codes = model_run.cv_spec.get("warnings")
            if codes:
                message += " Validation likely optimistic: " + "; ".join(
                    fold_warning_text(codes)) + "."
            self._readiness_status.setText(message)
            return

        target = self.current_target()
        if target is None:
            self._readiness_status.setText("Ready: define or select a target.")
            return

        labeled_roi_ids = {
            roi_id
            for roi_id, roi in self.project_state.rois.items()
            if (roi_id, target.target_id) in self.project_state.target_assignments
            and roi.job != "prediction"
        }
        if not self.project_state.rois:
            self._readiness_status.setText("Ready: draw ROIs in Explore.")
            return
        if not labeled_roi_ids:
            self._readiness_status.setText("Ready: assign target values to ROIs.")
            return
        if self._default_sample_set_id() is None:
            labeled_count = len(labeled_roi_ids)
            label = "ROI" if labeled_count == 1 else "ROIs"
            self._readiness_status.setText(
                f"Ready: fit a model from {labeled_count} labeled {label}."
            )
            return

        sample_set_id = self._default_sample_set_id()
        sample_set = (
            None if sample_set_id is None else self.project_state.sample_sets.get(sample_set_id)
        )
        if model_run is None:
            details: list[str] = []
            if sample_set is not None:
                roi_count = len(sample_set.roi_ids)
                roi_label = "ROI" if roi_count == 1 else "ROIs"
                n_roi_groups = self._sample_set_group_count(sample_set)
                group_label = "group" if n_roi_groups == 1 else "groups"
                details.append(
                    f"{roi_count} labeled {roi_label} across {n_roi_groups} ROI {group_label}"
                )
                stale_message = self.project_state.stale_sample_set_message(
                    sample_set.sample_set_id
                )
                if stale_message is not None:
                    details.append(stale_message)
                else:
                    if n_roi_groups < 2:
                        details.append("Grouped validation unavailable")
                    elif n_roi_groups < 7:
                        details.append("Auto validation will leave one ROI out at a time")
                    signal_domain_error = self._validate_signal_domain_preflight(
                        sample_set=sample_set,
                        required_signal_unit=str(self.state.y_unit_effective),
                    )
                    if signal_domain_error is not None:
                        details.append(
                            f"Current signal unit {self.state.y_unit_effective} is incompatible: {signal_domain_error}"
                        )
                    details += [f"Warning: {text}" for text in self.fold_warnings_for(
                        sample_set, fit_spec=None) if "fewer than 7" not in text]
            message = "Ready: fit a model from the selected calibration set."
            if details:
                message += " " + " ".join(
                    detail if detail.endswith(".") else f"{detail}." for detail in details
                )
            self._readiness_status.setText(message)
            return

    def _job_result_is_stale(self, project_path: str, *, what: str) -> bool:
        """Return True (and inform the user) when the project changed mid-job."""
        if self.project_state.project_path == project_path:
            return False
        msg = "Discarding {} computed for project '{}' because project '{}' is now open"
        logger.warning(msg, what, project_path, self.project_state.project_path)
        self._service_status.set_idle("Service: idle")
        QMessageBox.information(
            self,
            "Result discarded",
            msg.format(what, project_path, self.project_state.project_path) + ".",
        )
        return True

    def _on_fit_model_success(
        self,
        *,
        model_run_result: Any,
        signal_domain: SignalDomainContract,
        project_path: str,
        sample_set: SampleSet | None = None,
        recipe: PreprocessingRecipe | None = None,
    ) -> None:
        """Store the fitted run with its new sample set and recipe (``None``: already stored)."""
        if self._job_result_is_stale(project_path, what="model"):
            return
        state = self.project_state
        try:
            if sample_set is not None and recipe is not None \
                    and sample_set.preprocessing_recipe_id not in state.recipes:
                state.register_recipe(sample_set.preprocessing_recipe_id, recipe.to_dict())
            if sample_set is not None and sample_set.sample_set_id not in state.sample_sets:
                state.create_sample_set(sample_set)
            self.project_state.finalize_model_run_result(
                model_run_result=model_run_result,
                signal_domain=signal_domain,
            )
        except Exception as exc:
            logger.exception("Persisting the fitted model failed")
            self._on_job_error("Fit model failed", str(exc))
            return
        self._last_sample_set_id = model_run_result.model_run.sample_set_id
        fitted = model_run_result.model_run
        if fitted is not None:  # the stored run carries the name given when it was saved
            stored = self.project_state.model_runs.get(fitted.model_run_id) or fitted
            self._service_status.set_idle(f"Service: model {model_label(stored)}")
        self._update_model_status()

    def _on_job_error(self, title: str, message: str) -> None:
        self._service_status.set_idle("Service: idle")
        logger.error("{}: {}", title, message)
        QMessageBox.warning(self, title, message)

    def service_status_snapshot(self) -> tuple[str, bool]:
        """Return the current service label text and busy/progress visibility."""
        return self._service_status.label.text(), self._service_status.progress.isVisible()

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._btn_target.setEnabled(not busy)
        self._btn_assign_target.setEnabled(not busy)
        self._btn_fit_model.setEnabled(not busy)
        if busy:
            self._service_status.set_busy(text or "Working...")
        elif text is not None:
            self._service_status.set_idle(text)
        else:
            self._service_status.set_idle()

    def _clear_job(self) -> None:
        self._job_thread = None
        self._job_worker = None
        self._job_relay = None
        self._btn_target.setEnabled(True)
        self._btn_assign_target.setEnabled(True)
        self._btn_fit_model.setEnabled(True)

    def _selected_workspace_model_run(self) -> ModelRun | None:
        if self._selected_model_run_id is None:
            return None
        model_run = self.project_state.model_runs.get(self._selected_model_run_id)
        if model_run is None:
            self._selected_model_run_id = None
            return None
        return model_run

    def _resolved_workspace_model_run(self) -> ModelRun | None:
        selected_model_run = self._selected_workspace_model_run()
        if selected_model_run is not None:
            return selected_model_run

        model_run_id = self.project_state.active_model_name
        if model_run_id is None and self.project_state.model_runs:
            model_run_id = next(reversed(self.project_state.model_runs))
        if model_run_id is None:
            return None
        model_run = self.project_state.model_runs.get(model_run_id)
        if model_run is None:
            return None

        sample_set_id = self._default_sample_set_id()
        current_target = self.current_target()
        if (
            sample_set_id is not None
            and current_target is not None
            and model_run.sample_set_id != sample_set_id
        ):
            return None
        return model_run

    def _resolved_workspace_sample_set(self, *, model_run: ModelRun | None) -> SampleSet | None:
        if model_run is not None:
            return self.project_state.sample_sets.get(model_run.sample_set_id)
        sample_set_id = self._default_sample_set_id()
        if sample_set_id is None:
            return None
        return self.project_state.sample_sets.get(sample_set_id)

    def _resolved_workspace_target(
        self,
        *,
        model_run: ModelRun | None,
        sample_set: SampleSet | None,
    ) -> TargetDefinition | None:
        del model_run
        if sample_set is not None:
            target = self.project_state.targets.get(sample_set.target_id)
            if target is not None:
                return target
        return self.current_target()

    @staticmethod
    def _sample_set_group_count(sample_set: Any) -> int:
        return len({str(roi_id) for roi_id in sample_set.roi_ids})

    @staticmethod
    def _validate_cv_preflight(*, n_roi_groups: int, cv: int | str) -> str | None:
        # Round-robin folds over fewer ROIs than requested folds hold out one
        # ROI each (sample-set-model-run contract §4); only 2 ROIs are needed.
        if n_roi_groups < 2:
            return ("Cross-validation requires at least 2 ROIs in the selected calibration set, "
                    f"it has {n_roi_groups}.")
        return None

    def _validate_signal_domain_preflight(
        self,
        *,
        sample_set: SampleSet,
        required_signal_unit: str,
    ) -> str | None:
        sample_set_id = sample_set.sample_set_id
        stale_message = self._stale_message(sample_set)
        if stale_message is not None:
            return stale_message
        mismatch = self._signal_unit_mismatch(sample_set, required_signal_unit)
        if mismatch is not None:
            return mismatch
        try:
            signal_domain = self._build_signal_domain_contract(
                sample_set=sample_set,
                required_signal_unit=required_signal_unit,
            )
        except ValueError as exc:
            return str(exc)

        checked_contexts: set[tuple[str, str]] = set()
        for roi_id in sample_set.roi_ids:
            if roi_id not in self.project_state.rois:
                return f"SampleSet '{sample_set_id}' references missing roi_id '{roi_id}'"
            roi = self.project_state.rois[roi_id]
            context_key = (roi.dataset_id, roi.calibration_id)
            if context_key in checked_contexts:
                continue
            checked_contexts.add(context_key)
            if roi.dataset_id not in self.project_state.datasets:
                return f"ROI '{roi.roi_id}' references unknown dataset_id '{roi.dataset_id}'"
            if roi.calibration_id not in self.project_state.calibrations:
                return (
                    f"ROI '{roi.roi_id}' references unknown calibration_id '{roi.calibration_id}'"
                )
            try:
                check_signal_domain_compatibility(
                    dataset=self.project_state.datasets[roi.dataset_id],
                    calibration=self.project_state.calibrations[roi.calibration_id],
                    signal_domain=signal_domain,
                )
            except ValueError as exc:
                return str(exc)
        return None


__all__ = [
    "FitModelRunDialog",
    "ModelingView",
    "NewROIDialog",
    "TargetSelectionDialog",
]
