"""Band ranges of a recipe chosen on a plot (viewer-workspace-ui contract, band selection).

:class:`BandRangeDialog` shows the mean spectrum of the sample set's ROIs on
the recipe's axis after Savitzky-Golay, with an optional guide (the selected
model's VIP or coefficients) below it, and lets the user drag keep and
exclude regions. The spectra load in a background job.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .._logging import get_logger
from ..modeling import BandProposal
from ..preprocessing.recipe import format_ranges
from ._jobs import start_background_job
from ._theme import GUIDE_COLOR, panel_colors, secondary_text_style

logger = get_logger(__name__)

Ranges = tuple[tuple[float, float], ...]

#: Common NIR water bands, offered as exclude suggestions (never applied automatically).
WATER_BANDS: tuple[tuple[float, float], ...] = ((1400.0, 1500.0), (1900.0, 2000.0))

_KEEP_BRUSH = (60, 170, 90, 60)
_EXCLUDE_BRUSH = (210, 70, 70, 70)


@dataclass(frozen=True)
class BandSource:
    """What the range dialog shows.

    ``load`` runs in a worker thread and returns ``(axis, mean_spectrum)`` on
    the recipe's axis after Savitzky-Golay. ``guide`` is ``(axis, values,
    label)`` of the selected model, or ``None``.
    """

    load: Callable[[], tuple[np.ndarray, np.ndarray]]
    guide: tuple[np.ndarray, np.ndarray, str] | None = None
    axis_label: str = "Wavelength (nm)"


class BandRangeDialog(QDialog):
    """Drag keep and exclude regions on the mean spectrum."""

    def __init__(self, source: BandSource, *, keep: Ranges = (), exclude: Ranges = (),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select bands")
        self.resize(820, 520)
        self._regions: list[tuple[str, pg.LinearRegionItem]] = []
        self._axis: np.ndarray | None = None
        colors = panel_colors()
        layout = QVBoxLayout(self)
        hint = QLabel("Drag a region's edges or body. Green regions are kept (none: every "
                      "band), red regions are left out. The model sees the kept bands only.")
        hint.setWordWrap(True)
        hint.setStyleSheet(secondary_text_style())
        layout.addWidget(hint)

        self._spectrum_plot = pg.PlotWidget()
        self._spectrum_plot.setBackground(colors["bg"])
        self._spectrum_plot.setLabel("left", "Mean spectrum")
        self._spectrum_plot.setMenuEnabled(False)
        layout.addWidget(self._spectrum_plot, 3)
        self._guide_plot: pg.PlotWidget | None = None
        if source.guide is not None:
            axis, values, label = source.guide
            self._guide_plot = pg.PlotWidget()
            self._guide_plot.setBackground(colors["bg"])
            self._guide_plot.setLabel("left", label)
            self._guide_plot.setLabel("bottom", source.axis_label)
            self._guide_plot.setMenuEnabled(False)
            self._guide_plot.setXLink(self._spectrum_plot)
            self._guide_plot.plot(np.asarray(axis, dtype=float), np.asarray(values, dtype=float),
                                  pen=pg.mkPen(colors["accent"], width=1.5), connect="finite")
            if label == "VIP":
                self._guide_plot.addLine(y=1.0, pen=pg.mkPen(GUIDE_COLOR, width=1))
            layout.addWidget(self._guide_plot, 2)
        else:
            self._spectrum_plot.setLabel("bottom", source.axis_label)

        self._status = QLabel("Loading the spectra of the calibration set...")
        self._status.setStyleSheet(secondary_text_style())
        layout.addWidget(self._status)

        buttons_row = QHBoxLayout()
        self._add_keep = QPushButton("Add Keep Range")
        self._add_exclude = QPushButton("Add Exclude Range")
        self._clear = QPushButton("Clear")
        buttons_row.addWidget(self._add_keep)
        buttons_row.addWidget(self._add_exclude)
        for low, high in WATER_BANDS:
            water = QPushButton(f"Exclude Water {low:.0f}-{high:.0f}")
            water.setToolTip("A common NIR water band; added only when you click.")
            water.clicked.connect(lambda _checked=False, r=(low, high): self.add_region(
                "exclude", r))
            buttons_row.addWidget(water)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self._clear)
        layout.addLayout(buttons_row)
        self._add_keep.clicked.connect(lambda: self.add_region("keep"))
        self._add_exclude.clicked.connect(lambda: self.add_region("exclude"))
        self._clear.clicked.connect(self.clear_regions)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

        for kind, ranges in (("keep", keep), ("exclude", exclude)):
            for pair in ranges:
                self.add_region(kind, pair)
        self._job = start_background_job(owner=self, fn=source.load, on_success=self._loaded,
                                         on_error=self._failed, on_finished=lambda: None)

    # -- data -------------------------------------------------------------

    def _loaded(self, result: tuple[np.ndarray, np.ndarray]) -> None:
        axis, mean = (np.asarray(part, dtype=float).reshape(-1) for part in result)
        self._axis = axis
        self._spectrum_plot.plot(axis, mean, pen=pg.mkPen(panel_colors()["fg"], width=1.3))
        self._status.setText(f"{axis.size} bands, {axis.min():g} to {axis.max():g}")

    def _failed(self, message: str) -> None:
        logger.error("Band selection spectra could not be loaded: {}", message)
        self._status.setText(f"Could not load the spectra: {message}")

    # -- regions ----------------------------------------------------------

    def add_region(self, kind: str, bounds: tuple[float, float] | None = None) -> None:
        """Add a keep or exclude region (the middle fifth of the view when no bounds)."""
        if bounds is None:
            (low, high), _y = self._spectrum_plot.getViewBox().viewRange()
            width = (high - low) / 5.0
            bounds = (low + 2 * width, low + 3 * width)
        region = pg.LinearRegionItem(values=tuple(float(v) for v in bounds),
                                     brush=_KEEP_BRUSH if kind == "keep" else _EXCLUDE_BRUSH)
        region.setZValue(-10)
        self._spectrum_plot.addItem(region)
        self._regions.append((kind, region))

    def clear_regions(self) -> None:
        """Remove every region."""
        for _kind, region in self._regions:
            self._spectrum_plot.removeItem(region)
        self._regions.clear()

    def ranges(self, kind: str) -> Ranges:
        """The ``"keep"`` or ``"exclude"`` regions as sorted ``(low, high)`` pairs."""
        pairs = []
        for region_kind, region in self._regions:
            if region_kind == kind:
                low, high = np.asarray(region.getRegion(), dtype=float).reshape(2)
                pairs.append((round(float(low), 2), round(float(high), 2)))
        return tuple(sorted(pairs))


ProposalRun = Callable[[Callable[[int, int], None], Callable[[], bool]], BandProposal]


class ProposeBandsDialog(QDialog):
    """Run the band proposal in the background and show its result.

    ``run(progress, cancel)`` calls :meth:`enginir.Project.propose_bands`
    with the fit dialog's choices. Accepting the dialog means "use the
    proposed ranges as keep ranges" (:attr:`proposal`).
    """

    def __init__(self, run: ProposalRun, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Propose bands")
        self.resize(640, 460)
        self.proposal: BandProposal | None = None
        self._progress_state = (0, 0)
        self._cancelled = False
        layout = QVBoxLayout(self)
        intro = QLabel("Forward interval PLS: each step adds the interval that lowers the "
                       "RMSECV (per ROI) most, until the gain is within one standard error. "
                       "The whole search is repeated inside every outer fold, so the nested "
                       "RMSECV (per ROI) includes the selection.")
        intro.setWordWrap(True)
        intro.setStyleSheet(secondary_text_style())
        layout.addWidget(intro)
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        layout.addWidget(self._bar)
        self._status = QLabel("Starting...")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._plot = pg.PlotWidget()
        self._plot.setBackground(panel_colors()["bg"])
        self._plot.setLabel("bottom", "Step (intervals chosen)")
        self._plot.setLabel("left", "RMSECV (per ROI)")
        self._plot.setMenuEnabled(False)
        self._plot.hide()
        layout.addWidget(self._plot, 1)
        self._buttons = QDialogButtonBox()
        self._use = self._buttons.addButton("Use as Keep Ranges",
                                            QDialogButtonBox.ButtonRole.AcceptRole)
        self._cancel = self._buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        if self._use is not None:
            self._use.setEnabled(False)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self._cancel_or_close)
        layout.addWidget(self._buttons)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._show_progress)
        self._timer.start()
        self._running = True
        self._job = start_background_job(
            owner=self, fn=lambda: run(self._report, lambda: self._cancelled),
            on_success=self._done, on_error=self._failed, on_finished=self._finished)

    def _report(self, done: int, total: int) -> None:  # worker thread: store only
        self._progress_state = (done, total)

    def _show_progress(self) -> None:
        done, total = self._progress_state
        if total:
            self._bar.setRange(0, total)
            self._bar.setValue(min(done, total))
            self._status.setText(f"{done} of at most {total} candidate evaluations "
                                 "(each a cross-validated fit per fold)")

    def _cancel_or_close(self) -> None:
        if self._running:
            self._cancelled = True
            self._status.setText("Canceling...")
            return
        self.reject()

    def _finished(self) -> None:
        self._running = False
        self._timer.stop()

    def _done(self, proposal: BandProposal) -> None:
        self.proposal = proposal
        self._bar.hide()
        last = proposal.steps[-1]
        lines = [f"Proposed keep ranges: {format_ranges(proposal.ranges) or 'none'}",
                 f"RMSECV (per ROI) with these ranges: {last.rmsecv:.4g} ({last.n_components} "
                 "components)",
                 f"Nested RMSECV (per ROI) (selection included): {proposal.nested_rmsecv:.4g}",
                 f"RMSECV (per ROI) with all bands: {proposal.all_bands_rmsecv:.4g}"]
        if not proposal.ranges:
            lines.append("No interval beat the mean of the target values.")
        self._status.setText("\n".join(lines))
        steps = np.arange(len(proposal.steps), dtype=float)
        rmsecv = np.array([step.rmsecv for step in proposal.steps])
        se = np.array([step.rmsecv_se for step in proposal.steps])
        colors = panel_colors()
        self._plot.plot(steps, rmsecv, pen=pg.mkPen(colors["accent"], width=1.5),
                        symbol="o", symbolSize=6, symbolBrush=colors["accent"])
        self._plot.addItem(pg.ErrorBarItem(x=steps, y=rmsecv, height=2 * se,
                                           pen=pg.mkPen(GUIDE_COLOR)))
        self._plot.addLine(y=proposal.nested_rmsecv, pen=pg.mkPen(colors["fg"], width=1,
                                                                   style=Qt.PenStyle.DashLine))
        self._plot.show()
        if self._use is not None:
            self._use.setEnabled(bool(proposal.ranges))
        if self._cancel is not None:
            self._cancel.setText("Close")

    def _failed(self, message: str) -> None:
        self._bar.hide()
        if self._cancelled:
            self._status.setText("Canceled.")
        else:
            logger.error("Band proposal failed: {}", message)
            self._status.setText(f"Band proposal failed: {message}")
        if self._cancel is not None:
            self._cancel.setText("Close")


__all__ = ["BandRangeDialog", "BandSource", "ProposeBandsDialog", "WATER_BANDS"]
