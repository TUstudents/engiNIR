"""The workflow strip: Data › ROIs › Values › Model › Predict › Report (plan gui-1.6).

One button per step of the workflow, under the menu bar. A click opens the
step's workspace; the state of each step (done, open checks, not started)
comes from :meth:`enginir.Project.checks` through :meth:`WorkflowStrip.set_states`.
The strip is navigation, not a toolbar: it holds no actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

from ._icons import icon


@dataclass(frozen=True)
class Step:
    """One step of the strip: its key (the check step), label and tooltip."""

    key: str
    label: str
    tooltip: str


#: The steps in workflow order; keys match the steps of ``Project.checks()``
#: (``report`` has no checks).
STEPS = (
    Step("data", "Data", "Datasets and their reference corrections"),
    Step("rois", "ROIs", "Regions of interest in the datasets"),
    Step("values", "Values", "Target values and sample labels of the ROIs"),
    Step("model", "Model", "Fit and judge models"),
    Step("predict", "Predict", "Predict ROIs and whole datasets"),
    Step("report", "Report", "Export the plots of a workspace"),
)

#: The state of a step: ``done`` (only ok checks), ``open`` (checks to look
#: at), ``blocked`` (a blocking check), ``idle`` (nothing to say yet).
STATES = ("idle", "done", "open", "blocked")


class WorkflowStrip(QWidget):
    """The row of workflow steps with their state; emits :attr:`step_clicked`."""

    #: The key of the step the user clicked.
    step_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowStrip")
        self._buttons: dict[str, QToolButton] = {}
        self._states: dict[str, tuple[str, int]] = {step.key: ("idle", 0) for step in STEPS}
        self._current = ""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(0)
        label = QLabel("Workflow")
        label.setObjectName("WorkflowLabel")
        layout.addWidget(label)
        layout.addSpacing(8)
        for number, step in enumerate(STEPS, 1):
            button = QToolButton()
            button.setObjectName("WorkflowStep")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setIconSize(QSize(16, 16))
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.setProperty("number", number)
            button.clicked.connect(lambda _checked=False, key=step.key: self._clicked(key))
            self._buttons[step.key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh()

    def _clicked(self, key: str) -> None:
        self.set_current(key)
        self.step_clicked.emit(key)

    def set_current(self, key: str) -> None:
        """Mark ``key`` as the step shown."""
        self._current = key
        for step_key, button in self._buttons.items():
            button.setChecked(step_key == key)

    def current(self) -> str:
        """The key of the step shown."""
        return self._current

    def set_states(self, states: dict[str, tuple[str, int]]) -> None:
        """Set ``{step: (state, number of open checks)}``; missing steps become idle."""
        self._states = {step.key: states.get(step.key, ("idle", 0)) for step in STEPS}
        self._refresh()

    def state(self, key: str) -> tuple[str, int]:
        """The state and number of open checks of step ``key``."""
        return self._states[key]

    def button(self, key: str) -> QToolButton:
        """The button of step ``key`` (for tests and keyboard focus)."""
        return self._buttons[key]

    def refresh_theme(self) -> None:
        """Redraw the state icons in the current theme's colours."""
        self._refresh()

    def _refresh(self) -> None:
        for number, step in enumerate(STEPS, 1):
            button = self._buttons[step.key]
            state, count = self._states[step.key]
            if state == "done":
                button.setIcon(icon("circle-check", "ok"))
            elif state == "blocked":
                button.setIcon(icon("circle-x", "error"))
            elif state == "open":
                button.setIcon(icon("alert-triangle", "attention"))
            else:
                button.setIcon(QIcon())
            text = step.label if state not in ("open", "blocked") or not count \
                else f"{step.label}  {count}"
            button.setText(text if state != "idle" else f"{number}  {step.label}")
            tip = step.tooltip
            if state in ("open", "blocked") and count:
                tip += f" ({count} to look at)"
            elif state == "done":
                tip += " (done)"
            button.setToolTip(tip)
