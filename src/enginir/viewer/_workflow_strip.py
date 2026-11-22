"""The workflow strip: Data → ROIs → Values → Model → Predict (plan gui-1.6).

One button per step of the workflow, under the menu bar, with an arrow
between the steps: they are done in this order. A click opens the step's
workspace; the state of each step (done, open checks in a round badge, not
started) comes from :meth:`enginir.Project.checks` through
:meth:`WorkflowStrip.set_states`. The strip is navigation, not a toolbar:
it holds no actions. A Report step comes with the report itself
(engiNIR 2.0); exporting plots is a menu item.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

from ._icons import icon
from ._ui_helpers import wrap_tooltip


@dataclass(frozen=True)
class Step:
    """One step of the strip: its key (the check step), label and tooltip."""

    key: str
    label: str
    tooltip: str


#: The steps in workflow order; keys match the steps of ``Project.checks()``.
STEPS = (
    Step("data", "Data", "Datasets and their reference corrections"),
    Step("rois", "ROIs", "Regions of interest in the datasets"),
    Step("values", "Values", "Target values and sample labels of the ROIs"),
    Step("model", "Model", "Fit and judge models"),
    Step("predict", "Predict", "Predict ROIs and whole datasets"),
)

#: The state of a step: ``done`` (only ok checks), ``open`` (checks to look
#: at), ``blocked`` (a blocking check), ``idle`` (nothing to say yet).
STATES = ("idle", "done", "open", "blocked")


class _StepButton(QToolButton):
    """A step of the strip; open checks show as a round badge at its right end."""

    def __init__(self) -> None:
        super().__init__()
        self._badge = QLabel(self)
        self._badge.setObjectName("StepBadge")
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._badge.hide()

    def set_badge(self, count: int, level: str = "open") -> None:
        """Show ``count`` in the badge (0: no badge); ``level`` is ``open`` or ``blocked``."""
        shown = count > 0
        if (shown, level) != (self.property("badge"), self._badge.property("level")):
            self.setProperty("badge", shown)  # the style sheet makes room at the right
            self._badge.setProperty("level", level)
            for widget in (self, self._badge):
                style = widget.style()
                if style is not None:
                    style.unpolish(widget)
                    style.polish(widget)
        self._badge.setText(str(count) if shown else "")
        self._badge.setVisible(shown)
        self._place_badge()

    def badge_text(self) -> str:
        """The number in the badge, ``""`` without a badge."""
        return self._badge.text()

    def _place_badge(self) -> None:
        self._badge.adjustSize()
        self._badge.move(self.width() - self._badge.width() - 10,
                         (self.height() - self._badge.height()) // 2)

    def resizeEvent(self, a0: QEvent | None) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(a0)  # type: ignore[arg-type]
        self._place_badge()


class WorkflowStrip(QWidget):
    """The row of workflow steps with their state; emits :attr:`step_clicked`."""

    #: The key of the step the user clicked.
    step_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowStrip")
        self._buttons: dict[str, _StepButton] = {}
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
            if number > 1:  # the steps follow each other in this order
                arrow = QLabel("→")
                arrow.setObjectName("WorkflowArrow")
                layout.addWidget(arrow)
            button = _StepButton()
            button.setObjectName("WorkflowStep")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setIconSize(QSize(16, 16))
            button.setCheckable(True)
            button.setAutoRaise(True)
            button.clicked.connect(lambda _checked=False, key=step.key: self._clicked(key))
            self._buttons[step.key] = button
            layout.addWidget(button)
        layout.addStretch(1)
        #: What the window works on: the project and the model shown (right end).
        self._context = QLabel("")
        self._context.setObjectName("WorkflowContext")
        self._context.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._context)
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

    def button(self, key: str) -> _StepButton:
        """The button of step ``key`` (for tests and keyboard focus)."""
        return self._buttons[key]

    def set_context(self, project: str = "", model: str = "") -> None:
        """Name the open project and the model shown at the right end of the strip."""
        parts = [part for part in (project, f"Model: {model}" if model else "") if part]
        self._context.setText("   ·   ".join(parts))
        self._context.setToolTip(wrap_tooltip("The open project and the model shown in the Model step"
                                 if parts else ""))

    def context(self) -> str:
        """The text at the right end of the strip."""
        return self._context.text()

    def refresh_theme(self) -> None:
        """Redraw the state icons in the current theme's colours."""
        self._refresh()

    def _refresh(self) -> None:
        for step in STEPS:
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
            button.setText(step.label)
            open_count = count if state in ("open", "blocked") else 0
            button.set_badge(open_count, state if open_count else "open")
            tip = step.tooltip
            if open_count:
                tip += f" ({count} open check{'s' if count != 1 else ''}: see the Checks tab)"
            elif state == "done":
                tip += " (done)"
            button.setToolTip(wrap_tooltip(tip))
