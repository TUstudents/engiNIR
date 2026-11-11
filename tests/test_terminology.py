"""The viewer speaks the terminology contract (``contracts/terminology_v3.md``, Enforcement).

Every string literal of ``src/enginir/viewer`` (f-string parts included) is
scanned, except docstrings. A string may contain no word of "Words not
used" and no British spelling; identifier-like keys (one token of lower case
letters, digits, ``_ . : -``) are checked for British spelling only, since
one-word labels look the same; strings shown on buttons and menu items
are in title case, titles in sentence case. Justified exceptions are listed
in ``ALLOWED`` with their reason.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "src" / "enginir" / "viewer"

#: Word not used -> what to use (terminology contract, Words not used in user-visible text).
BANNED: dict[str, str] = {
    r"\bdata ?files?\b": "Dataset (the object) or file (on disk)",
    r"\bdata sets?\b": "Dataset",
    r"\btraining\b": "calibration, Calibration set",
    r"\bsample sets?\b": "Calibration set",
    r"\bmodel runs?\b": "Model",
    r"\bsignal domains?\b": "Input contract",
    r"\bfiltering\b": "Smoothing and derivatives",
    r"\buncertainty\b": "Prediction interval",
    r"\braw dn\b": "Raw counts (DN)",
    r"(?<!\()\bdn\b(?!\))": "Raw counts (DN)",
    r"\bobservations?\b": "pixel, spectrum, 'Fitted on: ...'",
    r"(?<!latent )\bvariables?\b": "Band",
    r"\breference values?\b": "Target value",
    r"\boutliers?\b": "above DCrit, above T²Crit, outside the model",
    r"\bpixel samples?\b": "pixel subsample",
    r"%\s*removed|\bremove(?:d)?\s+(?:the\s+)?pixels|\bpixels?\b[^.;]{0,40}\bremoved\b":
        "excluded, exclude (pixels)",
    r"\brmse(?:cv|e|p)\b(?! \(per (?:roi|pixel)\))": "RMSECV (per ROI), RMSECV (per pixel)",
}

#: British spelling -> US spelling (terminology contract, General rules 1).
BRITISH: dict[str, str] = {
    r"\bcolou?rs?\b(?<!colors)(?<!color)": "color",
    r"\bgreys?\b|\bgreyed\b": "gray",
    r"\bnormalis": "normaliz",
    r"\banalys(?:e|ed|es|ing)\b": "analyze",
    r"\bcentres?\b|\bcentred\b": "center, centered",
    r"\bbehaviours?\b": "behavior",
    r"\bmodelling\b": "modeling",
    r"\blabelled\b|\blabelling\b": "labeled, labeling",
    r"\bcancelled\b|\bcancelling\b": "canceled, canceling",
    r"\boptimis(?:e|ed|es|ing|ation)\b": "optimiz",
    r"\bvisualis(?:e|ed|es|ing|ation)\b": "visualiz",
    r"\binitialis(?:e|ed|es|ing|ation)\b": "initializ",
    r"\bfavour": "favor",
    r"\bneighbour": "neighbor",
}

#: Calls whose text is a button or menu item (title case) or a title (sentence case).
TITLE_CASE_CALLS = {"QPushButton", "QToolButton", "QAction", "addAction", "addMenu", "addButton",
                    "add_menu_action"}
SENTENCE_CASE_CALLS = {"set_title", "setWindowTitle", "setTitle", "QGroupBox", "suptitle"}
#: Plot-spec builders and the position of their title argument.
TITLE_ARGUMENT = {"PlotSpec": 1, "_spectral_spec": 2, "_distance_spec": 3, "PlotFrame": 1,
                  "QDockWidget": 0, "_plot_dock": 0, "getOpenFileName": 1,
                  "getSaveFileName": 1, "getExistingDirectory": 1,
                  "_build_deferred_workspace_page": 0}
_SMALL = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "per", "vs", "for", "with",
          "at", "by", "from", "as", "into", "off"}
#: Capitalized words allowed inside a sentence-case title (names and fixed terms).
PROPER = {"ROI", "ROIs", "Savitzky-Golay", "Hotelling's", "SIMCA", "DModX", "DModXPS", "DCrit", "PLS",
          "PCR", "LV", "PC", "SNV", "MSC", "EMSC", "VIP", "T²", "T²PS", "Q²", "R²", "R²X", "R²Y",
          "RMSEE", "RMSECV", "RMSEP", "RPD", "RPIQ", "CSV", "PNG", "SVG", "NIR", "HSI", "ENVI",
          "EIFF", "HDF5", "CV", "AsLS", "SG", "engiNIR", "I", "X", "Y", "Q", "T²Crit", "LV1",
          "LV2", "PC1", "PC2", "WSLg", "Wayland", "Explore", "Model", "Predict", "nm", "Q²ₐ",
          "ΔR²X"}

#: (file name, exact string) -> reason, for strings the scan cannot judge.
ALLOWED: dict[tuple[str, str], str] = {
    ("_browser.py", "DN"): "signal-unit key (SIGNAL_UNITS), not shown",
    ("_predict_dialogs.py", "rmsep {}"): "CSV metadata key (machine-readable, kept)",
    ("_predict_dialogs.py", "rmsep "): "CSV metadata key (machine-readable, kept)",
    ("_welcome.py", "engiNIR — HSI Viewer"): "the product name",
    ("_layout_settings.py", "Could not remove plot layout settings {}: {}"): "a file, not pixels",
    ("_viewer.py", "DN"): "signal-unit key (SIGNAL_UNITS), not shown",
    ("_modeling.py", "DN"): "signal-unit key (SIGNAL_UNITS), not shown",
    ("_eda.py", "DN"): "signal-unit key (SIGNAL_UNITS), not shown",
    ("_state.py", "DN"): "signal-unit key (SIGNAL_UNITS), not shown",
    ("_data_files.py", "DN"): "signal-unit key (SIGNAL_UNITS), not shown",
    ("_viewer.py", "engiNIR - HSI Viewer"): "the product name",
    ("_project_rois.py", "colour"): "CSV metadata key (machine-readable, kept until 2.0)",
}

_IDENTIFIER = re.compile(r"^[a-z0-9_.:\-/]*$")


def _strings(tree: ast.AST) -> Iterator[tuple[int, str, str | None]]:
    """(line, text, enclosing call name) for every string literal outside docstrings."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    call_of: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "setText" and isinstance(func, ast.Attribute) and "btn" in ast.unparse(
                    func.value).lower():
                name = "QPushButton"  # self._btn_fit.setText("Fit Model"): a button
            # A plot's title is an argument of PlotSpec or of a spec helper.
            position = TITLE_ARGUMENT.get(name or "")
            shown = node.args[position:position + 1] if position is not None else node.args[:1]
            for arg in shown:
                for inner in ast.walk(arg):
                    call_of.setdefault(id(inner), "set_title" if position is not None
                                       else name or "")
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "plot_title"
                for target in node.targets):
            call_of.setdefault(id(node.value), "set_title")
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(part.value if isinstance(part, ast.Constant) else "{}"
                           for part in node.values)
            yield node.lineno, text, call_of.get(id(node))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docstrings):
            parent_joined = False
            yield node.lineno, node.value, call_of.get(id(node)) if not parent_joined else None


def _case_problem(text: str, call: str | None) -> str | None:
    words = [w for w in re.split(r"[\s/]+", text.replace("...", "").replace("…", "")) if w]
    if not words or call is None or "{}" in text or len(words) > 8:
        return None
    if call in TITLE_CASE_CALLS:
        for i, word in enumerate(words):
            core = word.strip("()[],:;.!?'\"")
            if not core or not core[0].isalpha() or core in PROPER:
                continue
            if core[0].islower() and (i == 0 or core.lower() not in _SMALL):
                return f"title case for buttons and menu items: {word!r}"
    if call in SENTENCE_CASE_CALLS:
        for word in words[1:]:
            core = word.strip("()[],:;.!?'\"")
            if (core and core[0].isupper() and core not in PROPER and core[1:].islower()):
                return f"sentence case for titles: {word!r}"
    return None


def violations() -> list[str]:
    """Every banned word, British spelling or capitalization problem, as report lines."""
    found: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(VIEWER.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line, text, call in _strings(tree):
            if not text.strip() or (path.name, text) in ALLOWED:
                continue
            # A one-word string may be a key or a label ("centre" shown in the
            # Inspector): it is checked for spelling only.
            identifier = bool(_IDENTIFIER.match(text))
            problems = [f"{m.group(0)!r} -> {use}"
                        for table in ((BRITISH,) if identifier else (BANNED, BRITISH))
                        for pattern, use in table.items()
                        for m in [re.search(pattern, text, re.IGNORECASE)] if m]
            case = None if identifier else _case_problem(text, call)
            if case:
                problems.append(case)
            for problem in problems:
                key = (path.name, line, problem)
                if key not in seen:
                    seen.add(key)
                    found.append(f"{path.name}:{line}: {problem}   in {text[:70]!r}")
    return found


def test_the_viewer_uses_the_terms_of_the_contract():
    found = violations()
    assert not found, (f"{len(found)} wording problems (contracts/terminology_v3.md):\n"
                       + "\n".join(found))


if __name__ == "__main__":
    for item in violations():
        print(item)
