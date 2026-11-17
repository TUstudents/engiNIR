"""Project Browser: each model holds its calibration set and input contract (terminology contract)."""

from __future__ import annotations

import pytest

from tests.test_prediction_rois import tray as _tray


@pytest.fixture(scope="module")
def qapp():
    from enginir.viewer._app import ensure_app

    return ensure_app()


@pytest.fixture()
def tray(tmp_path):
    return _tray.__wrapped__(tmp_path)


def test_a_model_shows_its_calibration_set_and_input_contract_beneath_it(qapp, tray):
    from PyQt6.QtCore import Qt

    from enginir.viewer._browser import ProjectBrowser
    from enginir.viewer._state import ProjectState

    project, _scan, _path = tray
    model = project.fit("API", n_components=2, cv=3)
    state = ProjectState()
    state.open_project(str(project.path))
    browser = ProjectBrowser(state)
    browser.refresh()

    shown = [browser.topLevelItem(i).text(0) for i in range(browser.topLevelItemCount())]
    assert shown[0] == "Datasets (1)" and shown[-2:] == ["Models (1)", "Predictions"], \
        "a section counts its objects; an empty one has no count"
    sections = [text.split(" (")[0] for text in shown]
    assert sections == ["Datasets", "Reference corrections", "ROIs", "Targets", "Target values",
                        "Models", "Predictions"]
    assert not browser.topLevelItem(0).icon(0).isNull()
    models = browser.topLevelItem(sections.index("Models"))
    assert models.childCount() == 1, "nothing left over: the set and the contract are used"
    item = models.child(0)
    assert item.text(0).startswith(model.name)
    kinds = {item.child(i).data(0, Qt.ItemDataRole.UserRole)["kind"]: item.child(i).text(0)
             for i in range(item.childCount())}
    assert set(kinds) == {"sample_set", "signal_domain"}
    assert kinds["sample_set"].startswith("Calibration set: 7 ROIs, API")
    assert kinds["signal_domain"].startswith("Input contract: ")

    browser.select_payload({"kind": "signal_domain", "id": model.ref.signal_domain_id})
    assert browser.currentItem() is not None
    assert browser.currentItem().parent() is item, "selection finds the nested item"


def test_leaves_show_names_in_natural_order_and_keep_the_id_in_the_tooltip(qapp, tmp_path):
    from enginir import Project
    from enginir.viewer._browser import ProjectBrowser
    from enginir.viewer._state import ProjectState
    from tests.test_review_fixes import _scan

    project = Project.create(tmp_path / "p.h5")
    project.add_target("API", unit="%")
    scan = project.add_dataset(_scan(tmp_path / "s.h5"), dark_frames=(0, 1),
                               white_frames=(18, 19))
    for name in ("r10", "r2", "r1"):
        scan.add_roi(name, frames=(3, 5), rows=(0, 5), values={"API": 1.5})
    state = ProjectState()
    state.open_project(str(project.path))
    browser = ProjectBrowser(state)
    browser.refresh()

    def leaves(section: str) -> list[str]:
        top = next(browser.topLevelItem(i) for i in range(browser.topLevelItemCount())
                   if browser.topLevelItem(i).text(0).startswith(section))
        return [top.child(i).text(0) for i in range(top.childCount())]

    assert leaves("ROIs") == ["r1", "r2", "r10"], "names in natural order, no ids"
    assert leaves("Targets") == ["API (%)"]
    assert leaves("Target values") == ["r1: API = 1.5", "r2: API = 1.5", "r10: API = 1.5"]
    assert leaves("Reference corrections") == ["s.h5: dark/white references"]
    rois = next(browser.topLevelItem(i) for i in range(browser.topLevelItemCount())
                if browser.topLevelItem(i).text(0).startswith("ROIs"))
    roi_ids = {roi.name: roi_id for roi_id, roi in state.rois.items()}
    assert rois.child(0).toolTip(0).endswith(f"id {roi_ids['r1']}"), "the id stays at hand"
    every = [text for section in ("Datasets", "ROIs", "Targets", "Models") for text in leaves(section)]
    assert not any(object_id in text for text in every for object_id in state.rois), every
