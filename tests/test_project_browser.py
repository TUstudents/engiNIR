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

    sections = [browser.topLevelItem(i).text(0) for i in range(browser.topLevelItemCount())]
    assert sections == ["Datasets", "Reference corrections", "ROIs", "Targets", "Target values",
                        "Models", "Predictions"]
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
