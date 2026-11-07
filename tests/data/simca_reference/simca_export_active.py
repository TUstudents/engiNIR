"""Export every fitted model of the active SIMCA project (runs inside SIMCA 18).

Use it for settings that Python cannot make, such as the CV-groups dialog:
open ``simca_exports/simca_reference.usp``, change a model (or make a new
one), fit it, then run this script (Favorites or
``SIMCA.exe /execpython <full path>\\simca_export_active.py``). It writes the
same files as ``simca_export.py``, named by SIMCA's model name, into
``simca_exports/manual/``, with ``manual/manifest.json``.
"""

import importlib.util
import json
import os

import umetrics

HERE = os.path.dirname(os.path.realpath(__file__))
_spec = importlib.util.spec_from_file_location("simca_export", os.path.join(HERE, "simca_export.py"))
simca_export = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(simca_export)

OUT = os.path.join(simca_export.OUT, "manual")


def y_variable(project, number):
    """Name of the model's (single) Y variable."""
    workset = project.edit_workset(number)
    names = list(workset.get_data_collection().get_var_names())
    ys = list(workset.get_y_variables())
    if len(ys) != 1:
        raise ValueError("model %d has %d Y variables, expected 1" % (number, len(ys)))
    y = ys[0]
    return names[y] if isinstance(y, int) else str(y)


def run():
    project = umetrics.SimcaApp.get_active_project()
    os.makedirs(OUT, exist_ok=True)
    simca_export.OUT = OUT  # the export functions write into OUT
    has_prediction = bool(project.get_predictionsets())
    manifest = {"project": project.get_name(), "models": {}}
    for info in project.get_model_infos():
        if not info.isfitted or info.components < 1:
            continue
        label, number, a = info.name, info.number, info.components
        simca_export.log("model " + label)
        y_name = simca_export.attempt(label + ": Y variable", y_variable, project, number)
        if y_name is None:
            continue
        manifest["models"][label] = {
            "number": number, "observations": info.observations,
            "x_variables": info.Xvariables, "components": a, "y_variable": y_name,
            "options": simca_export.dmodx_options(project, number),
        }
        simca_export.attempt(label + ": export", simca_export.export_model, project, label,
                             number, a, has_prediction, y_name)
    manifest["errors"] = simca_export.errors
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    simca_export.log("done, %d step(s) failed; see simca_exports/manual/manifest.json"
                     % len(simca_export.errors))


if __name__ == "__main__":
    run()
