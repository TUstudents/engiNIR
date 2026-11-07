"""Build the SIMCA reference models and export their results (runs inside SIMCA 18).

Run it in SIMCA (needs a licence with Python scripting):

    SIMCA.exe /execpython <full path>\\simca_export.py

or add it to Favorites (Developer > Add script to Favorites) and click it.
It reads ``pixels.csv``, ``roi_means.csv`` and ``pixels_test.csv`` next to
this file, creates the project ``simca_exports/simca_reference.usp`` with
the models M1-M5 of ``README.md``, and writes every result as CSV into
``simca_exports/``, at the precision SIMCA computes (no list formatting).
``simca_exports/manifest.json`` records the SIMCA version, the settings and
every step that failed; the Python console shows the same messages.

Cross-validation groups cannot be set from Python. SIMCA's default is
every 7th observation in the same group, so the pixel table is imported
interleaved: row k comes from fold k mod 7, where fold f holds the ROIs at
positions f and f + 7 of the sample set (engiNIR's round-robin rule). The
default groups are then exactly the round-robin ROI folds; the exported
``CVGroups`` lets the tests check it. The ROI means (14 rows in ROI order)
need no reordering.

Only the standard library and ``umetrics`` are used.
"""

import csv
import json
import os
import traceback

import umetrics

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, "simca_exports")
N_FOLDS = 7
FIXED_COMPONENTS = 4
TARGET = "conc"
PROBABILITIES = (0.95, 0.99)

# label: (table, filter, components (-1 = Autofit), prediction set)
MODELS = {
    "M1": ("pixels", None, FIXED_COMPONENTS, True),
    "M2": ("pixels", "snv", FIXED_COMPONENTS, True),
    "M3": ("pixels", "derivative", FIXED_COMPONENTS, True),
    "M4": ("roi_means", None, -1, False),
    "M5": ("pixels", "asls", FIXED_COMPONENTS, True),
}

FILTER_SETTINGS = {
    "snv": {},
    "derivative": {"order": "first", "polynomial": "quadratic", "points": 15, "distance": 1.0},
    "asls": {"lambda": 10000, "p": 0.001},
}

errors = []


def log(message):
    print("[simca_export] " + message)


def attempt(what, function, *args, **kwargs):
    """Run one step; on failure record it and carry on with the others."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded and reported
        errors.append({"step": what, "error": repr(exc), "traceback": traceback.format_exc()})
        log("FAILED " + what + ": " + repr(exc))
        return None


# --- input -----------------------------------------------------------------

def read_table(name):
    with open(os.path.join(HERE, name + ".csv"), newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def interleave_by_fold(body):
    """Order the rows so that row k belongs to round-robin fold k mod 7."""
    rois = []
    for row in body:
        if row[1] not in rois:
            rois.append(row[1])
    folds = [[] for _ in range(min(N_FOLDS, len(rois)))]
    for row in body:
        folds[rois.index(row[1]) % len(folds)].append(row)
    sizes = {len(fold) for fold in folds}
    if len(sizes) != 1:
        raise ValueError("folds differ in size " + str(sorted(sizes)) +
                         "; every-Nth groups cannot reproduce them")
    return [folds[k % len(folds)][k // len(folds)] for k in range(len(body))]


def import_data(header, body):
    matrix = [list(header)]
    for row in body:
        matrix.append([row[0], row[1]] + [float(value) for value in row[2:]])
    data = umetrics.impdata.ImportData(matrix)
    spec = data.importspec
    spec.set_row_type(0, umetrics.impdata.rowtype.primaryvarid)
    spec.set_col_type(0, umetrics.impdata.columntype.primaryobsid)
    spec.set_col_type(1, umetrics.impdata.columntype.secondaryobsid)
    spec.set_col_block(2, umetrics.impdata.datablock.Y)
    return data


def create_dataset(project, name, header, body):
    number = project.create_dataset(import_data(header, body), name)
    log("dataset " + name + ": " + str(len(body)) + " observations")
    return number


# --- models ----------------------------------------------------------------

def filtered_dataset(project, source, kind, label):
    """Apply one spectral filter to the wavelengths; returns the new dataset's name.

    The filtered dataset holds only the filtered variables; the model takes Y
    from the unfiltered dataset (see :func:`build_model`).
    """
    simca = umetrics.simca
    variables = list(project.get_dataset(source).get_var_names())
    filt = project.create_filter(source)
    filt.set_variables([name for name in variables if name != TARGET])
    if kind == "snv":
        filt.add_filter_type(simca.filtertype.snv)
    elif kind == "derivative":
        settings = FILTER_SETTINGS["derivative"]
        filt.add_filter_type(simca.filtertype.derivatives)
        filt.set_derivative_order(simca.derivativeorder.firstderivative)
        filt.set_polynomial_order(simca.polynomialorder.quadratic)
        filt.set_submodel_points(settings["points"])
        filt.set_distance_between_points(settings["distance"])
    elif kind == "asls":
        settings = FILTER_SETTINGS["asls"]
        filt.add_filter_type(simca.filtertype.aslsbaselinecorrection)
        filt.set_lambda_asls(settings["lambda"])
        filt.set_p(settings["p"])
    else:
        raise ValueError("unknown filter " + kind)
    name = source + "_" + label + "_" + kind
    filt.apply_filter(name)
    log(label + ": filtered dataset " + name)
    return name


def build_model(project, label, table, filtered=None):
    """PLS workset: X from ``filtered`` (or ``table``), Y ``conc`` from ``table``.

    With a filter, the workset holds both datasets (SIMCA matches the
    observations by ID) and the unfiltered wavelengths are excluded.
    """
    simca = umetrics.simca
    if filtered is None:
        workset = project.create_workset(table)
    else:
        workset = project.create_workset([filtered, table])
        raw = [name for name in project.get_dataset(table).get_var_names() if name != TARGET]
        workset.exclude_variables(raw)
    workset.set_y(TARGET)
    workset.set_variable_scale_type([], simca.scaletype.ctr)  # [] = all variables
    workset.set_variable_scale_type([TARGET], simca.scaletype.uv)
    workset.set_type(simca.modeltype.pls)
    number = workset.create_model(remove_bad_data=False)[0]
    return number


def fit(project, number, components):
    if components < 0:
        result = project.fit_model(number)
    else:
        result = project.fit_model(number, numcomp=components)
    return int(result[0])


# --- export ----------------------------------------------------------------

def series(pd):
    """(ids per alias, [(series name, values)]) of a ProjectData."""
    vids = pd.get_value_ids()
    ids = [list(vids.get_names(alias)) for alias in range(1, vids.size_aliases() + 1)]
    names = list(pd.series_names())
    rows = [list(row) for row in pd.matrix()]
    return ids, list(zip(names, rows))


def write_columns(path, requests, builder):
    """One CSV: ID columns, then one column per series, rows matched by primary ID.

    Vectors may cover different rows (e.g. SSX also has component 0); a
    value a vector does not have is left empty.
    """
    order, ids_of, columns = [], {}, []
    n_aliases = 0
    for title, kwargs in requests:
        pd = attempt(os.path.basename(path) + ": " + title, builder.create, **kwargs)
        if pd is None:
            continue
        ids, data = series(pd)
        n_aliases = max(n_aliases, len(ids))
        for i, key in enumerate(ids[0]):
            if key not in ids_of:
                order.append(key)
                ids_of[key] = [alias[i] for alias in ids]
        for name, values in data:
            column = {key: values[i] for i, key in enumerate(ids[0])}
            columns.append((title if len(data) == 1 else title + " " + name, name, column))
    if not columns:
        log("nothing to write for " + os.path.basename(path))
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id" + str(i + 1) for i in range(n_aliases)] +
                        [title for title, _, _ in columns])
        writer.writerow([""] * n_aliases + [name for _, name, _ in columns])
        for key in order:
            ids = ids_of[key] + [""] * (n_aliases - len(ids_of[key]))
            writer.writerow(ids + [repr(float(column[key])) if key in column else ""
                                   for _, _, column in columns])
    log("wrote " + os.path.basename(path))


def export_model(project, label, number, a, has_prediction, y_name):
    builder = project.data_builder()
    prefix = os.path.join(OUT, label + "_")
    comps = list(range(1, a + 1))
    y = {"variable": y_name}

    write_columns(prefix + "components.csv", [
        (name, {"name": name, "model": number})
        for name in ("R2X", "R2Xcum", "R2Y", "R2Ycum", "Q2", "Q2cum",
                     "SSX", "SSY", "Eig", "SDt", "RMSEcv")
    ], builder)

    write_columns(prefix + "observations.csv", [
        ("CVGroups", {"name": "CVGroups", "model": number}),
        ("YVar", dict(name="YVar", model=number, transformed=False, scaled=False, **y)),
        ("YPred", dict(name="YPred", model=number, comp=a, transformed=False, scaled=False, **y)),
        ("YPredcv", dict(name="YPredcv", model=number, comp=a, **y)),
    ] + [("t[%d]" % c, {"name": "t", "model": number, "comp": c}) for c in comps] + [
        ("T2Range", {"name": "T2Range", "model": number, "comp": (1, a)}),
        ("DModX[%d]" % a, {"name": "DModX", "model": number, "comp": a}),
        ("DModX[0]", {"name": "DModX", "model": number, "comp": 0}),
    ], builder)

    write_columns(prefix + "variables.csv", [
        ("Xavg", {"name": "Xavg", "model": number}),
        ("Xws", {"name": "Xws", "model": number}),
    ] + [("w[%d]" % c, {"name": "w", "model": number, "comp": c}) for c in comps]
      + [("w*[%d]" % c, {"name": "w*", "model": number, "comp": c}) for c in comps]
      + [("p[%d]" % c, {"name": "p", "model": number, "comp": c}) for c in comps] + [
        ("CoeffCS", dict(name="CoeffCS", model=number, comp=a, **y)),
        ("Coeff", dict(name="Coeff", model=number, comp=a, **y)),
        ("VIP", {"name": "VIP", "model": number, "comp": a}),
        ("MPowX", {"name": "MPowX", "model": number, "comp": a}),
    ], builder)

    error_vectors = ["RMSEE", "RMSEcv", "MBEE", "MBEcv"]
    if has_prediction:
        error_vectors += ["RMSEP", "MBEP"]
    write_columns(prefix + "errors.csv",
                  [("Yavg", {"name": "Yavg", "model": number}),
                   ("Yws", {"name": "Yws", "model": number})] +
                  [(name, {"name": name, "model": number, "comp": a}) for name in error_vectors],
                  builder)

    if has_prediction:
        write_columns(prefix + "prediction.csv", [
            ("YVarPS", dict(name="YVarPS", model=number, transformed=False, scaled=False, **y)),
            ("YPredPS", dict(name="YPredPS", model=number, comp=a, transformed=False,
                             scaled=False, **y)),
        ] + [("tPS[%d]" % c, {"name": "tPS", "model": number, "comp": c}) for c in comps] + [
            ("T2RangePS", {"name": "T2RangePS", "model": number, "comp": (1, a)}),
            ("DModXPS[%d]" % a, {"name": "DModXPS", "model": number, "comp": a}),
            ("DModXPS+[%d]" % a, {"name": "DModXPS+", "model": number, "comp": a}),
        ], builder)

    rows = []
    for probability in PROBABILITIES:
        for name, kwargs in (("T2Crit", {"comp": a}),
                             ("DModXCrit", {"comp": a, "normalized_dcrit": False}),
                             ("DModXCrit normalized", {"comp": a, "normalized_dcrit": True})):
            pd = attempt(label + " limits " + name, builder.limits, name.split()[0], number,
                         probability, **kwargs)
            if pd is not None:
                ids, data = series(pd)
                for series_name, values in data:
                    for i, value in enumerate(values):
                        rows.append([name, probability, series_name,
                                     ids[0][i] if ids else "", repr(float(value))])
    with open(prefix + "limits.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["limit", "probability", "series", "id", "value"])
        writer.writerows(rows)


def dmodx_options(project, number):
    options = project.get_model_options(number)
    return {key: attempt("option " + key, getattr, options, key)
            for key in ("normalize_DModX", "weighted_DModX", "significance_for_DModX_T2",
                        "cross_validation_rounds", "adjusted_R2")}


def export_unweighted_dmodx(project, label, number, a, has_prediction):
    """DModX without modelling-power weighting (the default is weighted)."""
    options = project.get_model_options(number)
    weighted = options.weighted_DModX
    options.weighted_DModX = False
    options.apply()
    try:
        builder = project.data_builder()
        requests = [("DModX[%d]" % a, {"name": "DModX", "model": number, "comp": a}),
                    ("DModX[0]", {"name": "DModX", "model": number, "comp": 0})]
        write_columns(os.path.join(OUT, label + "_dmodx_unweighted.csv"), requests, builder)
        if has_prediction:
            write_columns(os.path.join(OUT, label + "_dmodxps_unweighted.csv"),
                          [("DModXPS[%d]" % a, {"name": "DModXPS", "model": number, "comp": a})],
                          builder)
    finally:
        options = project.get_model_options(number)
        options.weighted_DModX = weighted
        options.apply()


def export_dataset(project, name, path):
    data = project.get_dataset(name)
    obs = data.get_obs_names()
    variables = data.get_var_names()
    values = [list(row) for row in data.get_data()]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["obs_id"] + list(variables))
        for i, row in enumerate(values):
            writer.writerow([obs[i]] + [repr(float(v)) for v in row])
    log("wrote " + os.path.basename(path))


# --- main ------------------------------------------------------------------

def run():
    os.makedirs(OUT, exist_ok=True)
    project_path = os.path.join(OUT, "simca_reference.usp")
    umetrics.simca.ProjectHandler.close_all_projects()
    if os.path.isfile(project_path):
        os.remove(project_path)
    project = umetrics.simca.ProjectHandler.create_project(project_path)

    header, pixels = read_table("pixels")
    _, means = read_table("roi_means")
    _, test = read_table("pixels_test")
    create_dataset(project, "pixels", header, interleave_by_fold(pixels))
    create_dataset(project, "roi_means", header, means)
    create_dataset(project, "pixels_test", header, test)
    umetrics.SimcaApp.set_active_project(project)

    manifest = {
        "simca_version": attempt("version", lambda: list(umetrics.SimcaApp.get_version())),
        "pixel_order": "interleaved: row k from round-robin fold k mod 7",
        "filters": FILTER_SETTINGS,
        "scaling": {"X": "ctr", TARGET: "uv"},
        "models": {},
    }

    for label, (table, kind, components, has_prediction) in MODELS.items():
        log("model " + label)
        dataset, filtered = table, None
        if kind is not None:
            filtered = attempt(label + ": filter " + kind, filtered_dataset, project, table,
                               kind, label)
            if filtered is None:
                continue
            dataset = filtered
            attempt(label + ": export filtered", export_dataset, project, dataset,
                    os.path.join(OUT, label + "_filtered.csv"))
        number = attempt(label + ": create model", build_model, project, label, table,
                         filtered)
        if number is None:
            continue
        a = attempt(label + ": fit", fit, project, number, components)
        if a is None:
            continue
        if components > 0 and a != components:
            errors.append({"step": label + ": fit",
                           "error": "fitted %d components, asked for %d" % (a, components)})
        if has_prediction:
            ps = project.create_predictionset(number)
            attempt(label + ": prediction set", ps.as_dataset, "pixels_test")
            attempt(label + ": save prediction set", ps.save, "PS_" + label)
            attempt(label + ": use prediction set", project.set_predictionset, "PS_" + label)
        info = project.get_model_info(number)
        manifest["models"][label] = {
            "number": number,
            "simca_name": info.name,
            "dataset": dataset,
            "observations": info.observations,
            "x_variables": info.Xvariables,
            "components": a,
            "autofit": components < 0,
            "options": dmodx_options(project, number),
        }
        attempt(label + ": export", export_model, project, label, number, a, has_prediction,
                TARGET)
        attempt(label + ": unweighted DModX", export_unweighted_dmodx, project, label, number,
                a, has_prediction)

    project.save()
    manifest["errors"] = errors
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    log("done, %d step(s) failed; see simca_exports/manifest.json" % len(errors))


if __name__ == "__main__":
    run()
