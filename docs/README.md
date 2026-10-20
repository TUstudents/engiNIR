# engiNIR documentation

This directory contains the publishable documentation for the engiNIR release.
The release version is read from the project metadata in `pyproject.toml`.

Human-facing product name: `engiNIR`

Technical identifiers that stay unchanged:

- Python package: `enginir`
- CLI: `enginir-viewer`
- source tree: `src/enginir/`

## Published documentation surface

The main published docs are organized into:

- `index.rst`
  the documentation landing page
- `user_guide/`
  installation, workflows, viewer use, modeling, validation, and I/O
- `api/`
  curated API reference for the public package surface
- top-level references such as:
  - `eiff_format.md`
  - `spc_format.md`
  - `hdf5_format.md`
  - `hsi_viewer.md`
  - `workspace_domain_model.md`
  - `viewer_architecture.md`

`examples/` holds scripts included in the guide; `tests/test_docs_examples.py`
runs them.

## Build

```bash
uv sync --extra docs
uv run python -m sphinx -W -b html docs docs/_build/html
```

## Documentation goal

The docs should present engiNIR as a durable HSI chemometric workbench:

- clear to install
- clear to start using
- clear on current workflow strengths
- honest about current product scope
- precise enough for API and file-format reference use
