Installation
============

Requirements
------------

- Python >= 3.13 and < 3.14
- `uv <https://docs.astral.sh/uv/>`_ (recommended)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/TUstudents/engiNIR.git
   cd engiNIR

   # runtime dependencies only
   uv sync

   # with development tools (pytest, ruff, pyright)
   uv sync --extra dev

   # with documentation tools (sphinx, sphinx-rtd-theme, myst-parser)
   uv sync --extra docs

   # everything
   uv sync --extra dev --extra docs

Running tests
-------------

.. code-block:: bash

   uv run ruff check src/enginir tests
   uv run pyright src/enginir
   uv run pytest

Building the docs
-----------------

.. code-block:: bash

   uv run python -m sphinx -b html docs docs/_build/html
   # open docs/_build/html/index.html
