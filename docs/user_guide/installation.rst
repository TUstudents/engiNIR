Installation
============

Requirements
------------

- Python >= 3.12
- `uv <https://docs.astral.sh/uv/>`_ (recommended)

Install from PyPI
-----------------

.. code-block:: bash

   uv add enginir

With pip:

.. code-block:: bash

   pip install enginir

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

   uv run pytest
   uv run ruff check src/enginir tests docs/examples
   uv run pyright src/enginir

Set ``QT_QPA_PLATFORM=offscreen`` to run the viewer tests on a machine
without a display.

Building the docs
-----------------

.. code-block:: bash

   uv run python -m sphinx -b html docs docs/_build/html
   # open docs/_build/html/index.html
