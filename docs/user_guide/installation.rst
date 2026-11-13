Installation
============

Install the viewer on Windows
-----------------------------

To use the viewer only, without writing Python: open **PowerShell** (Start
menu, type "PowerShell") and paste this one line:

.. code-block:: powershell

   powershell -ExecutionPolicy ByPass -c "irm https://github.com/TUstudents/engiNIR/releases/latest/download/install-enginir-for-windows.ps1 | iex"

Then start **engiNIR Viewer** from the Start menu. No administrator rights
are needed, and everything stays in your user profile. The script, attached
to each engiNIR release, does four things:

1. It installs `uv <https://docs.astral.sh/uv/>`_, the tool that installs
   engiNIR, with Astral's own installer, unless uv is already there.
2. It installs engiNIR as a uv *tool*: in a folder of its own, with its own
   Python 3.13, which uv downloads. Other Python installations are not
   touched.
3. It adds the tool folder to your ``PATH``, for ``enginir-viewer-console``.
4. It adds "engiNIR Viewer" with the engiNIR icon to the Start menu.

The same steps by hand:

.. code-block:: powershell

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   uv tool install --python 3.13 enginir
   uv tool update-shell
   enginir-viewer-console --install-shortcut      # add --desktop for a desktop icon

**Update** by running the one-line command again, or with
``uv tool upgrade enginir`` (close the viewer first). The shortcut stays
valid.

**Uninstall** with the two commands below, in this order (the first removes
the shortcuts, the second engiNIR):

.. code-block:: powershell

   enginir-viewer-console --remove-shortcut
   uv tool uninstall enginir

If the viewer does not start, start ``enginir-viewer-console`` in a
PowerShell window to see its messages, and look in the log file
``%USERPROFILE%\.enginir\enginir.log`` (:doc:`logging`). The viewer itself
is described in :doc:`viewer`.

On Linux and macOS, ``uv tool install enginir`` installs the viewer the same
way (``enginir-viewer`` on the ``PATH``).

Requirements
------------

- Python >= 3.12
- `uv <https://docs.astral.sh/uv/>`_ (recommended)

Install from PyPI
-----------------

To use engiNIR from Python, add it to a project:

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
