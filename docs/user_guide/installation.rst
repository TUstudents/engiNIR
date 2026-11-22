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
   touched. The Python files are compiled during the install, so the first
   start is not slower than the others.
3. It adds the tool folder to your ``PATH``, for ``enginir-viewer-console``.
4. It adds "engiNIR Viewer" with the engiNIR icon to the Start menu.

The same steps by hand:

.. code-block:: powershell

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   uv tool install --python 3.13 --compile-bytecode enginir
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

If the start is slow, the lines ``Startup timing`` in the log file say
where the time went, counted from the moment Windows started the process:
until the launcher ran, until the welcome window was shown and until the
first project window was shown. A long time before the launcher ran comes
from outside engiNIR, usually an antivirus program scanning the files of a
new install or a busy computer.

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

pandas is not installed with engiNIR; only
``metrics_report(..., as_dataframe=True)`` needs it. Add it with the extra:

.. code-block:: bash

   pip install "enginir[pandas]"

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
