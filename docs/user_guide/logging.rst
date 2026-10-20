Logging
=======

engiNIR logs through `loguru <https://github.com/Delgan/loguru>`_. As a
library it is **silent by default**: importing ``enginir`` adds no log sinks,
creates no files and installs no exception hooks, so it never changes the
logging of the script, notebook or application that uses it. Errors still
reach you as ordinary Python exceptions.

Logging is switched on explicitly with :func:`enginir.enable_logging`. The HSI
viewer does this automatically when it starts.

Log levels
----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Level
     - What is logged
   * - **ERROR**
     - Unrecoverable failures: file not found, corrupt data that halts loading
   * - **WARNING**
     - Recoverable fallbacks: skipped corrupt chunk, missing metadata, fallback calibration
   * - **INFO**
     - High-level operations: file loaded (format, shape, path), model fitted, project saved
   * - **DEBUG**
     - Internals: preprocessing steps applied, cache invalidation, signal emission, ROI appended

Telemetry rule
--------------

enginir treats logging as part of its error-reporting contract, not as optional
debug output.  The viewer Log panel and the persisted log file are used as the
primary telemetry trail when users report failures.

That means:

- modules must log failures with ``logger.error(...)`` immediately before
  raising user-facing exceptions
- recoverable fallbacks and epsilon guards should emit ``logger.warning(...)``
- logger messages should include enough context to diagnose the problem later:
  file paths, ids, shapes, parameter values, thresholds, or sample indices

If code raises without logging first, the exception may still surface to the
caller, but the saved log will be incomplete and much less useful as an error
report.

Enabling log output
-------------------

.. code-block:: python

   import enginir

   enginir.enable_logging()                  # WARNING and above to stderr
   enginir.enable_logging("DEBUG")           # everything to stderr
   enginir.enable_logging("INFO", file="analysis.log")   # plus a log file
   enginir.enable_logging(file=True)         # log file at the default location
   enginir.disable_logging()                 # back to silent

``enable_logging`` arguments:

``level``
    Minimum level printed to stderr (default ``"WARNING"``).
``file``
    ``False`` (default), ``True`` for the default log file, or a path. The
    default log file is the first writable of ``$ENGINIR_LOG_FILE`` (if that
    environment variable is set), ``~/.enginir/enginir.log`` and
    ``/tmp/enginir/enginir.log``. Files rotate at 10 MB; three old
    files are kept.
``file_level``
    Minimum level written to the file (default ``"INFO"``).
``catch_uncaught``
    Also log uncaught exceptions of the whole process at CRITICAL level
    (default ``False``). The previous exception hooks still run, so tracebacks
    are printed as usual. :func:`enginir.disable_logging` restores the
    previous hooks.

Calling ``enable_logging`` again replaces the previous configuration. If
Loguru's default stderr handler is still installed, it is replaced by an
identical handler that skips engiNIR records, so engiNIR messages are not
printed twice and other Loguru messages are unaffected.

Using the logger in your own code
----------------------------------

Modules inside the engiNIR package obtain their logger with
``get_logger(__name__)``:

.. code-block:: python

   from enginir._logging import get_logger
   logger = get_logger(__name__)

   def my_custom_loader(path):
       logger.info("Loading custom format from {}", path)
       ...

Only records of modules whose ``__name__`` starts with ``enginir`` follow the
configuration of :func:`enginir.enable_logging` and appear in the engiNIR log
file and the viewer log panel. Code in your own package logs under its own
name and needs its own Loguru sink, e.g. ``logger.add(sys.stderr)``.

Viewer logging
--------------

``enginir-viewer``, :func:`enginir.viewer.open_viewer` and
:func:`enginir.viewer.open_project` call ``enable_logging("WARNING", file=True,
catch_uncaught=True)`` unless logging was already enabled. Uncaught exceptions,
including exceptions raised inside Qt event handlers and background jobs, are
therefore written to the log file with their traceback. The exception hook also
prevents PyQt from aborting the process when an exception escapes a Qt slot.
The viewer therefore logs to ``~/.enginir/enginir.log`` (or the file named by
``ENGINIR_LOG_FILE``). **Help > Log File...** shows the path of the active log
file, copies it to the clipboard (**Copy Path**) and opens its folder in the
file manager (**Open Location**; under WSL in Windows Explorer, with the file
selected).

A **Log** dock widget at the bottom of the viewer window shows all ``enginir``
messages at DEBUG level and above in real time (read-only, 1000-line ring
buffer). It can be hidden via the **View** menu.

What gets logged where
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Module
     - Level
     - Examples
   * - ``enginir.io.*``
     - INFO
     - ``Loaded HDF5: shape (100, 256) from data.h5``
   * - ``enginir.io.eiff_io``
     - WARNING
     - ``Skipping corrupt EIFF PROP chunk: ...``
   * - ``enginir.io.eiff_io``
     - DEBUG
     - ``EIFF timestamp parse failed``
   * - ``enginir.preprocessing.*``
     - WARNING
     - ``SNV: 1 spectrum(a) are (near-)constant with std < 1e-10; dividing by 1e-10 instead``
   * - ``enginir.models.*``
     - INFO
     - ``PLSRegression fitted: n_components=6, X=(100, 256), y=(100,)``
   * - ``enginir.viewer._state``
     - DEBUG
     - ``frame -> 42 (cache invalidated)``
