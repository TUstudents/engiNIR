enginir._logging
================

engiNIR is silent when imported. Applications opt in with
:func:`enginir.enable_logging` (also available as ``enginir.enable_logging``);
the viewer does so at startup. See :doc:`../user_guide/logging`.

.. automodule:: enginir._logging
   :members: enable_logging, disable_logging, logging_enabled, log_file_path,
             default_log_files, get_logger, add_log_sink, remove_log_sink
   :no-index:

``get_logger(__name__)`` is the standard module-level entry point for engiNIR
modules and extensions.
