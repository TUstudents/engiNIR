"""Tests for enginir logging: silent as a library, opt-in for applications."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

import enginir
from enginir._logging import get_logger


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _run_python(code: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "HOME": str(tmp_path / "home"), **env}
    environment.pop("ENGINIR_LOG_FILE", None)
    environment.update(env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )


def test_import_does_not_touch_host_logging(tmp_path):
    result = _run_python(
        """
        import json, sys, threading
        from loguru import logger
        hooks = (sys.excepthook, threading.excepthook)
        import numpy as np
        import enginir
        from enginir.preprocessing import SNV
        SNV().fit_transform(np.ones((2, 5)))  # emits an enginir warning
        logger.info("host message")
        print(json.dumps({
            "handlers": list(logger._core.handlers),
            "hooks_unchanged": hooks == (sys.excepthook, threading.excepthook),
            "threads": [t.name for t in threading.enumerate()],
        }))
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state["handlers"] == [0]
    assert state["hooks_unchanged"]
    assert state["threads"] == ["MainThread"]
    assert "host message" in result.stderr
    assert "SNV" not in result.stderr
    assert not (tmp_path / "home" / ".enginir").exists()


def test_enable_logging_routes_enginir_records_once_and_keeps_host_output(tmp_path):
    log_file = tmp_path / "run.log"
    result = _run_python(
        f"""
        import numpy as np
        from loguru import logger
        import enginir
        from enginir._logging import get_logger
        assert enginir.enable_logging("WARNING", file={str(log_file)!r}) is not None
        enginir.enable_logging("WARNING", file={str(log_file)!r})  # reconfiguring must not duplicate
        get_logger("enginir.probe").info("info probe")
        get_logger("enginir.probe").warning("warning probe")
        logger.info("host message")
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("warning probe") == 1
    assert "info probe" not in result.stderr
    assert result.stderr.count("host message") == 1
    written = log_file.read_text()
    assert "info probe" in written and "warning probe" in written
    assert "host message" not in written


def test_catch_uncaught_logs_to_file_without_duplicating_the_traceback(tmp_path):
    log_file = tmp_path / "crash.log"
    result = _run_python(
        """
        import enginir
        enginir.enable_logging(file=True, catch_uncaught=True)
        raise RuntimeError("sentinel-crash")
        """,
        tmp_path,
        ENGINIR_LOG_FILE=str(log_file),
    )
    assert result.returncode == 1
    assert result.stderr.count("Traceback (most recent call last)") == 1
    written = log_file.read_text()
    assert "Uncaught exception" in written and "sentinel-crash" in written


def test_disable_logging_restores_hooks_and_mutes_the_package(tmp_path):
    enginir.disable_logging()  # earlier viewer tests may have started a logging session
    hooks = (sys.excepthook, __import__("threading").excepthook)
    path = enginir.enable_logging("ERROR", file=tmp_path / "x.log", catch_uncaught=True)
    try:
        assert path == tmp_path / "x.log"
        assert enginir._logging.logging_enabled()
        assert sys.excepthook is not hooks[0]
    finally:
        enginir.disable_logging()
    assert (sys.excepthook, __import__("threading").excepthook) == hooks
    assert not enginir._logging.logging_enabled()
    assert enginir._logging.log_file_path() is None

    records: list = []
    sink_id = logger.add(records.append, level="DEBUG")
    try:
        from enginir.preprocessing import SNV

        SNV().fit_transform(np.ones((2, 5)))  # would warn about constant spectra
    finally:
        logger.remove(sink_id)
    assert records == []


def test_unwritable_log_file_is_reported_and_skipped(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    try:
        assert enginir.enable_logging("CRITICAL", file=blocker / "enginir.log") is None
        assert enginir._logging.logging_enabled()
    finally:
        enginir.disable_logging()


def test_get_logger_attributes_records_to_the_given_module():
    records: list = []
    sink_id = logger.add(lambda msg: records.append(msg.record), level="INFO", filter="enginir")
    try:
        get_logger("enginir.external_extension").info("extension message")
    finally:
        logger.remove(sink_id)
    assert [record["name"] for record in records] == ["enginir.external_extension"]


def test_errors_are_logged_before_they_are_raised():
    from enginir.preprocessing import SNV

    records: list = []
    sink_id = logger.add(lambda msg: records.append(msg.record), level="ERROR")
    try:
        try:
            SNV().fit_transform(np.ones((2, 1)))
        except ValueError:
            pass
    finally:
        logger.remove(sink_id)
    assert [record["name"] for record in records] == ["enginir.preprocessing.scatter"]


def test_viewer_session_enables_file_logging_and_exception_hook(qapp):
    from enginir.viewer._app import enable_viewer_logging

    enginir.disable_logging()
    previous_hook = sys.excepthook
    try:
        enable_viewer_logging()
        assert enginir._logging.log_file_path() == Path(os.environ["ENGINIR_LOG_FILE"])
        assert sys.excepthook is not previous_hook
    finally:
        enginir.disable_logging()


def test_background_job_failures_are_logged_with_traceback(qapp):
    from enginir.viewer._jobs import BackgroundWorker

    def failing_job():
        raise RuntimeError("job exploded")

    records: list = []
    sink_id = logger.add(lambda msg: records.append(msg.record), level="ERROR", filter="enginir")
    try:
        BackgroundWorker(failing_job).run()
    finally:
        logger.remove(sink_id)
    assert records and records[0]["exception"] is not None
    assert "job exploded" in str(records[0]["exception"].value)


def test_default_log_file_falls_back_to_tmp_when_home_is_unwritable(monkeypatch):
    monkeypatch.delenv("ENGINIR_LOG_FILE", raising=False)
    monkeypatch.setattr(enginir._logging.Path, "home",
                        lambda: Path("/nonexistent_home_xyz_abc_does_not_exist"))
    try:
        assert enginir.enable_logging("CRITICAL", file=True) == Path("/tmp/enginir/enginir.log")
    finally:
        enginir.disable_logging()


def test_failed_enable_logging_leaves_enginir_muted():
    from enginir.utils import flatten_cube

    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), level="DEBUG")
    try:
        with pytest.raises(ValueError):
            enginir.enable_logging("NOT_A_LEVEL")
        with pytest.raises(ValueError):
            flatten_cube([])  # logs an error, which must stay muted
    finally:
        logger.remove(sink_id)
    assert not any("X_3d" in record for record in records)
    assert not enginir._logging.logging_enabled()


# The thread ends with SystemExit on purpose; pytest's own thread hook, which ours chains
# to, reports it.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_exit_is_not_logged_and_later_hooks_survive_disable(tmp_path):
    import threading

    log_file = tmp_path / "threads.log"
    original_hook = sys.excepthook
    enginir.enable_logging("CRITICAL", file=log_file, catch_uncaught=True)
    try:
        thread = threading.Thread(target=sys.exit, args=(0,))
        thread.start()
        thread.join()

        def application_hook(*args):
            pass

        sys.excepthook = application_hook  # installed after ours: must survive disable
        enginir.disable_logging()
        assert sys.excepthook is application_hook
    finally:
        enginir.disable_logging()
        sys.excepthook = original_hook
    assert "Uncaught exception" not in log_file.read_text()
