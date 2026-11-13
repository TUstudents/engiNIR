"""The viewer on Windows (plan windows-1.5.2, Phase 3): shortcuts and the launcher without a console.

Shortcut tests write into a temporary folder, never the user's Start menu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import enginir
from enginir._logging import get_logger
from enginir.viewer import __main__ as launcher
from enginir.viewer import _windows_shell as shell

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows shell links")


@windows_only
def test_a_shortcut_reads_back_with_its_target_icon_and_taskbar_id(tmp_path):
    target = Path(sys.executable)
    path = tmp_path / "sub folder" / "engiNIR Viewer.lnk"
    shell.write_shortcut(path, target, icon=shell.icon_path(), app_id=shell.WINDOWS_APP_ID,
                         description="test", working_dir=tmp_path, arguments="--version")
    read = shell.read_shortcut(path)
    assert Path(read.target) == target
    assert read.arguments == "--version"
    assert Path(read.icon) == shell.icon_path() and shell.icon_path().is_file()
    assert read.app_id == shell.WINDOWS_APP_ID


@windows_only
def test_install_points_at_the_windowless_launcher_and_remove_keeps_foreign_files(
        tmp_path, monkeypatch):
    start_menu, desktop = tmp_path / "Programs", tmp_path / "Desktop"
    monkeypatch.setattr(shell, "shortcut_folders", lambda *, desktop: (
        [start_menu, tmp_path / "Desktop"] if desktop else [start_menu]))

    assert shell.install_shortcuts() == [start_menu / shell.SHORTCUT_NAME]
    link = shell.read_shortcut(start_menu / shell.SHORTCUT_NAME)
    assert Path(link.target).name == "enginir-viewer.exe" and Path(link.target).is_file()
    assert link.app_id == shell.WINDOWS_APP_ID

    # A shortcut of the same name that is not ours stays.
    shell.write_shortcut(desktop / shell.SHORTCUT_NAME, Path(sys.executable),
                         icon=shell.icon_path(), app_id="Someone.Else")
    assert shell.remove_shortcuts() == [start_menu / shell.SHORTCUT_NAME]
    assert not (start_menu / shell.SHORTCUT_NAME).exists()
    assert (desktop / shell.SHORTCUT_NAME).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="refusal off Windows")
@pytest.mark.parametrize("option", ["--install-shortcut", "--remove-shortcut"])
def test_shortcut_options_refuse_cleanly_off_windows(option, capsys):
    with pytest.raises(SystemExit) as exit_:
        launcher.main([option])
    assert exit_.value.code == 1
    enginir.disable_logging()


def test_without_a_console_logging_still_writes_its_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", None)
    log_file = tmp_path / "enginir.log"
    try:
        assert enginir.enable_logging("INFO", file=log_file) == log_file
        get_logger("enginir.test").warning("seen only in the file")
    finally:
        enginir.disable_logging()
    assert "seen only in the file" in log_file.read_text(encoding="utf-8")


def test_without_a_console_messages_and_a_failed_start_reach_a_message_box(monkeypatch):
    shown: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(launcher, "_message_box",
                        lambda title, text, *, error: shown.append((title, text, error)))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    with pytest.raises(SystemExit) as exit_:
        launcher.main(["--version"])
    assert exit_.value.code == 0
    assert shown[-1] == ("engiNIR Viewer", f"engiNIR {enginir.__version__}", False)

    with pytest.raises(SystemExit) as exit_:
        launcher.main(["--desktop"])
    assert exit_.value.code == 2
    assert shown[-1][2] and "--desktop needs --install-shortcut" in shown[-1][1]

    def fail(_argv):
        raise RuntimeError("no display")

    monkeypatch.setattr(launcher, "_main", fail)
    with pytest.raises(SystemExit) as exit_:
        launcher.main([])
    assert exit_.value.code == 1
    title, text, error = shown[-1]
    assert error and "could not start" in title and "no display" in text and "log file" in text
    enginir.disable_logging()
