"""Start menu and desktop shortcuts for the viewer on Windows (plan windows-1.5.2).

``enginir-viewer --install-shortcut`` writes an "engiNIR Viewer" shortcut
(``.lnk``) to the Start menu (with ``--desktop`` also to the desktop):
target the windowless ``enginir-viewer.exe`` launcher of this installation,
the engiNIR icon and the taskbar id :data:`WINDOWS_APP_ID`, the id the
running viewer gives itself. With the same id, Windows groups the window under
the shortcut, so pinning it to the taskbar pins engiNIR, not ``pythonw.exe``.

The shortcuts are written through the Windows shell's COM interfaces
(``IShellLinkW``, ``IPersistFile``, ``IPropertyStore``) with :mod:`ctypes`,
so no extra package is needed. This module imports no Qt.
"""

from __future__ import annotations

import ctypes
import importlib.resources
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from .._checks import log_and_raise
from .._logging import get_logger

logger = get_logger(__name__)

#: The Windows taskbar groups windows by this id, not by python.exe, and shows
#: the window icon for it (otherwise Python's icon, or none). Shortcuts carry
#: the same id.
WINDOWS_APP_ID = "TUstudents.engiNIR.viewer"

SHORTCUT_NAME = "engiNIR Viewer.lnk"
_DESCRIPTION = "engiNIR viewer for NIR spectra and hyperspectral images"
_LAUNCHER = "enginir-viewer.exe"

# COM identifiers (Windows SDK: ShObjIdl_core.h, ObjIdl.h, propkey.h, KnownFolders.h).
_CLSID_SHELL_LINK = "{00021401-0000-0000-C000-000000000046}"
_IID_SHELL_LINK_W = "{000214F9-0000-0000-C000-000000000046}"
_IID_PERSIST_FILE = "{0000010B-0000-0000-C000-000000000046}"
_IID_PROPERTY_STORE = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_PKEY_APP_USER_MODEL = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
_PID_APP_USER_MODEL_ID = 5
_FOLDERID_PROGRAMS = "{A77F5D77-2E2B-44C3-A6A2-ABA601054A51}"
_FOLDERID_DESKTOP = "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"

# Method slots in the interface vtables (IUnknown takes 0-2).
_RELEASE = 2
_LINK_GET_ARGUMENTS, _LINK_SET_DESCRIPTION, _LINK_SET_WORKING_DIRECTORY = 10, 7, 9
_LINK_SET_ARGUMENTS, _LINK_GET_ICON_LOCATION, _LINK_SET_ICON_LOCATION = 11, 16, 17
_LINK_GET_PATH, _LINK_SET_PATH = 3, 20
_PERSIST_LOAD, _PERSIST_SAVE = 5, 6
_STORE_GET_VALUE, _STORE_SET_VALUE, _STORE_COMMIT = 5, 6, 7

_CLSCTX_INPROC_SERVER = 1
_STGM_READ = 0
_VT_LPWSTR = 31
_MAX_PATH = 32768  # long paths


@dataclass(frozen=True)
class Shortcut:
    """What a ``.lnk`` file points to."""

    target: str
    arguments: str
    icon: str
    app_id: str


def launcher_path() -> Path:
    """The windowless ``enginir-viewer.exe`` of this installation.

    The launcher next to the running Python (a ``uv tool`` or project
    environment) comes first; otherwise the one on ``PATH``.
    """
    beside = Path(sys.executable).parent / _LAUNCHER
    if beside.is_file():
        return beside
    found = shutil.which(_LAUNCHER)
    if found is None:
        log_and_raise("The viewer launcher {} was not found next to {} or on PATH; "
                      "install engiNIR with 'uv tool install enginir'",
                      _LAUNCHER, sys.executable, error=FileNotFoundError)
    return Path(found)


def icon_path() -> Path:
    """The engiNIR icon file of the installed package."""
    return Path(str(importlib.resources.files("enginir._resources").joinpath("enginir.ico")))


def shortcut_folders(*, desktop: bool) -> list[Path]:
    """The user's Start menu programs folder, and the desktop if ``desktop``."""
    if sys.platform != "win32":
        _refuse()
    folders = [_known_folder(_FOLDERID_PROGRAMS)]
    if desktop:
        folders.append(_known_folder(_FOLDERID_DESKTOP))
    return folders


def install_shortcuts(*, desktop: bool = False) -> list[Path]:
    """Write the viewer's shortcut to the Start menu (and the desktop); return the files."""
    if sys.platform != "win32":
        _refuse()
    target, icon = launcher_path(), icon_path()
    written = []
    for folder in shortcut_folders(desktop=desktop):
        path = folder / SHORTCUT_NAME
        write_shortcut(path, target, icon=icon, app_id=WINDOWS_APP_ID,
                       description=_DESCRIPTION, working_dir=Path.home())
        logger.info("Shortcut written: {} -> {}", path, target)
        written.append(path)
    return written


def remove_shortcuts() -> list[Path]:
    """Remove the viewer's shortcuts from the Start menu and the desktop; return the files.

    Only a shortcut with engiNIR's taskbar id is removed; a file of the same
    name that is not ours stays (with a warning).
    """
    if sys.platform != "win32":
        _refuse()
    removed = []
    for folder in shortcut_folders(desktop=True):
        path = folder / SHORTCUT_NAME
        if not path.is_file():
            continue
        if read_shortcut(path).app_id != WINDOWS_APP_ID:
            logger.warning("Not removing {}: it is not an engiNIR shortcut", path)
            continue
        path.unlink()
        logger.info("Shortcut removed: {}", path)
        removed.append(path)
    return removed


def write_shortcut(path: Path, target: Path, *, icon: Path, app_id: str,
                   description: str = "", working_dir: Path | None = None,
                   arguments: str = "") -> None:
    """Write a ``.lnk`` file (replacing one of the same name)."""
    if sys.platform != "win32":  # also narrows ctypes' Windows-only names for pyright
        _refuse()
    with _com():
        link = _create_shell_link()
        try:
            _call(link, _LINK_SET_PATH, [ctypes.c_wchar_p], str(target))
            _call(link, _LINK_SET_ARGUMENTS, [ctypes.c_wchar_p], arguments)
            _call(link, _LINK_SET_DESCRIPTION, [ctypes.c_wchar_p], description)
            _call(link, _LINK_SET_ICON_LOCATION, [ctypes.c_wchar_p, ctypes.c_int], str(icon), 0)
            if working_dir is not None:
                _call(link, _LINK_SET_WORKING_DIRECTORY, [ctypes.c_wchar_p], str(working_dir))
            store = _query(link, _IID_PROPERTY_STORE)
            try:
                buffer = ctypes.create_unicode_buffer(app_id)
                value = _PropVariant(vt=_VT_LPWSTR)
                value.value.pwszVal = ctypes.cast(buffer, ctypes.c_wchar_p)
                _call(store, _STORE_SET_VALUE,
                      [ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant)],
                      ctypes.byref(_app_id_key()), ctypes.byref(value))
                _call(store, _STORE_COMMIT, [])
            finally:
                _release(store)
            path.parent.mkdir(parents=True, exist_ok=True)
            persist = _query(link, _IID_PERSIST_FILE)
            try:
                _call(persist, _PERSIST_SAVE, [ctypes.c_wchar_p, ctypes.c_int], str(path), 1)
            finally:
                _release(persist)
        finally:
            _release(link)


def read_shortcut(path: Path) -> Shortcut:
    """Read target, arguments, icon and taskbar id of a ``.lnk`` file."""
    if sys.platform != "win32":
        _refuse()
    with _com():
        link = _create_shell_link()
        try:
            persist = _query(link, _IID_PERSIST_FILE)
            try:
                _call(persist, _PERSIST_LOAD, [ctypes.c_wchar_p, ctypes.c_uint],
                      str(path), _STGM_READ)
            finally:
                _release(persist)
            text = ctypes.create_unicode_buffer(_MAX_PATH)
            _call(link, _LINK_GET_PATH,
                  [ctypes.c_wchar_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint],
                  text, _MAX_PATH, None, 0)
            target = text.value
            _call(link, _LINK_GET_ARGUMENTS, [ctypes.c_wchar_p, ctypes.c_int], text, _MAX_PATH)
            arguments = text.value
            index = ctypes.c_int()
            _call(link, _LINK_GET_ICON_LOCATION,
                  [ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)],
                  text, _MAX_PATH, ctypes.byref(index))
            icon = text.value
            store = _query(link, _IID_PROPERTY_STORE)
            try:
                value = _PropVariant()
                _call(store, _STORE_GET_VALUE,
                      [ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant)],
                      ctypes.byref(_app_id_key()), ctypes.byref(value))
                app_id = (value.value.pwszVal or "") if value.vt == _VT_LPWSTR else ""
                ctypes.windll.ole32.PropVariantClear(ctypes.byref(value))
            finally:
                _release(store)
        finally:
            _release(link)
    return Shortcut(target, arguments, icon, app_id)


def _refuse() -> NoReturn:
    log_and_raise("Shortcuts are only made on Windows (this is {})", sys.platform,
                  error=OSError)


if sys.platform == "win32":
    import contextlib
    from collections.abc import Iterator

    class _Guid(ctypes.Structure):
        _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                    ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    class _PropertyKey(ctypes.Structure):
        _fields_ = [("fmtid", _Guid), ("pid", ctypes.c_uint32)]

    class _PropVariantValue(ctypes.Union):
        # Large enough for every PROPVARIANT member (two pointers' worth).
        _fields_ = [("pwszVal", ctypes.c_wchar_p), ("_pad", ctypes.c_byte * 16)]

    class _PropVariant(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("_reserved1", ctypes.c_ushort),
                    ("_reserved2", ctypes.c_ushort), ("_reserved3", ctypes.c_ushort),
                    ("value", _PropVariantValue)]

    def _guid(text: str) -> _Guid:
        guid = _Guid()
        result = ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(guid))
        if result != 0:
            log_and_raise("Invalid GUID {} (HRESULT {:#x})", text, result & 0xFFFFFFFF,
                          error=OSError)
        return guid

    def _app_id_key() -> _PropertyKey:
        return _PropertyKey(_guid(_PKEY_APP_USER_MODEL), _PID_APP_USER_MODEL_ID)

    @contextlib.contextmanager
    def _com() -> Iterator[None]:
        # S_OK or S_FALSE (already initialized); a thread in another apartment
        # mode (RPC_E_CHANGED_MODE) can still use the in-process shell link.
        result = ctypes.windll.ole32.CoInitialize(None)
        try:
            yield
        finally:
            if result in (0, 1):
                ctypes.windll.ole32.CoUninitialize()

    def _create_shell_link() -> ctypes.c_void_p:
        link = ctypes.c_void_p()
        result = ctypes.windll.ole32.CoCreateInstance(
            ctypes.byref(_guid(_CLSID_SHELL_LINK)), None, _CLSCTX_INPROC_SERVER,
            ctypes.byref(_guid(_IID_SHELL_LINK_W)), ctypes.byref(link))
        if result != 0:
            log_and_raise("Cannot create a Windows shell link (HRESULT {:#x})",
                          result & 0xFFFFFFFF, error=OSError)
        return link

    def _method(interface: ctypes.c_void_p, slot: int, argtypes: list) -> Any:
        vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
        return prototype(vtable[slot])

    def _call(interface: ctypes.c_void_p, slot: int, argtypes: list, *args) -> None:
        """Call a COM method; a failing HRESULT raises OSError (ctypes.HRESULT)."""
        try:
            _method(interface, slot, argtypes)(interface, *args)
        except OSError as exc:
            log_and_raise("Windows shell call failed: {}", exc, error=OSError, cause=exc)

    def _query(interface: ctypes.c_void_p, iid: str) -> ctypes.c_void_p:
        other = ctypes.c_void_p()
        _call(interface, 0, [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)],
              ctypes.byref(_guid(iid)), ctypes.byref(other))
        return other

    def _release(interface: ctypes.c_void_p) -> None:
        if interface:
            prototype = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
            vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
            prototype(vtable[_RELEASE])(interface)

    def _known_folder(folder_id: str) -> Path:
        text = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(_guid(folder_id)), 0, None, ctypes.byref(text))
        try:
            if result != 0 or not text.value:
                log_and_raise("Cannot find the Windows folder {} (HRESULT {:#x})", folder_id,
                              result & 0xFFFFFFFF, error=OSError)
            return Path(text.value)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(text)
