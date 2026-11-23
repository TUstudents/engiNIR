"""The time since the operating system started this process, for the startup log lines.

A user's log then shows the whole start, including the time before Python ran
any of engiNIR (a cold start, an antivirus scan, a busy machine). Standard library only:
the launcher imports this before anything else.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from .._logging import get_logger

logger = get_logger(__name__)

#: ``time.perf_counter()`` at the moment the process started (set by :func:`mark_start`).
_ORIGIN: float | None = None


def process_age() -> float | None:
    """Seconds since the operating system started this process, or ``None`` if unknown."""
    try:
        if sys.platform == "win32":
            return _age_windows()
        if sys.platform.startswith("linux"):
            return _age_linux()
        return _age_ps()
    except Exception as exc:  # a log line must never stop the start
        logger.warning("The process start time is not known: {!r}", exc)
        return None


def _age_windows() -> float:
    if sys.platform != "win32":  # also narrows ctypes for the type checker
        raise OSError("GetProcessTimes is Windows only")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Without argtypes ctypes passes the process handle (the 64-bit pseudo handle -1)
    # as a C int, which overflows.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    filetime_p = ctypes.POINTER(wintypes.FILETIME)
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, filetime_p, filetime_p, filetime_p,
                                         filetime_p]
    kernel32.GetSystemTimeAsFileTime.restype = None
    kernel32.GetSystemTimeAsFileTime.argtypes = [filetime_p]
    times = [wintypes.FILETIME() for _ in range(4)]  # creation, exit, kernel, user
    if not kernel32.GetProcessTimes(kernel32.GetCurrentProcess(),
                                    *(ctypes.byref(t) for t in times)):
        raise OSError(f"GetProcessTimes failed (error {ctypes.get_last_error()})")
    now = wintypes.FILETIME()
    kernel32.GetSystemTimeAsFileTime(ctypes.byref(now))

    def ticks(value: wintypes.FILETIME) -> int:  # 100 ns since 1601
        return (value.dwHighDateTime << 32) | value.dwLowDateTime

    return (ticks(now) - ticks(times[0])) / 1e7


def _age_linux() -> float:
    with open("/proc/self/stat", encoding="ascii") as handle:
        stat = handle.read()
    # The fields after the command name (in parentheses, which may contain spaces);
    # the start time is field 22, in clock ticks since the boot.
    start_ticks = int(stat[stat.rindex(")") + 2:].split()[19])
    with open("/proc/uptime", encoding="ascii") as handle:
        uptime = float(handle.read().split()[0])
    return uptime - start_ticks / os.sysconf("SC_CLK_TCK")


def _age_ps() -> float:
    """macOS and other systems: ``ps -o etime`` ([[dd-]hh:]mm:ss), to the second."""
    out = subprocess.run(["ps", "-o", "etime=", "-p", str(os.getpid())], capture_output=True,
                         text=True, timeout=5, check=True).stdout.strip()
    days, _, clock = out.rpartition("-")
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds + (int(days) * 86400 if days else 0)


def mark_start() -> None:
    """Note when the process started and log how long it took to reach the launcher."""
    global _ORIGIN
    age = process_age()
    now = time.perf_counter()
    _ORIGIN = now - (age if age is not None else 0.0)
    if age is None:
        logger.info("Startup timing: the process start time is not known; times below count "
                    "from the launcher")
    else:
        logger.info("Startup timing: the launcher ran {:.2f}s after the process started "
                    "({} modules loaded)", age, len(sys.modules))


def since_start() -> float:
    """Seconds since the process started (since :func:`mark_start` if that is unknown)."""
    if _ORIGIN is None:
        mark_start()
    return time.perf_counter() - (_ORIGIN or 0.0)


__all__ = ["mark_start", "process_age", "since_start"]
