"""A unit's build step in a process of its own (change.md §4.6):
``python -m core.change.build <module>:<function> <seconds>``, started by the participant as the
leader of a session of its own (``Participant._run_builds``).

SIGTERM ends the step with ``SystemExit``, so its cleanup runs: PyTorch's extension builder deletes
the lock it holds on its build directory, and kills the ``ninja`` it waits for. The step gets
SIGTERM from the participant when the change is undone, and from the kernel when the participant's
process ends, however it ends (``PR_SET_PDEATHSIG``; should that process end before the step set
it, the step builds to the end, and lets its locks go then).

What the step started ends with it, by session and not by process group: ``ninja`` runs each
compiler in a group of its own. On SIGTERM the step stops the rest of its own session, even if the
parent PID is still visible: Linux can deliver the parent-death signal when the spawning thread
exits. The participant also enforces cleanup on an undo (``_stop_build``). The step sends SIGTERM
to the rest of its session and SIGKILL to what is left of it after the given seconds
(``change.build_stop_grace_s``).
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import os
import signal
import sys
import time
from types import FrameType

_PR_SET_PDEATHSIG = 1


def signal_session(sid: int, sig: int, *, leader: bool = True) -> int:
    """Send ``sig`` (0: nothing, only count) to every live process of session ``sid``
    (``/proc/<pid>/stat``), with or without its leader; how many."""
    sent = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or (not leader and int(entry.name) == sid):
            continue
        try:
            with open(f"/proc/{entry.name}/stat", "rb") as f:
                fields = f.read().rsplit(b")", 1)[1].split()  # after "pid (comm)": state, ppid, pgrp, session
        except OSError:
            continue
        if int(fields[3]) == sid and fields[0] != b"Z":
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(entry.name), sig)
                sent += 1
    return sent


def run(entry: str, grace_s: float) -> None:
    sid, parent = os.getsid(0), os.getppid()
    owns_session = os.getpid() == sid
    stopping = False

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # once: a second one would cut the cleanup short
        if owns_session:  # the compilers first, then (SystemExit) the lock on what they write
            signal_session(sid, signal.SIGTERM, leader=False)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    ctypes.CDLL(None).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    module, _, function = entry.partition(":")
    try:
        getattr(importlib.import_module(module), function)()
    finally:
        if owns_session and (stopping or os.getppid() != parent):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal_session(sid, signal.SIGTERM, leader=False)
            deadline = time.monotonic() + grace_s
            while signal_session(sid, 0, leader=False) and time.monotonic() < deadline:
                time.sleep(0.02)
            for _ in range(10):  # until nothing of it is left: a process may fork meanwhile
                if not signal_session(sid, signal.SIGKILL, leader=False):
                    break
                time.sleep(0.02)


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]))
