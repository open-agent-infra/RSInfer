"""Native I/O failure is bounded and its owned process is reaped."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from state.cache.io import StoreProcess


@pytest.mark.covers(edges=("call",), kinds=("logic",))
def test_a_stalled_native_client_is_killed_and_reaped(monkeypatch: pytest.MonkeyPatch) -> None:
    real = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def stalled(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        # The child must use the newly applied snapshot, not an inherited old PYTHONPATH.
        assert Path(args[0][1]).is_absolute() and Path(args[0][1]).name == "io.py"
        child = cast("subprocess.Popen[bytes]", real([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs))
        children.append(child)
        return child

    monkeypatch.setattr("state.cache.io.subprocess.Popen", stalled)
    client = StoreProcess({}, 0.1, 1024)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        client.call("get", "a-key")
    assert time.monotonic() - start < 3
    assert children and children[0].poll() is not None
    assert client.closed and client.process is None and client.socket is None
    with pytest.raises(ConnectionError, match="reload"):
        client.call("get", "a-key")
    client.close()


@pytest.mark.covers(edges=("call",), kinds=("logic",))
def test_normal_shutdown_acknowledges_close_and_exits_without_being_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    real = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []
    program = """
import socket, struct, sys, msgspec
s = socket.socket(fileno=int(sys.argv[1]))
def read(n):
    data = b''
    while len(data) < n:
        data += s.recv(n - len(data))
    return data
def receive():
    return msgspec.msgpack.decode(read(struct.unpack('!Q', read(8))[0]))
def send(value):
    data = msgspec.msgpack.encode(value)
    s.sendall(struct.pack('!Q', len(data)) + data)
receive()
send({'value': None})
assert receive() == ['get', 'key']
send({'value': b'found'})
assert receive() == ['close', None]
send({'value': None})
s.close()
"""

    def native(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = cast("subprocess.Popen[bytes]", real([sys.executable, "-c", program, str(args[0][2])], **kwargs))
        children.append(child)
        return child

    monkeypatch.setattr("state.cache.io.subprocess.Popen", native)
    client = StoreProcess({}, 2, 1024)
    assert client.call("get", "key") == b"found"
    client.close()
    assert client.closed and children[0].returncode == 0
