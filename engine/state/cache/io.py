"""Bounded Mooncake calls in an owned process; native stalls cannot hold an engine thread forever.

The socket is inherited, not a network listener. Native library output stays in the worker log,
separate from the framed protocol. No pickle or remote executable objects cross this boundary.
"""

from __future__ import annotations

import importlib
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import msgspec


def receive(sock: socket.socket, limit: int) -> Any:
    def read(n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            part = sock.recv(min(n - len(data), 1 << 20))
            if not part:
                raise ConnectionError("cache I/O process closed its channel")
            data.extend(part)
        return bytes(data)

    size = struct.unpack("!Q", read(8))[0]
    if size > limit:
        raise ValueError("cache I/O frame exceeds configured limit")
    return msgspec.msgpack.decode(read(size))


def send(sock: socket.socket, value: Any, limit: int) -> None:
    data = msgspec.msgpack.encode(value)
    if len(data) > limit:
        raise ValueError("cache I/O frame exceeds configured limit")
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


class StoreProcess:
    def __init__(self, config: dict[str, str], timeout: float, max_bytes: int) -> None:
        self.config, self.timeout, self.limit = config, timeout, max_bytes + (1 << 20)
        self.lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None
        self.socket: socket.socket | None = None
        self.closed = False

    def call(self, operation: str, value: Any) -> Any:
        if not self.lock.acquire(timeout=self.timeout):
            raise TimeoutError("cache I/O queue timed out")
        try:
            if self.closed:
                raise ConnectionError("cache I/O unavailable until unit reload")
            if self.socket is None:
                parent, child = socket.socketpair()
                try:
                    self.process = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), str(child.fileno()), str(self.limit)],
                        pass_fds=(child.fileno(),))
                    self.socket = parent
                    parent.settimeout(self.timeout)
                    send(parent, self.config, self.limit)
                    answer = receive(parent, self.limit)
                    if answer.get("error"):
                        raise RuntimeError(answer["error"])
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
            send(self.socket, [operation, value], self.limit)
            answer = receive(self.socket, self.limit)
            if answer.get("error"):
                raise RuntimeError(answer["error"])
            return answer["value"]
        except Exception:
            self._close()
            raise
        finally:
            self.lock.release()

    def _close(self) -> None:
        self.closed = True
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            self.process = None

    def close(self) -> None:
        with self.lock:
            try:
                if self.socket is not None and self.process is not None and self.process.poll() is None:
                    send(self.socket, ["close", None], self.limit)
                    receive(self.socket, self.limit)
                    self.process.wait(timeout=self.timeout)
            except Exception:
                pass  # Native shutdown also has a deadline; an unresponsive child is killed below.
            finally:
                self._close()


def main() -> None:
    sock = socket.socket(fileno=int(sys.argv[1]))
    limit = int(sys.argv[2])
    try:
        config = receive(sock, limit)
        store = importlib.import_module("mooncake.store").MooncakeDistributedStore()
        code = store.setup(config)
        if code != 0:
            raise RuntimeError(f"Mooncake setup returned {code}")
        send(sock, {"value": None}, limit)
        while True:
            op, value = receive(sock, limit)
            if op == "exists":
                result = store.batch_is_exist(value)
            elif op == "get":
                result = store.get(value)
            elif op == "put":
                code = store.put(value[0], value[1])
                if code != 0:
                    raise RuntimeError(f"Mooncake put returned {code}")
                result = None
            elif op == "close":
                store.close()
                send(sock, {"value": None}, limit)
                return
            elif op == "remove":
                result = store.remove(value)
            else:
                raise ValueError("unknown cache operation")
            send(sock, {"value": result}, limit)
    except Exception as exc:
        try:
            send(sock, {"error": f"{type(exc).__name__}: {exc}"}, limit)
        except OSError:
            pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
