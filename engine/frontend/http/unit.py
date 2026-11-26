"""Unit lifecycle adapter for mini-sglang's FastAPI/uvicorn serving path.

All app/manager state belongs to this unit instance. The process host retains its
control loop; the frontend's asyncio server owns one registered background thread, and
its record streams one I/O thread (``frontend.transport``).
"""

from __future__ import annotations

import threading
import time
from typing import Any, cast

import msgspec
import uvicorn

from core.common.ids import ProcId
from core.ipc.channels import Channels
from core.ipc.config import IpcTunables
from core.ipc.stream import StreamAddress
from frontend.interfaces import Detokenizer, Tokenizer
from frontend.transport import Directory, StepChannel

from .api import FrontendManager, Gate, create_app


class Server:
    def __init__(self, manager: FrontendManager, host: str, port: int,
                 start_timeout: float, shutdown_timeout: float) -> None:
        self.manager = manager
        self.app = create_app(manager)
        self.server = uvicorn.Server(uvicorn.Config(
            self.app, host=host, port=port, loop="auto", http="auto", log_level="warning",
            timeout_graceful_shutdown=max(1, int(shutdown_timeout / 2))))
        self.thread = threading.Thread(target=self.server.run, name="frontend-http", daemon=True)
        self.start_timeout, self.shutdown_timeout = start_timeout, shutdown_timeout
        self.host, self.port = host, port

    def start(self) -> Any:
        self.thread.start()
        deadline = time.monotonic() + self.start_timeout
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("frontend HTTP startup failed")
            time.sleep(0.01)
        self.port = int(self.server.servers[0].sockets[0].getsockname()[1])
        return self.close

    def close(self) -> None:
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(self.shutdown_timeout)
        if self.thread.is_alive():
            raise RuntimeError("frontend HTTP shutdown did not finish")


class Unit:
    def __init__(self, ctx: Any, config: Any) -> None:
        tokenizer = cast(Tokenizer, ctx.use("frontend.tokenizer@1"))
        detok = cast(Detokenizer, ctx.use("frontend.detok@1"))
        control = ctx.use("proc.control@1")
        streams = cast(Channels, ctx.use("proc.streams@1"))
        if config["frontend.requests_address"]:
            fixed = msgspec.convert(config["frontend.requests_address"], StreamAddress)
            directory, scheduler_host = Directory(fixed=fixed), fixed.host
        else:
            scheduler = msgspec.convert(config["frontend.scheduler"], ProcId)
            directory = Directory(supervisor=str(control.address), scheduler=scheduler, me=ctx.instance.proc)
            scheduler_host = scheduler.host
        # The host keeps this reader across reloads of this unit: the return route stays valid.
        tokens = streams.reader("tokens", scheduler_host)
        io = config["frontend.io_timeout_s"]
        channel = StepChannel(ctx.instance.proc.host, tokens, directory,
                              tun=IpcTunables(rpc_timeout_s=io, close_timeout_s=io))
        manager = FrontendManager(config["frontend.model"], tokenizer, detok, channel,
                                  lambda: str(control.identity), max_tokens=config["frontend.max_tokens"],
                                  max_requests=config["frontend.max_requests"],
                                  response_timeout_s=config["frontend.response_timeout_s"],
                                  paused=lambda: bool(getattr(control, "paused", False)),
                                  cache=ctx.get("kv.cache@1"))
        server = Server(manager, config["frontend.http_bind"], config["frontend.http_port"],
                        config["frontend.start_timeout_s"], config["frontend.shutdown_timeout_s"])
        ctx.effect(server.start, "HTTP server and record streams")
        ctx.on_action("drain", Gate(manager))  # the engine's gate for changes (doc/specs/empty-apply.md)
        ctx.provide("frontend.http@1", server)
