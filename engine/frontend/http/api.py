# Copyright (c) 2026 sgl-project
"""Adapted from mini-sglang/server/api_server.py (9a91cfa, MIT).

See ../LICENSE.mini-sglang. Retains FrontendManager's per-request ack/event maps,
listener, wait_for_ack and StreamingResponse cancellation flow. Changes are the
rsinfer record channel, instance-owned app/lifespan, identity metadata, actual
finish reasons and token counts, and disconnect/stop cleanup.

The frontend is the gate of the engine for changes (doc/specs/empty-apply.md): while one runs it
takes no new requests (503 with ``Retry-After``), and ``Gate`` tells the change how many are open.
"""

from __future__ import annotations

import asyncio
import gc
import json
import secrets
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.contracts.columns import Column
from core.contracts.requests import ENDED_BY_CHANGE, AbortRequest, FinishReason, NewRequest, Requests, TokenEvents
from core.contracts.sampling import SamplingParams
from frontend.interfaces import Detokenizer, Message, TextStream, Tokenizer
from frontend.transport import Channel

CLOSED = "the engine is being changed; retry shortly"
"""Why a request is turned away while a change empties the engine."""
RETRY_AFTER_S = 1
HINT = ("A change applies to an empty engine. Run the same apply with --wait-empty [SECONDS] to take no new "
        "requests and apply once these ended (3600 s at most by default), or with --force to end them now with "
        "an error and apply.")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_k: int = Field(default=-1, ge=-1, lt=2**31)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    n: int = 1
    stream: bool = False
    stop: str | list[str] = Field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    seed: int | None = Field(default=None, ge=-(2**63), lt=2**63)


@dataclass
class UserReply:
    uid: int
    incremental_output: str
    finished: bool
    reason: FinishReason | None = None
    error: str | None = None
    error_status: int = 503


@dataclass
class RequestState:
    decoder: TextStream
    prompt_tokens: int
    identity: str
    completion_tokens: int = 0
    identities: list[str] = field(default_factory=list[str])
    last_epoch: int = -1
    finished: bool = False
    abort_sent: bool = False
    incarnation: int | None = None
    """The scheduler instance it was sent to; ``None`` until it was."""
    last_active: float = field(default_factory=time.monotonic)


class FrontendManager:
    def __init__(self, model: str, tokenizer: Tokenizer, detokenizer: Detokenizer, channel: Channel,
                 identity: Callable[[], str], *, max_tokens: int = 4096, max_requests: int = 1024,
                 response_timeout_s: float = 120.0, paused: Callable[[], bool] = lambda: False) -> None:
        self.model, self.tokenizer, self.detokenizer, self.channel = model, tokenizer, detokenizer, channel
        self._model_json = json.dumps(model, ensure_ascii=False)
        self.identity, self.max_tokens, self.max_requests = identity, max_tokens, max_requests
        self.response_timeout_s = response_timeout_s
        self.uid_counter = secrets.randbits(30) << 32
        self.ack_map: dict[int, list[UserReply]] = {}
        self.event_map: dict[int, asyncio.Event] = {}
        self.states: dict[int, RequestState] = {}
        self.listener: asyncio.Task[None] | None = None
        self.watchdog: asyncio.Task[None] | None = None
        self._now_sec: int = int(time.time())
        self.failure: str | None = None
        """The token stream broke for good (two schedulers writing at once); a reload fixes it."""
        self.incarnation = 0
        self.gate: str | None = None
        """Closed (the change's mode) while a change empties the engine: new requests are turned away."""
        self.paused = paused
        self.built_paused = paused()
        """Built while its process was paused: by a change's Execute, which reloads this unit (or at
        start). The drain closed the instance this one replaces; this one stays closed until the
        process resumes, after the change committed or was undone (Codex, #42)."""
        self.entering = 0
        """Requests between the gate and ``states``: a change that closes the gate waits for them too."""

    def open_requests(self) -> int:
        return len(self.states) + self.entering

    def closed(self) -> bool:
        if self.built_paused and not self.paused():
            self.built_paused = False  # resumed once: later pauses are of changes that keep it
        return self.gate is not None or self.built_paused

    async def start(self) -> None:
        gc.set_threshold(100_000, 50, 50)
        self._now_sec = int(time.time())
        await self.channel.start()
        self.listener = asyncio.create_task(self.listen())
        self.watchdog = asyncio.create_task(self._watchdog())

    async def shutdown(self) -> None:
        for uid in list(self.states):
            with suppress(Exception):
                await self.abort_user(uid, "frontend unloaded")
        for task in (self.watchdog, self.listener):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        await self.channel.close()

    async def _watchdog(self) -> None:
        while True:
            limit = self.response_timeout_s
            await asyncio.sleep(min(0.25, max(0.02, limit * 0.25)))
            now = time.monotonic()
            self._now_sec = int(time.time())
            for uid, state in list(self.states.items()):
                if not state.finished and now - state.last_active >= limit:
                    state.finished = True
                    with suppress(Exception):
                        await self._send_abort(uid, state, "scheduler response timed out")
                    self._ack(UserReply(uid, "", True, error="scheduler response timed out"))

    def _ack(self, reply: UserReply) -> None:
        if reply.uid in self.ack_map:
            self.ack_map[reply.uid].append(reply)
            self.event_map[reply.uid].set()

    def _lost(self, uid: int, state: RequestState) -> None:
        state.finished = True
        self._ack(UserReply(uid, "", True, error="the scheduler was replaced; this request was lost"))

    def _end_replaced(self) -> None:
        """A new scheduler instance knows nothing of the requests sent to an earlier one. Scans
        only when the instance changed; a request not sent yet has no instance and stays."""
        if self.channel.incarnation == self.incarnation:
            return
        self.incarnation = self.channel.incarnation
        for uid, state in self.states.items():
            if not state.finished and state.incarnation is not None and state.incarnation < self.incarnation:
                self._lost(uid, state)

    async def listen(self) -> None:
        try:
            while True:
                epoch, events = await self.channel.receive()
                self._end_replaced()
                await self.deliver(epoch, events)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = str(exc)
            for uid, state in self.states.items():
                if not state.finished:
                    state.finished = True
                    self._ack(UserReply(uid, "", True, error=self.failure))

    async def deliver(self, epoch: int, events: TokenEvents) -> None:
        ids = [int(x) for x in events.requests.values()]
        offsets = [int(x) for x in events.offsets.values()]
        tokens = [int(x) for x in events.tokens.values()]
        if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(tokens) \
                or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("malformed TokenEvents offsets")
        now_mono = time.monotonic()
        self._now_sec = int(time.time())
        pieces = {uid: tokens[offsets[i]:offsets[i + 1]] for i, uid in enumerate(ids)}
        for uid in events.finished:
            pieces.setdefault(uid, [])
        for uid in events.errors:
            pieces.setdefault(uid, [])
        for uid, new_tokens in pieces.items():
            state = self.states.get(uid)
            if state is None or state.finished:
                continue
            state.last_active = now_mono
            if uid in events.errors:
                state.finished = True
                # A change forcing the engine empty ended it: the server's doing, retry later (Devin, #42).
                # Known by the error itself: the scheduler may see the change before this frontend (Codex).
                status = 503 if events.errors[uid] == ENDED_BY_CHANGE else 400
                self._ack(UserReply(uid, "", True, error=events.errors[uid], error_status=status))
                continue
            state.last_epoch = epoch
            if events.identity not in state.identities:
                state.identities.append(events.identity)
            state.completion_tokens += len(new_tokens)
            reason = events.finished.get(uid)
            piece = state.decoder.push(new_tokens, final=reason is not None)
            if piece.stopped:
                reason = "stop"
                await self._send_abort(uid, state, "stop string matched")
            state.finished = reason is not None
            self._ack(UserReply(uid, piece.text, state.finished, reason))

    async def submit(self, request: OpenAICompletionRequest, *, chat: bool) -> int:
        self.entering += 1  # before the gate: a change closing it now still sees this request
        try:
            if self.closed():
                raise HTTPException(503, CLOSED, headers={"Retry-After": str(RETRY_AFTER_S)})
            return await self._submit(request, chat=chat)
        finally:
            self.entering -= 1

    async def _submit(self, request: OpenAICompletionRequest, *, chat: bool) -> int:
        if self.failure is not None:
            raise HTTPException(503, self.failure)
        if request.model != self.model:
            raise HTTPException(404, "unknown model")
        if request.n != 1 or request.presence_penalty or request.frequency_penalty:
            raise HTTPException(400, "this frontend supports n=1 and zero repetition penalties")
        if request.max_tokens > self.max_tokens or request.top_k < -1:
            raise HTTPException(400, "generation limit or top_k is out of range")
        stops = [request.stop] if isinstance(request.stop, str) else request.stop
        if any(not s for s in stops):
            raise HTTPException(400, "stop strings must not be empty")
        if chat:
            if not request.messages:
                raise HTTPException(400, "messages must not be empty")
            prompt = await asyncio.to_thread(self.tokenizer.chat,
                                            [Message(m.role, m.content) for m in request.messages])
        else:
            if request.prompt is None:
                raise HTTPException(400, "prompt is required")
            prompt = await asyncio.to_thread(self.tokenizer.encode, request.prompt)
        if not prompt:
            raise HTTPException(400, "tokenized prompt must not be empty")
        if len(self.states) >= self.max_requests:
            raise HTTPException(503, "frontend request capacity reached")
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid], self.event_map[uid] = [], asyncio.Event()
        identity = self.identity()
        self.states[uid] = RequestState(self.detokenizer.open(stops), len(prompt), identity, identities=[identity])
        new = NewRequest(uid, Column.of("i64", prompt),
                         SamplingParams(request.temperature, request.top_p, request.top_k, request.seed),
                         request.max_tokens, [] if request.ignore_eos else list(self.tokenizer.stop_token_ids))
        try:
            sent = await self.channel.send(Requests(new=[new]))
        except asyncio.CancelledError:
            self.forget(uid)  # the client went away while it was being sent
            raise
        except Exception as exc:  # noqa: BLE001 - this request alone fails, and says why
            self.forget(uid)
            raise HTTPException(500, f"the request could not be sent: {type(exc).__name__}: {exc}") from exc
        if sent is None:
            self.forget(uid)
            raise HTTPException(503, "no scheduler is reachable right now")
        state = self.states[uid]
        state.incarnation = sent
        self._end_replaced()
        if sent < self.incarnation:  # it went to an instance already known replaced
            self._lost(uid, state)
        return uid

    async def wait_for_ack(self, uid: int) -> AsyncGenerator[UserReply]:
        event = self.event_map[uid]
        ack_list = self.ack_map[uid]
        while True:
            if not ack_list:
                await event.wait()
            event.clear()
            pending = ack_list[:]
            ack_list.clear()
            for ack in pending:
                yield ack
                if ack.finished:
                    return

    async def _send_abort(self, uid: int, state: RequestState, reason: str) -> None:
        """Best effort: a scheduler that cannot be reached holds nothing to abort for long, and a
        replaced one never had the request."""
        if not state.abort_sent:
            state.abort_sent = True
            await self.channel.send(Requests(abort=[AbortRequest(uid, reason)]))
            self._end_replaced()

    async def abort_user(self, uid: int, reason: str = "client disconnected") -> bool:
        state = self.states.get(uid)
        if state is None or state.finished:
            return False
        await self._send_abort(uid, state, reason)
        state.finished = True
        self._ack(UserReply(uid, "", True, "abort"))
        return True

    def forget(self, uid: int) -> None:
        self.states.pop(uid, None)
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        if not self.states:
            gc.collect(0)

    def usage(self, uid: int) -> dict[str, int]:
        state = self.states[uid]
        return dict(prompt_tokens=state.prompt_tokens, completion_tokens=state.completion_tokens,
                    total_tokens=state.prompt_tokens + state.completion_tokens)

    async def stream_completions(self, uid: int, chat: bool) -> AsyncGenerator[bytes]:
        first_chunk = True
        cmpl_prefix = f'data: {{"id": "cmpl-{uid}", "object": "text_completion", "created": '
        chat_prefix = f'data: {{"id": "chatcmpl-{uid}", "object": "chat.completion.chunk", "created": '
        model_mid = f', "model": {self._model_json}, "choices": [{{"index": 0, "finish_reason": null, '
        try:
            async for ack in self.wait_for_ack(uid):
                if ack.error:
                    kind = "invalid_request_error" if ack.error_status < 500 else "server_error"
                    yield self.sse({"error": {"message": ack.error, "type": kind, "code": ack.error_status}})
                    break
                if not ack.finished and not (first_chunk and chat):
                    first_chunk = False
                    now = self._now_sec
                    if not chat:
                        txt = json.dumps(ack.incremental_output, ensure_ascii=False)
                        yield f'{cmpl_prefix}{now}{model_mid}"text": {txt}}}]}}\n\n'.encode()
                    else:
                        dlt = (f'{{"content": {json.dumps(ack.incremental_output, ensure_ascii=False)}}}'
                               if ack.incremental_output else "{}")
                        yield f'{chat_prefix}{now}{model_mid}"delta": {dlt}}}]}}\n\n'.encode()
                    continue
                delta: dict[str, str] = {}
                if first_chunk and chat:
                    delta["role"] = "assistant"
                first_chunk = False
                if ack.incremental_output:
                    delta["content"] = ack.incremental_output
                reason = ack.reason if ack.reason in ("stop", "length") else ("stop" if ack.finished else None)
                choice: dict[str, Any] = {"index": 0, "finish_reason": reason}
                choice.update({"delta": delta} if chat else {"text": ack.incremental_output})
                chunk: dict[str, Any] = {
                    "id": f"{'chatcmpl' if chat else 'cmpl'}-{uid}",
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": self._now_sec, "model": self.model, "choices": [choice],
                }
                if ack.finished:
                    chunk.update(usage=self.usage(uid), engine_finish_reason=ack.reason,
                                 engine_identities=self.states[uid].identities)
                yield self.sse(chunk)
            yield b"data: [DONE]\n\n"
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await self.abort_user(uid)
            finally:
                self.forget(uid)

    @staticmethod
    def sse(value: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(value, ensure_ascii=False)}\n\n".encode()


def create_app(manager: FrontendManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        try:
            await manager.start()
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(title="rsinfer", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        kind = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse({"error": {"message": str(exc.detail), "type": kind}},
                            status_code=exc.status_code, headers=exc.headers)

    @app.get("/health")
    async def health() -> dict[str, str]:
        if manager.failure:
            raise HTTPException(503, manager.failure)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": manager.model, "object": "model",
                "created": int(time.time()), "owned_by": "rsinfer"}]}

    @app.delete("/v1/requests/{uid}")
    async def cancel(uid: int) -> dict[str, Any]:
        return {"id": uid, "cancelled": await manager.abort_user(uid, "explicit cancellation")}

    async def completion(req: OpenAICompletionRequest, request: Request, chat: bool) -> Any:
        uid = await manager.submit(req, chat=chat)
        headers = {"x-engine-identity": manager.states[uid].identity, "x-request-id": str(uid)}
        if req.stream:
            return StreamingResponse(manager.stream_completions(uid, chat), media_type="text/event-stream",
                                     headers=headers)

        async def collect() -> tuple[str, UserReply]:
            chunks: list[str] = []
            async for ack in manager.wait_for_ack(uid):
                if ack.error:
                    raise HTTPException(ack.error_status, ack.error, headers={"Retry-After": str(RETRY_AFTER_S)}
                                        if ack.error_status == 503 else None)
                chunks.append(ack.incremental_output)
                if ack.finished:
                    return "".join(chunks), ack
            raise HTTPException(503, "scheduler ended without a final result")

        task = asyncio.create_task(collect())
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.05)
                if await request.is_disconnected():
                    await manager.abort_user(uid)
                    raise HTTPException(499, "client disconnected")
            content, last = task.result()
            reason = last.reason if last.reason in ("stop", "length") else "stop"
            choice: dict[str, Any] = {"index": 0, "finish_reason": reason}
            choice.update({"message": {"role": "assistant", "content": content}} if chat else {"text": content})
            return JSONResponse({"id": f"{'chatcmpl' if chat else 'cmpl'}-{uid}",
                                 "object": "chat.completion" if chat else "text_completion",
                                 "created": int(time.time()), "model": manager.model, "choices": [choice],
                                 "usage": manager.usage(uid), "engine_finish_reason": last.reason,
                                 "engine_identities": manager.states[uid].identities}, headers=headers)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError, HTTPException):
                await task
            try:
                with anyio.CancelScope(shield=True):
                    await manager.abort_user(uid)
            finally:
                manager.forget(uid)

    @app.post("/v1/chat/completions")
    async def chat(req: OpenAICompletionRequest, request: Request) -> Any:
        return await completion(req, request, True)

    @app.post("/v1/completions")
    async def text(req: OpenAICompletionRequest, request: Request) -> Any:
        return await completion(req, request, False)

    return app


class Gate:
    """``frontend.http``'s ``drain`` action (doc/specs/empty-apply.md). Called by the process host
    at its safe points, while the requests run on the server's thread: it only sets and reads
    plain attributes. In every mode the gate closes until the change committed or was undone;
    ``refuse`` fails the change if a request is open. With ``force`` the scheduler ends them,
    and their responses close here."""

    def __init__(self, manager: FrontendManager) -> None:
        self.manager = manager

    def start_drain(self, args: dict[str, Any]) -> None:
        mode = str(args.get("mode", "refuse"))
        # Closed first, counted after: ``submit`` counts itself before it looks at the gate, so a
        # request either is counted here or finds the gate closed. ``abort()`` opens it again.
        self.manager.gate = mode
        if mode == "refuse" and (n := self.manager.open_requests()):
            raise RuntimeError(f"the frontend has {n} open request(s). {HINT}")

    def drained(self) -> bool:
        return self.manager.gate is None or self.manager.open_requests() == 0

    def drain_deadline(self) -> dict[str, int]:
        raise TimeoutError(f"the frontend still has {self.manager.open_requests()} open request(s); nothing was "
                           "ended for this change, and requests come in again. Retry with a longer --wait-empty, "
                           "or with --force.")

    def commit(self) -> None:
        self.manager.gate = None

    def abort(self) -> None:
        self.manager.gate = None
