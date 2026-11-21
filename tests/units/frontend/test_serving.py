"""The real frontend units over real HTTP and record streams, against a scripted scheduler."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

import frontend.transport as transport
from frontend.detok.unit import Detokenizer
from frontend.http.api import OpenAICompletionRequest
from tests.units.frontend.support import ByteTokenizer, FakeSupervisor, Frontend, ScriptScheduler


@pytest.fixture
def serving() -> Generator[tuple[str, ScriptScheduler]]:
    scheduler = ScriptScheduler()
    front = Frontend(scheduler)
    try:
        yield front.url, scheduler
    finally:
        front.close()
        scheduler.close()


def test_completions_and_chat_use_the_real_units_and_record_streams(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    with httpx.Client(base_url=url, timeout=10) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/openapi.json").json()["info"]["title"] == "rsinfer"
        model = client.get("/v1/models").json()["data"][0]
        assert model["id"] == "toy"
        assert model["owned_by"] == "rsinfer"
        for path, body, chat in [
            ("/v1/completions", {"prompt": "hello"}, False),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]}, True),
        ]:
            response = client.post(path, json={"model": "toy", "max_tokens": 64, **body})
            assert response.status_code == 200, response.text
            data = response.json()
            choice = data["choices"][0]
            assert (choice["message"]["content"] if chat else choice["text"]) == scheduler.text
            assert choice["finish_reason"] == "stop"
            assert response.headers["x-engine-identity"] == "v1"
            assert data["engine_identities"] == ["v1"]
            assert data["usage"]["completion_tokens"] == len(scheduler.text.encode()) + 1
            sent = scheduler.new.get(timeout=2)
            assert len(sent.prompt) == data["usage"]["prompt_tokens"]


def test_chat_sse_and_stop_across_record_boundaries(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.text = "你好END不能泄漏"
    with httpx.Client(base_url=url, timeout=10) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "toy", "messages": [{"role": "user", "content": "hi"}],
            "stream": True, "max_tokens": 128, "stop": ["END"],
        }) as response:
            assert response.status_code == 200
            frames = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        chunks = [json.loads(line) for line in frames[:-1]]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
        assert text == "你好"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert scheduler.aborts.get(timeout=2).reason == "stop string matched"


def test_disconnect_before_first_token_sends_abort(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.hold = True
    with httpx.Client(base_url=url, timeout=10) as client:
        with client.stream("POST", "/v1/completions", json={"model": "toy", "prompt": "wait", "stream": True}):
            sent = scheduler.new.get(timeout=2)
        assert scheduler.aborts.get(timeout=3).request == sent.request


def test_explicit_cancel_finishes_an_idle_stream(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.hold = True
    with httpx.Client(base_url=url, timeout=10) as client:
        with client.stream("POST", "/v1/completions", json={"model": "toy", "prompt": "wait", "stream": True}) as stream:
            uid = int(stream.headers["x-request-id"])
            assert client.delete(f"/v1/requests/{uid}").json()["cancelled"]
            frames = [line[6:] for line in stream.iter_lines() if line.startswith("data: ")]
            assert json.loads(frames[0])["engine_finish_reason"] == "abort"
            assert frames[-1] == "[DONE]"
        assert scheduler.aborts.get(timeout=2).request == uid
        assert not client.delete(f"/v1/requests/{uid}").json()["cancelled"]


def test_length_and_unsupported_parameters_are_reported(serving: tuple[str, ScriptScheduler]) -> None:
    url, scheduler = serving
    scheduler.text = "abcdef"
    with httpx.Client(base_url=url, timeout=10) as client:
        data = client.post("/v1/completions", json={"model": "toy", "prompt": "x", "max_tokens": 3}).json()
        assert data["choices"][0] == {"index": 0, "finish_reason": "length", "text": "abc"}
        assert data["usage"]["completion_tokens"] == 3
        assert client.post("/v1/completions", json={"model": "toy", "prompt": "x", "n": 2}).status_code == 400
        assert client.post("/v1/completions", json={"model": "missing", "prompt": "x"}).status_code == 404


def test_a_refusal_by_the_scheduler_is_an_error_in_http_and_sse() -> None:
    scheduler = ScriptScheduler(refuse="prompt plus max_tokens exceeds state capacity")
    front = Frontend(scheduler)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            body = {"model": "toy", "prompt": "x", "max_tokens": 2}
            response = client.post("/v1/completions", json=body)
            assert response.status_code == 400
            assert "exceeds state capacity" in response.json()["error"]["message"]
            with client.stream("POST", "/v1/completions", json={**body, "stream": True}) as stream:
                frames = [line[6:] for line in stream.iter_lines() if line.startswith("data: ")]
            assert json.loads(frames[0])["error"]["code"] == 400 and frames[-1] == "[DONE]"
    finally:
        front.close()
        scheduler.close()


def test_a_silent_scheduler_times_out_the_http_request_and_sends_cancel() -> None:
    scheduler = ScriptScheduler(hold=True)
    front = Frontend(scheduler, response_timeout_s=0.1)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "hello"})
        assert response.status_code == 503
        assert "timed out" in response.json()["error"]["message"]
        assert scheduler.aborts.get(timeout=2).request == scheduler.new.get(timeout=2).request
    finally:
        front.close()
        scheduler.close()


@pytest.mark.covers(edges=("data", "call"), kinds=("logic",))
def test_response_timeout_is_refreshed_while_tokens_keep_arriving() -> None:
    scheduler = ScriptScheduler()
    scheduler.text = "x" * 64
    front = Frontend(scheduler, response_timeout_s=0.1)
    try:
        started = time.monotonic()
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "hello", "max_tokens": 128})
        assert time.monotonic() - started > 0.1
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["text"] == scheduler.text
    finally:
        front.close()
        scheduler.close()


def test_without_a_scheduler_a_request_fails_fast_and_the_frontend_stays_up() -> None:
    """Not a permanent failure: the frontend answers 503 for this request and keeps serving."""
    scheduler = ScriptScheduler()
    front = Frontend(scheduler, io_timeout_s=0.3)
    scheduler.close()
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "x"})
            assert response.status_code == 503
            assert "no scheduler" in response.json()["error"]["message"]
            assert client.get("/health").json() == {"status": "ok"}
    finally:
        front.close()


def test_a_replaced_scheduler_ends_what_it_lost_and_the_next_one_serves() -> None:
    """The scheduler's process is replaced: the new one has a new reader address, which the
    supervisor names. The request in flight at the old one is lost and ends with an error;
    requests sent while the frontend finds the new one all reach it and are served, none
    mistaken for the old instance's (Codex and Antigravity on #26)."""
    supervisor = FakeSupervisor()
    old = ScriptScheduler(hold=True)
    supervisor.target = old.address
    front = Frontend(supervisor=supervisor)
    new: ScriptScheduler | None = None
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client, ThreadPoolExecutor(max_workers=9) as pool:
            lost = pool.submit(client.post, "/v1/completions", json={"model": "toy", "prompt": "wait"})
            old.new.get(timeout=2)
            new = ScriptScheduler(text="new")
            supervisor.target = new.address
            old.close()
            fresh = [pool.submit(client.post, "/v1/completions", json={"model": "toy", "prompt": str(i),
                                                                         "max_tokens": 64}) for i in range(8)]
            replies = [f.result(timeout=10) for f in fresh]
            assert [r.status_code for r in replies] == [200] * 8, [r.text for r in replies]
            assert all(r.json()["choices"][0]["text"] == "new" for r in replies)
            assert new.new.qsize() == 8
            failed = lost.result(timeout=5)
            assert failed.status_code == 503
            assert "scheduler was replaced" in failed.json()["error"]["message"]
    finally:
        front.close()
        if new is not None:
            new.close()
        supervisor.close()


def test_a_request_that_fails_to_go_out_frees_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever stops a request after it was registered, it alone fails: it does not keep a
    place among ``frontend.max_requests`` (Codex on #26)."""
    scheduler = ScriptScheduler()
    front = Frontend(scheduler)
    manager = front.server.manager
    manager.max_requests = 2
    send = manager.channel.send
    failures = 0

    async def failing(requests: Any) -> int | None:
        nonlocal failures
        if failures < 2:
            failures += 1
            raise RuntimeError("the send broke")
        return await send(requests)

    monkeypatch.setattr(manager.channel, "send", failing)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            for _ in range(2):
                assert client.post("/v1/completions", json={"model": "toy", "prompt": "x"}).status_code == 500
                assert not manager.states and not manager.ack_map and not manager.event_map
            for _ in range(2):  # a seed msgpack cannot carry is refused before anything is registered
                body = {"model": "toy", "prompt": "x", "seed": 2**100}
                assert client.post("/v1/completions", json=body).status_code == 422
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "x", "max_tokens": 64})
            assert response.status_code == 200, response.text
            assert scheduler.new.qsize() == 1
    finally:
        front.close()
        scheduler.close()


def test_requests_waiting_for_a_scheduler_each_keep_their_own_deadline() -> None:
    """With no scheduler published, every request waits ``io_timeout_s`` from when it was queued,
    not behind the others: the I/O thread keeps going while sends wait (Codex on #26)."""
    supervisor = FakeSupervisor()
    front = Frontend(supervisor=supervisor, io_timeout_s=0.2)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client, ThreadPoolExecutor(max_workers=6) as pool:
            def request(i: int) -> tuple[int, float]:
                started = time.monotonic()
                r = client.post("/v1/completions", json={"model": "toy", "prompt": str(i)})
                return r.status_code, time.monotonic() - started

            results = list(pool.map(request, range(6)))
        assert [code for code, _ in results] == [503] * 6
        assert max(elapsed for _, elapsed in results) < 0.6, results
    finally:
        front.close()
        supervisor.close()


def wait_until(check: Any, within: float = 2.0) -> None:
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_a_request_given_up_before_it_went_out_never_reaches_the_scheduler() -> None:
    """The submitting coroutine is cancelled while its request waits for a scheduler: the
    request leaves the queue, and the scheduler found later never sees it (Codex on #26)."""
    supervisor = FakeSupervisor()
    front = Frontend(supervisor=supervisor, io_timeout_s=2.0)
    manager = front.server.manager
    scheduler: ScriptScheduler | None = None
    try:
        submit = asyncio.run_coroutine_threadsafe(
            manager.submit(OpenAICompletionRequest(model="toy", prompt="given up"), chat=False),
            getattr(manager.channel, "_loop"))
        wait_until(lambda: bool(manager.states))
        submit.cancel()
        wait_until(lambda: not manager.states)
        scheduler = ScriptScheduler()
        supervisor.target = scheduler.address
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "x", "max_tokens": 64})
        assert response.status_code == 200, response.text
        assert scheduler.new.qsize() == 1  # only the request still wanted
        assert scheduler.aborts.empty()
    finally:
        front.close()
        if scheduler is not None:
            scheduler.close()
        supervisor.close()


def test_a_request_given_up_after_it_went_out_is_aborted_at_the_scheduler() -> None:
    """The request went out, but its coroutine is cancelled before it learns so (its event loop
    is busy meanwhile): the scheduler has it, so it is told to drop it."""
    supervisor = FakeSupervisor()
    front = Frontend(supervisor=supervisor, io_timeout_s=2.0)
    manager = front.server.manager
    loop = getattr(manager.channel, "_loop")
    scheduler = ScriptScheduler(hold=True)
    try:
        submit = asyncio.run_coroutine_threadsafe(
            manager.submit(OpenAICompletionRequest(model="toy", prompt="given up late"), chat=False), loop)
        wait_until(lambda: bool(manager.states))
        loop.call_soon_threadsafe(time.sleep, 0.5)  # the event loop is busy: the answer waits
        supervisor.target = scheduler.address
        uid = scheduler.new.get(timeout=2).request  # it went out meanwhile
        submit.cancel()
        assert scheduler.aborts.get(timeout=3).request == uid
        wait_until(lambda: not manager.states)
    finally:
        front.close()
        scheduler.close()
        supervisor.close()


def test_a_waiting_request_is_encoded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries reuse the encoded record: a long prompt is not encoded again every turn."""
    encodes = 0
    encode = transport.encode_record

    def counting(record: Any) -> bytes:
        nonlocal encodes
        encodes += 1
        return encode(record)

    monkeypatch.setattr(transport, "encode_record", counting)
    supervisor = FakeSupervisor()
    front = Frontend(supervisor=supervisor, io_timeout_s=0.3)
    try:
        with httpx.Client(base_url=front.url, timeout=10) as client:
            response = client.post("/v1/completions", json={"model": "toy", "prompt": "x" * 10000})
        assert response.status_code == 503
        assert encodes == 1
    finally:
        front.close()
        supervisor.close()


def test_concurrent_requests_keep_their_own_limits_and_return_identity_history(
    serving: tuple[str, ScriptScheduler],
) -> None:
    url, scheduler = serving
    scheduler.text, scheduler.identity = "abcdefghij", "v2"
    with httpx.Client(base_url=url, timeout=10) as client, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.post, "/v1/completions", json={
            "model": "toy", "prompt": "x", "max_tokens": length}) for length in (3, 7)]
        responses = [f.result(timeout=5) for f in futures]
        assert [r.json()["choices"][0]["text"] for r in responses] == ["abc", "abcdefg"]
        assert all(r.json()["engine_identities"] == ["v1", "v2"] for r in responses)
        assert len({r.headers["x-request-id"] for r in responses}) == 2


def test_incremental_decoder_preserves_unicode_and_partial_stop_at_finish() -> None:
    detok = Detokenizer(ByteTokenizer())
    decoder = detok.open(["END"])
    parts = [decoder.push([token]).text for token in "中文EN".encode()]
    parts.append(decoder.push([], final=True).text)
    assert "".join(parts) == "中文EN"
    decoder = detok.open(["END"])
    assert decoder.push(list(b"aE")).text == "a"
    assert decoder.push(list(b"NDignored")).stopped
    assert decoder.push(list(b"late")).text == ""
