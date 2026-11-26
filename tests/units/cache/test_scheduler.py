"""Row-level cache failures retire resources without stopping the scheduler."""

from __future__ import annotations

import msgspec
import pytest

from core.contracts.columns import Column
from core.contracts.records import Sampled
from tests.units.scheduler.support import bed, request


@pytest.mark.covers(edges=("data", "state"), kinds=("logic", "state"))
def test_failed_row_reports_error_releases_its_items_and_later_requests_run(monkeypatch: pytest.MonkeyPatch) -> None:
    with bed(ranks=1) as b:
        worker = b.workers[0].fixture
        answer = worker._answer
        injected = False

        def fail_one(epoch: int, section: str, value: msgspec.Struct, due: float | None = None) -> None:
            nonlocal injected
            if isinstance(value, Sampled) and not injected:
                injected = True
                rows = [int(x) for x in value.rows.values()]
                assert len(rows) == 2
                value = Sampled(value.rows, Column.of("i32", [0, 1]),
                                Column.of("i64", list(value.tokens.values())[1:]), {rows[0]: "cache entry evicted"})
            answer(epoch, section, value, due)

        monkeypatch.setattr(worker, "_answer", fail_one)
        b.client.send([request(1, [1, 2, 3], 4), request(2, [5, 6, 7], 4)])
        b.run_until(b.finished(1, 2))
        b.run_until(b.idle)
        assert injected and b.client.outputs[1].error == "cache entry evicted"
        assert b.client.outputs[1].tokens == [] and len(b.client.outputs[2].tokens) == 4
        assert b.engine.loop.report().fault is None and b.engine.ledger.check() == []
        b.client.send([request(3, [9, 8], 3)])
        b.run_until(b.finished(3))
        assert b.client.outputs[3].error is None and len(b.client.outputs[3].tokens) == 3
        b.run_until(b.idle)
