"""``core.sched`` with scheduling domains (``sched.domains``, doc/specs/v1-ds1017.md stage 3): ranks
that hold different requests, each with a ledger of its own, all stepping together.

As in ``test_scheduling.py`` the fixture workers check every plan against what they hold, so a row
continued at a wrong position, or given another domain's items, fails in the worker.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.contracts.requests import NewRequest
from tests.units.scheduler.support import CAPACITY, Bed, bed, request
from tests.units.scheduler.units.schedfix.worker.unit import expected

TWO = {"sched.domains": 2}


def want(r: NewRequest) -> list[int]:
    assert r.sampling.seed is not None
    return expected([int(x) for x in r.prompt.values()], r.sampling.seed, r.max_tokens, r.stop_token_ids)


def check(b: Bed, requests: Sequence[NewRequest]) -> None:
    for r in requests:
        assert b.client.outputs[r.request].tokens == want(r), r.request
    b.run_until(b.idle, what="idle")
    assert b.errors() == [] and all(ledger.check() == [] for ledger in b.engine.ledgers)
    assert b.engine.loop.report().fault is None


def served(b: Bed, rank: int) -> set[int]:
    """The requests whose rows ``rank`` was given."""
    return {a.request for _, _, delta in b.workers[rank].fixture.plans_seen for a in delta.added}


def rows_run(b: Bed, rank: int) -> list[int]:
    return [len(plan.rows) for _, plan, _ in b.workers[rank].fixture.plans_seen]


@pytest.mark.parametrize("overlap", [False, True])
def test_each_domain_runs_requests_of_its_own_and_every_rank_gets_every_step(overlap: bool) -> None:
    reqs = [request(1, [5, 6, 7, 8, 9], 6), request(2, list(range(1, 30)), 4), request(3, [3, 1, 4], 9),
            request(4, [2, 7, 1, 8, 2, 8], 5)]
    with bed(overlap=overlap, delay_s=0.001, config=TWO) as b:
        b.client.send(reqs)
        b.run_until(b.finished(1, 2, 3, 4))
        check(b, reqs)
        one, other = served(b, 0), served(b, 1)
        assert one and other and not one & other and one | other == {1, 2, 3, 4}
        assert len(one) == len(other) == 2  # a waiting request goes where most rows are free
        # One step is one plan to every rank: the ranks compute each layer together.
        assert b.workers[0].fixture.received == b.workers[1].fixture.received
        # Each rank numbers its rows from 0.
        for w in b.workers:
            assert all(a.row < 2 for _, _, delta in w.fixture.plans_seen for a in delta.added)


def test_a_domain_with_nothing_to_run_gets_an_empty_plan() -> None:
    r = request(1, [1, 2, 3], 5)
    with bed(config=TWO) as b:
        b.client.send([r])
        b.run_until(b.finished(1))
        check(b, [r])
        assert served(b, 0) == {1} and served(b, 1) == set()
        assert b.workers[1].fixture.received == b.workers[0].fixture.received
        assert set(rows_run(b, 1)) == {0} and set(rows_run(b, 0)) == {1}


def test_a_domain_composes_its_own_step() -> None:
    """One domain computes a long prompt in chunks while the other already decodes: the budget of
    prompt tokens and the choice between prompts and decodes are each domain's."""
    long, short = request(1, list(range(1, 60)), 3), request(2, [4, 5], 12)
    with bed(config=TWO, token_budget=16) as b:
        b.client.send([long, short])
        b.run_until(b.finished(1, 2))
        check(b, [long, short])
        phases = [{rank: plan.phase for rank in (0, 1)
                   for epoch, plan, _ in b.workers[rank].fixture.plans_seen if epoch == e and len(plan.rows)}
                  for e in b.workers[0].fixture.received]
        assert {0: "prefill", 1: "decode"} in phases


def test_each_domain_has_the_rows_and_the_items_of_its_own_ranks() -> None:
    small = {**CAPACITY, "kv.paged": 8}
    with bed(config=TWO, max_batch=2, capacity=[dict(CAPACITY), small]) as b:
        assert [led.pools["kv.paged"].capacity for led in b.engine.ledgers] == [CAPACITY["kv.paged"], 8]
        reqs = [request(i, [i, i + 1, i + 2], 6) for i in range(1, 7)]
        b.client.send(reqs)
        b.run_until(b.finished(*range(1, 7)))
        check(b, reqs)
        assert max(rows_run(b, 0)) == 2 and max(rows_run(b, 1)) == 2  # max_batch rows on each
        assert len(served(b, 0)) == len(served(b, 1)) == 3


def test_a_request_only_the_larger_domain_holds_runs_there() -> None:
    small = {**CAPACITY, "kv.paged": 2}  # 8 tokens
    r = request(1, list(range(1, 20)), 6)
    with bed(config=TWO, capacity=[small, dict(CAPACITY)]) as b:
        b.client.send([r])
        b.run_until(b.finished(1))
        check(b, [r])
        assert served(b, 1) == {1}


def test_a_request_one_domain_has_no_room_for_now_tries_the_other() -> None:
    """Rows are free on both, but the first domain's pages are taken: the request it turned away is
    admitted by the other while the first is still busy."""
    small = {**CAPACITY, "kv.paged": 8}  # 32 tokens
    reqs = [request(i, [i, i + 1, i + 2], 25) for i in (1, 2, 3)]
    with bed(delay_s=0.002, config=TWO, capacity=[small, dict(CAPACITY)]) as b:
        b.client.send(reqs[:2])
        b.run_until(lambda: all(len(b.client.outputs[r].tokens) >= 1 for r in (1, 2)))
        assert served(b, 0) == {1} and served(b, 1) == {2}
        b.client.send(reqs[2:])
        b.run_until(lambda: len(b.client.outputs[3].tokens) >= 1)
        assert b.client.outputs[1].finished is None  # the first domain's pages are still taken
        assert served(b, 1) == {2, 3}
        b.run_until(b.finished(1, 2, 3))
        check(b, reqs)


@pytest.mark.parametrize("overlap", [False, True])
def test_a_step_one_domain_refused_ran_on_the_others(overlap: bool) -> None:
    """The worker of one domain lost its persistent batch: it runs nothing until a full delta. What
    the other domain ran meanwhile counts, and the refused rows go on from where they were."""
    reqs = [request(1, [5, 6, 7, 8, 9], 9), request(2, [3, 1, 4, 1, 5], 9)]
    with bed(overlap=overlap, delay_s=0.001, config=TWO) as b:
        b.client.send(reqs)
        b.run_until(lambda: all(len(b.client.outputs[r].tokens) >= 2 for r in (1, 2)))
        at = b.engine.table.last_submitted + 2
        b.workers[1].fixture.script[at] = "resync"
        b.run_until(b.finished(1, 2))
        check(b, reqs)
        assert any("not run by domain(s) [1]" in d for d in b.engine.loop.report().diagnostics)
        # The other domain was not held up: its request ran in the refused step.
        assert any(epoch == at and len(plan.rows) for epoch, plan, _ in b.workers[0].fixture.plans_seen)


def test_a_replaced_worker_takes_its_own_domains_requests_only() -> None:
    reqs = [request(1, [5, 6, 7, 8, 9], 40), request(2, [3, 1, 4, 1, 5], 40)]
    with bed(delay_s=0.002, config=TWO) as b:
        b.client.send(reqs)
        b.run_until(lambda: all(len(b.client.outputs[r].tokens) >= 3 for r in (1, 2)))
        lost = next(iter(served(b, 1)))
        kept = next(iter(served(b, 0)))
        b.replace_worker(1)
        b.run_until(b.finished(1, 2))
        assert b.client.outputs[lost].finished == "abort" and "new" in (b.client.outputs[lost].error or "")
        assert b.client.outputs[kept].error is None
        assert b.client.outputs[kept].tokens == want(reqs[kept - 1])
        # The new worker serves the next request of its domain.
        again = [request(3, [9, 9, 9], 5), request(4, [8, 8], 5)]
        b.client.send(again)
        b.run_until(b.finished(3, 4))
        check(b, again)
        assert b.engine.loop.report().fault is None


def test_the_ranks_of_one_domain_run_the_same_plan() -> None:
    """Two domains of two ranks each: within a domain every rank gets the same plan and must
    sample the same."""
    reqs = [request(i, [i, i + 3, i + 5], 5) for i in range(1, 5)]
    with bed(ranks=4, config=TWO) as b:
        b.client.send(reqs)
        b.run_until(b.finished(1, 2, 3, 4))
        check(b, reqs)
        assert served(b, 0) == served(b, 1) and served(b, 2) == served(b, 3)
        assert not served(b, 0) & served(b, 2) and served(b, 0) | served(b, 2) == {1, 2, 3, 4}


def test_a_failed_worker_of_one_domain_stops_the_engine() -> None:
    """Ranks that compute together cannot go on without one of them."""
    reqs = [request(1, [5, 6, 7], 30), request(2, [3, 1, 4], 30)]
    with bed(config=TWO) as b:
        b.client.send(reqs)
        b.run_until(lambda: all(len(b.client.outputs[r].tokens) >= 2 for r in (1, 2)))
        b.workers[1].fixture.script[b.engine.table.last_submitted + 1] = "fail"
        b.run_until(lambda: b.engine.loop.report().fault is not None)
        b.run_until(b.finished(1, 2))
        assert all(b.client.outputs[r].error for r in (1, 2))


def test_speculation_with_domains() -> None:
    reqs = [request(1, [5, 6, 7, 8, 9], 17), request(2, list(range(1, 30)), 9), request(3, [3, 1, 4], 21)]
    with bed(delay_s=0.001, overlap=True, config={**TWO, "spec.algorithm": "eagle", "spec.num_draft_tokens": 4}) as b:
        b.client.send(reqs)
        b.run_until(b.finished(1, 2, 3))
        check(b, reqs)


def test_domains_must_divide_the_ranks() -> None:
    with pytest.raises(Exception, match="sched.domains"):
        with bed(ranks=2, config={"sched.domains": 3}):
            pass


def test_balance_prefill_and_coordinated_prefill_interval() -> None:
    """With balance_prefill, long waiting prompts route to the domain with fewer queued prompt tokens;
    with prefill_interval, arriving prefills batch behind pure decode steps while a domain decodes."""
    cfg = {**TWO, "sched.balance_prefill": True, "sched.prefill_interval": 4, "policy.min_chunk_tokens": 8}
    first = [request(1, list(range(1, 13)), 10), request(2, [5, 6], 10)]
    later = [request(3, list(range(1, 11)), 4), request(4, list(range(1, 11)), 4)]
    with bed(delay_s=0.001, overlap=True, token_budget=16, config=cfg) as b:
        b.client.send(first)
        b.run_until(lambda: all(len(b.client.outputs[r].tokens) >= 2 for r in (1, 2)))
        b.client.send(later)
        b.run_until(b.finished(1, 2, 3, 4))
        check(b, [*first, *later])
        assert served(b, 0) & {3, 4} and served(b, 1) & {3, 4}


@pytest.mark.parametrize("lengths,uniform", [((3, 3), True), ((3, 4), False)])
def test_ep_domains_share_one_prefill_geometry_hint(lengths: tuple[int, int], uniform: bool) -> None:
    reqs = [request(i + 1, list(range(n)), 1) for i, n in enumerate(lengths)]
    with bed(config=TWO) as b:
        b.client.send(reqs)
        b.run_until(b.finished(1, 2))
        first = [worker.fixture.plans_seen[0][1] for worker in b.workers]
        assert [plan.uniform_prefill for plan in first] == [uniform, uniform]
