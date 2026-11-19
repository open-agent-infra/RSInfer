"""131K shared-prefix Decode sweep for agxinfer.

Verifies every request hits the 126,976-token shared prefix via ``AGENTX_ADMIT`` and times 60
consecutive full-DP pure-decode scheduler retirements after 110 warmup decode steps.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import secrets
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "engine").is_dir())
sys.path[:0] = [str(REPO), str(REPO / "engine")]

from core.contracts.columns import Column  # noqa: E402
from core.contracts.requests import NewRequest, Requests  # noqa: E402
from core.contracts.sampling import SamplingParams  # noqa: E402
from tests.gpu.ds1017_bench.dsbench_pareto_sweep import (  # noqa: E402
    _load_non_special_vocab,
    _open_step_channel,
)


def read_windows(
    path: Path,
    offset: int,
    base: int,
    total: int,
    d: int,
    *,
    dp_size: int = 4,
    cached_tokens: int = 126976,
    warmup_steps: int = 110,
    measure_steps: int = 60,
) -> dict[str, Any]:
    admissions: dict[int, int] = {}
    epochs: list[int] = []
    stamps: list[float] = []
    intervals: list[dict[int, tuple[float, float, float]]] = []
    with path.open(encoding="utf-8") as stream:
        stream.seek(offset)
        for line in stream:
            if line.startswith("AGENTX_ADMIT "):
                _, rid, prefix = line.split()
                req_id = int(rid)
                if base <= req_id < base + total:
                    admissions[req_id] = int(prefix)
            elif line.startswith("DSBENCH_STEP "):
                _, epoch_s, stamp_s, payload = line.split(" ", 3)
                counts_s, timing_s = payload.split("] ", 1)
                if ast.literal_eval(counts_s + "]") == [d] * dp_size:
                    epochs.append(int(epoch_s))
                    stamps.append(float(stamp_s))
                    intervals.append(ast.literal_eval(timing_s))
    assert len(admissions) == total and set(admissions.values()) == {cached_tokens}, (
        len(admissions),
        set(admissions.values()),
    )
    need = warmup_steps + measure_steps + 1
    assert len(stamps) >= need, (len(stamps), d)
    epochs = epochs[warmup_steps : warmup_steps + measure_steps + 1]
    stamps = stamps[warmup_steps : warmup_steps + measure_steps + 1]
    intervals = intervals[warmup_steps + 1 : warmup_steps + measure_steps + 1]
    assert epochs == list(range(epochs[0], epochs[0] + measure_steps + 1))
    steps = [1000.0 * (b - a) for a, b in zip(stamps[:-1], stamps[1:], strict=True)]
    sps = measure_steps / (stamps[-1] - stamps[0])
    device_ms = (
        statistics.mean(max(v[1] for v in entry.values()) / 1000.0 for entry in intervals)
        if all(intervals)
        else None
    )
    return {
        "cached_admissions": len(admissions),
        "decode_window_steps": measure_steps,
        "decode_warmup_steps": warmup_steps,
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
        "decode_avg_sps": sps,
        "decode_tokens_per_s_per_gpu": d * sps,
        "decode_max_step_ms": max(steps),
        "step_ms": steps,
        "device_mean_ms": device_ms,
    }


async def run_decode(args: argparse.Namespace) -> None:
    d: int = args.d
    dp_size: int = args.dp_size
    total = d * dp_size
    prefix: int = args.context_length - args.non_cached_tokens
    suffix: int = args.non_cached_tokens
    rng = np.random.default_rng(args.seed_base + d)
    vocab = np.array(_load_non_special_vocab(args.model)[0], dtype=np.int64)
    bases = [rng.choice(vocab, prefix).tobytes() for _ in range(dp_size)]
    sampling = SamplingParams(0.3, 0.95, -1, 7)
    base = secrets.randbits(30) << 32
    channel, reader, ctx, _cfg = await _open_step_channel(args.run)
    (log,) = args.run.glob("*/snapshots/.run/sched-*/log")
    begin = log.stat().st_size

    def prompt(i: int, n: int) -> Column:
        return Column("i64", bases[i % dp_size] + rng.choice(vocab, n).tobytes())

    async def finish(ids: set[int]) -> None:
        while ids:
            _, event = await channel.receive()
            if event.errors:
                raise RuntimeError(str(event.errors))
            ids.difference_update(event.finished)

    try:
        primer = [NewRequest(base - i - 1, prompt(i, args.yoco_clip), sampling, 1, []) for i in range(dp_size)]
        assert await channel.send(Requests(new=primer)) is not None
        await finish({r.request for r in primer})
        # Enqueue promptly so admission and send time stay outside the steady decode window.
        cols = [prompt(i, suffix) for i in range(total)]
        for j in range(0, total, dp_size):
            reqs = [NewRequest(base + i, cols[i], sampling, args.decode_tokens, []) for i in range(j, j + dp_size)]
            assert await channel.send(Requests(new=reqs)) is not None
        await finish(set(range(base, base + total)))
        result = read_windows(
            log,
            begin,
            base,
            total,
            d,
            dp_size=dp_size,
            cached_tokens=prefix,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        result.update(
            model_name="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            dp_size=dp_size,
            ep_size=dp_size,
            cp=1,
            context_length=prefix + suffix,
            non_cached_tokens=suffix,
            cached_tokens=prefix,
            yoco_clip=args.yoco_clip,
            decode_num_requests=d,
            prefill_batch_size=args.prefill_batch_size,
            nextn=0,
            num_extra_experts=32,
            seed=args.seed_base + d,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")
        print(f"D={d} decode_tokens_per_s_per_gpu={result['decode_tokens_per_s_per_gpu']:.2f} "
              f"decode_avg_sps={result['decode_avg_sps']:.2f}", flush=True)
    finally:
        await channel.close()
        reader.close()
        ctx.destroy(linger=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--d", type=int, required=True, help="Concurrent decode requests per DP rank")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dp-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=131072)
    p.add_argument("--non-cached-tokens", type=int, default=4096)
    p.add_argument("--yoco-clip", type=int, default=140)
    p.add_argument("--prefill-batch-size", type=int, default=32768)
    p.add_argument("--decode-tokens", type=int, default=320)
    p.add_argument("--warmup-steps", type=int, default=110)
    p.add_argument("--measure-steps", type=int, default=60)
    p.add_argument("--seed-base", type=int, default=20000)
    asyncio.run(run_decode(p.parse_args()))


if __name__ == "__main__":
    main()
