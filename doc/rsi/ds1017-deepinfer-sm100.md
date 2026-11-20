# DeepInfer prefill on four SM100 GPUs

The optional AKA package provides DS1017 implementations for FP8 projections, MLA, Indexer, shifted mHC and EP4 MegaMoE. The DeepInfer unit owns the implementation choice per instance. Calls keep the existing output buffers, scales, strides and process group. The official backend remains the default and works without the package.

The integration starts from `4e26636`. The original benchmark snapshot's DeepInfer `ops.py` and `mega.py` match that commit byte for byte. The head of PR #67 at the time of integration, `f0d6692`, has different versions of both files.

## Install and select a profile

Build the optional wheel from [the package source](../../third_party/aka-ds1017-kernels/README.md). Install it with the same Python executable that will start the workers. On this release the supported runtime uses PyTorch 2.13.0+cu132, CUDA 13.2, NVIDIA SM100 and the DS1017 DeepInfer dependencies. Select the profile with the ordinary deployment settings.

```sh
uv build --wheel third_party/aka-ds1017-kernels --out-dir dist
python -m pip install --no-deps dist/aka_ds1017_kernels-0.1.0-py3-none-any.whl
python -m aka_ds1017 explain --profile expanded --budget 8192
python -m aka_ds1017 warmup --profile validated --budget 32768
```

| Setting | Behavior |
|---|---|
| `--set deepinfer.aka_profile=official` | Original DeepInfer implementations |
| `--set deepinfer.aka_profile=validated` | Select verified paths at per-rank budgets 32,768 and 65,536 |
| `--set deepinfer.aka_profile=expanded` | Also select verified paths at per-rank budgets 4,096, 8,192 and 16,384 |
| `--set deepinfer.aka_disable=mhc,megamoe` | Keep the two named operators official in either tuned profile |

Set `AKA_DS1017_CACHE` to a writable, private cache directory for the workers. The first build needs a CUDA development toolchain. The build uses the source shipped in the wheel; the runtime compiles shape-specific DeepGEMM kernels as the model warms up. Optional `AKA_DS1017_TRACE=1` logs the chosen implementation the first time a worker sees a dispatch key. The explain command reports policy eligibility. Check the worker trace to see actual invocations.

## Dispatch scope

The measured workload uses 131,072-token requests, 126,976 cached tokens and a 4,096-token increment on DP4/EP4 with a 140-token retained suffix. At a 32K budget, the producer operates on 32,768 rows while eight compact requests together contribute 1,120 rows. The equivalent 64K shapes are 65,536 and 2,240.

- The FP8 projection call checks the actual projection dimensions. At small budgets the full-row QB projection uses the candidate; QKV and output projections have separate rules.
- MLA selects its sparse prefill and paged implementations separately. At 4K and 8K the new full-row rule requires 640 sparse indices.
- Indexer takes the caller's index buffer and selects by the dense, source or user mode.
- Shifted mHC selects by physical rows and scale layout. Compact rows 140, 280 and 560 use the official kernel.
- At a 32K budget, MegaMoE buffer layout and execution share the EP4 eligibility check. The scheduler marks a prefill geometry uniform only if all domains have the same ordered token counts, positions and prompt lengths. Idle, mixed and nonuniform steps use the official collective. 64K uses official MegaMoE.

Apply a profile change to the complete deployment and allow agxinfer to drain the old plan and recapture graphs. For a change to the wheel itself, use fresh workers. To roll back, select `official`. The package's changing-input graph regression can be run on one SM100 with `python third_party/aka-ds1017-kernels/tests/replay_mhc.py`.
