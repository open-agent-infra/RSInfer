# DS1017 DeepInfer prefill kernels

This optional package supplies the ten AKA producer and consumer candidates and the integration fixes selected for DS1017 prefill. The validated deployment target is Linux x86-64, four NVIDIA B200 GPUs, PyTorch 2.13.0+cu132 and CUDA 13.2 with the deployment's DeepInfer libraries. Python 3.12, the CUDA compiler with SM100a support, a C++ compiler, Ninja, CUDA development headers and linkable CUDA driver and cuBLASLt libraries are required to build in a new environment. The package reuses the deployment's DeepGEMM, FlashMLA and DeepSelect official fallback implementations.

## Install

Build with `uv build` in this repository. Install the wheel in the worker Python environment with `python -m pip install --no-deps <wheel>`. The wheel contains the CUDA source, C++ bindings and upstream license files. It needs no AKA agent, credentials, source checkout or network connection on the worker. The host build and JIT use a writable cache. Set `AKA_DS1017_CACHE` to a private directory on a local volume or use the default `~/.cache/aka-ds1017`. Run warmup under the same Python environment, device class and cache as the serving workers.

```sh
python -m aka_ds1017 explain --profile expanded --budget 8192
python -m aka_ds1017 warmup --profile validated --budget 32768
```

Warmup builds host extensions and standalone CUDA libraries. Input-dependent DeepGEMM kernels are compiled when their shapes first execute. Source is under `native/common` and `native/variants`; `native/layout.json` records the components of each variant; the wheel deduplicates identical headers, restores them under a versioned cache path and imports the individual Python bindings within the package namespace. Each original license remains with its component. The upstream AKA revisions and source tree digests are in `src/aka_ds1017/provenance.json`.

## agxinfer settings

Set `deepinfer.aka_profile` to `official` for the original backend (the package is optional), `validated` for 32K and 64K, or `expanded` to also enable the 4K, 8K and 16K paths. The framework passes the actual per-rank scheduler token budget. Set `deepinfer.aka_disable` to a comma-separated subset of `projection,index,mla,mhc,megamoe` to keep those operators official. Apply settings on all ranks; use a fresh worker and graph capture when changing the wheel. INFO logs from `aka_ds1017.runtime` identify the implementation on first dispatch for a shape. The explain command enumerates policy matches; actual calls also check their tensor layout.

Full and compact token counts, projection dimensions, MLA index width, mHC scale layout, Indexer mode and EP configuration affect selection. The 32K MegaMoE fast path requires a uniform prefill step across the four EP domains. 64K MegaMoE, unvalidated small mHC shapes and off-policy shapes use the official implementation. Numeric or CUDA launch failures propagate to the worker. To roll back, select the official profile and rebuild the serving plans.

## Tests

`PYTHONPATH=src python -m unittest discover -s tests` checks dispatch without CUDA. Verify a newly built wheel from a fresh cache, with the versioned agxinfer integration and four-rank matched sweep and answer tests before publishing a change.
