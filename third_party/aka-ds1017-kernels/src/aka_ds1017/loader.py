"""Load a private, content-addressed source tree on a writable local cache volume."""
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
from functools import cache
from pathlib import Path
import platform
import sys
import tarfile
import tempfile
import threading

from . import _vendor

_PACKAGE = Path(__file__).resolve().parent
_guard = threading.RLock()


@contextlib.contextmanager
def locked(path):
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@cache
def source_tree():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    import subprocess

    digest = json.loads((_PACKAGE / "provenance.json").read_text())["archive_sha256"]
    compiler = subprocess.check_output([str(Path(CUDA_HOME) / "bin/nvcc"), "--version"])
    abi = (platform.machine(), sys.implementation.cache_tag, torch.__version__, torch.version.cuda,
           torch.compiled_with_cxx11_abi(), torch.cuda.get_device_capability(), compiler.decode())
    base = Path(os.getenv("AKA_DS1017_CACHE", Path.home() / ".cache" / "aka-ds1017")).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / (digest + "-" + hashlib.sha256(repr(abi).encode()).hexdigest()[:16])
    with locked(base / (root.name + ".lock")):
        if not root.exists():
            archive = _PACKAGE / "kernels.tar.gz"
            with archive.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                    raise ValueError("DS1017 kernel source checksum mismatch")
            with tempfile.TemporaryDirectory(dir=base) as temporary:
                stage = Path(temporary) / "sources"
                stage.mkdir()
                with tarfile.open(archive) as bundle:
                    bundle.extractall(stage, filter="data")
                stage.rename(root)
    _vendor.__path__.append(str(root))
    return root


def paged_bridge(root):
    from torch.utils.cpp_extension import CUDA_HOME, load
    src = root / "paged_bridge"
    headers = root / "mla_r1720" / "include"
    header = next((src / "include").rglob("kernel.cuh"))
    build = root / "paged-build"
    build.mkdir(exist_ok=True)
    import subprocess
    cuda = Path(CUDA_HOME)
    obj = build / "paged.o"
    if not obj.exists():
        temporary = obj.with_suffix(".part.o")
        subprocess.run([
            str(cuda/"bin/nvcc"), str(src/"paged.cu"), "-c", "-o", str(temporary), "-Xcompiler=-fPIC",
            "-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "-lineinfo", "--ptxas-options=-v,--register-usage-level=10", "-DFLASH_MLA_IS_BUILD_ON_CUDA",
            "-DKERUTILS_IS_BUILD_ON_CUDA", "-DNDEBUG", "-gencode=arch=compute_100a,code=sm_100a",
            "-I"+str(src/"include"), "-I"+str(headers/header.relative_to(src/"include").parent),
            "-I"+str(headers), "-I"+str(cuda/"include/cccl"),
        ], check=True)
        temporary.replace(obj)
    return load(name="aka_ds1017_paged", sources=[str(src/"pr69_paged.cpp")],
                build_directory=str(build), with_cuda=True, extra_cflags=["-O2"],
                extra_ldflags=[str(obj), "-lcuda"])


@cache
def load(name):
    root = source_tree()
    # Existing source builders set environment variables as part of DeepGEMM JIT initialization.
    # Keep those settings private to the build, including when another variant was loaded earlier.
    keys = ("TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "DG_JIT_CACHE_DIR", "AKA_BUILD_CACHE_DIR", "AKA_JIT_BUILD_KEY")
    with _guard, locked(root / (name + ".lock")):
        previous = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["AKA_BUILD_CACHE_DIR"] = str(root/"build")
            os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
            os.environ["MAX_JOBS"] = "2"
            if name == "paged_bridge":
                return paged_bridge(root)
            mod = importlib.import_module(f"{_vendor.__name__}.{name}.module")
            native = mod.build()
            if not name.startswith("mla") and hasattr(native, "get_jit"):
                native.get_jit()
            mod.native = native
            if name.startswith("projection"):
                mod.qkv = mod.build_official_qkv().run
            return mod
        finally:
            for k, value in previous.items():
                if value is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = value
