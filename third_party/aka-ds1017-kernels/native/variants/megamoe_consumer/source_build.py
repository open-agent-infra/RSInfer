"""Compile the supplied C++ dispatcher and CUDA headers, never import system DeepGEMM."""
import atexit
import fcntl
import hashlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_module = None

@contextmanager
def compile_lock(path):
    with path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield

def jit_compile():
    return compile_lock(Path(os.environ['DG_JIT_CACHE_DIR']).parent / 'prepare.lock')

def build():
    global _module
    if _module is not None:
        return _module
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load
    root = Path(__file__).resolve().parent
    paths = [root / 'bindings.cpp', *sorted((root / 'csrc').rglob('*')),
             *sorted((root / 'include').rglob('*'))]
    digest = hashlib.sha256()
    for path in [Path(__file__), *paths]:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    digest.update(repr((torch.__version__,torch.version.cuda,torch.compiled_with_cxx11_abi(),
                        sys.implementation.cache_tag,CUDA_HOME,torch.cuda.get_device_capability())).encode())
    key = digest.hexdigest()[:16]
    cache = root / '.aka-build' / key
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['DG_JIT_CACHE_DIR'] = str(cache / 'jit')
    os.environ['MAX_JOBS'] = '2'
    _module = load(name='aka_dg_' + key, sources=[str(root / 'bindings.cpp')],
        extra_include_paths=[str(root), str(root / 'include'), str(Path(CUDA_HOME) / 'include'),
                             str(Path(CUDA_HOME) / 'include/cccl')],
        extra_cflags=['-O3', '-std=c++20','-fno-gnu-unique','-fvisibility=hidden', '-Wno-psabi', '-Wno-deprecated-declarations'],
        extra_ldflags=['-L' + str(Path(CUDA_HOME) / 'lib64'), '-Wl,-Bsymbolic','-lcudart', '-lcuda', '-lcublasLt'],
        with_cuda=True, build_directory=str(cache), verbose=True)
    _module.init(str(root))
    atexit.register(_module.shutdown)
    return _module
