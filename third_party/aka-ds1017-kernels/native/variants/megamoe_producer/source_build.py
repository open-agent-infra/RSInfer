"""Compile the supplied C++ dispatcher and CUDA headers, never import system DeepGEMM."""
import atexit
import hashlib
import os
from pathlib import Path

_module = None

def build():
    global _module
    if _module is not None:
        return _module
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load
    root = Path(__file__).resolve().parent
    paths = [root / 'bindings.cpp', *sorted((root / 'csrc').rglob('*')),
             *sorted((root / 'include').rglob('*'))]
    if os.environ.get('AKA_BUILD_CACHE_DIR'):
        from .aka_cache import load_torch_extension
        _module=load_torch_extension(root,paths,
            ['-O3','-std=c++20','-fno-gnu-unique','-fvisibility=hidden','-Wno-psabi','-Wno-deprecated-declarations'],
            ['-L'+str(Path(CUDA_HOME)/'lib64'),'-Wl,-Bsymbolic','-lcudart','-lcuda','-lcublasLt'])
        _module.init(str(root))
        atexit.register(_module.shutdown)
        return _module
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    digest.update(repr((torch.__version__,torch.version.cuda,torch.compiled_with_cxx11_abi())).encode())
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
