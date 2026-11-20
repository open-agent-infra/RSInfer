"""Compile supplied source through the bundled, auditable shared build cache."""
import atexit
import os
from pathlib import Path
_module=None
def build():
    global _module
    if _module is not None:return _module
    os.environ['TORCH_CUDA_ARCH_LIST']='10.0a'
    os.environ['MAX_JOBS']='2'
    from torch.utils.cpp_extension import CUDA_HOME
    from ..aka_cache import load_torch_extension
    root=Path(__file__).resolve().parent
    paths=[root/'bindings.cpp',*sorted((root/'csrc').rglob('*')),*sorted((root/'include').rglob('*'))]
    _module=load_torch_extension(root,paths,
        ['-O3','-std=c++20','-fno-gnu-unique','-fvisibility=hidden','-fvisibility=hidden','-Wno-psabi','-Wno-deprecated-declarations'],
        ['-L'+str(Path(CUDA_HOME)/'lib64'),'-Wl,-Bsymbolic','-lcudart','-lcuda','-lcublasLt'])
    _module.init(str(root));atexit.register(_module.shutdown)
    return _module
