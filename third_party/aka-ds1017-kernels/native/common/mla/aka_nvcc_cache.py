"""Auditable NVCC shared-library cache; caller supplies the complete compiler argv."""
import ctypes
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import torch
from .aka_cache import atomic,document,digest,lock,pin_build,gc_builds

def load(root,files,command):
    root=Path(root)
    recipe={'kind':'ctypes-nvcc-v1','sources':{str(p.relative_to(root)):digest(p.read_bytes()) for p in files if p.is_file()},
        'builder':digest(inspect.getsource(load).encode()),'support':digest((root/'aka_cache.py').read_bytes()),
        'argv':command,'nvcc':subprocess.run([command[0],'--version'],capture_output=True,text=True,check=True).stdout,
        'torch':torch.__version__,'cuda':torch.version.cuda,'abi':torch.compiled_with_cxx11_abi(),
        'python':list(sys.version_info[:2]),'arch':list(torch.cuda.get_device_capability()),
        'sms':torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        'environment':{k:os.environ.get(k,'') for k in ['CXX','CC','CFLAGS','CXXFLAGS','LDFLAGS','CUDAHOSTCXX','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS']}}
    key=digest(document(recipe));base=Path(os.environ.get('AKA_BUILD_CACHE_DIR',str(root/'.aka-build')))
    directory=base/'host'/key;library=directory/'prefill.so';proof=directory/'complete.json';pin_build(base,key)
    with lock(base/'locks'/(key+'.lock')):
        value=json.loads(proof.read_text()) if proof.exists() else {}
        hit=(value.get('recipe')==recipe and library.is_file() and value.get('sha256')==digest(library.read_bytes()))
        if not hit:
            directory.mkdir(parents=True,exist_ok=True);temporary=directory/('prefill-'+uuid.uuid4().hex+'.so')
            argv=[str(temporary) if x=='{output}' else str(root/x[2:]) if x.startswith('-R') else x for x in command]
            # -R represents an input path relative to this immutable source snapshot.
            argv=[('-I'+str(root/x[3:])) if x.startswith('-IR') else x for x in argv]
            try:
                subprocess.run(argv,check=True);temporary.replace(library)
                atomic(proof,document({'recipe':recipe,'library':library.name,'sha256':digest(library.read_bytes())}))
            finally:temporary.unlink(missing_ok=True)
        module=ctypes.CDLL(str(library));os.utime(proof,None)
    print('AKA_BUILD_CACHE='+json.dumps({'key':key,'hit':hit,'library':str(library)}),flush=True)
    gc_builds(base,int(os.environ.get('AKA_BUILD_CACHE_LIMIT',str(2*1024**3))),keep={key})
    return module
