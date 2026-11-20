"""Optional DS1017 DeepInfer acceleration. Importing this module does not initialize CUDA."""

from .policy import Policy

__version__ = "0.1.0"

def create(profile, budget, disabled=""):
    from .runtime import Kernels
    return Kernels(Policy.create(profile, budget, disabled))
