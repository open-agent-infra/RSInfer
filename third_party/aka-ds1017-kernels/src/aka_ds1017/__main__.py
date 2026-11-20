"""Inspect a profile without CUDA, or compile its selected operators before serving."""
import argparse
from .policy import Policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("explain","warmup"))
    parser.add_argument("--profile", choices=("official","validated","expanded"), required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--disable", default="")
    args = parser.parse_args()
    policy = Policy.create(args.profile, args.budget, args.disable)
    layouts = {
        "projection": ("qkv","qb","wo_a","wo_b"),
        "mla": ("prefill","sparse640","paged"),
        "index": ("dense","source","user"),
        "mhc": ("attn","moe","bf16"),
        "megamoe": ("uniform-ep4",),
    }
    candidates=set()
    for rows in sorted({policy.budget, 140*(policy.budget//4096)}):
        for op, paths in layouts.items():
            for path in paths:
                impl = policy.select(op, rows, path)
                print(f"{op:12} rows={rows:6} {path:12} {impl or 'official'}")
                if impl:
                    candidates.add(impl)
    if args.action == "warmup" and candidates:
        from .runtime import Kernels
        from .loader import load
        Kernels(policy)
        for name in sorted(candidates):
            load(name)
            print("built",name,flush=True)


if __name__ == "__main__":
    main()
