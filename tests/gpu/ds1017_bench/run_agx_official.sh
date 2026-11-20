#!/usr/bin/env bash
# Local one-command DS1017 official Prefill & Decode benchmark for agxinfer (4x SM100, DP4/EP4).
#
# Run directly on a GPU host inside a repository checkout; probes are applied only to a
# disposable staged copy so the git working tree stays completely clean.
#
# Usage:
#   ./run_agx_official.sh prefill --model <MODEL_DIR> [OPTIONS] [BUDGET ...]
#       Default budgets: 4096 8192 16384 32768 65536
#   ./run_agx_official.sh decode  --model <MODEL_DIR> [OPTIONS] [D ...]
#       Default concurrencies (per GPU): 4 8 16 32 64 128 256
#   ./run_agx_official.sh serve-prefill --model <MODEL_DIR> [OPTIONS]
#   ./run_agx_official.sh serve-decode  --model <MODEL_DIR> [OPTIONS]
#   ./run_agx_official.sh status --run <RUN_DIR> [--python <PY>]
#   ./run_agx_official.sh stop   --run <RUN_DIR> [--python <PY>]
#
# Options (all can also be passed via environment variables):
#   --model <DIR>        DS1017 model directory (env: AGX_MODEL / MODEL_DIR)
#   --gpus <CSV>         4 GPU indices for DP4/EP4 (default: 0,1,2,3; env: GPUS / N1_GPUS)
#   --python <PY>        Python interpreter (default: active python/python3; env: AGX_PY)
#   --port <PORT>        HTTP port (default: 24739 prefill / 23619 decode; env: PORT / N1_PORT)
#   --run <DIR>          Existing or target engine run dir (env: RUN_DIR / N1_RUN)
#   --out <DIR>          Output directory for JSON/JSONL results (env: OUT_DIR)
#   --work-dir <DIR>     Disposable staging directory (default: <repo>/.runs/bench-...)
#   --cuda-env <FILE>    Optional shell script to source for CUDA toolchain (env: CUDA_ENV_SCRIPT)
#   --keep-engine        Leave the engine running after prefill/decode finishes (env: KEEP_ENGINE=1)
#   --keep-work          Keep the disposable staged source directory on exit (env: KEEP_WORK=1)
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

cmd=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$cmd" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac

find_repo_root() {
    local d="$script_dir"
    while [[ "$d" != "/" ]]; do
        if [[ -d "$d/engine" && -d "$d/deploy" ]]; then
            echo "$d"
            return 0
        fi
        d=$(dirname "$d")
    done
    echo "error: could not find repository root containing engine/ and deploy/" >&2
    exit 2
}

repo=$(find_repo_root)

# ---- Defaults & option parsing ----
py="${AGX_PY:-}"
model="${AGX_MODEL:-${MODEL_DIR:-}}"
gpus_csv="${GPUS:-${N1_GPUS:-0,1,2,3}}"
port="${PORT:-${N1_PORT:-}}"
run="${RUN_DIR:-${N1_RUN:-}}"
out_dir="${OUT_DIR:-}"
work_dir="${WORK_DIR:-}"
cuda_env="${CUDA_ENV_SCRIPT:-}"
keep_engine="${KEEP_ENGINE:-0}"
keep_work="${KEEP_WORK:-0}"
extra_serve_args=()
pos=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --model) model=${2:?missing value for --model}; shift 2 ;;
        --gpus) gpus_csv=${2:?missing value for --gpus}; shift 2 ;;
        --python) py=${2:?missing value for --python}; shift 2 ;;
        --port) port=${2:?missing value for --port}; shift 2 ;;
        --run) run=${2:?missing value for --run}; shift 2 ;;
        --out) out_dir=${2:?missing value for --out}; shift 2 ;;
        --work-dir) work_dir=${2:?missing value for --work-dir}; shift 2 ;;
        --cuda-env) cuda_env=${2:?missing value for --cuda-env}; shift 2 ;;
        --extra-serve-arg) extra_serve_args+=("${2:?missing value for --extra-serve-arg}"); shift 2 ;;
        --keep-engine) keep_engine=1; shift ;;
        --keep-work) keep_work=1; shift ;;
        --) shift; pos+=("$@"); break ;;
        -*) echo "error: unknown option '$1' (see --help)" >&2; exit 2 ;;
        *) pos+=("$1"); shift ;;
    esac
done

# ---- Environment setup ----
if [[ -z "$py" ]]; then
    py=$(command -v python || command -v python3 || true)
fi
if [[ -z "$py" || ! -x "$py" ]]; then
    echo "error: python interpreter not found; activate your venv or pass --python <path>" >&2
    exit 2
fi

if [[ -n "$cuda_env" ]]; then
    if [[ ! -f "$cuda_env" ]]; then
        echo "error: --cuda-env file not found: $cuda_env" >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    source "$cuda_env"
else
    torch_cuda=$("$py" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
    cuda_root=""
    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        if [[ -z "$torch_cuda" ]] || "${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -q "release ${torch_cuda}"; then
            cuda_root="$CUDA_HOME"
        fi
    fi
    if [[ -z "$cuda_root" && -n "$torch_cuda" && -x "/usr/local/cuda-${torch_cuda}/bin/nvcc" ]]; then
        cuda_root="/usr/local/cuda-${torch_cuda}"
    elif [[ -z "$cuda_root" ]] && ! command -v nvcc >/dev/null 2>&1; then
        for d in /usr/local/cuda-13.2 /usr/local/cuda; do
            if [[ -x "$d/bin/nvcc" ]]; then
                cuda_root="$d"
                break
            fi
        done
    fi
    if [[ -n "$cuda_root" ]]; then
        export CUDA_HOME="$cuda_root"
        export CUDA_PATH="$cuda_root"
        export CUDACXX="$cuda_root/bin/nvcc"
        export PATH="$cuda_root/bin:$PATH"
        export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        if [[ -n "${TRITON_PTXAS_PATH:-}" && -x "$cuda_root/bin/ptxas" ]]; then
            export TRITON_PTXAS_PATH="$cuda_root/bin/ptxas"
        fi
    fi
fi

cu13_inc=$("$py" -c '
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia.cu13")
if spec and spec.submodule_search_locations:
    p = pathlib.Path(next(iter(spec.submodule_search_locations))) / "include"
    if p.is_dir():
        print(p)
' 2>/dev/null || true)
if [[ -n "$cu13_inc" ]]; then
    export CPATH="${cu13_inc}${CPATH:+:$CPATH}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONUNBUFFERED=1

require_model() {
    if [[ -z "$model" ]]; then
        echo "error: model directory required; pass --model /path/to/DeepSeek-V4.1-Flash-1017 or set AGX_MODEL" >&2
        exit 2
    fi
    if [[ ! -d "$model" || ! -f "$model/config.json" ]]; then
        echo "error: invalid model directory (missing config.json): $model" >&2
        exit 2
    fi
}

assert_gpus_free() {
    local vals
    vals=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus_csv")
    awk 'NF{n++; if ($1>1500) busy=1} END{exit !(n>0 && !busy)}' <<< "$vals" || {
        echo "error: GPUs ${gpus_csv} are busy (MiB used: $(tr '\n' ' ' <<< "$vals"))" >&2
        return 1
    }
}

patch_staged_tree() {
    local dst=$1 mode=$2
    local bdir="$dst/tests/gpu/ds1017_bench"
    command -v patch >/dev/null 2>&1 || { echo "error: 'patch' utility is required" >&2; exit 2; }
    local patchfile="$bdir/top5-measurement.patch"
    if ! (cd "$dst" && patch --force -s -R -p1 --dry-run < "$patchfile" >/dev/null 2>&1); then
        (cd "$dst" && patch -s -p1 < "$patchfile")
    fi
    "$py" "$bdir/admit_probe.py" "$dst"
    "$py" "$bdir/dsbench_step_probe.py" "$dst"
    if [[ "$mode" == "decode" ]]; then
        "$py" "$bdir/decode_counts_probe.py" "$dst"
    fi
}

stage_tree() {
    local mode=$1 target=${2:-}
    if [[ -f "$repo/source-version.txt" ]]; then
        patch_staged_tree "$repo" "$mode" >&2
        echo "$repo"
        return 0
    fi
    local dst="${target:-$work_dir}"
    if [[ -z "$dst" ]]; then
        dst="$repo/.runs/bench-agx-${mode}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    fi
    mkdir -p "$dst"
    tar cf - --exclude='__pycache__' --exclude='*.pyc' --exclude='._*' --exclude='extension' \
        -C "$repo" engine deploy tests/gpu/ds1017_bench | tar xf - -C "$dst"
    {
        git -C "$repo" rev-parse HEAD 2>/dev/null || echo "unknown"
        git -C "$repo" status --short -- engine deploy tests/gpu/ds1017_bench 2>/dev/null || true
    } > "$dst/source-version.txt"
    patch_staged_tree "$dst" "$mode" >&2
    echo "$dst"
}

apply_config() {
    local src=$1 run_dir=$2 cfg=$3 out_json=$4
    PYTHONPATH="$src/engine:$src" "$py" "$src/deploy/agxinfer.py" control --run "$run_dir" apply \
        --source "$src" --config "$cfg" --wait-empty 180 --json > "$out_json"
    "$py" -c 'import json,sys; r=json.load(open(sys.argv[1]))["result"]; assert r["status"]=="committed", r.get("diagnostics", r["status"])' "$out_json"
}

serve_engine() {
    local mode=$1 src=$2 run_dir=$3 srv_port=$4
    assert_gpus_free
    export PYTHONPATH="$src/engine:$src"
    if [[ "$mode" == "prefill" ]]; then
        "$py" "$src/deploy/agxinfer.py" serve "$model" --source "$src" --run "$run_dir" --detach --port "$srv_port" \
            --gpus "$gpus_csv" --ds-backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 \
            --max-context 135168 --max-rows 64 --max-slots 320 --token-budget 4096 --overlap --mem-fraction .75 \
            --yoco-clip 140 --timeout 1800 --proc-tunables '{"ipc.shm_capacity":134217728}' \
            --set eplb.extra_experts=32 --set deepinfer.force_random_gate=true --set sampler.force_topk=1024 \
            --set sched.prefill_interval=108 --set sched.prefill_occupancy_target=0 \
            "${extra_serve_args[@]}"
    else
        "$py" "$src/deploy/agxinfer.py" serve "$model" --source "$src" --run "$run_dir" --detach --port "$srv_port" \
            --gpus "$gpus_csv" --ds-backend deepinfer --cuda-graph auto \
            --graph-buckets 1 2 4 8 16 32 64 128 256 \
            --max-context 135168 --max-rows 256 --max-slots 1440 --token-budget 32768 --overlap --mem-fraction .78 \
            --yoco-clip 140 --timeout 1800 --proc-tunables '{"ipc.shm_capacity":16777216}' \
            --set frontend.response_timeout_s=1800.0 --set eplb.extra_experts=32 \
            --set deepinfer.force_random_gate=true --set sampler.force_topk=1024 --set sched.prefill_interval=1 \
            "${extra_serve_args[@]}"
    fi
}

case "$cmd" in
    status)
        [[ -n "$run" ]] || { echo "error: pass --run <RUN_DIR>" >&2; exit 2; }
        PYTHONPATH="$repo/engine:$repo" exec "$py" "$repo/deploy/agxinfer.py" status --run "$run"
        ;;
    stop)
        [[ -n "$run" ]] || { echo "error: pass --run <RUN_DIR>" >&2; exit 2; }
        PYTHONPATH="$repo/engine:$repo" exec "$py" "$repo/deploy/agxinfer.py" stop --run "$run" --timeout 90
        ;;
    serve-prefill|serve-decode)
        require_model
        mode="${cmd#serve-}"
        run="${run:-$repo/.runs/agx-${mode}-run}"
        mkdir -p "$run"
        staged=$(stage_tree "$mode" "$run/staged-src")
        srv_port="${port:-$([[ "$mode" == "prefill" ]] && echo 24739 || echo 23619)}"
        serve_engine "$mode" "$staged" "$run" "$srv_port"
        ;;
    prefill)
        require_model
        if [[ ${#pos[@]} -eq 0 ]]; then pos=(4096 8192 16384 32768 65536); fi
        for b in "${pos[@]}"; do
            case "$b" in 4096|8192|16384|32768|65536) ;; *) echo "error: invalid prefill budget: $b" >&2; exit 2 ;; esac
        done

        created_stage=0
        started_here=0
        if [[ -n "$run" && -f "$run/engine.json" ]]; then
            staged="${run}/staged-src"
            [[ -d "$staged/engine" ]] || staged=$(stage_tree prefill)
        else
            staged=$(stage_tree prefill)
            [[ "$staged" != "$repo" ]] && created_stage=1
            run="${run:-$staged/run-prefill}"
            srv_port="${port:-24739}"
            serve_engine prefill "$staged" "$run" "$srv_port"
            started_here=1
        fi
        cleanup() {
            local rc=$?
            if [[ $started_here -eq 1 && "$keep_engine" != "1" ]]; then
                PYTHONPATH="$staged/engine:$staged" "$py" "$staged/deploy/agxinfer.py" stop --run "$run" --timeout 90 || true
            fi
            if [[ $created_stage -eq 1 && "$keep_work" != "1" && "$keep_engine" != "1" ]]; then
                if [[ $rc -eq 0 ]]; then
                    rm -rf "$staged"
                else
                    echo "note: benchmark exited with code ${rc}; staged work dir kept for inspection: ${staged}" >&2
                fi
            fi
        }
        trap cleanup EXIT

        out="${out_dir:-$repo/.runs/results/agx-prefill-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        mkdir -p "$out"
        cp "$staged/source-version.txt" "$out/source-version.txt" 2>/dev/null || true
        nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"

        export PYTHONPATH="$staged/engine:$staged"
        for b in "${pos[@]}"; do
            echo "$(date -Is) AGX prefill budget=${b} (gpus=${gpus_csv})"
            for cache in false true; do
                cfg="{\"sched.token_budget\":$b,\"alloc.radix_cache\":$cache,\"sched.max_batch\":64,\"batch.max_rows\":64,\"holder.max_slots\":320,\"plan.cuda_graph\":\"auto\",\"plan.graph_buckets\":[1,2,4,8,16,32],\"sched.prefill_interval\":108,\"sched.prefill_occupancy_target\":0,\"eplb.extra_experts\":32,\"deepinfer.force_random_gate\":true,\"sampler.force_topk\":1024,\"dsv4.yoco_mode\":\"compact\"}"
                apply_config "$staged" "$run" "$cfg" "$out/apply-${b}-${cache}.json"
            done
            "$py" "$staged/tests/gpu/ds1017_bench/dsbench_pareto_sweep.py" --run "$run" --model "$model" --mode prefill \
                --context-length 131072 --non-cached-tokens 4096 --yoco-clip 140 --prefill-batch-size "$b" \
                --seed-base 20000 --prefill-measure-steps 24 --prefill-tail-steps 4 --step-stats \
                --result-jsonl "$out/agx-prefill-${b}.jsonl"
            echo "RESULT $out/agx-prefill-${b}.jsonl"
        done
        if [[ $started_here -eq 1 && "$keep_engine" == "1" ]]; then
            echo "Engine left running at ${run}"
        fi
        ;;
    decode)
        require_model
        if [[ ${#pos[@]} -eq 0 ]]; then pos=(4 8 16 32 64 128 256); fi

        created_stage=0
        started_here=0
        if [[ -n "$run" && -f "$run/engine.json" ]]; then
            staged="${run}/staged-src"
            [[ -d "$staged/engine" ]] || staged=$(stage_tree decode)
        else
            staged=$(stage_tree decode)
            [[ "$staged" != "$repo" ]] && created_stage=1
            run="${run:-$staged/run-decode}"
            srv_port="${port:-23619}"
            serve_engine decode "$staged" "$run" "$srv_port"
            started_here=1
        fi
        cleanup() {
            local rc=$?
            if [[ $started_here -eq 1 && "$keep_engine" != "1" ]]; then
                PYTHONPATH="$staged/engine:$staged" "$py" "$staged/deploy/agxinfer.py" stop --run "$run" --timeout 90 || true
            fi
            if [[ $created_stage -eq 1 && "$keep_work" != "1" && "$keep_engine" != "1" ]]; then
                if [[ $rc -eq 0 ]]; then
                    rm -rf "$staged"
                else
                    echo "note: benchmark exited with code ${rc}; staged work dir kept for inspection: ${staged}" >&2
                fi
            fi
        }
        trap cleanup EXIT

        out="${out_dir:-$repo/.runs/results/agx-decode-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        mkdir -p "$out"
        cp "$staged/source-version.txt" "$out/source-version.txt" 2>/dev/null || true
        nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"

        export PYTHONPATH="$staged/engine:$staged"
        for d in "${pos[@]}"; do
            echo "$(date -Is) AGX decode D=${d} (gpus=${gpus_csv})"
            for cache in false true; do
                apply_config "$staged" "$run" "{\"alloc.radix_cache\":$cache}" "$out/apply-d${d}-${cache}.json"
            done
            "$py" "$staged/tests/gpu/ds1017_bench/dsbench_decode.py" --run "$run" --model "$model" --d "$d" \
                --out "$out/agx-decode-${d}.json"
            echo "RESULT $out/agx-decode-${d}.json"
        done
        if [[ $started_here -eq 1 && "$keep_engine" == "1" ]]; then
            echo "Engine left running at ${run}"
        fi
        ;;
    *)
        echo "error: unknown command '$cmd'" >&2
        usage >&2
        exit 2
        ;;
esac
