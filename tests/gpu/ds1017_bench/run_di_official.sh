#!/usr/bin/env bash
# Local one-command DS1017 official Prefill & Decode benchmark for stock DeepInfer (4x SM100, DP4/EP4).
#
# Usage:
#   ./run_di_official.sh prefill --model <MODEL_DIR> --di-root <DI_DIR> [OPTIONS] [BUDGET ...]
#       Default budgets: 4096 8192 16384 32768 65536
#   ./run_di_official.sh decode  --model <MODEL_DIR> --di-root <DI_DIR> [OPTIONS] [D ...]
#       Default concurrencies (per GPU): 4 8 16 32 64 128 256
#
# Options (all can also be passed via environment variables):
#   --model <DIR>        DS1017 model directory (env: AGX_MODEL / MODEL_DIR)
#   --di-root <DIR>      Stock DeepInfer repo directory (env: DI_ROOT)
#   --python <PY>        Python interpreter (default: active python/python3; env: DI_PY)
#   --gpus <CSV>         4 GPU indices for DP4/EP4 (default: 0,1,2,3; env: GPUS / N1_GPUS)
#   --mode <MODE>        Prefill mode: resident (default) or cold (env: DI_MODE)
#   --engram-qps <N>     Optional Engram QP override (env: ENGRAM_QPS)
#   --master-port <PORT> Rendezvous port (default: 36995 prefill / 35927 decode; env: MASTER_PORT)
#   --out <DIR>          Output directory for JSONL results (env: OUT_DIR)
#   --cuda-env <FILE>    Optional shell script to source for CUDA toolchain (env: CUDA_ENV_SCRIPT)
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

usage() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
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
    pwd
}

repo=$(find_repo_root)

# ---- Defaults & option parsing ----
di="${DI_ROOT:-}"
py="${DI_PY:-}"
model="${AGX_MODEL:-${MODEL_DIR:-}}"
gpus_csv="${GPUS:-${N1_GPUS:-0,1,2,3}}"
di_mode="${DI_MODE:-resident}"
engram_qps="${ENGRAM_QPS:-}"
master_port="${MASTER_PORT:-}"
out_dir="${OUT_DIR:-}"
cuda_env="${CUDA_ENV_SCRIPT:-}"
pos=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --model) model=${2:?missing value for --model}; shift 2 ;;
        --di-root) di=${2:?missing value for --di-root}; shift 2 ;;
        --python) py=${2:?missing value for --python}; shift 2 ;;
        --gpus) gpus_csv=${2:?missing value for --gpus}; shift 2 ;;
        --mode) di_mode=${2:?missing value for --mode}; shift 2 ;;
        --engram-qps) engram_qps=${2:?missing value for --engram-qps}; shift 2 ;;
        --master-port) master_port=${2:?missing value for --master-port}; shift 2 ;;
        --out) out_dir=${2:?missing value for --out}; shift 2 ;;
        --cuda-env) cuda_env=${2:?missing value for --cuda-env}; shift 2 ;;
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
if [[ -z "$model" || ! -d "$model" || ! -f "$model/config.json" ]]; then
    echo "error: valid DS1017 model directory required; pass --model <DIR> or set AGX_MODEL" >&2
    exit 2
fi
if [[ -z "$di" || ! -d "$di" ]]; then
    echo "error: stock DeepInfer directory required; pass --di-root <DIR> or set DI_ROOT" >&2
    exit 2
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

device_offset="${DEVICE_OFFSET:-${gpus_csv%%,*}}"
export CUDA_VISIBLE_DEVICES="$gpus_csv" DEVICE_OFFSET="$device_offset" WORLD_SIZE=1 PROFILE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
export DEEPINFER_HF_MODEL_DIR="$model"
export DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_ENABLE_MTP_VARLEN=0
export DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=32 DEEPINFER_USE_MULTIMODAL_POOL=0
export DEEPINFER_ROOT="$di" PYTHONPATH="$di:$di/examples"
if [[ -n "$engram_qps" ]]; then
    export ENGRAM_QPS="$engram_qps"
fi
unset DEEPINFER_ENGRAM_LOCAL NVSHMEM_IB_QP_DEPTH NVSHMEM_IBGDA_NUM_RC_PER_PE

assert_gpus_free() {
    local vals
    vals=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus_csv")
    awk 'NF{n++; if ($1>1500) busy=1} END{exit !(n>0 && !busy)}' <<< "$vals" || {
        echo "error: GPUs ${gpus_csv} are busy (MiB used: $(tr '\n' ' ' <<< "$vals"))" >&2
        return 1
    }
}

out="${out_dir:-$repo/.runs/results/di-${cmd}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$out"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"
git -c safe.directory="$di" -C "$di" rev-parse HEAD > "$out/source.txt" 2>/dev/null || true

case "$cmd" in
    prefill)
        if [[ ${#pos[@]} -eq 0 ]]; then pos=(4096 8192 16384 32768 65536); fi
        assert_gpus_free
        port="${master_port:-36995}"
        if [[ "$di_mode" == "cold" ]]; then
            for b in "${pos[@]}"; do
                echo "$(date -Is) DeepInfer cold prefill budget=${b} (gpus=${gpus_csv})"
                if [[ -n "$engram_qps" ]]; then
                    SWEEP_MODE=prefill SWEEP_POINTS="$b" SWEEP_OUT="$out" SWEEP_PREFIX="di-prefill" \
                        MASTER_PORT="$port" \
                        "$py" "$script_dir/di_official_sweep.py" \
                        --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size "$b" \
                        --decode_num_requests 32 --max_model_len 135168 --yoco_clip 140 \
                        --sampler_force_topk 1024 --result_jsonl "$out/init-${b}.jsonl"
                else
                    (cd "$di" && MASTER_PORT="$port" "$py" examples/benchmark.py \
                        --model_name deepseek-ai/DeepSeek-V4.1-Flash-1017 --cp 1 \
                        --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size "$b" \
                        --decode_num_requests 32 --max_model_len 135168 --yoco_clip 140 \
                        --sampler_force_topk 1024 --result_jsonl "$out/di-prefill-${b}.jsonl")
                fi
                echo "RESULT $out/di-prefill-${b}.jsonl"
                port=$((port + 1))
            done
        else
            max_b=0
            points_csv=""
            for b in "${pos[@]}"; do
                if (( b > max_b )); then max_b=$b; fi
                points_csv="${points_csv:+${points_csv},}${b}"
            done
            echo "$(date -Is) DeepInfer resident prefill points=${points_csv} (init_budget=${max_b}, gpus=${gpus_csv})"
            SWEEP_MODE=prefill SWEEP_POINTS="$points_csv" SWEEP_OUT="$out" SWEEP_PREFIX="di-prefill" \
                MASTER_PORT="$port" \
                "$py" "$script_dir/di_official_sweep.py" \
                --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size "$max_b" \
                --decode_num_requests 32 --max_model_len 135168 --yoco_clip 140 \
                --sampler_force_topk 1024 --result_jsonl "$out/init.jsonl"
        fi
        echo "$(date -Is) completed prefill -> ${out}"
        ;;
    decode)
        if [[ ${#pos[@]} -eq 0 ]]; then pos=(4 8 16 32 64 128 256); fi
        assert_gpus_free
        resident_csv=""
        cold_list=()
        for d in "${pos[@]}"; do
            if (( d <= 248 )); then
                resident_csv="${resident_csv:+${resident_csv},}${d}"
            else
                cold_list+=("$d")
            fi
        done
        port="${master_port:-35927}"
        if [[ -n "$resident_csv" ]]; then
            first_d="${resident_csv%%,*}"
            echo "$(date -Is) DeepInfer resident decode D=${resident_csv} (gpus=${gpus_csv})"
            SWEEP_MODE=decode SWEEP_POINTS="$resident_csv" SWEEP_OUT="$out" SWEEP_PREFIX="di-decode" \
                MASTER_PORT="$port" \
                "$py" "$script_dir/di_official_sweep.py" \
                --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size 32768 \
                --decode_num_requests "$first_d" --max_model_len 135168 --yoco_clip 140 \
                --sampler_force_topk 1024 --result_jsonl "$out/init-decode.jsonl"
            port=$((port + 1))
        fi
        for d in "${cold_list[@]}"; do
            echo "$(date -Is) DeepInfer cold decode D=${d} (gpus=${gpus_csv})"
            SWEEP_MODE=decode SWEEP_POINTS="$d" SWEEP_OUT="$out" SWEEP_PREFIX="di-decode" \
                MASTER_PORT="$port" \
                "$py" "$script_dir/di_official_sweep.py" \
                --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size 32768 \
                --decode_num_requests "$d" --max_model_len 135168 --yoco_clip 140 \
                --sampler_force_topk 1024 --result_jsonl "$out/init-decode-${d}.jsonl"
            port=$((port + 1))
        done
        echo "$(date -Is) completed decode -> ${out}"
        ;;
    *)
        echo "error: unknown command '$cmd'" >&2
        usage >&2
        exit 2
        ;;
esac
