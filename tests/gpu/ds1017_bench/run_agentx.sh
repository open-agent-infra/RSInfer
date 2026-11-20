#!/usr/bin/env bash
# Local one-command AgentX dataset generation and replay benchmark for agxinfer / optimized-deepinfer & DeepInfer (4x SM100, DP4/EP4).
#
# Run directly on a GPU host inside a repository checkout; the admission probe is applied only
# to a disposable staged copy so the git working tree stays completely clean.
#
# Usage:
#   ./run_agentx.sh prepare  --model <MODEL_DIR> <WEKA_TRACES_JSONL> <OUT_DATA_DIR> [OPTIONS]
#   ./run_agentx.sh run-agx  --model <MODEL_DIR> [OPTIONS] <SUITE_PREFIX> [CONCURRENCIES ...]
#       Default concurrencies: 16 24 48 (pass e.g. `32 32` to run cold + JIT-warm back-to-back)
#   ./run_agentx.sh run-di   --model <MODEL_DIR> --di-root <DI_DIR> [OPTIONS] <SUITE_PREFIX> [CONCURRENCIES ...]
#   ./run_agentx.sh serve-agx --model <MODEL_DIR> [OPTIONS]
#   ./run_agentx.sh status   --run <RUN_DIR> [--python <PY>]
#   ./run_agentx.sh stop-agx --run <RUN_DIR> [--python <PY>]
#
# Notes:
#   <SUITE_PREFIX> refers to <prefix>.json and <prefix>.i32 (e.g. ./agentx-data/agentx_eval).
#
# Options (all can also be passed via environment variables):
#   --model <DIR>        DS1017 model directory (env: AGX_MODEL / MODEL_DIR)
#   --gpus <CSV>         4 GPU indices for DP4/EP4 (default: 0,1,2,3; env: GPUS / N1_GPUS)
#   --python <PY>        Python interpreter for agxinfer (default: active python/python3; env: AGX_PY)
#   --aka-profile <P>    AKA kernel policy: auto (default), official, validated, expanded (env: AKA_PROFILE)
#   --aka-disable <OPS>  Comma-separated operators to keep official (env: AKA_DISABLE)
#   --aka-cache <DIR>    Writable cache dir for AKA kernels (env: AKA_DS1017_CACHE)
#   --aka-warmup         Run `python -m aka_ds1017 warmup --budget 32768` before serving (env: AKA_WARMUP=1)
#   --di-root <DIR>      Stock DeepInfer repo path for run-di (env: DI_ROOT)
#   --di-python <PY>     Python interpreter for run-di (default: --python; env: DI_PY)
#   --port <PORT>        HTTP port for agxinfer (default: 24919; env: PORT / N1_PORT)
#   --master-port <PORT> Rendezvous port for run-di (default: 36897; env: MASTER_PORT)
#   --run <DIR>          Existing or target agxinfer run dir (env: RUN_DIR / N1_RUN)
#   --out <DIR>          Output directory for results (env: OUT_DIR)
#   --work-dir <DIR>     Disposable staging directory (default: <repo>/.runs/bench-...)
#   --max-context <N>    Max context length (default: 131072; env: MAX_CONTEXT)
#   --yoco-clip <N>      YoCo retained suffix tokens, 0 to disable (default: 140; env: YOCO_CLIP)
#   --dev-sessions <N>   Sessions in agentx_dev for prepare (default: 16; env: DEV_SESSIONS)
#   --eval-sessions <N>  Sessions in agentx_eval for prepare (default: 48; env: EVAL_SESSIONS)
#   --cuda-env <FILE>    Optional shell script to source for CUDA toolchain (env: CUDA_ENV_SCRIPT)
#   --keep-engine        Leave agxinfer running after run-agx finishes (env: KEEP_ENGINE=1)
#   --keep-work          Keep disposable staged source directory on exit (env: KEEP_WORK=1)
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
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
di="${DI_ROOT:-}"
dipy="${DI_PY:-}"
model="${AGX_MODEL:-${MODEL_DIR:-}}"
gpus_csv="${GPUS:-${N1_GPUS:-0,1,2,3}}"
port="${PORT:-${N1_PORT:-24919}}"
master_port="${MASTER_PORT:-36897}"
run="${RUN_DIR:-${N1_RUN:-}}"
out_dir="${OUT_DIR:-}"
work_dir="${WORK_DIR:-}"
max_ctx="${MAX_CONTEXT:-131072}"
yoco_clip="${YOCO_CLIP:-140}"
dev_sessions="${DEV_SESSIONS:-16}"
eval_sessions="${EVAL_SESSIONS:-48}"
cuda_env="${CUDA_ENV_SCRIPT:-}"
keep_engine="${KEEP_ENGINE:-0}"
keep_work="${KEEP_WORK:-0}"
aka_profile="${AKA_PROFILE:-auto}"
aka_disable="${AKA_DISABLE:-}"
aka_cache="${AKA_DS1017_CACHE:-}"
aka_warmup="${AKA_WARMUP:-0}"
extra_serve_args=()
pos=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --model) model=${2:?missing value for --model}; shift 2 ;;
        --gpus) gpus_csv=${2:?missing value for --gpus}; shift 2 ;;
        --python) py=${2:?missing value for --python}; shift 2 ;;
        --aka-profile) aka_profile=${2:?missing value for --aka-profile}; shift 2 ;;
        --aka-disable) aka_disable=${2:?missing value for --aka-disable}; shift 2 ;;
        --aka-cache) aka_cache=${2:?missing value for --aka-cache}; shift 2 ;;
        --aka-warmup) aka_warmup=1; shift ;;
        --di-root) di=${2:?missing value for --di-root}; shift 2 ;;
        --di-python) dipy=${2:?missing value for --di-python}; shift 2 ;;
        --port) port=${2:?missing value for --port}; shift 2 ;;
        --master-port) master_port=${2:?missing value for --master-port}; shift 2 ;;
        --run) run=${2:?missing value for --run}; shift 2 ;;
        --out) out_dir=${2:?missing value for --out}; shift 2 ;;
        --work-dir) work_dir=${2:?missing value for --work-dir}; shift 2 ;;
        --max-context) max_ctx=${2:?missing value for --max-context}; shift 2 ;;
        --yoco-clip) yoco_clip=${2:?missing value for --yoco-clip}; shift 2 ;;
        --dev-sessions) dev_sessions=${2:?missing value for --dev-sessions}; shift 2 ;;
        --eval-sessions) eval_sessions=${2:?missing value for --eval-sessions}; shift 2 ;;
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
dipy="${dipy:-$py}"

detect_cu13() {
    local interp=$1
    "$interp" -c '
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia.cu13")
if spec and spec.submodule_search_locations:
    p = pathlib.Path(next(iter(spec.submodule_search_locations))) / "include"
    if p.is_dir():
        print(p)
' 2>/dev/null || true
}

cu13_inc=$(detect_cu13 "$py")
if [[ -n "$cu13_inc" ]]; then
    export CPATH="${cu13_inc}${CPATH:+:$CPATH}"
fi
if [[ -n "$aka_cache" ]]; then
    mkdir -p "$aka_cache"
    export AKA_DS1017_CACHE="$aka_cache"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONUNBUFFERED=1

resolve_aka_profile() {
    local has_aka=0
    if "$py" -c "import aka_ds1017" >/dev/null 2>&1; then
        has_aka=1
    fi
    if [[ "$aka_profile" == "auto" ]]; then
        if [[ $has_aka -eq 1 ]]; then
            aka_profile="validated"
            echo "AKA profile: validated (auto-detected installed aka_ds1017)" >&2
        else
            aka_profile="official"
            echo "AKA profile: official (aka_ds1017 not installed in $py; install third_party/aka-ds1017-kernels wheel to enable tuned kernels)" >&2
        fi
    elif [[ "$aka_profile" == "validated" || "$aka_profile" == "expanded" ]]; then
        if [[ $has_aka -ne 1 ]]; then
            echo "error: --aka-profile $aka_profile requires aka-ds1017-kernels installed in $py" >&2
            echo "  Build & install:" >&2
            echo "    uv build --wheel third_party/aka-ds1017-kernels --out-dir dist" >&2
            echo "    $py -m pip install --no-deps dist/aka_ds1017_kernels-0.1.0-py3-none-any.whl" >&2
            exit 2
        fi
    elif [[ "$aka_profile" != "official" ]]; then
        echo "error: invalid --aka-profile '$aka_profile' (choose auto, official, validated, expanded)" >&2
        exit 2
    fi
}

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

stage_tree() {
    local target=${1:-}
    if [[ -f "$repo/source-version.txt" ]]; then
        "$py" "$repo/tests/gpu/ds1017_bench/admit_probe.py" "$repo" >&2
        echo "$repo"
        return 0
    fi
    local dst="${target:-$work_dir}"
    if [[ -z "$dst" ]]; then
        dst="$repo/.runs/bench-agentx-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    fi
    mkdir -p "$dst"
    tar cf - --exclude='__pycache__' --exclude='*.pyc' --exclude='._*' --exclude='extension' \
        -C "$repo" engine deploy tests/gpu/ds1017_bench | tar xf - -C "$dst"
    {
        git -C "$repo" rev-parse HEAD 2>/dev/null || echo "unknown"
        git -C "$repo" status --short -- engine deploy tests/gpu/ds1017_bench 2>/dev/null || true
    } > "$dst/source-version.txt"
    "$py" "$dst/tests/gpu/ds1017_bench/admit_probe.py" "$dst" >&2
    echo "$dst"
}

serve_agx() {
    local src=$1 run_dir=$2
    assert_gpus_free
    export PYTHONPATH="$src/engine:$src"
    local aka_flags=(--set "deepinfer.aka_profile=${aka_profile}")
    if [[ -n "$aka_disable" ]]; then
        aka_flags+=(--set "deepinfer.aka_disable=${aka_disable}")
    fi
    "$py" "$src/deploy/agxinfer.py" serve "$model" --source "$src" --run "$run_dir" --detach --port "$port" \
        --gpus "$gpus_csv" --ds-backend deepinfer --cuda-graph auto \
        --max-rows 128 --graph-buckets 1 2 4 8 12 16 24 32 40 48 56 64 80 96 112 128 \
        --max-context "$max_ctx" --token-budget 32768 --overlap --max-slots 512 --mem-fraction .88 \
        --yoco-clip "$yoco_clip" --timeout 2400 --proc-tunables '{"ipc.shm_capacity":134217728}' \
        --set state.kv.pages=65536 --set frontend.response_timeout_s=900.0 \
        --set sampler.force_topk=0 --set deepinfer.force_random_gate=false \
        --set eplb.extra_experts=0 --set dsv4.graph_idle_experts=true --set sched.prefill_occupancy_target=0 \
        "${aka_flags[@]}" "${extra_serve_args[@]}"
}

case "$cmd" in
    prepare)
        require_model
        traces=${pos[0]:-}
        prep_out=${pos[1]:-$out_dir}
        if [[ -z "$traces" || -z "$prep_out" ]]; then
            echo "usage: $0 prepare --model <MODEL_DIR> <WEKA_TRACES_JSONL> <OUT_DATA_DIR>" >&2
            exit 2
        fi
        PYTHONPATH="$repo/engine:$repo" "$py" "$repo/tests/gpu/ds1017_bench/agentx_driver.py" prepare \
            --traces-jsonl "$traces" --tokenizer "$model" --out-dir "$prep_out" \
            --max-context "$max_ctx" --dev-sessions "$dev_sessions" --eval-sessions "$eval_sessions"
        ;;
    status)
        [[ -n "$run" ]] || { echo "error: pass --run <RUN_DIR>" >&2; exit 2; }
        PYTHONPATH="$repo/engine:$repo" exec "$py" "$repo/deploy/agxinfer.py" status --run "$run"
        ;;
    stop|stop-agx)
        [[ -n "$run" ]] || { echo "error: pass --run <RUN_DIR>" >&2; exit 2; }
        PYTHONPATH="$repo/engine:$repo" exec "$py" "$repo/deploy/agxinfer.py" stop --run "$run" --timeout 90
        ;;
    serve-agx)
        require_model
        resolve_aka_profile
        if [[ "$aka_warmup" == "1" && "$aka_profile" != "official" ]]; then
            "$py" -m aka_ds1017 warmup --profile "$aka_profile" --budget 32768
        fi
        run="${run:-$repo/.runs/agentx-agx-run}"
        mkdir -p "$run"
        staged=$(stage_tree "$run/staged-src")
        serve_agx "$staged" "$run"
        ;;
    run-agx)
        require_model
        resolve_aka_profile
        suite_prefix=${pos[0]:-}
        if [[ -z "$suite_prefix" ]]; then
            echo "usage: $0 run-agx --model <MODEL_DIR> [OPTIONS] <SUITE_PREFIX> [CONCURRENCIES ...]" >&2
            exit 2
        fi
        if [[ ! -f "${suite_prefix}.json" || ! -f "${suite_prefix}.i32" ]]; then
            echo "error: suite files not found: ${suite_prefix}.json / ${suite_prefix}.i32" >&2
            exit 2
        fi
        concs=("${pos[@]:1}")
        if [[ ${#concs[@]} -eq 0 ]]; then concs=(16 24 48); fi
        if [[ "$aka_warmup" == "1" && "$aka_profile" != "official" ]]; then
            echo "$(date -Is) Warming up aka_ds1017 (profile=${aka_profile}, budget=32768)..." >&2
            "$py" -m aka_ds1017 warmup --profile "$aka_profile" --budget 32768
        fi

        created_stage=0
        started_here=0
        if [[ -n "$run" && -f "$run/engine.json" ]]; then
            staged="${run}/staged-src"
            [[ -d "$staged/engine" ]] || staged=$(stage_tree)
        else
            staged=$(stage_tree)
            [[ "$staged" != "$repo" ]] && created_stage=1
            run="${run:-$staged/agx-run}"
            serve_agx "$staged" "$run"
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

        out="${out_dir:-$repo/.runs/results/agentx-agx-${aka_profile}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        mkdir -p "$out"
        cp "$staged/source-version.txt" "$out/source-version.txt" 2>/dev/null || true
        nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"

        export PYTHONPATH="$staged/engine:$staged"
        idx=0
        for c in "${concs[@]}"; do
            idx=$((idx + 1))
            tag="c${c}"
            if [[ -f "$out/agx-${tag}.json" ]]; then
                tag="c${c}-run${idx}"
            fi
            echo "$(date -Is) AGX AgentX suite=${suite_prefix} C=${c} profile=${aka_profile} (gpus=${gpus_csv})"
            for enabled in false true; do
                f="$out/apply-${tag}-${enabled}.json"
                "$py" "$staged/deploy/agxinfer.py" control --run "$run" apply --source "$staged" \
                    --config "{\"alloc.radix_cache\":$enabled}" --wait-empty 120 --json > "$f"
                "$py" -c 'import json,sys; d=json.load(open(sys.argv[1]))["result"]; assert d["status"]=="committed", d' "$f"
            done
            "$py" "$staged/tests/gpu/ds1017_bench/agentx_driver.py" agx-direct --run "$run" \
                --suite "${suite_prefix}.json" --corpus "${suite_prefix}.i32" \
                --concurrencies "$c" --out "$out/agx-${tag}.json"
            echo "RESULT $out/agx-${tag}.json"
        done
        if [[ $started_here -eq 1 && "$keep_engine" == "1" ]]; then
            echo "Engine left running at ${run}"
        fi
        ;;
    run-di)
        require_model
        if [[ -z "$di" || ! -d "$di" ]]; then
            echo "error: DeepInfer repo directory required; pass --di-root <DIR> or set DI_ROOT" >&2
            exit 2
        fi
        suite_prefix=${pos[0]:-}
        if [[ -z "$suite_prefix" ]]; then
            echo "usage: $0 run-di --model <MODEL_DIR> --di-root <DI_DIR> [OPTIONS] <SUITE_PREFIX> [CONCURRENCIES ...]" >&2
            exit 2
        fi
        if [[ ! -f "${suite_prefix}.json" || ! -f "${suite_prefix}.i32" ]]; then
            echo "error: suite files not found: ${suite_prefix}.json / ${suite_prefix}.i32" >&2
            exit 2
        fi
        concs=("${pos[@]:1}")
        if [[ ${#concs[@]} -eq 0 ]]; then concs=(16 24 48); fi
        assert_gpus_free

        di_cu13=$(detect_cu13 "$dipy")
        if [[ -n "$di_cu13" ]]; then
            export CPATH="${di_cu13}${CPATH:+:$CPATH}"
        fi
        device_offset="${DEVICE_OFFSET:-${gpus_csv%%,*}}"
        export CUDA_VISIBLE_DEVICES="$gpus_csv" DEVICE_OFFSET="$device_offset" WORLD_SIZE=1 PROFILE=0
        export DEEPINFER_NUM_STATE_SLOTS_RATIO=4 DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_ENABLE_MTP_VARLEN=0
        export DEEPINFER_USE_MULTIMODAL_POOL=0 DEEPINFER_FORCE_LOAD_BALANCE=0 DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0
        export DEEPINFER_HF_MODEL_DIR="$model" DEEPINFER_ROOT="$di"
        export PYTHONPATH="$script_dir:$di"
        export MASTER_PORT="$master_port"
        unset DEEPINFER_ENGRAM_LOCAL
        if ! compgen -G "$script_dir/extension/ds_prefix_pages*.so" >/dev/null; then
            "$dipy" "$script_dir/build_di_prefix_ext.py"
        fi
        out="${out_dir:-$repo/.runs/results/agentx-di-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
        mkdir -p "$out"
        nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"
        "$dipy" "$script_dir/agentx_driver.py" deepinfer \
            --model "$model" --deepinfer "$di" \
            --suite "${suite_prefix}.json" --corpus "${suite_prefix}.i32" \
            --concurrencies "${concs[@]}" --out "$out/di.json" \
            --mem .88 --token-budget 32768 --max-seqs 128 --max-model-len "$max_ctx" --yoco-clip "$yoco_clip"
        echo "RESULT $out/di.json"
        ;;
    *)
        echo "error: unknown command '$cmd'" >&2
        usage >&2
        exit 2
        ;;
esac
