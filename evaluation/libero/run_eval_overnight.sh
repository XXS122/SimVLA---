#!/bin/bash
# =============================================================================
# ATTC 串行评估脚本（单卡 · libero_goal · 过夜跑）
# -----------------------------------------------------------------------------
# 一个一个实验顺序跑：每个实验先后台起服务端，等就绪后跑客户端，
# 跑完杀掉服务端再进入下一个。全部用同一块 GPU。
#
# 用法：
#   source paths.env          # 先加载 SIMVLA_CHECKPOINTS / SIMVLA_SMOLVLM_MODEL
#   bash evaluation/libero/run_eval_overnight.sh [num_trials] [gpu_id]
#
# 默认 num_trials=20, gpu_id=0
# 结果输出到 evaluation/libero/eval_overnight_<时间戳>/
# =============================================================================

set -u  # 未定义变量报错（不加 -e，单个实验失败不影响后续）

# ---------- 路径设置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 若调用前没 source，这里兜底 source 一次
if [ -f "${REPO_ROOT}/paths.env" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/paths.env"
fi

# LIBERO 仿真环境
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"

# ---------- conda 环境 ----------
# 服务端（SimVLA 策略模型）跑在 base，客户端（LIBERO 评估）跑在 libero。
# 可用环境变量覆盖：SIMVLA_SERVER_ENV / SIMVLA_CLIENT_ENV
SERVER_ENV="${SIMVLA_SERVER_ENV:-base}"
CLIENT_ENV="${SIMVLA_CLIENT_ENV:-libero}"

# 加载 conda（让 conda activate 在脚本里可用）
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -n "$CONDA_BASE" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "⚠️  找不到 conda，假设当前环境已正确（服务端和客户端将用同一个 python）"
fi

# ---------- 参数 ----------
NUM_TRIALS=${1:-20}
GPU=${2:-0}

CKPT="${SIMVLA_CHECKPOINTS:-./runs/simvla_libero_small/step_200000}"
NORM="${REPO_ROOT}/norm_stats/libero_norm.json"
SERVE="${REPO_ROOT}/evaluation/libero/serve_smolvlm_libero.py"
SMOLVLM="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
TASK_SUITE="libero_goal"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/eval_overnight_${TS}"
mkdir -p "$OUT_DIR"

echo "========================================================"
echo " ATTC 串行评估（过夜）"
echo "   checkpoint : $CKPT"
echo "   smolvlm    : $SMOLVLM"
echo "   norm_stats : $NORM"
echo "   task_suite : $TASK_SUITE"
echo "   num_trials : $NUM_TRIALS"
echo "   GPU        : $GPU"
echo "   服务端环境 : $SERVER_ENV"
echo "   客户端环境 : $CLIENT_ENV"
echo "   输出目录   : $OUT_DIR"
echo "========================================================"
echo ""

if [ ! -e "$CKPT" ]; then
    echo "⚠️  警告：checkpoint 路径不存在：$CKPT"
    echo "    请确认已 source paths.env，或检查 SIMVLA_CHECKPOINTS。"
fi

# ---------- 单个实验执行函数 ----------
# run_one <实验名> <端口> [服务端额外参数...]
run_one() {
    local name="$1"; local port="$2"; shift 2
    local server_args=("$@")

    local server_log="${OUT_DIR}/${name}_server.log"
    local client_log="${OUT_DIR}/${name}_goal.txt"

    echo "--------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] 开始实验: ${name}  (port ${port})"
    echo "   服务端参数: ${server_args[*]:-<无, baseline>}"

    # 1) 切到服务端环境(base)，后台启动服务端
    conda activate "$SERVER_ENV" 2>/dev/null || echo "   (无法 activate $SERVER_ENV，沿用当前环境)"
    CUDA_VISIBLE_DEVICES="$GPU" python -u "$SERVE" \
        --checkpoint "$CKPT" \
        --norm_stats "$NORM" \
        --smolvlm_model "$SMOLVLM" \
        --port "$port" \
        "${server_args[@]}" > "$server_log" 2>&1 &
    local server_pid=$!

    # 2) 等待服务端就绪（最多 600s，模型加载可能较慢）
    local ready=0
    for _ in $(seq 1 600); do
        if grep -q "server listening on" "$server_log" 2>/dev/null; then
            ready=1; break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "   ❌ 服务端进程提前退出，见 $server_log"
            break
        fi
        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        echo "   ❌ 服务端未就绪，跳过 ${name}"
        kill "$server_pid" 2>/dev/null
        wait "$server_pid" 2>/dev/null
        return
    fi
    echo "   ✓ 服务端就绪，开始评估..."

    # 3) 切到客户端环境(libero)，跑客户端（libero_goal，单连接）
    #    服务端已作为独立进程在跑，这里切换环境不影响它。
    conda activate "$CLIENT_ENV" 2>/dev/null || echo "   (无法 activate $CLIENT_ENV，沿用当前环境)"
    ( cd "$SCRIPT_DIR" && CUDA_VISIBLE_DEVICES="$GPU" python -u libero_client.py \
        --host 127.0.0.1 --port "$port" --client_type websocket \
        --task_suite "$TASK_SUITE" --num_trials "$NUM_TRIALS" --no_video ) \
        > "$client_log" 2>&1

    # 4) 等几秒让服务端打印连接关闭时的 [Latency summary] / hit rate
    sleep 6

    # 5) 杀掉服务端
    kill "$server_pid" 2>/dev/null
    wait "$server_pid" 2>/dev/null

    # 6) 打印本实验关键结果
    echo "   结果:"
    grep -i "Total success rate" "$client_log" | tail -1 | sed 's/^/      SR: /'
    grep    "Latency summary"    "$server_log" | tail -1 | sed 's/^.*\[Latency/      [Latency/'
    grep    "cumulative hit rate" "$server_log" | tail -1 | sed 's/^.*\[ATTC/      [ATTC/'
    echo "[$(date +%H:%M:%S)] 完成实验: ${name}"
    echo ""
}

# =============================================================================
# 实验列表（串行）
# =============================================================================

# ---- 主实验 ----
run_one "B0_baseline"   8200
run_one "C2_attc"       8201  --use_cache --cache_alpha 1.0 --cache_beta 0.9 --cache_warmup 3
run_one "C1_fixed"      8202  --use_cache --cache_fixed_threshold 0.10 --cache_warmup 3

# ---- Ablation α（β=0.9 固定）----
run_one "A_alpha05"     8210  --use_cache --cache_alpha 0.5 --cache_beta 0.9 --cache_warmup 3
run_one "A_alpha15"     8211  --use_cache --cache_alpha 1.5 --cache_beta 0.9 --cache_warmup 3
run_one "A_alpha20"     8212  --use_cache --cache_alpha 2.0 --cache_beta 0.9 --cache_warmup 3

# ---- Ablation β（α=1.0 固定）----
run_one "A_beta07"      8220  --use_cache --cache_alpha 1.0 --cache_beta 0.7 --cache_warmup 3
run_one "A_beta099"     8221  --use_cache --cache_alpha 1.0 --cache_beta 0.99 --cache_warmup 3

# =============================================================================
# 汇总
# =============================================================================
echo "========================================================"
echo " 全部实验完成！汇总："
echo "========================================================"
SUMMARY="${OUT_DIR}/SUMMARY.txt"
{
    echo "ATTC 评估汇总 ($TASK_SUITE, num_trials=$NUM_TRIALS)"
    echo "checkpoint: $CKPT"
    echo "时间: $(date)"
    echo ""
    printf "%-14s | %-22s | %-50s | %s\n" "实验" "成功率" "延迟" "命中率"
    echo "-------------------------------------------------------------------------------------------------"
    for name in B0_baseline C2_attc C1_fixed A_alpha05 A_alpha15 A_alpha20 A_beta07 A_beta099; do
        sr=$(grep -i "Total success rate" "${OUT_DIR}/${name}_goal.txt" 2>/dev/null | tail -1 | sed 's/Total success rate: //I')
        lat=$(grep "Latency summary" "${OUT_DIR}/${name}_server.log" 2>/dev/null | tail -1 | sed 's/^.*\[Latency summary\] //')
        hr=$(grep "cumulative hit rate" "${OUT_DIR}/${name}_server.log" 2>/dev/null | tail -1 | sed 's/^.*cumulative hit rate: //;s/ .*//')
        printf "%-14s | %-22s | %-50s | %s\n" "$name" "${sr:-N/A}" "${lat:-N/A}" "${hr:-—}"
    done
} | tee "$SUMMARY"

echo ""
echo "详细日志在: $OUT_DIR"
echo "汇总文件:   $SUMMARY"
