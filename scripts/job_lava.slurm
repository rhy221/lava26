#!/bin/bash
# Usage: sbatch scripts/job_lava.slurm
#        OVERRIDES="data.sample=20" sbatch scripts/job_lava.slurm   # quick smoke test
#
# Two-phase pipeline:
#   Phase 1 — pymupdf parse + Dense encode + ColQwen page/query encode → disk cache  (~12 min)
#   Phase 2 — vLLM Qwen 27B inference, MaxSim retrieval from cache                  (~62 min)
#   Total budget: ~77 min < 2h ✓

#SBATCH --job-name=lava_2026
#SBATCH --output=/datastore/%u/LAVA2026_2/lava26/runs/slurm-%j-early.out
#SBATCH --error=/datastore/%u/LAVA2026_2/lava26/runs/slurm-%j-early.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=mps:a100:1
#SBATCH --time=08:00:00

set -euo pipefail

# ── 1. LOAD MODULES ───────────────────────────────────────────────────────────
module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# ── 2. KHAI BÁO BIẾN ─────────────────────────────────────────────────────────
REQUIRED_VRAM=20000
PROJECT_DIR=/datastore/$USER/LAVA2026_2/lava26
PYTHON=/datastore/$USER/miniconda3/envs/vqa-jv/bin/python
VLLM=/datastore/$USER/miniconda3/envs/vqa-jv/bin/vllm
MODEL_DIR=/datastore/$USER/models/Qwen3.6-27B
RUN_DIR="$PROJECT_DIR/runs/$SLURM_JOB_ID"
VLLM_PORT=$((8000 + SLURM_JOB_ID % 1000))
VLLM_LOG="$RUN_DIR/vllm_server.log"

# ── 3. KIỂM TRA GPU ──────────────────────────────────────────────────────────
unset CUDA_VISIBLE_DEVICES

CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then echo "$CHECK_OUT"; exit 0; fi
if [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$CHECK_OUT
fi

# ── 4. KHỞI TẠO MPS ──────────────────────────────────────────────────────────
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"

# ── 5. SETUP DIRS & LOGGING ───────────────────────────────────────────────────
export TORCH_HOME=/datastore/$USER/.torch_cache
export HF_HOME=/datastore/$USER/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$RUN_DIR"

# Per-job log goes into runs/<job_id>/
exec > >(tee -a "$RUN_DIR/job.out") 2> >(tee -a "$RUN_DIR/job.err" >&2)

echo "Run dir    : $RUN_DIR"
echo "Project dir: $PROJECT_DIR"
echo "Model dir  : $MODEL_DIR"

# ── 6. HF TOKEN ──────────────────────────────────────────────────────────────
HF_TOKEN_FILE="/datastore/$USER/.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_TOKEN_FILE}" ]; then
    export HF_TOKEN="$(cat "${HF_TOKEN_FILE}")"
fi

# ── 7. KIỂM TRA MÔI TRƯỜNG ───────────────────────────────────────────────────
if [ ! -x "$PYTHON" ]; then echo "ERROR: Python not found: $PYTHON"; exit 1; fi
if [ ! -d "$PROJECT_DIR" ]; then echo "ERROR: Project not found: $PROJECT_DIR"; exit 1; fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -u -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")'

# ── 8. DOWNLOAD MODEL (nếu chưa có) ──────────────────────────────────────────
HF_CLI=/datastore/$USER/miniconda3/envs/vqa-jv/bin/hf

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "Model not found. Downloading Qwen/Qwen3.6-27B ..."
    mkdir -p "$MODEL_DIR"
    "$HF_CLI" download Qwen/Qwen3.6-27B \
        --local-dir "$MODEL_DIR"
    echo "Download complete."
    ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null | tail -5 || echo "(no safetensors found)"
else
    echo "Model already exists at $MODEL_DIR — skipping download."
    ls -lh "$MODEL_DIR"/*.safetensors 2>/dev/null | tail -3 || true
fi

# ── 9. PHASE 1: PREPROCESSING (ColQwen only, Qwen 27B chưa start) ────────────
# ColQwen (~7 GiB) loads alone — plenty of room, no OOM risk.
# Encodes: page images → cache/visual/*.pt
#          all queries → cache/query_embs.pt   (MaxSim in Phase 2, no model needed)
#
# --clear-cache wipes all cache dirs before Phase 1 → accurate end-to-end timing.
# Remove it if you want to reuse cached embeddings from a previous job.
OVERRIDES="${OVERRIDES:-}"
COLQWEN_DIR=/datastore/$USER/models/colqwen2.5-v0.2

if [ ! -d "$COLQWEN_DIR" ]; then
    echo "WARNING: ColQwen model not found at $COLQWEN_DIR — visual retrieval disabled"
    COLQWEN_OVERRIDE="visual.enabled=false"
else
    COLQWEN_OVERRIDE="visual.enabled=true visual.retriever_model=$COLQWEN_DIR"
fi

PREPROCESS_OVERRIDES="model.qwen.path=$MODEL_DIR \
    retriever.tokenizer=mecab \
    $COLQWEN_OVERRIDE"

echo ""
echo "============================================================"
echo "  PHASE 1 — Preprocessing (pymupdf + Dense + ColQwen)"
echo "============================================================"
"$PYTHON" -u run.py \
    --config config.yaml \
    --output "runs/$SLURM_JOB_ID/results.json" \
    --preprocess \
    --clear-cache \
    $PREPROCESS_OVERRIDES $OVERRIDES
echo "Phase 1 done."

# ── 10. KHỞI ĐỘNG vLLM SERVER (sau khi preprocessing xong) ──────────────────
echo "Starting vLLM server ..."
mkdir -p "$RUN_DIR"

# Kill any stale process holding the port before starting fresh
STALE_PID=$(lsof -ti tcp:$VLLM_PORT 2>/dev/null || true)
if [ -n "$STALE_PID" ]; then
    echo "Killing stale process on port $VLLM_PORT (PID: $STALE_PID)"
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 2
fi

# ColQwen unloaded after Phase 1 — Qwen can use 0.50 × 80GB = 40GB.
# --enforce-eager disables CUDA graph capture to avoid the ~14 GiB memory spike.
# VLLM_PLUGINS="" disables paddlex plugin (expects ernie45 module missing in this vLLM version).
VLLM_PLUGINS="" "$VLLM" serve "$MODEL_DIR" \
    --port $VLLM_PORT \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.20 \
    --enforce-eager \
    --generation-config vllm \
    --override-generation-config '{"enable_thinking": false}' \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID (log: $VLLM_LOG)"

# ── 11. PHASE 2: INFERENCE (cache + MaxSim, ColQwen không load lại) ──────────
# Visual retrieval dùng pre-computed query embs + MaxSim tensor math (no model).
BASE_OVERRIDES="model.qwen.path=$MODEL_DIR model.qwen.backend=vllm \
    model.qwen.vllm_url=http://localhost:$VLLM_PORT/v1 \
    retriever.tokenizer=mecab \
    visual.enabled=false \
    vlm.enabled=true model.qwen.mode=vlm"

echo ""
echo "============================================================"
echo "  PHASE 2 — Inference (Qwen 27B vLLM + MaxSim cache)"
echo "============================================================"
"$PYTHON" -u run.py \
    --config config.yaml \
    --output "runs/$SLURM_JOB_ID/results.json" \
    $BASE_OVERRIDES $OVERRIDES

# ── 12. DỌN DẸP ──────────────────────────────────────────────────────────────
kill $VLLM_PID 2>/dev/null || true
echo ""
echo "============================================================"
echo "  Job $SLURM_JOB_ID completed — results in $RUN_DIR"
echo "============================================================"
