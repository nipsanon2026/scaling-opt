#!/bin/bash

echo "Job started on $(date)"
echo "Host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID}"

echo "GPU info:"
nvidia-smi || true

module load anaconda3/2023.07
# module load cuda/10.2
# module load cudnn

echo "Setting up isolated eval environment..."
python -m venv .venv_sq_alt
source .venv_sq_alt/bin/activate
python -m pip install --upgrade pip setuptools wheel

PERSIST_ROOT=""
PERSIST_HF_HOME="${PERSIST_ROOT}/hf_cache"
PERSIST_HF_HUB="${PERSIST_HF_HOME}/hub"
PERSIST_HF_DATASETS="${PERSIST_HF_HOME}/datasets"
PERSIST_MODEL_DIR="${PERSIST_ROOT}/persistent_models/Qwen3-8B-Base"

export HF_HOME="${PERSIST_HF_HOME}"
export HF_DATASETS_CACHE="${PERSIST_HF_DATASETS}"
export HUGGINGFACE_HUB_CACHE="${PERSIST_HF_HUB}"
export TRANSFORMERS_CACHE="${PERSIST_HF_HUB}"

# More robust on HPC than accelerated transfer paths
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${PERSIST_MODEL_DIR}"

echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "PERSIST_MODEL_DIR=${PERSIST_MODEL_DIR}"

python -m pip install \
  "numpy<2" \
  "scipy<1.13" \
  "matplotlib" \
  "transformers<5" \
  "datasets<4.0" \
  "accelerate" \
  "lm_eval" \
  "huggingface_hub"

echo "Python: $(which python)"
python --version

echo "Torch sanity (must be CUDA build):"
python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cuda.is_built:", torch.backends.cuda.is_built())
print("torch.cuda.is_available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if not torch.backends.cuda.is_built():
    raise SystemExit("FATAL: CPU-only torch build is being used in this job.")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA torch build present, but CUDA unavailable.")
print("gpu:", torch.cuda.get_device_name(0))
PY

export PYTHONUNBUFFERED=1

BASE_SAVE_DIR="${PERSIST_ROOT}"
SCRIPT="SQ_alternating.py"
RUN_PREFIX="qwen3_8b_alt"
ACT_MODE="${ACT_MODE:-asymmetric}"
W_MODE="${W_MODE:-symmetric}"
ACT_BITS="${ACT_BITS:-6}"
W_BITS="${W_BITS:-6}"

if [[ "${ACT_MODE}" == "asymmetric" ]]; then
  ACT_MODE_TAG="asymmA"
else
  ACT_MODE_TAG="symmA"
fi

if [[ "${W_MODE}" == "asymmetric" ]]; then
  W_MODE_TAG="asymmW"
else
  W_MODE_TAG="symmW"
fi

SCHEME_TAG="${ACT_MODE_TAG}_${W_MODE_TAG}_W${W_BITS}A${ACT_BITS}"
RUN_TAG="${RUN_PREFIX}_${SCHEME_TAG}"
OUT_DIR="${BASE_SAVE_DIR}/${RUN_TAG}"
FINAL_MODEL_DIR="${OUT_DIR}/qwen3_8b_alt_final_${SCHEME_TAG}"

UPLOAD_TO_HF=1
HF_TOKEN="${HF_TOKEN:-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}"
HF_REPO_OWNER="${HF_REPO_OWNER:-}"
HF_REPO_ID="${HF_REPO_ID:-${HF_REPO_OWNER}/qwen3_8b_alt_final_${SCHEME_TAG}}"
HF_PRIVATE_REPO="${HF_PRIVATE_REPO:-1}"

mkdir -p "${OUT_DIR}"
mkdir -p "${FINAL_MODEL_DIR}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "FATAL: Script not found: ${SCRIPT}"
  exit 1
fi

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
  if [[ -z "${HF_REPO_ID}" ]]; then
    echo "FATAL: HF_REPO_ID must be set to upload the final model to Hugging Face."
    exit 1
  fi
  python - <<'PY'
import importlib.util
if importlib.util.find_spec("huggingface_hub") is None:
    raise SystemExit("FATAL: huggingface_hub is not installed in this environment.")
PY
fi

echo "Ensuring persistent local snapshot exists..."
export HF_TOKEN
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen3-8B-Base"
local_dir = Path("")
local_dir.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN", None)

path = snapshot_download(
    repo_id=repo_id,
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
    token=token,
    resume_download=True,
)
print(f"[snapshot] ready at {path}")
PY

echo "============================================================"
echo "Running SQ alternating job at $(date)"
echo "SCRIPT=${SCRIPT}"
echo "RUN_TAG=${RUN_TAG}"
echo "OUT_DIR=${OUT_DIR}"
echo "FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
echo "ACT_QUANT_MODE=${ACT_MODE}"
echo "W_QUANT_MODE=${W_MODE}"
echo "ACT_BITS=${ACT_BITS}"
echo "W_BITS=${W_BITS}"
echo "HF_REPO_ID=${HF_REPO_ID}"
echo "LOCAL_MODEL_PATH=${PERSIST_MODEL_DIR}"
echo "============================================================"

python "${SCRIPT}" \
  --model "${PERSIST_MODEL_DIR}" \
  --model_family qwen \
  --out_dir "${OUT_DIR}" \
  --tag "${RUN_TAG}" \
  --alt_s_init GD \
  --calib_rows 128 \
  --calib_batch_size 8 \
  --calib_max_length 2048 \
  --calib_max_rows_X 262144 \
  --act_bits "${ACT_BITS}" \
  --w_bits "${W_BITS}" \
  --act_quant_mode "${ACT_MODE}" \
  --w_quant_mode "${W_MODE}" \
  --alt_iters 2 \
  --s_sweeps 1 \
  --golden_max_iter 15 \
  --coord_backtrack_steps 5 \
  --ppl_seq_len 2048 \
  --run_final_eval_ppl_suite \
  --final_eval_ppl_datasets wikitext2,c4 \
  --run_final_zeroshot \
  --final_eval_batch_size 4 \
  --save_patches \
  --final_model_dir "${FINAL_MODEL_DIR}"

if [[ "${UPLOAD_TO_HF}" == "1" ]]; then
  echo "Uploading final model to Hugging Face: ${HF_REPO_ID}"
  export FINAL_MODEL_DIR HF_REPO_ID HF_PRIVATE_REPO
  python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
token = os.environ["HF_TOKEN"]
folder_path = os.environ["FINAL_MODEL_DIR"]
private = os.environ.get("HF_PRIVATE_REPO", "1").lower() not in {"0", "false", "no"}

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder_path,
    commit_message=f"Upload quantized model from {os.path.basename(folder_path)}",
)
print(f"[upload] pushed {folder_path} -> https://huggingface.co/{repo_id}")
PY
fi

echo "Job finished at $(date)"
