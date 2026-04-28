#!/usr/bin/env bash
# One-shot setup: create a conda env and install all dependencies.
#
# Usage:
#     bash setup_env.sh
# Or run in the background with a log:
#     nohup bash setup_env.sh > setup.log 2>&1 &
#     tail -f setup.log
#
# By default this uses the standard PyPI / PyTorch / HuggingFace endpoints.
# If you are in mainland China and want to speed up downloads, uncomment the
# blocks marked "[CN MIRRORS]" below.

set -euo pipefail

ENV_NAME="multitq"
PY_VER="3.10"

# Override the env name from the command line if you want, e.g.:
#     ENV_NAME=faming2.3 bash setup_env.sh
ENV_NAME="${ENV_NAME_OVERRIDE:-${ENV_NAME}}"

echo "==> [1/5] Loading conda"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Cannot find conda.sh; please install Miniconda or Anaconda first."
    exit 1
fi

echo "==> [2/5] Creating/reusing conda env ${ENV_NAME} (Python ${PY_VER})"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi
conda activate "${ENV_NAME}"

echo "==> [3/5] Upgrading pip and pinning setuptools<81 (qa_datasets.py uses pkg_resources)"
python -m pip install --upgrade pip

# [CN MIRRORS] Uncomment to switch pip to a domestic Chinese mirror:
# python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# python -m pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com"

python -m pip install "setuptools<81"

echo "==> [4/5] Installing torch 2.1.2 (CUDA 12.1, works on RTX 30xx/40xx)"
# Default: install from the regular PyPI wheel (bundles CUDA 12.1 since torch 2.1).
python -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2

# [CN MIRRORS] Faster alternative for users in mainland China; comment the line
# above and uncomment the block below to use the Aliyun PyTorch wheel mirror
# (CUDA 11.8 build):
#
# python -m pip install --no-cache-dir \
#     --index-url https://mirrors.aliyun.com/pytorch-wheels/cu118/ \
#     --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
#     torch==2.1.2 torchvision==0.16.2

echo "==> [5/5] Installing the rest of the dependencies"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements.txt"

echo
echo "==> Self-check"
python - <<'PY'
import torch, transformers, flair, numpy
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("transformers:", transformers.__version__)
print("flair:", flair.__version__)
print("numpy:", numpy.__version__)
PY

cat <<TIP

==> Done!

To use the environment later:
  conda activate ${ENV_NAME}

[CN] If HuggingFace downloads (distilbert, flair/ner-english-large, ...)
are slow, set the mirror:
  export HF_ENDPOINT=https://hf-mirror.com

To run the pipeline (from MultiQA/):
  cd MultiQA
  python ner_task.py                          # NER preprocessing
  python train_qa_model.py --model multiqa    # Train the QA model

TIP
