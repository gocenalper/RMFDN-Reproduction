# R-MFDN reproduction image. The same file builds the CPU image (default) and
# the GPU image:
#
#   docker build -t rmfdn .                                   # CPU, any OS
#   docker build -t rmfdn:gpu \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 .
#
FROM python:3.11-slim

ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

WORKDIR /app

# PyTorch first: it is the large layer and changes least often. torchvision
# is installed here too (timm depends on it) so that a later step cannot pull
# a different PyTorch build from PyPI.
RUN pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX_URL}"

COPY requirements.txt ./
RUN pip install -r requirements.txt \
    && python -c "import torch, torchaudio; assert torch.__version__.startswith('${TORCH_VERSION}'), torch.__version__"

COPY pyproject.toml README.md ./
COPY rmfdn_ablation ./rmfdn_ablation
RUN pip install --no-deps -e .

COPY . .

# Pretrained weights downloaded at run time land here; mount a volume to keep them.
VOLUME ["/cache"]

CMD ["python", "scripts/smoke_test.py"]
