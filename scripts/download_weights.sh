#!/usr/bin/env bash
# ImageNet-pretrained weights used for transfer learning (official timm releases on GitHub).
# Downloading them explicitly keeps training reproducible and working behind proxies that block
# the Hugging Face hub. Set model.weights_path: null in the config to let timm fetch from HF instead.
set -euo pipefail
DEST=${1:-weights}
BASE=https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights
mkdir -p "$DEST"
for f in efficientnet_b0_ra-3dd342df.pth mobilenetv3_large_100_ra-f55367f5.pth; do
  [[ -s "$DEST/$f" ]] && { echo "$f already present"; continue; }
  curl -fsSL -o "$DEST/$f" "$BASE/$f"
  echo "downloaded $DEST/$f"
done
