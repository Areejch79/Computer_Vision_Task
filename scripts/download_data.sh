#!/usr/bin/env bash
# Download the "Casting product image data for quality inspection" dataset (512x512 originals,
# 1300 images: 781 def_front / 519 ok_front) into data/raw/{def_front,ok_front}.
#
# Original source (Kaggle, requires an account):
#   https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
#   -> kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product
#      and copy casting_512x512/casting_512x512/{def_front,ok_front} to data/raw/
#
# This script uses a public GitHub mirror of the same files (no credentials needed).
set -euo pipefail
DEST=${1:-data/raw}
MIRROR=${DATA_MIRROR:-https://github.com/peotrannnn/Casting-Quality-Inspection-using-ML-DL}
SUBDIR=data/raw/casting_512x512/casting_512x512

if [[ -d "$DEST/def_front" && -d "$DEST/ok_front" ]]; then
  echo "Dataset already present in $DEST"; exit 0
fi
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --depth 1 --filter=blob:none --no-checkout "$MIRROR" "$TMP/repo"
git -C "$TMP/repo" sparse-checkout set --no-cone "$SUBDIR/"
git -C "$TMP/repo" checkout --quiet
mkdir -p "$DEST"
cp -r "$TMP/repo/$SUBDIR/def_front" "$TMP/repo/$SUBDIR/ok_front" "$DEST/"
echo "def_front: $(ls "$DEST/def_front" | wc -l) images, ok_front: $(ls "$DEST/ok_front" | wc -l) images -> $DEST"
