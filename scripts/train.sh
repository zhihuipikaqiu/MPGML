#!/usr/bin/env bash
# Usage: bash scripts/train.sh [dataset] [seed] [gpu] [extra run.py arguments]
set -euo pipefail

dataset=${1:-toxcast-TOX21}
if (( $# > 0 )); then shift; fi
seed=${1:-2}
if (( $# > 0 )); then shift; fi
gpu=${1:-0}
if (( $# > 0 )); then shift; fi

case "$dataset" in
  toxcast-TOX21) inner_lr=1e-1; meta_lr=1e-3 ;;
  muv) inner_lr=5e-1; meta_lr=5e-4 ;;
  toxcast-Tanguay) inner_lr=1e-1; meta_lr=5e-4 ;;
  *) echo "No supplied preset for '$dataset'; configure run.py directly." >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python run.py \
  --dataset "$dataset" --random_seed "$seed" --n_support 10 --gpu "$gpu" \
  --mol_pretrain_load_path ./pretrained/supervised_contextpred.pth \
  --inner_lr "$inner_lr" --meta_lr "$meta_lr" --dump_path dump \
  --use_fingerprint --use_descriptor --fp_type mixed --fp_encoder_type typewise \
  --fusion_method transformer --fusion_layers 3 --use_cross_modal_attn --use_l2_norm \
  "$@"
