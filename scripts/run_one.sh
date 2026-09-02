#!/usr/bin/env bash
# Run row N of the manifest (1-based, excluding the header).
#
#   bash scripts/run_one.sh 7                    # train row 7
#   MANIFEST=configs/manifest_T1.csv bash scripts/run_one.sh 7
#
# Resumes automatically from the highest saved checkpoint, so the same
# command can be re-issued after a walltime kill without losing progress.
set -euo pipefail

ROW="${1:?usage: run_one.sh <row-number>}"
MANIFEST="${MANIFEST:-configs/manifest.csv}"
PYTHON="${PYTHON:-python}"

line=$(tail -n +2 "$MANIFEST" | sed -n "${ROW}p")
[ -n "$line" ] || { echo "no row $ROW in $MANIFEST" >&2; exit 1; }

run_id=$(printf '%s' "$line" | cut -d, -f1)
saving_dir=$(printf '%s' "$line" | cut -d, -f8)
command=$(printf '%s' "$line" | cut -d, -f9-)

echo "[$run_id] saving_dir=$saving_dir"

# Resume from the newest complete checkpoint quartet (G/D/V/Z all present):
# a job killed mid-save can leave a G with no matching Z.
resume=""
model_dir="$saving_dir/train/model_dir"
if [ -d "$model_dir" ]; then
    for e in $(ls "$model_dir" | grep -oE '^[0-9]+-G\.ckpt$' | sed 's/-G\.ckpt//' | sort -rn); do
        if [ -f "$model_dir/$e-D.ckpt" ] && [ -f "$model_dir/$e-V.ckpt" ] && [ -f "$model_dir/$e-Z.ckpt" ]; then
            resume="--resume_epoch $e"
            echo "[$run_id] resuming from epoch $e"
            break
        fi
    done
fi

eval "${command/#python/$PYTHON} $resume"
