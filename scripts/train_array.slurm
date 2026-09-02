#!/bin/bash
#SBATCH --job-name=qmolgan_train
#SBATCH --account=pfw-cs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/train_%A_%a.out
#SBATCH --error=slurm_logs/train_%A_%a.err
#
# Train one manifest row per array task, chaining across walltime windows.
#
#   mkdir -p slurm_logs
#   python scripts/build_manifest.py --out configs/manifest.csv
#   sbatch --array=1-120%20 scripts/train_array.slurm
#
# The %20 throttle keeps at most 20 tasks resident. Every task resumes from
# its own newest checkpoint, so a task killed at the walltime limit simply
# resubmits itself and continues.
#
# NOTE ON RESOURCES: this workload is CPU-bound, not GPU-bound. The generator
# is ~8k parameters and the VQC runs on PennyLane's default.qubit state-vector
# simulator on CPU; a GPU buys nothing and costs queue time. Request GPUs only
# if you switch to lightning.gpu.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-configs/manifest.csv}"
MAX_CHAIN="${MAX_CHAIN:-6}"
CHAIN_COUNT="${CHAIN_COUNT:-0}"
ROW="${SLURM_ARRAY_TASK_ID}"

module purge
module load anaconda
source activate molgan-pt
cd "$REPO_ROOT"

line=$(tail -n +2 "$MANIFEST" | sed -n "${ROW}p")
run_id=$(printf '%s' "$line" | cut -d, -f1)
saving_dir=$(printf '%s' "$line" | cut -d, -f8)

echo "[$run_id] chain $CHAIN_COUNT/$MAX_CHAIN"
bash scripts/run_one.sh "$ROW"
status=$?

target_epochs=$(python - "$saving_dir" <<'PY'
import json, sys
print(json.load(open(sys.argv[1] + '/config.json'))['num_epochs'])
PY
)
done_epoch=$(ls "$saving_dir/train/model_dir" 2>/dev/null \
             | grep -oE '^[0-9]+-G\.ckpt$' | sed 's/-G\.ckpt//' | sort -n | tail -1)
done_epoch=${done_epoch:-0}
echo "[$run_id] reached epoch $done_epoch / $target_epochs (exit $status)"

if [ "$done_epoch" -lt "$target_epochs" ] && [ "$CHAIN_COUNT" -lt "$MAX_CHAIN" ]; then
    sbatch --dependency=afterany:${SLURM_JOB_ID} --array=${ROW} \
           --export=ALL,CHAIN_COUNT=$((CHAIN_COUNT + 1)),REPO_ROOT="$REPO_ROOT",MANIFEST="$MANIFEST" \
           scripts/train_array.slurm
else
    [ "$done_epoch" -lt "$target_epochs" ] && \
        echo "[$run_id] WARNING: chain cap reached at epoch $done_epoch; evaluating anyway"
    sbatch --dependency=afterany:${SLURM_JOB_ID} \
           --export=ALL,REPO_ROOT="$REPO_ROOT",RUN_DIR="$saving_dir" \
           scripts/eval_one.slurm
fi
