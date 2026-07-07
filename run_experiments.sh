#!/bin/bash
#SBATCH --job-name=qumolgan_multiseed
#SBATCH --account=pfw-cs
#SBATCH --partition=ai            # TODO: confirm GPU partition name for the pfw-cs account,
                                   #       e.g. via `sacctmgr show assoc user=$USER` or `sinfo -o "%P %G"`
#SBATCH --gres=gpu:a100:1         # TODO: confirm exact GPU gres syntax/name on Gilbreth for A100-80GB
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# 3 seeds x 2 reward presets = 6 array tasks. Each task trains 300 epochs of
# Quantum + Ablation B (or the clean-validity-boosted preset).
#
# A single 300-epoch quantum run takes ~11-12h wall-clock (observed ~2.3
# min/epoch from the original single-seed run in git history: d463d86 ->
# 89a32f5 reached epoch 141 in ~5.5h). A flat 4h walltime cannot finish 300
# epochs, so this script self-chains: at the end of each 4h window it checks
# how far training got and, if incomplete, resubmits itself (same array
# index) as a dependent job. MAX_CHAIN caps the total chain length as a
# safety stop. Once training reaches num_epochs (or the chain cap is hit),
# it submits a separate CPU-only honest-evaluation job (find_best_epoch.py)
# instead of continuing to train.
#
# Submit with:
#   mkdir -p slurm_logs
#   sbatch run_experiments.sh
# ---------------------------------------------------------------------------

set -uo pipefail

REPO_ROOT=/scratch/gilbreth/quaiqa01/QuantumDrugDiscovery
DATASET=$REPO_ROOT/data/qm9_5k_py37.sparsedataset
NUM_EPOCHS=300
MAX_CHAIN=6                       # safety cap: 6 x 4h = 24h max per seed/preset

SEEDS=(42 123 456)
PRESETS=(ablation_b ablation_b_clean)

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_IDX=$(( TASK_ID / 2 ))
PRESET_IDX=$(( TASK_ID % 2 ))
SEED=${SEEDS[$SEED_IDX]}
PRESET=${PRESETS[$PRESET_IDX]}
CHAIN_COUNT=${CHAIN_COUNT:-0}

SAVING_DIR=$REPO_ROOT/results/quantum_multiseed/seed_${SEED}_${PRESET}
MODEL_DIR=$SAVING_DIR/train/model_dir
mkdir -p "$SAVING_DIR" "$REPO_ROOT/slurm_logs"

echo "[task $TASK_ID] seed=$SEED preset=$PRESET chain=$CHAIN_COUNT/$MAX_CHAIN saving_dir=$SAVING_DIR"

module purge
module load anaconda               # TODO: adjust to Gilbreth's actual anaconda module name if different
source activate molgan-pt          # conda env name confirmed from environment.yml

cd "$REPO_ROOT"

# ---- Helper: highest completed epoch (empty string if none) ----
latest_epoch() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    ls "$dir" 2>/dev/null | grep -oE '^[0-9]+-G\.ckpt$' | sed 's/-G\.ckpt//' | sort -n | tail -1
}

RESUME_ARG=""
LATEST=$(latest_epoch "$MODEL_DIR")
if [ -n "$LATEST" ]; then
    RESUME_ARG="--resume_epoch $LATEST"
    echo "[task $TASK_ID] resuming from epoch $LATEST"
fi

python main.py \
    --seed "$SEED" \
    --reward_preset "$PRESET" \
    --saving_dir "$SAVING_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    $RESUME_ARG

# ---- How far did this chunk get? ----
DONE_EPOCH=$(latest_epoch "$MODEL_DIR")
DONE_EPOCH=${DONE_EPOCH:-0}
echo "[task $TASK_ID] reached epoch $DONE_EPOCH / $NUM_EPOCHS"

if [ "$DONE_EPOCH" -lt "$NUM_EPOCHS" ] && [ "$CHAIN_COUNT" -lt "$MAX_CHAIN" ]; then
    echo "[task $TASK_ID] not finished, resubmitting (chain $((CHAIN_COUNT + 1))/$MAX_CHAIN)"
    sbatch --dependency=afterany:${SLURM_JOB_ID} \
           --array=${TASK_ID} \
           --export=ALL,CHAIN_COUNT=$((CHAIN_COUNT + 1)) \
           "$0"
else
    if [ "$DONE_EPOCH" -lt "$NUM_EPOCHS" ]; then
        echo "[task $TASK_ID] WARNING: chain cap reached at epoch $DONE_EPOCH (< $NUM_EPOCHS) — evaluating what we have"
    fi
    echo "[task $TASK_ID] submitting honest post-hoc evaluation (find_best_epoch.py, CPU-only)"
    ANALYSIS_DIR=$SAVING_DIR/analysis
    mkdir -p "$ANALYSIS_DIR"
    sbatch --dependency=afterany:${SLURM_JOB_ID} \
           --job-name=qumolgan_eval \
           --account=pfw-cs \
           --partition=cpu \
           --mem=8G \
           --cpus-per-task=4 \
           --time=04:00:00 \
           --output="$REPO_ROOT/slurm_logs/eval_%j.out" \
           --wrap="module purge && module load anaconda && source activate molgan-pt && cd $REPO_ROOT && python find_best_epoch.py --model_dir $MODEL_DIR --dataset $DATASET --analysis_dir $ANALYSIS_DIR --max_epoch $DONE_EPOCH --n_generate 500 --seed $SEED --reward_preset $PRESET"
fi
