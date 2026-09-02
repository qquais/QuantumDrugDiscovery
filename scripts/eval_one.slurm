#!/bin/bash
#SBATCH --job-name=qmolgan_eval
#SBATCH --account=pfw-cs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err
#
# Evaluate one finished run under the pre-registered protocol.
#   sbatch --export=ALL,RUN_DIR=results/runs/T1_vqc_ablation_b_s42 scripts/eval_one.slurm
#
# Or evaluate everything at once from a login node / interactive session:
#   for d in results/runs/*/; do python evaluate_run.py --run_dir "$d"; done
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
module purge
module load anaconda
source activate molgan-pt
cd "$REPO_ROOT"

# --epoch_stride 1 sweeps every checkpoint. Raise it to 2-5 if the sweep
# dominates wall-clock; the selection floor and rule are unaffected, but say
# so in the paper if you do, since a coarser sweep can miss the argmax.
python evaluate_run.py --run_dir "$RUN_DIR" --epoch_stride 1
