#!/usr/bin/env bash
# List (and, with --force, delete) the superseded v1 result trees.
#
#   bash scripts/clean_results.sh            # dry run: show what would go
#   bash scripts/clean_results.sh --force    # actually delete
#
# These are ~13 GB of checkpoints from runs whose numbers are all invalidated
# by docs/ERRATA.md. Nothing in the new pipeline reads them. The small
# artifacts worth keeping have already been copied to results/v1_reference/,
# and scripts/reproduce_errata.py needs results/quantum/ablation_300 (kept
# below by default) to demonstrate the defects.
set -euo pipefail

FORCE=0
KEEP_ERRATA=1
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --drop-errata-checkpoint) KEEP_ERRATA=0 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

TARGETS=(
    results/classical/GAN
    results/classical/experiments
    results/classical/QDISC_Fixed_0405_0118
    results/classical/QDISC_Fixed_0405_0119
    results/classical/logs
    results/classical/plots
    results/quantum/GAN
    results/quantum/QDISC_300epochs_5k
    results/quantum/QDISC_300epochs_all_data
    results/quantum/QDISC_30epochs_all_data
    results/quantum/QDISC_ExactKao_30epochs
    results/quantum/QDISC_KaoLoss_30epochs
    results/quantum/QDISC_Simple_30epochs
    results/quantum/QDISC_TEST
    results/quantum/QUANTUM_Fresh_0405_0222
    results/quantum/QUANTUM_Fresh_0405_0225
    results/quantum/QUANTUM_Fresh_0405_0231
    results/classical_baseline_lw05_weighted
    results/quantum_disc_lw05_weighted
)
[ "$KEEP_ERRATA" -eq 0 ] && TARGETS+=(results/quantum/ablation_300)

total=0
for t in "${TARGETS[@]}"; do
    [ -e "$t" ] || continue
    size=$(du -sk "$t" | cut -f1)
    total=$((total + size))
    printf '%8s MB  %s\n' "$((size / 1024))" "$t"
done
printf '\n%8s MB  TOTAL\n\n' "$((total / 1024))"

if [ "$KEEP_ERRATA" -eq 1 ]; then
    echo "KEEPING results/quantum/ablation_300 (scripts/reproduce_errata.py needs it)."
    echo "Pass --drop-errata-checkpoint once the errata are written up."
fi

if [ "$FORCE" -eq 1 ]; then
    read -r -p "Delete the trees listed above? [y/N] " reply
    case "$reply" in
        [yY]) for t in "${TARGETS[@]}"; do [ -e "$t" ] && rm -rf "$t" && echo "removed $t"; done ;;
        *) echo "aborted" ;;
    esac
else
    echo "Dry run. Re-run with --force to delete."
fi
