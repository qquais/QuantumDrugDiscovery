"""
aggregate_results.py — Aggregate honest multi-seed evaluation results.

Scans results/quantum_multiseed/seed_{seed}_{preset}/analysis/best_epoch_summary.json
(written by find_best_epoch.py's fixed-rule epoch selection) for both reward
presets, computes mean +/- std across whichever seeds are available, and
prints a human-readable table plus LaTeX-ready rows. Tolerates missing or
still-running seeds rather than failing.

Run from the project root:
    python aggregate_results.py
    python aggregate_results.py --results_dir results/quantum_multiseed --seeds 42 123 456
"""

import argparse
import json
import os

import numpy as np

METRICS = ['validity', 'clean_validity', 'uniqueness', 'novelty', 'QED', 'SA']
PRESETS = ['ablation_b', 'ablation_b_clean']
PRESET_LABELS = {
    'ablation_b': 'Quantum + Ablation B',
    'ablation_b_clean': 'Quantum + Ablation B (clean)',
}


def load_summaries(results_dir, seeds, presets):
    """Return {preset: [summary_dict, ...]} for whichever seed/preset JSONs exist,
    and a list of expected-but-missing paths."""
    found = {preset: [] for preset in presets}
    missing = []
    for preset in presets:
        for seed in seeds:
            path = os.path.join(results_dir, f'seed_{seed}_{preset}',
                                 'analysis', 'best_epoch_summary.json')
            if os.path.exists(path):
                with open(path) as f:
                    found[preset].append(json.load(f))
            else:
                missing.append(path)
    return found, missing


def summarize(summaries):
    """{metric: (mean, std, n)} across a list of summary dicts."""
    stats = {}
    for metric in METRICS:
        vals = [s[metric] for s in summaries if s.get(metric) is not None]
        if vals:
            stats[metric] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            stats[metric] = (float('nan'), float('nan'), 0)
    return stats


def print_table(all_stats):
    header = f"{'Preset':<28}" + ''.join(f"{m:>18}" for m in METRICS)
    print(header)
    print('-' * len(header))
    for preset, stats in all_stats.items():
        row = f"{PRESET_LABELS.get(preset, preset):<28}"
        for m in METRICS:
            mean, std, n = stats[m]
            cell = 'n/a' if n == 0 else f'{mean:.3f}±{std:.3f} (n={n})'
            row += cell.rjust(18)
        print(row)
    print()


def print_latex(all_stats):
    print('% LaTeX-ready rows (mean +/- std across seeds)')
    for preset, stats in all_stats.items():
        cells = []
        for m in METRICS:
            mean, std, n = stats[m]
            cells.append('--' if n == 0 else f'{mean:.3f} $\\pm$ {std:.3f}')
        print(f'{PRESET_LABELS.get(preset, preset)} & ' + ' & '.join(cells) + r' \\')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multi-seed quantum reward-preset results.')
    parser.add_argument('--results_dir', type=str, default='results/quantum_multiseed')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--presets', type=str, nargs='+', default=PRESETS)
    args = parser.parse_args()

    found, missing = load_summaries(args.results_dir, args.seeds, args.presets)

    if missing:
        print('Missing (not yet finished / evaluated):')
        for path in missing:
            print(f'  {path}')
        print()

    all_stats = {preset: summarize(summaries) for preset, summaries in found.items()}

    print_table(all_stats)
    print_latex(all_stats)


if __name__ == '__main__':
    main()
