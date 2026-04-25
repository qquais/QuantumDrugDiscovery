"""
compare_results.py — Compare QuMolGAN results against Kao et al. 2023 Table I baselines.

Loads quantum_results.csv produced by evaluate_quantum.py, adds hardcoded
paper baselines, prints a comparison table, and saves a grouped bar chart.

Run from the project root (or anywhere):
    python compare_results.py
    python compare_results.py --analysis_dir /path/to/analysis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hardcoded paper baselines — Kao et al. 2023, Table I
# ---------------------------------------------------------------------------
PAPER_BASELINES = {
    'Classical Baseline': {
        'validity':       0.824,
        'clean_validity': 0.630,
        'uniqueness':     0.590,
        'novelty':        0.795,
        'QED':            0.388,
        'logP':          -0.253,
        'SA':             5.400,
    },
    'Ablation B': {
        'validity':       0.843,
        'clean_validity': 0.590,
        'uniqueness':     0.539,
        'novelty':        0.805,
        'QED':            0.396,
        'logP':          -0.073,
        'SA':             5.123,
    },
}

METRICS = ['validity', 'clean_validity', 'uniqueness', 'novelty', 'QED', 'logP', 'SA']
METRIC_LABELS = {
    'validity':       'Validity',
    'clean_validity': 'Clean Validity',
    'uniqueness':     'Uniqueness',
    'novelty':        'Novelty',
    'QED':            'QED',
    'logP':           'logP',
    'SA':             'SA (↓ better)',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ANALYSIS_DIR = '/scratch/gilbreth/quaiqa01/QuantumDrugDiscovery/analysis'


def load_quantum_results(analysis_dir):
    csv_path = os.path.join(analysis_dir, 'quantum_results.csv')
    if not os.path.exists(csv_path):
        sys.exit(f'ERROR: quantum_results.csv not found at {csv_path}\n'
                 'Run evaluate_quantum.py first.')
    df = pd.read_csv(csv_path)
    row = df.iloc[-1]   # use the last (most recent) run if multiple epochs saved
    epoch = int(row.get('epoch', 0))
    metrics = {m: float(row[m]) for m in METRICS if m in row}
    return epoch, metrics


def build_comparison_table(quantum_metrics, epoch):
    """Return a DataFrame with one row per model."""
    rows = []
    rows.append({'Model': f'QuMolGAN (epoch {epoch})', **quantum_metrics})
    for name, vals in PAPER_BASELINES.items():
        rows.append({'Model': name, **vals})
    return pd.DataFrame(rows).set_index('Model')


def print_table(df):
    col_widths = {col: max(len(METRIC_LABELS[col]), 8) for col in METRICS}
    header = f"{'Model':<28}" + ''.join(
        f"  {METRIC_LABELS[c]:>{col_widths[c]}}" for c in METRICS)
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for model, row in df.iterrows():
        line = f'{model:<28}'
        for c in METRICS:
            v = row[c]
            line += f'  {v:>{col_widths[c]}.3f}'
        print(line)
    print(sep + '\n')


def save_comparison_chart(df, output_path):
    """Grouped bar chart — one group per metric, one bar per model."""
    models  = list(df.index)
    n_models = len(models)
    n_metrics = len(METRICS)

    # palette — distinct colours per model
    palette = ['#4C72B0', '#C44E52', '#55A868'][:n_models]

    x = np.arange(n_metrics)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (model, color) in enumerate(zip(models, palette)):
        vals = [df.loc[model, m] for m in METRICS]
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=color, edgecolor='white', alpha=0.88)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('QuMolGAN vs Kao et al. 2023 Table I Baselines', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)

    # SA note
    ax.annotate('* SA score: lower is better (1–10 scale)',
                xy=(0.01, 0.01), xycoords='axes fraction',
                fontsize=8, color='gray')

    # graceful y-axis: accommodate logP which can be negative
    all_vals = [df.loc[m, c] for m in df.index for c in METRICS if not np.isnan(df.loc[m, c])]
    ymin = min(0, min(all_vals)) - 0.3
    ymax = max(all_vals) * 1.25
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Comparison chart saved → {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare QuMolGAN evaluation results against paper baselines.')
    parser.add_argument('--analysis_dir', type=str, default=ANALYSIS_DIR,
                        help='Directory containing quantum_results.csv')
    args = parser.parse_args()

    print(f'Loading quantum_results.csv from {args.analysis_dir}...')
    epoch, quantum_metrics = load_quantum_results(args.analysis_dir)
    print(f'  Loaded results for epoch {epoch}.')

    df = build_comparison_table(quantum_metrics, epoch)

    print_table(df)

    # ---- Save comparison CSV ----
    csv_out = os.path.join(args.analysis_dir, 'comparison_table.csv')
    df.reset_index().to_csv(csv_out, index=False)
    print(f'Comparison CSV saved → {csv_out}')

    # ---- Save comparison chart ----
    png_out = os.path.join(args.analysis_dir, 'comparison_results.png')
    save_comparison_chart(df, png_out)


if __name__ == '__main__':
    main()
