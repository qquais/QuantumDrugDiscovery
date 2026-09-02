#!/usr/bin/env python
"""Regenerate every paper figure from saved results. No hardcoded numbers.

    python make_figures.py --results_root results/runs --out_dir figures

The figure script it replaces (`generate_figures.py`) carried the headline
table's values as Python literals, so a wrong table produced matching wrong
figures with nothing to flag the disagreement. Everything here is read from
results.json / history.csv / epoch_sweep.csv, and a figure whose inputs are
missing is skipped with a message rather than drawn from defaults.

Figures produced (each skipped if its inputs are absent):
  fig_factorial            latent x reward, mean +- std over seeds
  fig_uniqueness_vs_n      uniqueness against sample size (log x)
  fig_quality_diversity    clean-validity vs uniqueness front over presets
  fig_property_dists       generated vs QM9 property histograms
  fig_training_curves      per-epoch validation curves, seed band
  fig_epoch_sweep          selection sweep with the chosen epoch marked
  fig_circuit_ablation     qubit count / depth sweep
  fig_latent_geometry      latent participation ratio and space occupancy
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import warnings

import matplotlib
matplotlib.use('Agg')

# All-NaN slices are normal here: an untrained or collapsed run legitimately
# has no property values, and the figures render those as gaps.
warnings.filterwarnings('ignore', message='.*Mean of empty slice.*')
warnings.filterwarnings('ignore', message='.*Degrees of freedom <= 0.*')
warnings.filterwarnings('ignore', message='.*All-NaN.*')
import matplotlib.pyplot as plt

from qmolgan import chem

PALETTE = ['#4C6EF5', '#F76707', '#37B24D', '#AE3EC9', '#F03E3E', '#1098AD',
           '#868E96', '#F59F00']
plt.rcParams.update({'figure.dpi': 160, 'savefig.bbox': 'tight',
                     'axes.grid': True, 'grid.alpha': 0.25, 'font.size': 9})


def load_all(results_root, report='selected'):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_root, '**', 'results.json'),
                                 recursive=True)):
        with open(path) as f:
            blob = json.load(f)
        reports = blob.get('reports', {})
        if report not in reports:
            continue
        cfg = blob.get('config', {})
        runs.append({
            'dir': os.path.dirname(os.path.dirname(path)),
            'eval_dir': os.path.dirname(path),
            'latent': cfg.get('latent'),
            'reward': cfg.get('reward_preset'),
            'qubits': cfg.get('qubits'),
            'layers': cfg.get('layers'),
            'z_dim': cfg.get('z_dim'),
            'seed': blob.get('protocol', {}).get('run_seed', cfg.get('seed')),
            'metrics': reports[report],
            'latent_stats': blob.get('latent_statistics', {}),
            'blob': blob,
        })
    return runs


def group_mean_std(runs, key_fn, metric):
    groups = defaultdict(list)
    for r in runs:
        v = r['metrics'].get(metric)
        if v is not None and np.isfinite(v):
            groups[key_fn(r)].append(float(v))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
            for k, v in groups.items()}


def annotate_n(ax, labels, counts):
    """Print the seed count under each bar group. A figure that does not say
    how many seeds it averages is a figure that cannot be checked."""
    for i, (lab, n) in enumerate(zip(labels, counts)):
        ax.annotate(f'n={n}', (i, 0), xytext=(0, -22), textcoords='offset points',
                    ha='center', fontsize=7, color='#555')


# ---------------------------------------------------------------------------

def fig_factorial(runs, out_dir):
    metrics = [('clean_validity', 'Clean-validity', False),
               ('uniqueness_clean', 'Uniqueness (clean-valid)', False),
               ('QED', 'QED', False),
               ('SA', 'SA (lower is better)', True)]
    latents = sorted({r['latent'] for r in runs})
    rewards = sorted({r['reward'] for r in runs})
    if len(latents) < 2 and len(rewards) < 2:
        print('  skip fig_factorial: needs more than one condition')
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for ax, (metric, title, lower_better) in zip(np.atleast_1d(axes), metrics):
        stats = group_mean_std(runs, lambda r: (r['latent'], r['reward']), metric)
        width = 0.8 / max(1, len(rewards))
        for j, reward in enumerate(rewards):
            xs, ys, es, ns = [], [], [], []
            for i, latent in enumerate(latents):
                if (latent, reward) not in stats:
                    continue
                m, s, n = stats[(latent, reward)]
                xs.append(i + (j - (len(rewards) - 1) / 2) * width)
                ys.append(m); es.append(s); ns.append(n)
            if xs:
                ax.bar(xs, ys, width=width * 0.9, yerr=es, capsize=3,
                       label=reward, color=PALETTE[j % len(PALETTE)], alpha=0.9)
        ax.set_xticks(range(len(latents)))
        ax.set_xticklabels(latents, rotation=25, ha='right')
        ax.set_title(title + ('  (lower better)' if lower_better else ''))
    np.atleast_1d(axes)[0].legend(title='reward', fontsize=7)
    fig.suptitle('Latent source x reward shaping (mean $\\pm$ std over seeds)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_factorial.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_uniqueness_vs_n(runs, out_dir):
    curves = defaultdict(list)
    for r in runs:
        curve = r['metrics'].get('uniqueness_curve')
        if not curve:
            continue
        curves[f'{r["latent"]} / {r["reward"]}'].append(
            {int(k): float(v) for k, v in curve.items()})
    if not curves:
        print('  skip fig_uniqueness_vs_n: no uniqueness curves in results.json')
        return

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for i, (label, series) in enumerate(sorted(curves.items())):
        sizes = sorted({n for s in series for n in s})
        means = [np.nanmean([s.get(n, np.nan) for s in series]) for n in sizes]
        stds = [np.nanstd([s.get(n, np.nan) for s in series]) for n in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker='o', capsize=3,
                    color=PALETTE[i % len(PALETTE)], label=f'{label} (n={len(series)})')
    ax.set_xscale('log')
    ax.set_xlabel('molecules generated ($n$)')
    ax.set_ylabel('uniqueness among valid')
    ax.set_title('Uniqueness depends strongly on sample size\n'
                 '(comparisons are only meaningful at fixed $n$)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_uniqueness_vs_n.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_quality_diversity(runs, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    by_latent = defaultdict(list)
    for r in runs:
        cv, uq = r['metrics'].get('clean_validity'), r['metrics'].get('uniqueness_clean')
        if cv is None or uq is None or not (np.isfinite(cv) and np.isfinite(uq)):
            continue
        by_latent[r['latent']].append((cv, uq, r['reward']))
    if not by_latent:
        print('  skip fig_quality_diversity: no usable points')
        plt.close(fig)
        return
    for i, (latent, pts) in enumerate(sorted(by_latent.items())):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=45, alpha=0.85, color=PALETTE[i % len(PALETTE)],
                   label=latent)
        for x, y, reward in pts:
            ax.annotate(reward, (x, y), fontsize=6, xytext=(3, 3),
                        textcoords='offset points', color='#444')
    ax.set_xlabel('clean-validity (quality)')
    ax.set_ylabel('uniqueness among clean-valid (diversity)')
    ax.set_title('Quality-diversity front across reward presets')
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_quality_diversity.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_property_dists(runs, out_dir, dataset_path=None, max_runs=4):
    from rdkit import Chem
    smiles_sets = {}
    for r in runs[:max_runs]:
        files = sorted(glob.glob(os.path.join(r['eval_dir'], 'smiles_selected_*.txt')))
        if not files:
            continue
        with open(files[0]) as f:
            smiles_sets[f'{r["latent"]}/{r["reward"]}/s{r["seed"]}'] = \
                [s.strip() for s in f if s.strip()]
    if not smiles_sets:
        print('  skip fig_property_dists: no SMILES files')
        return

    reference = None
    if dataset_path and os.path.exists(dataset_path):
        from qmolgan.generate import load_dataset, training_smiles
        ref_smiles = training_smiles(load_dataset(dataset_path))
        reference = chem.raw_properties(
            [Chem.MolFromSmiles(s) for s in ref_smiles[:3000]])

    keys = ['QED', 'logP', 'SA', 'MW']
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.3))
    for ax, key in zip(axes, keys):
        if reference is not None:
            ax.hist(reference[key][np.isfinite(reference[key])], bins=40, density=True,
                    color='#ADB5BD', alpha=0.75, label='QM9 training set')
        for i, (label, smis) in enumerate(smiles_sets.items()):
            props = chem.raw_properties([Chem.MolFromSmiles(s) for s in smis])
            vals = props[key][np.isfinite(props[key])]
            if vals.size:
                ax.hist(vals, bins=40, density=True, histtype='step', lw=1.6,
                        color=PALETTE[i % len(PALETTE)], label=label)
        ax.set_xlabel(key + (' (lower better)' if key == 'SA' else ''))
        ax.set_ylabel('density')
    axes[0].legend(fontsize=6)
    fig.suptitle('Property distributions: generated vs training set')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_property_dists.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_training_curves(results_root, out_dir, metrics=('clean_validity',
                                                        'uniqueness_clean',
                                                        'validity')):
    histories = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_root, '*', 'history.csv'))):
        run_dir = os.path.dirname(path)
        cfg_path = os.path.join(run_dir, 'config.json')
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        histories[f'{cfg.get("latent")} / {cfg.get("reward_preset")}'].append(df)
    if not histories:
        print('  skip fig_training_curves: no history.csv files')
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.6))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for i, (label, dfs) in enumerate(sorted(histories.items())):
            n_epochs = min(len(d) for d in dfs)
            if n_epochs == 0:
                continue
            stack = np.vstack([d[metric].to_numpy()[:n_epochs] for d in dfs
                               if metric in d])
            if stack.size == 0:
                continue
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            xs = dfs[0]['epoch'].to_numpy()[:n_epochs]
            color = PALETTE[i % len(PALETTE)]
            ax.plot(xs, mean, color=color, lw=1.4, label=f'{label} (n={stack.shape[0]})')
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_xlabel('epoch'); ax.set_ylabel(metric)
        ax.set_title(metric)
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.suptitle('In-training validation (fixed sample size and noise seed per epoch)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_training_curves.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_epoch_sweep(runs, out_dir, max_panels=4):
    sweeps = []
    for r in runs:
        path = os.path.join(r['eval_dir'], 'epoch_sweep.csv')
        if os.path.exists(path):
            sweeps.append((r, pd.read_csv(path)))
        if len(sweeps) >= max_panels:
            break
    if not sweeps:
        print('  skip fig_epoch_sweep: no epoch_sweep.csv')
        return

    fig, axes = plt.subplots(len(sweeps), 1, figsize=(7.5, 2.7 * len(sweeps)),
                             sharex=True, squeeze=False)
    for ax, (r, df) in zip(axes[:, 0], sweeps):
        for i, col in enumerate(('validity', 'clean_validity', 'uniqueness_clean')):
            if col in df:
                ax.plot(df['epoch'], df[col], lw=1.2,
                        color=PALETTE[i % len(PALETTE)], label=col)
        sel = r['metrics'].get('epoch')
        if sel is not None:
            ax.axvline(sel, color='k', ls='--', lw=1,
                       label=f'selected epoch {sel}')
        ax.set_ylabel('metric')
        ax.set_title(f'{r["latent"]} / {r["reward"]} / seed {r["seed"]}', fontsize=9)
        ax.legend(fontsize=6, ncol=4)
    axes[-1, 0].set_xlabel('epoch')
    fig.suptitle('Checkpoint-selection sweep (selection noise stream)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_epoch_sweep.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_circuit_ablation(runs, out_dir):
    pts = [r for r in runs if str(r['latent']).startswith('vqc') and r['qubits']]
    if len({(r['qubits'], r['layers']) for r in pts}) < 2:
        print('  skip fig_circuit_ablation: needs >1 circuit shape')
        return
    metrics = ['clean_validity', 'uniqueness_clean', 'QED', 'SA']
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4))
    for ax, metric in zip(axes, metrics):
        stats = group_mean_std(pts, lambda r: (r['qubits'], r['layers']), metric)
        labels = sorted(stats)
        ys = [stats[k][0] for k in labels]
        es = [stats[k][1] for k in labels]
        ax.bar(range(len(labels)), ys, yerr=es, capsize=3, color=PALETTE[0], alpha=0.9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f'q={q}\nL={L}' for q, L in labels], fontsize=7)
        ax.set_title(metric)
        annotate_n(ax, labels, [stats[k][2] for k in labels])
    fig.suptitle('Circuit-shape ablation: qubit count and depth')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_circuit_ablation.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def fig_latent_geometry(latent_csv, out_dir):
    if not os.path.exists(latent_csv):
        print(f'  skip fig_latent_geometry: run analyze_latent.py first ({latent_csv})')
        return
    df = pd.read_csv(latent_csv)
    head = df[df.qubits.isna() | (df.qubits == 4)].drop_duplicates('latent')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(range(len(head)), head.participation_ratio, color=PALETTE[0], alpha=0.9)
    axes[0].axhline(head.dim.max(), color='k', ls=':', lw=1,
                    label='full-rank ceiling ($d$)')
    axes[0].set_ylabel('participation ratio')
    axes[0].set_title('Effective latent dimensionality')
    axes[0].legend(fontsize=7)
    axes[1].bar(range(len(head)), head.occupancy_frac * 100, color=PALETTE[1], alpha=0.9)
    axes[1].set_ylabel('% of $[-1,1]^d$ grid cells reached')
    axes[1].set_title('Latent-space coverage')
    for ax in axes:
        ax.set_xticks(range(len(head)))
        ax.set_xticklabels(head.latent, rotation=25, ha='right')
    fig.suptitle('The VQC latent is bounded AND low-rank; both differ from a Gaussian')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_latent_geometry.png')
    fig.savefig(path); plt.close(fig)
    print(f'  wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_root', default='results/runs')
    p.add_argument('--out_dir', default='figures')
    p.add_argument('--report', default='selected', choices=['selected', 'final'])
    p.add_argument('--dataset', default='data/qm9_5k_py37.sparsedataset',
                   help='reference dataset for property-distribution overlays')
    p.add_argument('--latent_csv', default='results/latent_analysis/latent_analysis.csv')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_all(args.results_root, args.report)
    print(f'Loaded {len(runs)} evaluated runs from {args.results_root}')

    fig_latent_geometry(args.latent_csv, args.out_dir)
    if not runs:
        print('No evaluated runs yet — only the latent-geometry figure was produced.')
        return
    fig_factorial(runs, args.out_dir)
    fig_uniqueness_vs_n(runs, args.out_dir)
    fig_quality_diversity(runs, args.out_dir)
    fig_property_dists(runs, args.out_dir, args.dataset)
    fig_training_curves(args.results_root, args.out_dir)
    fig_epoch_sweep(runs, args.out_dir)
    fig_circuit_ablation(runs, args.out_dir)


if __name__ == '__main__':
    main()
