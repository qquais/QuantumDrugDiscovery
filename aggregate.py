#!/usr/bin/env python
"""Aggregate per-run results.json files into multi-seed tables.

    python aggregate.py --results_root results/runs --out_dir results/tables

Groups runs by their experimental condition (latent x reward preset x latent
dimension x circuit shape x dataset), reports mean +- std and a bootstrap CI
over seeds, and runs the seed-paired comparisons that the conclusions rest
on. Single-seed point estimates were the single most repeated criticism of
the earlier pipeline; this script refuses to print a mean without also
printing n_seeds, so a table can never look multi-seed when it is not.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from qmolgan import protocol

TABLE_METRICS = ['validity', 'clean_validity', 'uniqueness', 'uniqueness_clean',
                 'novelty_clean', 'QED', 'logP', 'SA', 'int_div1',
                 'scaffold_diversity', 'snn_to_train', 'fcd']

LATEX_METRICS = ['validity', 'clean_validity', 'uniqueness_clean', 'novelty_clean',
                 'QED', 'logP', 'SA', 'int_div1', 'scaffold_diversity']

LOWER_IS_BETTER = {'SA', 'fcd', 'snn_to_train'}


def condition_key(cfg):
    """The experimental condition a run belongs to — everything except the seed."""
    latent = cfg.get('latent')
    parts = [f'latent={latent}',
             f'reward={cfg.get("reward_preset")}',
             f'z={cfg.get("z_dim")}']
    if latent in ('vqc', 'vqc_noent'):
        parts.append(f'q={cfg.get("qubits")}L{cfg.get("layers")}')
    parts.append(f'g={"-".join(str(x) for x in cfg.get("g_conv_dim", []))}')
    parts.append(f'data={os.path.basename(str(cfg.get("dataset")))}')
    return ' | '.join(parts)


def load_runs(results_root, report='selected'):
    """Collect (condition, seed, metrics, config) from every eval/results.json."""
    runs = []
    for path in sorted(glob.glob(os.path.join(results_root, '**', 'results.json'),
                                 recursive=True)):
        with open(path) as f:
            blob = json.load(f)
        cfg = blob.get('config', {})
        reports = blob.get('reports', {})
        if report not in reports:
            if not reports:
                print(f'  skip (no reports): {path}')
                continue
            report_used = 'selected' if 'selected' in reports else sorted(reports)[0]
        else:
            report_used = report
        seed = blob.get('protocol', {}).get('run_seed', cfg.get('seed'))
        runs.append({
            'path': path,
            'run_dir': blob.get('run_dir'),
            'condition': condition_key(cfg),
            'seed': seed,
            'epoch': reports[report_used].get('epoch'),
            'report': report_used,
            'metrics': reports[report_used],
            'config': cfg,
            'latent_statistics': blob.get('latent_statistics', {}),
        })
    return runs


def aggregate(runs):
    """condition -> {metric: (mean, std, n, ci_low, ci_high)} plus per-seed values."""
    by_cond = defaultdict(list)
    for r in runs:
        by_cond[r['condition']].append(r)

    table = {}
    for cond, group in by_cond.items():
        group = sorted(group, key=lambda r: (r['seed'] is None, r['seed']))
        entry = {'n_seeds': len(group),
                 'seeds': [r['seed'] for r in group],
                 'epochs': [r['epoch'] for r in group],
                 'per_seed': {}}
        for metric in TABLE_METRICS:
            vals = [r['metrics'].get(metric) for r in group]
            vals = [float(v) if v is not None else np.nan for v in vals]
            mean, std, n = protocol.mean_std(vals)
            lo, hi = protocol.bootstrap_ci(vals)
            entry[metric] = {'mean': mean, 'std': std, 'n': n, 'ci_low': lo, 'ci_high': hi}
            entry['per_seed'][metric] = vals
        # Latent geometry is a property of the condition, not of a seed, but
        # it is averaged anyway so a trained VQC's drift across seeds shows.
        prs = [r['latent_statistics'].get('participation_ratio') for r in group]
        prs = [float(p) for p in prs if p is not None and np.isfinite(p)]
        entry['participation_ratio'] = float(np.mean(prs)) if prs else float('nan')
        table[cond] = entry
    return table


def to_dataframe(table):
    rows = []
    for cond, entry in sorted(table.items()):
        row = {'condition': cond, 'n_seeds': entry['n_seeds'],
               'seeds': ','.join(str(s) for s in entry['seeds']),
               'epochs': ','.join(str(e) for e in entry['epochs']),
               'latent_participation_ratio': entry['participation_ratio']}
        for metric in TABLE_METRICS:
            row[f'{metric}_mean'] = entry[metric]['mean']
            row[f'{metric}_std'] = entry[metric]['std']
            row[f'{metric}_ci_low'] = entry[metric]['ci_low']
            row[f'{metric}_ci_high'] = entry[metric]['ci_high']
        rows.append(row)
    return pd.DataFrame(rows)


def short_label(condition):
    """Compact, LaTeX-safe model name from a condition string."""
    fields = dict(part.split('=', 1) for part in condition.split(' | ') if '=' in part)
    name = fields.get('latent', '?')
    if 'q' in fields:
        name += f' ({fields["q"]})'
    reward = fields.get('reward', '?')
    return f'{name} + {reward}'.replace('_', r'\_')


def to_latex(table, metrics=LATEX_METRICS, caption='', label='tab:main'):
    """LaTeX table body with mean +- std and an explicit seed count column."""
    header = ' & '.join(['Model', '$n$'] + [m.replace('_', r'\_') for m in metrics])
    lines = [r'\begin{tabular}{l' + 'c' * (len(metrics) + 1) + '}', r'\toprule',
             header + r' \\', r'\midrule']
    for cond, entry in sorted(table.items()):
        cells = [short_label(cond), str(entry['n_seeds'])]
        for m in metrics:
            mean, std = entry[m]['mean'], entry[m]['std']
            if not np.isfinite(mean):
                cells.append('--')
            elif entry['n_seeds'] > 1 and np.isfinite(std):
                cells.append(f'{mean:.3f} $\\pm$ {std:.3f}')
            else:
                cells.append(f'{mean:.3f}')
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    body = '\n'.join(lines)
    return (f'% {caption}\n\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n'
            f'\\label{{{label}}}\n{body}\n\\end{{table}}\n')


def paired_comparisons(runs, pairs, metrics=('clean_validity', 'uniqueness_clean',
                                             'QED', 'SA', 'int_div1',
                                             'scaffold_diversity')):
    """Seed-paired A-vs-B contrasts.

    ``pairs`` is a list of (label, condition_a, condition_b). Only seeds
    present on BOTH sides are used, and the number of usable pairs is
    reported, so a contrast built from one shared seed cannot masquerade as
    a multi-seed result.
    """
    by_cond_seed = {}
    for r in runs:
        by_cond_seed[(r['condition'], r['seed'])] = r['metrics']

    out = []
    for label, cond_a, cond_b in pairs:
        seeds_a = {s for (c, s) in by_cond_seed if c == cond_a}
        seeds_b = {s for (c, s) in by_cond_seed if c == cond_b}
        shared = sorted(seeds_a & seeds_b, key=lambda s: (s is None, s))
        if not shared:
            out.append({'comparison': label, 'metric': '-', 'n_pairs': 0,
                        'note': f'no shared seeds between {cond_a!r} and {cond_b!r}'})
            continue
        for metric in metrics:
            a = [by_cond_seed[(cond_a, s)].get(metric, np.nan) for s in shared]
            b = [by_cond_seed[(cond_b, s)].get(metric, np.nan) for s in shared]
            res = protocol.paired_difference(a, b)
            out.append({'comparison': label, 'metric': metric,
                        'A': cond_a, 'B': cond_b,
                        'mean_A': float(np.nanmean(a)), 'mean_B': float(np.nanmean(b)),
                        **res})
    return pd.DataFrame(out)


DEFAULT_PAIRS_DOC = """\
Suggested contrasts (pass with --pairs pairs.json as
[[label, condition_a, condition_b], ...]; run with --list_conditions first to
copy exact condition strings):

  "VQC vs matched Gaussian (same reward)"     latent=vqc ... vs latent=gaussian ...
  "VQC vs bounded uniform (same reward)"      latent=vqc ... vs latent=uniform ...
  "VQC vs rank-2 classical (same reward)"     latent=vqc ... vs latent=rank2 ...
  "VQC vs trig surrogate (same reward)"       latent=vqc ... vs latent=trig ...
  "Entanglement on vs off"                    latent=vqc ... vs latent=vqc_noent ...
  "Reward shaping vs none (quantum)"          reward=ablation_b vs reward=none
  "Reward shaping vs none (classical)"        reward=ablation_b vs reward=none
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_root', default='results/runs')
    p.add_argument('--out_dir', default='results/tables')
    p.add_argument('--report', default='selected', choices=['selected', 'final'],
                   help='which protocol report to tabulate')
    p.add_argument('--pairs', default=None, help='JSON file of [label, condA, condB]')
    p.add_argument('--list_conditions', action='store_true')
    args = p.parse_args()

    runs = load_runs(args.results_root, args.report)
    if not runs:
        raise SystemExit(f'no results.json under {args.results_root}; '
                         'run evaluate_run.py first')
    print(f'Loaded {len(runs)} runs from {args.results_root}')

    table = aggregate(runs)
    if args.list_conditions:
        for cond, entry in sorted(table.items()):
            print(f'  [{entry["n_seeds"]} seeds {entry["seeds"]}] {cond}')
        print('\n' + DEFAULT_PAIRS_DOC)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    df = to_dataframe(table)
    csv_path = os.path.join(args.out_dir, f'summary_{args.report}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Wrote {csv_path}')

    # Report the sample size the runs ACTUALLY used, not the protocol default:
    # an evaluation run with a reduced --n_report must not be captioned n=5000.
    n_used = sorted({int(r['metrics'].get('n_generated', 0)) for r in runs})
    n_text = str(n_used[0]) if len(n_used) == 1 else f'{min(n_used)}-{max(n_used)}'
    if len(n_used) > 1:
        print(f'WARNING: runs were evaluated at different sample sizes {n_used}; '
              'uniqueness is not comparable across them. Re-run evaluate_run.py '
              'with a single --n_report before using this table.')

    tex_path = os.path.join(args.out_dir, f'summary_{args.report}.tex')
    with open(tex_path, 'w') as f:
        f.write(to_latex(table,
                         caption=f'Multi-seed results ({args.report} checkpoint, '
                                 f'$n={n_text}$ generated molecules per run, '
                                 f'mean $\\pm$ std over seeds).',
                         label=f'tab:{args.report}'))
    print(f'Wrote {tex_path}')

    if args.pairs:
        with open(args.pairs) as f:
            pairs = [tuple(x) for x in json.load(f)]
        pdf = paired_comparisons(runs, pairs)
        pair_path = os.path.join(args.out_dir, f'paired_{args.report}.csv')
        pdf.to_csv(pair_path, index=False)
        print(f'Wrote {pair_path}')
        usable = pdf[pdf['n_pairs'] > 0] if 'n_pairs' in pdf else pdf
        skipped = pdf[pdf['n_pairs'] == 0]['comparison'].unique() if 'n_pairs' in pdf else []
        if len(usable):
            cols = ['comparison', 'metric', 'n_pairs', 'mean_A', 'mean_B',
                    'mean_diff', 'ci_low', 'ci_high', 'excludes_zero']
            with pd.option_context('display.width', 220, 'display.max_rows', 200):
                print(usable[cols].to_string(index=False))
        if len(skipped):
            print(f'\n{len(skipped)} contrast(s) skipped for lack of shared seeds: '
                  + ', '.join(map(str, skipped)))

    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        cols = ['condition', 'n_seeds'] + [f'{m}_mean' for m in
                                           ('validity', 'clean_validity',
                                            'uniqueness_clean', 'QED', 'SA')]
        print('\n' + df[cols].to_string(index=False))


if __name__ == '__main__':
    main()
