#!/usr/bin/env python
"""Generate the full experiment manifest.

    python scripts/build_manifest.py --out configs/manifest.csv

Emits one row per training run: a unique run id, the exact `python main.py`
arguments, a tier, and an estimated cost. The SLURM array script then runs
row N of this file for array index N, so the experiment grid lives in
version control as data rather than as a shell loop nobody can audit.

Tiers, and what each one answers:

  T1 headline    latent x reward factorial. Answers "is the gain from the
                 VQC or from the reward?" — the question v1 could not answer —
                 by varying exactly one factor at a time under an otherwise
                 identical protocol.
  T2 mechanism   classical controls matched to the VQC on support (uniform),
                 intrinsic rank (rank2), and function class (trig), plus the
                 entanglement ablation (vqc_noent).
  T3 reward-C    classical reward ablations (the original Table I).
  T4 reward-Q    the same reward ablations on the quantum model, which the
                 v1 pipeline never ran.
  T5 circuit     qubit count and depth sweep. Tests the claim that reward
                 design matters "as strongly as" architecture.
  T6 sensitivity one-at-a-time reward-weight sweeps.
  T7 scale       the same headline conditions on the full GDB9 dataset.

Every run in T1-T6 uses the SAME dataset, epoch budget, batch size, n_critic,
optimiser and learning rates. Only the named factor changes.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmolgan.rewards import weight_sweep_grid

# Shared training configuration. Changing anything here changes every run,
# which is the point: the comparison is only as good as its shared baseline.
COMMON = dict(
    mol_data_dir='data/qm9_5k_py37.sparsedataset',
    num_epochs=300,
    batch_size=16,
    n_critic=5,
    g_lr=1e-3,
    d_lr=1e-3,
    g_conv_dim='[16]',
    z_dim=4,
    val_n=1000,
    val_every=1,
    model_save_step=1,
)

GDB9 = dict(COMMON, mol_data_dir='data/gdb9_9nodes.sparsedataset', num_epochs=30)

HEADLINE_SEEDS = [42, 123, 456, 789, 1011]
ABLATION_SEEDS = [42, 123, 456]
SWEEP_SEEDS = [42, 123]

# Rough wall-clock per 300-epoch run, measured at ~5.7 ms per circuit
# evaluation and ~25 ms per classical step on one CPU core of the reference
# node. Used only to size SLURM time limits.
HOURS = {'classical': 3.0, 'quantum': 14.0, 'gdb9_classical': 8.0, 'gdb9_quantum': 40.0}


def is_quantum(latent):
    return latent.startswith('vqc')


def make_run(tier, latent, reward, seed, common=None, extra=None, tag=None):
    common = dict(common or COMMON)
    extra = dict(extra or {})
    parts = [tier, latent, reward]
    if tag:
        parts.append(tag)
    parts.append(f's{seed}')
    run_id = '_'.join(str(p) for p in parts)

    args = dict(common)
    args.update(latent=latent, reward_preset=reward, seed=seed)
    if is_quantum(latent):
        args.setdefault('qubits', args['z_dim'])
        args.setdefault('layer', 3)
        args.setdefault('qc_lr', 0.04)
    args.update(extra)

    dataset_tag = 'gdb9' if 'gdb9' in str(args['mol_data_dir']) else 'qm9'
    kind = ('gdb9_' if dataset_tag == 'gdb9' else '') + \
           ('quantum' if is_quantum(latent) else 'classical')

    flags = []
    for key, value in args.items():
        text = str(value)
        # Quote anything the shell would glob or split: '[16]' is a valid
        # bracket pattern and would be mangled before argparse sees it.
        if any(ch in text for ch in '[]{}*? '):
            text = f"'{text}'"
        flags.append(f'--{key} {text}')
    return {
        'run_id': run_id,
        'tier': tier,
        'latent': latent,
        'reward_preset': reward,
        'seed': seed,
        'dataset': dataset_tag,
        'est_hours': HOURS[kind],
        'args': ' '.join(flags),
    }


def build():
    runs = []

    # ---- T1: headline factorial ----------------------------------------
    for latent in ('gaussian', 'uniform', 'vqc'):
        for reward in ('none', 'ablation_b'):
            for seed in HEADLINE_SEEDS:
                runs.append(make_run('T1', latent, reward, seed))

    # ---- T2: mechanism controls ----------------------------------------
    for latent in ('rank2', 'trig', 'vqc_noent'):
        for seed in ABLATION_SEEDS:
            runs.append(make_run('T2', latent, 'ablation_b', seed))
    # The no-reward arm of the entanglement ablation, so T2 supports the same
    # factorial contrast T1 does.
    for seed in ABLATION_SEEDS:
        runs.append(make_run('T2', 'vqc_noent', 'none', seed))

    # ---- T3/T4: reward ablations, classical and quantum ------------------
    for reward in ('legacy', 'ablation_a', 'ablation_c',
                   'ablation_d_clean', 'ablation_e_diverse'):
        for seed in ABLATION_SEEDS:
            runs.append(make_run('T3', 'gaussian', reward, seed))
    for reward in ('ablation_a', 'ablation_c', 'ablation_d_clean',
                   'ablation_e_diverse'):
        for seed in ABLATION_SEEDS:
            runs.append(make_run('T4', 'vqc', reward, seed))

    # ---- T5: circuit shape ----------------------------------------------
    for qubits, layers in ((2, 3), (6, 3), (8, 3), (4, 1), (4, 6)):
        for seed in ABLATION_SEEDS:
            runs.append(make_run(
                'T5', 'vqc', 'ablation_b', seed,
                extra={'z_dim': qubits, 'qubits': qubits, 'layer': layers},
                tag=f'q{qubits}L{layers}'))

    # ---- T6: reward-weight sensitivity ----------------------------------
    # Full one-at-a-time sweeps on the cheap classical model; three
    # representative points repeated on the quantum model to confirm the
    # sensitivity transfers.
    for axis in ('rw_unique', 'rw_fragment_penalty'):
        values = ((0.0, 0.05, 0.10, 0.15, 0.25, 0.35) if axis == 'rw_unique'
                  else (0.0, 0.10, 0.20, 0.30, 0.40))
        for name, weights in weight_sweep_grid('ablation_b', axis, values):
            extra = {k: v for k, v in weights.items() if k.startswith('rw_')}
            tag = name.split('__')[1]
            for seed in SWEEP_SEEDS:
                runs.append(make_run('T6', 'gaussian', 'ablation_b', seed,
                                     extra=extra, tag=tag))
    for name, weights in weight_sweep_grid('ablation_b', 'rw_unique',
                                           (0.0, 0.15, 0.35)):
        extra = {k: v for k, v in weights.items() if k.startswith('rw_')}
        tag = name.split('__')[1]
        for seed in SWEEP_SEEDS:
            runs.append(make_run('T6', 'vqc', 'ablation_b', seed,
                                 extra=extra, tag=tag))

    # ---- T7: scale check on full GDB9 ------------------------------------
    for latent in ('gaussian', 'vqc'):
        for reward in ('none', 'ablation_b'):
            for seed in ABLATION_SEEDS[:2]:
                runs.append(make_run('T7', latent, reward, seed, common=GDB9))

    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='configs/manifest.csv')
    p.add_argument('--results_root', default='results/runs')
    p.add_argument('--tiers', nargs='*', default=None,
                   help='restrict to these tiers, e.g. --tiers T1 T2')
    args = p.parse_args()

    runs = build()
    if args.tiers:
        runs = [r for r in runs if r['tier'] in set(args.tiers)]
    for r in runs:
        r['saving_dir'] = os.path.join(args.results_root, r['run_id'])
        r['command'] = (f'python main.py --saving_dir {r["saving_dir"]} '
                        f'--run_name {r["run_id"]} {r["args"]}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fields = ['run_id', 'tier', 'latent', 'reward_preset', 'seed', 'dataset',
              'est_hours', 'saving_dir', 'command']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(runs)

    by_tier = {}
    for r in runs:
        t = by_tier.setdefault(r['tier'], {'n': 0, 'h': 0.0})
        t['n'] += 1
        t['h'] += r['est_hours']
    print(f'Wrote {args.out}: {len(runs)} runs\n')
    print(f'{"tier":6s} {"runs":>5s} {"est. core-hours":>16s}')
    for tier in sorted(by_tier):
        print(f'{tier:6s} {by_tier[tier]["n"]:>5d} {by_tier[tier]["h"]:>16.0f}')
    print(f'{"TOTAL":6s} {len(runs):>5d} {sum(t["h"] for t in by_tier.values()):>16.0f}')
    print('\nRun one row with:  bash scripts/run_one.sh <row-number>')
    print('Submit all with :  sbatch --array=1-%d scripts/train_array.slurm' % len(runs))


if __name__ == '__main__':
    main()
