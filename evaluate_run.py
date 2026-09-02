#!/usr/bin/env python
"""Evaluate one training run end-to-end under the pre-registered protocol.

    python evaluate_run.py --run_dir results/runs/quantum_vqc_ablation_b_seed42

Does, in order:
  1. sweep every saved epoch on the SELECTION noise stream (N_SELECT samples),
  2. apply the fixed selection rule to pick an epoch,
  3. re-evaluate the selected epoch AND the final epoch on the disjoint
     REPORT noise stream (N_REPORT samples) with the full metric suite,
  4. write epoch_sweep.csv, results.json and generated SMILES.

Nothing downstream (tables, figures, aggregation) reads anything other than
the results.json this produces, so a table cannot disagree with the protocol.

Legacy runs that have no config.json can be described on the command line
(--dataset/--latent/--z_dim/--qubits/--layers/--g_conv_dim); the values are
recorded into the output so the reconstruction is never implicit.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from qmolgan import chem, generate, latent as latent_mod, protocol


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True,
                   help='run root, or its train/ or train/model_dir subdirectory')
    p.add_argument('--out_dir', default=None,
                   help='defaults to <run_dir>/eval')
    p.add_argument('--seed', type=int, default=None,
                   help='training seed of this run; defaults to config.json seed')
    p.add_argument('--n_select', type=int, default=protocol.N_SELECT)
    p.add_argument('--n_report', type=int, default=protocol.N_REPORT)
    p.add_argument('--max_epoch', type=int, default=None)
    p.add_argument('--epoch_stride', type=int, default=1,
                   help='sweep every k-th epoch (the selection floor still applies)')
    p.add_argument('--only_epoch', type=int, default=None,
                   help='skip the sweep and report this epoch directly')
    p.add_argument('--skip_sweep', action='store_true',
                   help='reuse an existing epoch_sweep.csv in --out_dir')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--fcd', action='store_true', help='compute FCD (needs fcd_torch)')
    p.add_argument('--device', default='cpu')
    # Legacy-run description (ignored when config.json exists).
    p.add_argument('--dataset', default=None)
    p.add_argument('--latent', default=None, choices=list(latent_mod.LATENT_KINDS))
    p.add_argument('--z_dim', type=int, default=None)
    p.add_argument('--qubits', type=int, default=None)
    p.add_argument('--layers', type=int, default=None)
    p.add_argument('--g_conv_dim', default=None, help='JSON list, e.g. "[16]"')
    p.add_argument('--post_method', default='softmax')
    return p.parse_args()


def overrides_from_args(a):
    ov = {'dataset': a.dataset, 'latent': a.latent, 'z_dim': a.z_dim,
          'qubits': a.qubits, 'layers': a.layers}
    if a.g_conv_dim is not None:
        ov['g_conv_dim'] = json.loads(a.g_conv_dim)
    return {k: v for k, v in ov.items() if v is not None}


def main():
    args = parse_args()
    run_dir, train_dir, model_dir = generate.resolve_run_dir(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    overrides = overrides_from_args(args)
    cfg = generate.read_run_config(run_dir, overrides)
    run_seed = args.seed if args.seed is not None else int(cfg.get('seed', 0))

    print(f'Run       : {run_dir}')
    print(f'Latent    : {cfg["latent"]}  z_dim={cfg["z_dim"]}  g_conv_dim={cfg["g_conv_dim"]}')
    print(f'Dataset   : {cfg["dataset"]}')
    print(f'Run seed  : {run_seed}')

    data = generate.load_dataset(cfg['dataset'])
    train_smiles = generate.training_smiles(data)
    print(f'Train split: {len(train_smiles)} molecules (novelty reference)')

    epochs = generate.available_epochs(model_dir, args.max_epoch)
    if not epochs:
        sys.exit(f'no *-G.ckpt checkpoints in {model_dir}')
    n_epochs = max(epochs)
    print(f'Checkpoints: {len(epochs)} (epochs {min(epochs)}..{n_epochs})')

    sweep_path = os.path.join(out_dir, 'epoch_sweep.csv')

    # ---- 1. selection sweep ------------------------------------------------
    if args.only_epoch is not None:
        selected_epoch = args.only_epoch
        sweep_rows = []
        selection_info = {'mode': 'explicit_epoch', 'epoch': selected_epoch}
    elif args.skip_sweep and os.path.exists(sweep_path):
        sweep_rows = pd.read_csv(sweep_path).to_dict('records')
        best, scored = protocol.select_best_epoch(sweep_rows, n_epochs)
        selected_epoch = int(best['epoch'])
        selection_info = {'mode': 'reused_sweep', 'epoch': selected_epoch,
                          'selection_score': best.get('selection_score')}
    else:
        sweep_epochs = epochs[::max(1, args.epoch_stride)]
        if epochs[-1] not in sweep_epochs:
            sweep_epochs.append(epochs[-1])
        print(f'\nSelection sweep: {len(sweep_epochs)} epochs x {args.n_select} molecules '
              f'(seed {protocol.select_seed(run_seed)})')
        sweep_rows = []
        t0 = time.time()
        for i, epoch in enumerate(sweep_epochs):
            G, lat, data, _ = generate.load_run(run_dir, epoch, overrides,
                                                device=args.device, data=data)
            mols = generate.sample_molecules(
                G, lat, data, args.n_select, batch_size=args.batch_size,
                seed=protocol.select_seed(run_seed), post_method=args.post_method,
                device=args.device)
            # Selection uses only the two cheap structural metrics, so the
            # sweep never pays for fingerprints/scaffolds it will not use.
            row = quick_metrics(mols, set(train_smiles))
            row['epoch'] = epoch
            sweep_rows.append(row)
            done, total = i + 1, len(sweep_epochs)
            eta = (time.time() - t0) / done * (total - done) / 60
            print(f'  [{done:>4}/{total}] epoch {epoch:>4}  '
                  f'valid={row["validity"]:.3f} clean={row["clean_validity"]:.3f} '
                  f'uniq_clean={row["uniqueness_clean"]:.3f}  ETA {eta:.1f} min', flush=True)
            if done % 20 == 0 or done == total:
                pd.DataFrame(sweep_rows).to_csv(sweep_path, index=False)

        best, scored = protocol.select_best_epoch(sweep_rows, n_epochs)
        pd.DataFrame(scored).to_csv(sweep_path, index=False)
        selected_epoch = int(best['epoch'])
        selection_info = {'mode': 'protocol_sweep', 'epoch': selected_epoch,
                          'selection_score': best.get('selection_score'),
                          'fallback': best.get('selection_fallback')}
        print(f'\nSelected epoch {selected_epoch} '
              f'(score {best.get("selection_score", float("nan")):.4f}) '
              f'by rule: {protocol.SELECTION_RULE}')

    # ---- 2/3. report stream ------------------------------------------------
    final_epoch = epochs[-1]
    report_epochs = {'selected': selected_epoch}
    if final_epoch != selected_epoch:
        report_epochs['final'] = final_epoch

    reports = {}
    for label, epoch in report_epochs.items():
        print(f'\nReport ({label}) epoch {epoch}: {args.n_report} molecules '
              f'(seed {protocol.report_seed(run_seed)})')
        G, lat, data, load_cfg = generate.load_run(run_dir, epoch, overrides,
                                                   device=args.device, data=data)
        mols = generate.sample_molecules(
            G, lat, data, args.n_report, batch_size=args.batch_size,
            seed=protocol.report_seed(run_seed), post_method=args.post_method,
            device=args.device)
        metrics = chem.evaluate_molecules(mols, train_smiles, compute_fcd=args.fcd,
                                          seed=protocol.report_seed(run_seed),
                                          device=args.device)
        smi = [chem.canonical_smiles(m) for m in mols]
        clean = [s for m, s in zip(mols, smi) if chem.is_clean_valid(m) and s]
        metrics['uniqueness_curve'] = chem.uniqueness_vs_sample_size(
            smi, seed=protocol.report_seed(run_seed))
        metrics['epoch'] = epoch
        metrics['latent_state_source'] = load_cfg['latent_state_source']
        reports[label] = metrics
        with open(os.path.join(out_dir, f'smiles_{label}_epoch{epoch}.txt'), 'w') as f:
            f.write('\n'.join(clean))
        print('  ' + '  '.join(
            f'{k}={metrics[k]:.4f}' for k in chem.HEADLINE_KEYS
            if isinstance(metrics.get(k), float) and np.isfinite(metrics[k])))

    # ---- 4. latent-space analysis -----------------------------------------
    lat_stats = latent_mod.latent_statistics(
        lat, n=4096, seed=protocol.latent_seed(run_seed))

    out = {
        'run_dir': run_dir,
        'config': {k: v for k, v in cfg.items() if k != 'gen_circuit'},
        'protocol': protocol.protocol_metadata(run_seed, n_epochs),
        'selection': selection_info,
        'reports': reports,
        'latent_statistics': lat_stats,
        'n_checkpoints': len(epochs),
        'final_epoch': final_epoch,
    }
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'\nWrote {results_path}')


def quick_metrics(mols, train_set):
    """Cheap per-epoch metrics for the selection sweep only.

    Deliberately a subset of `chem.evaluate_molecules` — the same definitions,
    without the O(n^2) diversity work that the selection rule never consults.
    """
    n = len(mols)
    valid_smi, clean_smi = [], []
    for m in mols:
        if not chem.is_valid(m):
            continue
        s = chem.canonical_smiles(m)
        if s is None:
            continue
        valid_smi.append(s)
        if chem.is_clean_valid(m):
            clean_smi.append(s)
    nv, nc = len(valid_smi), len(clean_smi)
    props = chem.raw_properties([m for m in mols if chem.is_clean_valid(m)])
    return {
        'validity': nv / n if n else float('nan'),
        'clean_validity': nc / n if n else float('nan'),
        'uniqueness': len(set(valid_smi)) / nv if nv else float('nan'),
        'uniqueness_clean': len(set(clean_smi)) / nc if nc else float('nan'),
        'novelty': sum(s not in train_set for s in valid_smi) / nv if nv else float('nan'),
        'novelty_clean': (sum(s not in train_set for s in clean_smi) / nc
                          if nc else float('nan')),
        'QED': float(np.nanmean(props['QED'])) if nc else float('nan'),
        'logP': float(np.nanmean(props['logP'])) if nc else float('nan'),
        'SA': float(np.nanmean(props['SA'])) if nc else float('nan'),
    }


if __name__ == '__main__':
    main()
