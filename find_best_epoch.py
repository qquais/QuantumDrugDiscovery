"""
find_best_epoch.py — Sweep all saved checkpoints and find the best epoch.

For each epoch from 1 to MAX_EPOCH (where the checkpoint exists), generates
N_GENERATE molecules using the quantum noise circuit and computes key metrics.
Results are saved to analysis/epoch_sweep.csv.

Run from the project root:
    python find_best_epoch.py

    # Optional overrides:
    python find_best_epoch.py --max_epoch 284 --n_generate 500
"""

import os
import sys
import argparse
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pennylane as qml
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)
warnings.filterwarnings('ignore')

from models.models import Generator
from data.sparse_molecular_dataset import SparseMolecularDataset
from utils.utils import MolecularMetrics

# ---------------------------------------------------------------------------
# Fixed config — matches the training run in main.py
# ---------------------------------------------------------------------------
MODEL_DIR   = 'results/quantum_ablationB_300/GAN/20260425_191533/train/model_dir'
DATASET     = '/scratch/gilbreth/quaiqa01/QuantumDrugDiscovery/data/qm9_5k_py37.sparsedataset'
ANALYSIS_DIR = 'analysis'

MAX_EPOCH   = 284
N_GENERATE  = 500
BATCH_SIZE  = 16

QUBITS      = 4
LAYERS      = 3
Z_DIM       = 4
G_CONV_DIM  = [16]
DROPOUT     = 0.0


# ---------------------------------------------------------------------------
# Core helpers (self-contained, no dependency on evaluate_quantum.py)
# ---------------------------------------------------------------------------

def postprocess(inputs, method='softmax', temperature=1.0):
    def listify(x):
        return x if isinstance(x, (list, tuple)) else [x]
    def delistify(x):
        return x if len(x) > 1 else x[0]
    if method == 'hard_gumbel':
        out = [F.gumbel_softmax(
                   e.contiguous().view(-1, e.size(-1)) / temperature, hard=True
               ).view(e.size()) for e in listify(inputs)]
    elif method == 'soft_gumbel':
        out = [F.gumbel_softmax(
                   e.contiguous().view(-1, e.size(-1)) / temperature, hard=False
               ).view(e.size()) for e in listify(inputs)]
    else:
        out = [F.softmax(e / temperature, -1) for e in listify(inputs)]
    return [delistify(e) for e in out]


def build_gen_circuit(qubits, layers):
    """Exact QuMolGAN circuit from Kao et al. 2023 — built once, reused for all epochs."""
    dev = qml.device('default.qubit', wires=qubits)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def gen_circuit(w):
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)
        for i in range(qubits):
            qml.RY(np.arcsin(z1), wires=i)
            qml.RZ(np.arcsin(z2), wires=i)
        for _l in range(layers):
            for i in range(qubits):
                qml.RY(w[i], wires=i)
            for i in range(qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(w[i + qubits], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

    return gen_circuit


def generate_molecules(G, gen_circuit, gen_weights, data, n_generate, batch_size):
    """Generate n_generate molecules; returns a flat list of RDKit mol objects (or None)."""
    G.eval()
    device = next(G.parameters()).device
    all_mols = []
    n_batches = (n_generate + batch_size - 1) // batch_size

    for _ in range(n_batches):
        cur_batch = min(batch_size, n_generate - len(all_mols))
        sample_list = [gen_circuit(gen_weights) for _ in range(cur_batch)]
        z = torch.stack(tuple(sample_list)).to(device).float()

        with torch.no_grad():
            edges_logits, nodes_logits = G(z)

        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), 'softmax')
        edges_hard = torch.max(edges_hat, -1)[1]
        nodes_hard = torch.max(nodes_hat, -1)[1]

        batch_mols = [
            data.matrices2mol(n_.cpu().numpy(), e_.cpu().numpy(), strict=True)
            for e_, n_ in zip(edges_hard, nodes_hard)
        ]
        all_mols.extend(batch_mols)

    return all_mols[:n_generate]


def compute_metrics(mols, data):
    """Return a flat dict of all six metrics as floats in [0,1] or raw."""
    n_total = len(mols)
    if n_total == 0:
        return {k: float('nan') for k in
                ('validity', 'clean_validity', 'uniqueness', 'novelty', 'QED', 'SA')}

    valid_mask = np.array([MolecularMetrics.valid_lambda(m) for m in mols])
    clean_mask = np.array([MolecularMetrics.valid_lambda_special(m) for m in mols])

    validity       = float(valid_mask.sum()) / n_total
    clean_validity = float(clean_mask.sum()) / n_total
    uniqueness     = MolecularMetrics.unique_total_score(mols)

    try:
        novelty = MolecularMetrics.novel_total_score(mols, data)
        novelty = 0.0 if np.isnan(novelty) else float(novelty)
    except Exception:
        novelty = float('nan')

    valid_mols = [m for m, v in zip(mols, valid_mask) if v]
    if valid_mols:
        qed = float(np.nanmean(
            MolecularMetrics.quantitative_estimation_druglikeness_scores(valid_mols)))
        sa  = float(np.nanmean(
            MolecularMetrics.synthetic_accessibility_score_scores(valid_mols, norm=False)))
    else:
        qed, sa = float('nan'), float('nan')

    return {
        'validity':       round(validity,       4),
        'clean_validity': round(clean_validity, 4),
        'uniqueness':     round(uniqueness,     4),
        'novelty':        round(novelty,        4),
        'QED':            round(qed,            4),
        'SA':             round(sa,             4),
    }


def print_summary(df, n_show=20):
    """Print top-n epochs sorted by uniqueness descending."""
    cols = ['epoch', 'validity', 'clean_validity', 'uniqueness', 'novelty', 'QED', 'SA']
    df_sorted = df.sort_values('uniqueness', ascending=False)

    header = (f"{'Epoch':>6}  {'Validity':>9}  {'CleanVal':>9}  "
              f"{'Unique':>9}  {'Novelty':>9}  {'QED':>7}  {'SA':>7}")
    sep = '-' * len(header)
    print(f'\n{sep}')
    print(f'  Top {min(n_show, len(df_sorted))} epochs by Uniqueness')
    print(sep)
    print(header)
    print(sep)
    for _, row in df_sorted.head(n_show).iterrows():
        print(f"  {int(row['epoch']):>4}   "
              f"{row['validity']:>9.4f}  "
              f"{row['clean_validity']:>9.4f}  "
              f"{row['uniqueness']:>9.4f}  "
              f"{row['novelty']:>9.4f}  "
              f"{row['QED']:>7.4f}  "
              f"{row['SA']:>7.3f}")
    print(sep)

    best = df_sorted.iloc[0]
    print(f"\n  Best epoch by uniqueness: {int(best['epoch'])}")
    print(f"    uniqueness    = {best['uniqueness']:.4f}")
    print(f"    validity      = {best['validity']:.4f}")
    print(f"    clean_valid   = {best['clean_validity']:.4f}")
    print(f"    novelty       = {best['novelty']:.4f}")
    print(f"    QED           = {best['QED']:.4f}")
    print(f"    SA            = {best['SA']:.3f}  (lower = more synthesizable)")
    print()


def save_sweep_chart(df, output_path):
    """Line plot of all metrics across epochs."""
    df_s = df.sort_values('epoch')
    metrics = ['validity', 'clean_validity', 'uniqueness', 'novelty', 'QED']
    colors  = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: [0,1] metrics
    ax = axes[0]
    for m, c in zip(metrics, colors):
        ax.plot(df_s['epoch'], df_s[m], label=m, color=c, linewidth=1.4)
    ax.set_ylabel('Score [0–1]')
    ax.set_title('QuMolGAN Epoch Sweep — Metric Trajectories', fontsize=12)
    ax.legend(loc='upper left', fontsize=8, ncol=3)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Bottom panel: SA (raw, 1–10 scale)
    ax2 = axes[1]
    ax2.plot(df_s['epoch'], df_s['SA'], color='#E08462', linewidth=1.4, label='SA (↓ better)')
    ax2.set_ylabel('SA score (raw, 1–10)')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Sweep chart saved → {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Sweep all QuMolGAN checkpoints.')
    parser.add_argument('--model_dir',   default=MODEL_DIR)
    parser.add_argument('--dataset',     default=DATASET)
    parser.add_argument('--analysis_dir',default=ANALYSIS_DIR)
    parser.add_argument('--max_epoch',   type=int, default=MAX_EPOCH)
    parser.add_argument('--n_generate',  type=int, default=N_GENERATE)
    args = parser.parse_args()

    os.makedirs(args.analysis_dir, exist_ok=True)
    device = torch.device('cpu')

    # ---- Load dataset once ----
    print(f'Loading dataset: {args.dataset}')
    data = SparseMolecularDataset()
    data.load(args.dataset)
    print(f'  vertexes={data.vertexes}, bond_types={data.bond_num_types}, '
          f'atom_types={data.atom_num_types}')

    # ---- Build Generator architecture once ----
    G = Generator(G_CONV_DIM, Z_DIM,
                  data.vertexes, data.bond_num_types, data.atom_num_types, DROPOUT)
    G.to(device)

    # ---- Build quantum circuit once ----
    print(f'Building quantum circuit (qubits={QUBITS}, layers={LAYERS})...')
    gen_circuit = build_gen_circuit(QUBITS, LAYERS)

    # ---- Pre-read gen_weights CSV once ----
    weights_file = os.path.join(args.model_dir, 'molgan_red_weights.csv')
    if os.path.exists(weights_file):
        weights_df = pd.read_csv(weights_file, header=None)
        print(f'Loaded gen_weights CSV: {len(weights_df)} rows.')
    else:
        weights_df = None
        print('WARNING: molgan_red_weights.csv not found — random weights will be used.')

    # ---- Scan which checkpoints exist ----
    epochs_found = [
        e for e in range(1, args.max_epoch + 1)
        if os.path.exists(os.path.join(args.model_dir, f'{e}-G.ckpt'))
    ]
    print(f'\nFound {len(epochs_found)} checkpoints out of epochs 1–{args.max_epoch}.\n')
    if not epochs_found:
        sys.exit(f'ERROR: no checkpoints found in {args.model_dir}')

    # ---- Sweep ----
    results = []
    t_start = time.time()

    for idx, epoch in enumerate(epochs_found):
        ckpt_path = os.path.join(args.model_dir, f'{epoch}-G.ckpt')

        # Reload G weights (in-place; no re-allocation)
        G.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Get gen_weights for this epoch
        if weights_df is not None and (epoch - 1) < len(weights_df):
            row_vals = weights_df.iloc[epoch - 1, 1:].values.astype(float)
            gen_weights = torch.tensor(list(row_vals), requires_grad=False)
        else:
            n_w = LAYERS * (QUBITS * 2 - 1)
            gen_weights = torch.zeros(n_w, requires_grad=False)

        # Generate and score
        mols    = generate_molecules(G, gen_circuit, gen_weights, data,
                                     args.n_generate, BATCH_SIZE)
        metrics = compute_metrics(mols, data)

        row = {'epoch': epoch, **metrics}
        results.append(row)

        # Progress line
        elapsed = time.time() - t_start
        per_ep  = elapsed / (idx + 1)
        remaining = per_ep * (len(epochs_found) - idx - 1)
        print(f'[{idx+1:>3}/{len(epochs_found)}] epoch {epoch:>4} | '
              f'valid={metrics["validity"]:.3f}  '
              f'clean={metrics["clean_validity"]:.3f}  '
              f'unique={metrics["uniqueness"]:.3f}  '
              f'novel={metrics["novelty"]:.3f}  '
              f'QED={metrics["QED"]:.3f}  '
              f'SA={metrics["SA"]:.2f}  '
              f'| ETA {remaining/60:.1f}min',
              flush=True)

        # Checkpoint-save every 10 epochs so results survive early termination
        if (idx + 1) % 10 == 0 or (idx + 1) == len(epochs_found):
            df_partial = pd.DataFrame(results)
            csv_path = os.path.join(args.analysis_dir, 'epoch_sweep.csv')
            df_partial.to_csv(csv_path, index=False)

    # ---- Final save ----
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.analysis_dir, 'epoch_sweep.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved → {csv_path}')

    # ---- Summary ----
    print_summary(df, n_show=20)

    # ---- Chart ----
    chart_path = os.path.join(args.analysis_dir, 'epoch_sweep.png')
    save_sweep_chart(df, chart_path)

    total_min = (time.time() - t_start) / 60
    print(f'Total sweep time: {total_min:.1f} minutes.')


if __name__ == '__main__':
    main()
