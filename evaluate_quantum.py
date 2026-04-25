"""
evaluate_quantum.py — QuMolGAN evaluation script.

Loads a trained Generator checkpoint, generates molecules using the quantum
noise circuit (Kao et al. 2023 QuMolGAN), computes all paper metrics, and
saves results to the analysis directory.

Run from the project root:
    python evaluate_quantum.py \
        --model_dir results/quantum/GAN/20260425_160026/train/model_dir \
        --epoch 30
"""

import os
import sys
import argparse
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Must run from project root so data/ imports (NP_score, SA_score) resolve.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pennylane as qml
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.CRITICAL)
warnings.filterwarnings('ignore')

from models.models import Generator
from data.sparse_molecular_dataset import SparseMolecularDataset
from utils.utils import MolecularMetrics

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = '/scratch/gilbreth/quaiqa01/QuantumDrugDiscovery/analysis'
DATASET_PATH = '/scratch/gilbreth/quaiqa01/QuantumDrugDiscovery/data/qm9_5k_py37.sparsedataset'

# ---------------------------------------------------------------------------
# QuMolGAN config — must match what was used during training in main.py
# ---------------------------------------------------------------------------
QUBITS     = 4
LAYERS     = 3
Z_DIM      = 4
G_CONV_DIM = [16]
DROPOUT    = 0.0
BATCH_SIZE = 16
POST_METHOD = 'softmax'   # deterministic; solver default during training scoring


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def postprocess(inputs, method, temperature=1.0):
    """Mirror of Solver.postprocess — applies softmax/gumbel to each input."""
    def listify(x):
        return x if isinstance(x, (list, tuple)) else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == 'hard_gumbel':
        out = [F.gumbel_softmax(
                   e.contiguous().view(-1, e.size(-1)) / temperature,
                   hard=True).view(e.size())
               for e in listify(inputs)]
    elif method == 'soft_gumbel':
        out = [F.gumbel_softmax(
                   e.contiguous().view(-1, e.size(-1)) / temperature,
                   hard=False).view(e.size())
               for e in listify(inputs)]
    else:  # softmax
        out = [F.softmax(e / temperature, -1) for e in listify(inputs)]

    return [delistify(e) for e in out]


def build_gen_circuit(qubits, layers):
    """Exact QuMolGAN circuit from Kao et al. 2023 main branch."""
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


def load_gen_weights(model_dir, epoch):
    """
    Read gen_weights from molgan_red_weights.csv.
    Solver saves row [epoch_i, w0, ..., w_n] where epoch_i = epoch - 1 (0-based).
    load_gen_weights(epoch) reads iloc[epoch-1, 1:].
    """
    weights_path = os.path.join(model_dir, 'molgan_red_weights.csv')
    df = pd.read_csv(weights_path, header=None)
    weights = df.iloc[epoch - 1, 1:].values.astype(float)
    return torch.tensor(list(weights), requires_grad=False)


def generate_molecules(G, gen_circuit, gen_weights, data, n_generate, batch_size):
    """Generate n_generate molecules using the quantum noise circuit."""
    G.eval()
    device = next(G.parameters()).device
    all_mols = []
    n_batches = (n_generate + batch_size - 1) // batch_size

    print(f'Generating {n_generate} molecules in {n_batches} batches of {batch_size}...')
    for batch_idx in range(n_batches):
        cur_batch = min(batch_size, n_generate - len(all_mols))

        # Quantum noise: batch_size circuit evaluations stacked → (batch, qubits)
        sample_list = [gen_circuit(gen_weights) for _ in range(cur_batch)]
        z = torch.stack(tuple(sample_list)).to(device).float()

        with torch.no_grad():
            edges_logits, nodes_logits = G(z)

        # Postprocess: (batch, V, V, edges) and (batch, V, nodes)
        (edges_hat, nodes_hat) = postprocess((edges_logits, nodes_logits), POST_METHOD)
        edges_hard = torch.max(edges_hat, -1)[1]   # (batch, V, V)
        nodes_hard = torch.max(nodes_hat, -1)[1]   # (batch, V)

        batch_mols = [
            data.matrices2mol(n_.cpu().numpy(), e_.cpu().numpy(), strict=True)
            for e_, n_ in zip(edges_hard, nodes_hard)
        ]
        all_mols.extend(batch_mols)

        if (batch_idx + 1) % 50 == 0:
            print(f'  {len(all_mols)}/{n_generate} molecules generated...')

    return all_mols[:n_generate]


def compute_metrics(mols, data):
    """
    Compute all evaluation metrics over a molecule list.

    Returns a dict with:
      validity, clean_validity, uniqueness, novelty  — [0, 1] fractions
      QED                                             — [0, 1] raw
      logP                                            — raw (can be negative)
      SA                                              — raw SAS score [1, 10], lower = better
    """
    n_total = len(mols)

    # Basic validity: mol not None and SMILES not empty (matches valid_lambda)
    valid_mask = np.array([MolecularMetrics.valid_lambda(m) for m in mols])
    # Clean validity: valid AND no '.' AND no '*' (matches valid_lambda_special)
    clean_mask = np.array([MolecularMetrics.valid_lambda_special(m) for m in mols])

    validity       = float(valid_mask.sum()) / n_total
    clean_validity = float(clean_mask.sum()) / n_total
    uniqueness     = MolecularMetrics.unique_total_score(mols)

    try:
        novelty = MolecularMetrics.novel_total_score(mols, data)
        if np.isnan(novelty):
            novelty = 0.0
    except Exception:
        novelty = float('nan')

    valid_mols = [m for m, v in zip(mols, valid_mask) if v]

    if len(valid_mols) > 0:
        qed_arr  = MolecularMetrics.quantitative_estimation_druglikeness_scores(valid_mols)
        qed      = float(np.nanmean(qed_arr))

        logp_arr = MolecularMetrics.water_octanol_partition_coefficient_scores(valid_mols, norm=False)
        logp     = float(np.nanmean(logp_arr))

        sa_arr   = MolecularMetrics.synthetic_accessibility_score_scores(valid_mols, norm=False)
        sa       = float(np.nanmean(sa_arr))
    else:
        qed, logp, sa = float('nan'), float('nan'), float('nan')

    return {
        'validity':       round(validity, 4),
        'clean_validity': round(clean_validity, 4),
        'uniqueness':     round(uniqueness, 4),
        'novelty':        round(novelty, 4),
        'QED':            round(qed, 4),
        'logP':           round(logp, 4),
        'SA':             round(sa, 4),
    }


def save_bar_chart(metrics, epoch, output_path):
    """Save a bar chart of all metrics for a single model."""
    keys  = list(metrics.keys())
    vals  = [metrics[k] for k in keys]
    notes = ['↑ better'] * 6 + ['↓ better']   # SA is inverted

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#E08462']
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(keys)), vals, color=colors, width=0.6, edgecolor='white')

    for bar, v, note in zip(bars, vals, notes):
        label = f'{v:.3f}\n{note}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label, ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha='right')
    ax.set_title(f'QuMolGAN Evaluation — Epoch {epoch}  (n=5000)', fontsize=13)
    ax.set_ylabel('Score')

    ymax = max(v for v in vals if not np.isnan(v))
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Bar chart saved → {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate a QuMolGAN checkpoint.')
    parser.add_argument('--model_dir', required=True,
                        help='Path to the model_dir containing *.ckpt files')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Epoch number to evaluate (e.g. 30)')
    parser.add_argument('--n_generate', type=int, default=5000,
                        help='Number of molecules to generate (default: 5000)')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH,
                        help='Override dataset path')
    parser.add_argument('--analysis_dir', type=str, default=ANALYSIS_DIR,
                        help='Override output directory')
    args = parser.parse_args()

    os.makedirs(args.analysis_dir, exist_ok=True)
    device = torch.device('cpu')

    # ---- Dataset ----
    print(f'Loading dataset: {args.dataset}')
    data = SparseMolecularDataset()
    data.load(args.dataset)
    print(f'  vertexes={data.vertexes}, '
          f'bond_types={data.bond_num_types}, '
          f'atom_types={data.atom_num_types}, '
          f'train_mols={data.train_count}')

    # ---- Generator ----
    print(f'Building Generator (z_dim={Z_DIM}, g_conv_dim={G_CONV_DIM})...')
    G = Generator(G_CONV_DIM, Z_DIM,
                  data.vertexes, data.bond_num_types, data.atom_num_types, DROPOUT)
    G.to(device)

    ckpt_path = os.path.join(args.model_dir, f'{args.epoch}-G.ckpt')
    if not os.path.exists(ckpt_path):
        sys.exit(f'ERROR: checkpoint not found: {ckpt_path}')
    print(f'Loading checkpoint: {ckpt_path}')
    G.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ---- Quantum circuit ----
    print(f'Building quantum circuit (qubits={QUBITS}, layers={LAYERS})...')
    gen_circuit = build_gen_circuit(QUBITS, LAYERS)

    weights_file = os.path.join(args.model_dir, 'molgan_red_weights.csv')
    if os.path.exists(weights_file):
        print(f'Loading gen_weights from {weights_file} (row epoch={args.epoch})...')
        gen_weights = load_gen_weights(args.model_dir, args.epoch)
        print(f'  gen_weights shape: {gen_weights.shape}')
    else:
        n_w = LAYERS * (QUBITS * 2 - 1)
        print(f'WARNING: molgan_red_weights.csv not found — using random {n_w}-param weights.')
        gen_weights = torch.tensor(
            list(np.random.rand(n_w) * 2 * np.pi - np.pi), requires_grad=False)

    # ---- Generate ----
    mols = generate_molecules(G, gen_circuit, gen_weights, data,
                              args.n_generate, BATCH_SIZE)
    n_valid = sum(1 for m in mols if MolecularMetrics.valid_lambda(m))
    print(f'Generated {len(mols)} molecules, {n_valid} valid by RDKit.')

    # ---- Metrics ----
    print('Computing metrics...')
    metrics = compute_metrics(mols, data)

    print('\n===== QuMolGAN Evaluation Results =====')
    print(f'  Epoch          : {args.epoch}')
    print(f'  N generated    : {len(mols)}')
    for k, v in metrics.items():
        note = '  (↓ better)' if k == 'SA' else ''
        print(f'  {k:<16}: {v:.4f}{note}')
    print('========================================\n')

    # ---- Save CSV ----
    csv_path = os.path.join(args.analysis_dir, 'quantum_results.csv')
    pd.DataFrame([{'epoch': args.epoch, 'n_generated': len(mols), **metrics}]).to_csv(
        csv_path, index=False)
    print(f'Metrics CSV saved → {csv_path}')

    # ---- Save bar chart ----
    png_path = os.path.join(args.analysis_dir, 'quantum_results.png')
    save_bar_chart(metrics, args.epoch, png_path)

    # ---- Save valid SMILES ----
    smiles_path = os.path.join(args.analysis_dir, 'generated_smiles.txt')
    valid_smiles = [Chem.MolToSmiles(m)
                    for m in mols if MolecularMetrics.valid_lambda(m)]
    with open(smiles_path, 'w') as f:
        f.write('\n'.join(valid_smiles))
    print(f'Valid SMILES ({len(valid_smiles)}) saved → {smiles_path}')


if __name__ == '__main__':
    main()
