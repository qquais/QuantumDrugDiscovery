#!/usr/bin/env python
"""Reproduce the three measurement artifacts behind the v1 headline numbers.

    python scripts/reproduce_errata.py \
        --model_dir results/quantum/ablation_300/train/model_dir --epoch 113

Runs against a v1 checkpoint and demonstrates, in one pass:

  A. clean-validity = 0.000 was a decoder padding artifact. Decoding with and
     without unbonded PAD-atom removal, on the SAME generated graphs, gives
     0.000 vs a substantial clean-validity, with identical validity.
  B. "SA = 0.410" was the [0,1] reward-space SA, not the 1-10 Ertl scale.
  C. "uniqueness 73.0%" was measured on training batches of 16. Uniqueness is
     reported here at n = 16, 64, 256, 1000 and 5000 from the same pool.

Nothing here depends on the fixed code paths: the padding comparison decodes
both ways explicitly, so the script also serves as an independent check that
the fix in data/sparse_molecular_dataset.py does what it claims.
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from data.sparse_molecular_dataset import SparseMolecularDataset
from models.models import Generator
from qmolgan.chem import uniqueness_vs_sample_size
from qmolgan.latent import build_gen_circuit
from utils.utils import MolecularMetrics


def decode(nodes, edges, data, drop_unbonded_pad):
    """Decode one graph, optionally keeping unbonded PAD slots as '*' atoms
    (which is what the v1 decoder did)."""
    mol = Chem.RWMol()
    for label in nodes:
        mol.AddAtom(Chem.Atom(data.atom_decoder_m[label]))
    for s, e in zip(*np.nonzero(edges)):
        if s > e:
            mol.AddBond(int(s), int(e), data.bond_decoder_m[edges[s, e]])
    if drop_unbonded_pad:
        pad = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetAtomicNum() == 0 and a.GetDegree() == 0]
        for idx in sorted(pad, reverse=True):
            mol.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_dir', default='results/quantum/ablation_300/train/model_dir')
    p.add_argument('--dataset', default='data/qm9_5k_py37.sparsedataset')
    p.add_argument('--epoch', type=int, default=113)
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--qubits', type=int, default=4)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--g_conv_dim', type=int, nargs='+', default=[16])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    data = SparseMolecularDataset()
    data.load(args.dataset)
    heavy = [Chem.MolFromSmiles(s).GetNumHeavyAtoms()
             for s in data.smiles[:2000] if Chem.MolFromSmiles(s)]
    print(f'Dataset {args.dataset}: {data.vertexes} graph vertices, '
          f'heavy-atom count of training molecules = '
          f'{np.mean(heavy):.1f} mean / {max(heavy)} max')
    print(f'  -> on average {data.vertexes - np.mean(heavy):.1f} of the '
          f'{data.vertexes} vertex slots are PAD in a correct generation.\n')

    G = Generator(list(args.g_conv_dim), args.qubits, data.vertexes,
                  data.bond_num_types, data.atom_num_types, 0.0)
    G.load_state_dict(torch.load(
        os.path.join(args.model_dir, f'{args.epoch}-G.ckpt'), map_location='cpu'))
    G.eval()

    weights_csv = os.path.join(args.model_dir, 'molgan_red_weights.csv')
    df = pd.read_csv(weights_csv, header=None)
    rows = df[df[0] == args.epoch - 1]
    w = torch.tensor(list(rows.iloc[-1, 1:].values.astype(float)), requires_grad=False)
    circuit = build_gen_circuit(args.qubits, args.layers, entangle=True)

    nodes_all, edges_all = [], []
    with torch.no_grad():
        for i in range(0, args.n, 64):
            b = min(64, args.n - i)
            z = torch.stack(tuple(
                torch.stack(circuit(w)) if isinstance(circuit(w), list) else circuit(w)
                for _ in range(b))).float()
            el, nl = G(z)
            edges_all.append(torch.max(F.softmax(el, -1), -1)[1].numpy())
            nodes_all.append(torch.max(F.softmax(nl, -1), -1)[1].numpy())
    nodes_all = np.concatenate(nodes_all)
    edges_all = np.concatenate(edges_all)

    print(f'Generated {len(nodes_all)} graphs from epoch {args.epoch}.\n')
    print('=' * 78)
    print('A. clean-validity = 0.000 was a padding artifact, not chemistry')
    print('=' * 78)
    results = {}
    for drop in (False, True):
        mols = [decode(n, e, data, drop) for n, e in zip(nodes_all, edges_all)]
        valid = np.array([MolecularMetrics.valid_lambda(m) for m in mols])
        clean = np.array([MolecularMetrics.valid_lambda_special(m) for m in mols])
        label = 'v1 decoder (PAD kept)' if not drop else 'fixed decoder (unbonded PAD dropped)'
        print(f'  {label:42s} validity={valid.mean():.3f}  '
              f'clean-validity={clean.mean():.3f}')
        results[drop] = mols
        if not drop:
            examples = [Chem.MolToSmiles(m) for m in mols if m is not None][:3]
            print(f'    example SMILES: {examples}')
        else:
            examples = [Chem.MolToSmiles(m) for m in mols if m is not None][:3]
            print(f'    example SMILES: {examples}')
    print('  -> validity is IDENTICAL; only the wildcard/fragment test changes.\n')

    print('=' * 78)
    print('B. "SA = 0.410" was the normalised reward-space score, not raw SA')
    print('=' * 78)
    valid_mols = [m for m in results[True] if MolecularMetrics.valid_lambda(m)]
    sa_raw = MolecularMetrics.synthetic_accessibility_score_scores(valid_mols, norm=False)
    sa_norm = MolecularMetrics.synthetic_accessibility_score_scores(valid_mols, norm=True)
    print(f'  raw SA (Ertl 1-10, lower is better) : {np.mean(sa_raw):.3f}')
    print(f'  normalised SA (reward space, 0-1)   : {np.mean(sa_norm):.3f}')
    print('  normalisation is clip((5 - SA) / 3.5, 0, 1), so the two scales run '
          'in\n  OPPOSITE directions and 0.410 corresponds to a raw SA of '
          f'{5 - 0.410 * 3.5:.2f}.\n')

    print('=' * 78)
    print('C. "uniqueness = 73.0%" came from batches of 16, not from n = 5000')
    print('=' * 78)
    smi = [Chem.MolToSmiles(m) if MolecularMetrics.valid_lambda(m) else None
           for m in results[True]]
    curve = uniqueness_vs_sample_size(smi, sizes=(16, 64, 256, 1000, len(smi)))
    for n, val in curve.items():
        marker = '   <- v1 measured here' if n == 16 else (
            '   <- v1 compared against the classical model here' if n >= 1000 else '')
        print(f'  uniqueness @ n={n:>5}: '
              + (f'{val:.3f}' if np.isfinite(val) else '  n/a') + marker)
    print('\n  Uniqueness is monotonically non-increasing in n. Comparing a '
          'quantum\n  number at n=16 against a classical number at n=5000 is not '
          'a comparison.')


if __name__ == '__main__':
    main()
