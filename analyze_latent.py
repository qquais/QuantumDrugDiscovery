#!/usr/bin/env python
"""Latent-space analysis of the VQC noise generator and its classical controls.

    python analyze_latent.py --out_dir results/latent_analysis

The v1 analysis asserted that "entangled quantum states explore better latent
regions" without measuring it. This script measures it, and needs no trained
model: the geometry of the latent distribution is a property of the circuit,
not of the GAN.

For every latent source it reports
  * per-dimension mean/std and the correlation matrix,
  * the covariance eigenvalue spectrum, its participation ratio and the
    effective rank at 99% of variance -- i.e. how many dimensions the latent
    actually uses,
  * the energy distance to a standard Gaussian and to a bounded uniform, so
    "not Gaussian" is quantified rather than assumed,
  * the volume of [-1,1]^d the samples actually cover (occupied cells of a
    coarse grid),
and for quantum sources additionally
  * the mean single-qubit von Neumann entropy of the circuit state, averaged
    over the encoding distribution -- a direct measure of how much
    entanglement the circuit generates, which the entangling/non-entangling
    ablation can then be tested against.

The headline structural fact this exposes: the Kao circuit draws only two
random scalars (z1, z2) per sample and encodes them on every wire, so its
n_qubits outputs lie on a 2-dimensional manifold no matter how many qubits
are used. Latent capacity is bounded by the encoding, not by the qubit count.
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qmolgan.latent import make_latent, latent_statistics


def energy_distance(x, y, max_n=1500, seed=0):
    """Szekely's energy distance between two samples (0 iff same distribution)."""
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]

    def mean_dist(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return float(d.mean())

    return float(2 * mean_dist(x, y) - mean_dist(x, x) - mean_dist(y, y))


def occupancy(z, bins=8, box=(-1.0, 1.0)):
    """Fraction of a coarse grid over the latent box that samples reach.

    A latent whose samples trace a curve occupies O(bins) cells out of
    bins^d; a full-rank latent occupies a constant fraction of them. This is
    the crudest possible 'does it explore the space' statistic, and precisely
    because it is crude it is hard to argue with.
    """
    z = np.clip(z, box[0], box[1] - 1e-9)
    idx = ((z - box[0]) / (box[1] - box[0]) * bins).astype(int)
    cells = {tuple(row) for row in idx}
    return len(cells) / float(bins ** z.shape[1]), len(cells)


def circuit_entanglement(qubits, layers, entangle, n=64, seed=0):
    """Mean single-qubit von Neumann entropy of the circuit state.

    Averaged over the encoding distribution (z1, z2 ~ U(-1,1)) at randomly
    initialised weights. 0 means a product state (no entanglement); values up
    to ln 2 indicate a maximally mixed single-qubit reduced state.
    """
    try:
        import pennylane as qml
    except Exception:
        return {'mean_vn_entropy': float('nan'), 'available': False}

    rng = np.random.default_rng(seed)
    w = rng.random(layers * (2 * qubits - 1)) * 2 * np.pi - np.pi
    dev = qml.device('default.qubit', wires=qubits)

    @qml.qnode(dev)
    def entropy_circuit(z1, z2, w, wire):
        for i in range(qubits):
            qml.RY(np.arcsin(z1), wires=i)
            qml.RZ(np.arcsin(z2), wires=i)
        for _ in range(layers):
            for i in range(qubits):
                qml.RY(w[i], wires=i)
            if entangle:
                for i in range(qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(w[i + qubits], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
        return qml.vn_entropy(wires=[wire])

    vals = []
    for _ in range(n):
        z1, z2 = rng.uniform(-1, 1), rng.uniform(-1, 1)
        vals.extend(float(entropy_circuit(z1, z2, w, wire)) for wire in range(qubits))
    vals = np.asarray(vals, dtype=float)
    return {'mean_vn_entropy': float(np.nanmean(vals)),
            'max_vn_entropy': float(np.nanmax(vals)),
            'ln2_reference': float(np.log(2)), 'available': True}


def analyse(kind, dim, qubits, layers, n, seed):
    sampler = make_latent(kind, dim=dim, qubits=qubits, layers=layers, seed=seed)
    stats = latent_statistics(sampler, n=n, seed=seed)
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        z = sampler.sample(n).double().cpu().numpy()

    rng = np.random.default_rng(seed)
    gauss = rng.standard_normal((n, dim))
    unif = rng.uniform(-1, 1, size=(n, dim))
    frac, cells = occupancy(z)

    row = {
        'latent': kind, 'dim': dim,
        'qubits': qubits if kind.startswith('vqc') else None,
        'layers': layers if kind.startswith('vqc') else None,
        'participation_ratio': stats['participation_ratio'],
        'effective_rank_99pct': stats['effective_rank_99pct'],
        'log_det_cov': stats['log_det_cov'],
        'energy_dist_to_gaussian': energy_distance(z, gauss, seed=seed),
        'energy_dist_to_uniform': energy_distance(z, unif, seed=seed),
        'occupancy_frac': frac, 'occupied_cells': cells,
        'mean_abs_offdiag_corr': float(np.mean(np.abs(
            np.asarray(stats['corr'])[~np.eye(dim, dtype=bool)]))) if dim > 1 else 0.0,
        'std_mean': float(np.mean(stats['std'])),
    }
    if kind.startswith('vqc'):
        row.update({f'ent_{k}': v for k, v in
                    circuit_entanglement(qubits, layers, kind == 'vqc', seed=seed).items()})
    return row, z, stats


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out_dir', default='results/latent_analysis')
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--sweep_n', type=int, default=1024,
                   help='samples per circuit-shape sweep point (kept smaller: an '
                        '8-qubit simulation costs far more per sample)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--qubit_sweep', type=int, nargs='*', default=[2, 4, 6, 8])
    p.add_argument('--layer_sweep', type=int, nargs='*', default=[1, 2, 3, 6])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd
    rows, samples, spectra = [], {}, {}

    # Head-to-head at the paper's configuration (4 dims / 4 qubits, 3 layers).
    for kind in ('gaussian', 'uniform', 'rank2', 'trig', 'vqc', 'vqc_noent'):
        row, z, stats = analyse(kind, 4, 4, 3, args.n, args.seed)
        rows.append(row)
        samples[kind] = z
        spectra[kind] = np.asarray(stats['cov_eigenvalues'])
        print(f'{kind:10s} PR={row["participation_ratio"]:.3f} '
              f'eff_rank={row["effective_rank_99pct"]} '
              f'occupancy={row["occupancy_frac"]:.4f} '
              f'E-dist(N)={row["energy_dist_to_gaussian"]:.3f}'
              + (f' vN_entropy={row.get("ent_mean_vn_entropy", float("nan")):.4f}'
                 if kind.startswith('vqc') else ''))

    # Circuit-shape ablation: does adding qubits or depth buy latent capacity?
    print('\nCircuit shape sweep (entangling):')
    for q in args.qubit_sweep:
        for L in args.layer_sweep:
            row, _, _ = analyse('vqc', q, q, L, args.sweep_n, args.seed)
            rows.append(row)
            print(f'  q={q} L={L}: PR={row["participation_ratio"]:.3f} '
                  f'eff_rank={row["effective_rank_99pct"]}/{q} '
                  f'vN_entropy={row.get("ent_mean_vn_entropy", float("nan")):.4f}')

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'latent_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nWrote {csv_path}')

    # ---- Figures --------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, kind in zip(axes.ravel(), samples):
        z = samples[kind]
        ax.scatter(z[:800, 0], z[:800, 1], s=4, alpha=0.35, edgecolors='none')
        pr = df[df.latent == kind].iloc[0]['participation_ratio']
        ax.set_title(f'{kind}  (participation ratio {pr:.2f})', fontsize=10)
        ax.set_xlabel('$z_0$'); ax.set_ylabel('$z_1$')
        ax.grid(alpha=0.25)
    fig.suptitle('Latent samples, first two dimensions (n=800 shown)')
    fig.tight_layout()
    scatter_path = os.path.join(args.out_dir, 'latent_scatter.png')
    fig.savefig(scatter_path, dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in samples:
        eig = np.clip(spectra[kind], 1e-12, None)
        ax.semilogy(range(1, len(eig) + 1), eig / eig.sum(), 'o-', label=kind)
    ax.set_xlabel('component'); ax.set_ylabel('explained variance (log)')
    ax.set_title('Latent covariance spectrum: the VQC uses ~1 of its 4 dimensions')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    spec_path = os.path.join(args.out_dir, 'latent_spectrum.png')
    fig.savefig(spec_path, dpi=160); plt.close(fig)

    with open(os.path.join(args.out_dir, 'latent_analysis.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    print(f'Wrote {scatter_path}\nWrote {spec_path}')


if __name__ == '__main__':
    main()
