"""Reconstruct a trained run from disk and sample molecules from it.

A run directory is self-describing: `config.json` records the dataset, the
generator architecture, the latent source and the training seed, so offline
evaluation never has to be told (or guess) what the checkpoint was.
Mismatched hand-typed constants in the old find_best_epoch.py are how the
"quantum" epoch-30 row in Table II ended up being generated with a classical
z_dim=8 / g_conv_dim=[128,256,512] generator.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data.sparse_molecular_dataset import SparseMolecularDataset
from models.models import Generator
from qmolgan import chem
from qmolgan.latent import make_latent, n_circuit_weights

CONFIG_NAME = 'config.json'


# ---------------------------------------------------------------------------
# Run directory conventions
# ---------------------------------------------------------------------------

def resolve_run_dir(path):
    """Accept a run root, its train/ dir, or its model_dir; return all three."""
    path = os.path.abspath(path)
    if os.path.basename(path) == 'model_dir':
        train_dir = os.path.dirname(path)
        run_dir = os.path.dirname(train_dir)
    elif os.path.basename(path) == 'train':
        train_dir, run_dir = path, os.path.dirname(path)
    else:
        run_dir = path
        train_dir = os.path.join(run_dir, 'train')
    model_dir = os.path.join(train_dir, 'model_dir')
    return run_dir, train_dir, model_dir


def write_run_config(run_dir, config_dict):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, CONFIG_NAME), 'w') as f:
        json.dump(config_dict, f, indent=2, sort_keys=True, default=str)


def read_run_config(run_dir, overrides=None):
    """Load config.json, applying explicit overrides for legacy runs that
    predate it. Raises if neither exists, rather than guessing."""
    path = os.path.join(run_dir, CONFIG_NAME)
    cfg = {}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    required = ('dataset', 'latent', 'z_dim', 'g_conv_dim')
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(
            f'{path} is missing {missing}. This run predates config.json; pass the '
            'values explicitly (--dataset/--latent/--z_dim/--g_conv_dim) so the '
            'generator architecture is never guessed.')
    return cfg


def available_epochs(model_dir, max_epoch=None):
    """Sorted epochs that have a generator checkpoint on disk."""
    if not os.path.isdir(model_dir):
        return []
    epochs = []
    for name in os.listdir(model_dir):
        if name.endswith('-G.ckpt'):
            try:
                epochs.append(int(name.split('-')[0]))
            except ValueError:
                continue
    epochs.sort()
    if max_epoch is not None:
        epochs = [e for e in epochs if e <= max_epoch]
    return epochs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(path):
    data = SparseMolecularDataset()
    data.load(path)
    return data


def training_smiles(data):
    """Canonical SMILES of the TRAINING split only.

    Novelty must be measured against what the model actually saw. Using
    `data.smiles` (train + validation + test) makes held-out molecules count
    as 'novel' and inflates the metric.
    """
    from rdkit import Chem
    idx = getattr(data, 'train_idx', None)
    smiles = data.smiles if idx is None else np.asarray(data.smiles)[idx]
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToSmiles(m) if m is not None else s)
    return out


def build_generator(cfg, data, device='cpu'):
    g_conv_dim = cfg['g_conv_dim']
    if isinstance(g_conv_dim, str):
        g_conv_dim = json.loads(g_conv_dim)
    G = Generator(list(g_conv_dim), int(cfg['z_dim']), data.vertexes,
                  data.bond_num_types, data.atom_num_types, float(cfg.get('dropout', 0.0)))
    return G.to(device)


def build_latent(cfg, seed=0):
    return make_latent(cfg['latent'], dim=int(cfg['z_dim']),
                       qubits=cfg.get('qubits'), layers=int(cfg.get('layers', 3)),
                       seed=seed, n_freq=int(cfg.get('n_freq', 3)))


def load_latent_state(latent, model_dir, epoch):
    """Restore a trainable latent source for a given epoch.

    Prefers `{epoch}-Z.ckpt` (a real state_dict). Falls back to the legacy
    `molgan_red_weights.csv`, matching rows by the epoch label in column 0
    rather than by file position — resumed runs append duplicate rows, and
    positional indexing silently loads the wrong epoch's circuit once any
    duplicate precedes it.
    """
    z_path = os.path.join(model_dir, f'{epoch}-Z.ckpt')
    if os.path.exists(z_path):
        latent.load_state_dict(torch.load(z_path, map_location='cpu'))
        return 'state_dict'

    csv_path = os.path.join(model_dir, 'molgan_red_weights.csv')
    if os.path.exists(csv_path) and hasattr(latent, 'weights'):
        import pandas as pd
        df = pd.read_csv(csv_path, header=None)
        rows = df[df[0] == epoch - 1]
        if len(rows) == 0:
            raise ValueError(f'no circuit weights for epoch {epoch} in {csv_path}')
        vals = rows.iloc[-1, 1:].values.astype(float)
        expected = latent.weights.numel()
        if vals.size != expected:
            raise ValueError(f'{csv_path} epoch {epoch} has {vals.size} weights, '
                             f'latent expects {expected}')
        with torch.no_grad():
            latent.weights.copy_(torch.tensor(vals, dtype=latent.weights.dtype))
        return 'legacy_csv'

    if getattr(latent, 'trainable', False):
        raise FileNotFoundError(
            f'trainable latent {latent.kind!r} has no saved state for epoch {epoch} '
            f'in {model_dir}; refusing to evaluate with random circuit weights')
    return 'stateless'


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def postprocess(logits, method='softmax', temperature=1.0):
    if method == 'hard_gumbel':
        return F.gumbel_softmax(logits.view(-1, logits.size(-1)) / temperature,
                                hard=True).view(logits.size())
    if method == 'soft_gumbel':
        return F.gumbel_softmax(logits.view(-1, logits.size(-1)) / temperature,
                                hard=False).view(logits.size())
    return F.softmax(logits / temperature, -1)


def sample_molecules(G, latent, data, n, batch_size=256, seed=0,
                     post_method='softmax', device='cpu'):
    """Generate exactly ``n`` molecules with a fixed, reproducible noise stream.

    The seed controls every stochastic element (torch global RNG drives both
    the classical samplers and the VQC's z1/z2 draws), so two evaluations of
    the same checkpoint with the same seed return identical molecules — a
    prerequisite for the val/test noise split in qmolgan.protocol.
    """
    G.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    mols = []
    with torch.no_grad():
        while len(mols) < n:
            cur = min(batch_size, n - len(mols))
            z = latent.sample(cur, device=device).float()
            edges_logits, nodes_logits = G(z)
            edges_hat = postprocess(edges_logits, post_method)
            nodes_hat = postprocess(nodes_logits, post_method)
            edges_hard = torch.max(edges_hat, -1)[1].cpu().numpy()
            nodes_hard = torch.max(nodes_hat, -1)[1].cpu().numpy()
            mols.extend(chem.decode_batch(nodes_hard, edges_hard, data))
    return mols[:n]


def load_run(run_dir, epoch, overrides=None, device='cpu', data=None):
    """One call: run dir + epoch -> (G, latent, data, cfg) ready to sample."""
    run_dir, _, model_dir = resolve_run_dir(run_dir)
    cfg = read_run_config(run_dir, overrides)
    if data is None:
        data = load_dataset(cfg['dataset'])
    G = build_generator(cfg, data, device=device)
    g_path = os.path.join(model_dir, f'{epoch}-G.ckpt')
    if not os.path.exists(g_path):
        raise FileNotFoundError(g_path)
    G.load_state_dict(torch.load(g_path, map_location=device))
    latent = build_latent(cfg, seed=int(cfg.get('seed', 0)))
    source = load_latent_state(latent, model_dir, epoch)
    cfg = {**cfg, 'latent_state_source': source, 'model_dir': model_dir, 'epoch': epoch}
    return G, latent, data, cfg
