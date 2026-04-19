"""
Generate and visualise the best molecules from a saved generator checkpoint.

Usage (classical 300-epoch baseline):
    python generate_best_mols.py \
        --run_dir results/classical/GAN/20250126_234326 \
        --epoch 300 \
        --top_n 12 \
        --out molecules_best.png

Colab (quantum run, epoch 30):
    python generate_best_mols.py \
        --run_dir /content/drive/MyDrive/Quais/results/QDISC_WGAN_0405 \
        --epoch 30 \
        --top_n 12 \
        --out molecules_quantum_best.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw

from data.sparse_molecular_dataset import SparseMolecularDataset
from models.models import Generator
from utils.args import get_GAN_config

RDLogger.logger().setLevel(RDLogger.CRITICAL)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Path to run root, train dir, or model_dir")
    p.add_argument("--epoch", type=int, required=True,
                   help="Checkpoint epoch")
    p.add_argument("--num_samples", type=int, default=2000,
                   help="Molecules to sample before filtering (default 2000)")
    p.add_argument("--top_n", type=int, default=12,
                   help="Number of best molecules to draw")
    p.add_argument("--rank_by", type=str, default="qed",
                   choices=["qed", "sa", "logp"],
                   help="Property to rank by (default: qed, higher=better)")
    p.add_argument("--mol_data_dir", type=str, default=None)
    p.add_argument("--post_method", type=str, default="softmax",
                   choices=["softmax", "soft_gumbel", "hard_gumbel"])
    p.add_argument("--out", type=str, default="molecules_best.png",
                   help="Output image path")
    p.add_argument("--mols_per_row", type=int, default=4)
    p.add_argument("--img_size", type=int, default=300,
                   help="Sub-image size in pixels")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_model_dir(run_dir):
    run_dir = os.path.normpath(run_dir)
    for candidate in [
        run_dir,
        os.path.join(run_dir, "train", "model_dir"),
        os.path.join(run_dir, "model_dir"),
    ]:
        if os.path.isdir(candidate) and any(
            f.endswith("-G.ckpt") for f in os.listdir(candidate)
        ):
            return candidate
    # Try one level deeper (timestamped sub-dirs)
    for sub in sorted(os.listdir(run_dir), reverse=True):
        for suffix in ["train/model_dir", "model_dir"]:
            candidate = os.path.join(run_dir, sub, suffix)
            if os.path.isdir(candidate) and any(
                f.endswith("-G.ckpt") for f in os.listdir(candidate)
            ):
                return candidate
    raise FileNotFoundError(f"Cannot find model_dir under: {run_dir}")


def _infer_generator_dims(state_dict):
    import re
    layers = []
    for key, val in state_dict.items():
        m = re.match(r"multi_dense_layers\.(?:linear_layers\.)?(\d+)\.weight$", key)
        if m:
            layers.append((int(m.group(1)), val.shape))
    if not layers:
        raise RuntimeError("Could not infer generator dims from checkpoint keys.")
    layers.sort()
    z_dim = layers[0][1][1]
    g_conv_dim = [s[0] for _, s in layers]
    return z_dim, g_conv_dim


def _infer_dataset(state_dict, requested=None):
    if requested:
        return requested
    n = state_dict["nodes_layer.weight"].shape[0]
    if n == 54:
        return "data/gdb9_9nodes.sparsedataset"
    if n == 45:
        return "data/qm9_5k.sparsedataset"
    raise RuntimeError(f"Unexpected nodes_layer output size: {n}")


def _postprocess(inputs, method):
    def listify(x):
        return x if isinstance(x, (list, tuple)) else [x]
    def delistify(x):
        return x if len(x) > 1 else x[0]
    if method == "soft_gumbel":
        out = [F.gumbel_softmax(e.contiguous().view(-1, e.size(-1)), hard=False).view(e.size())
               for e in listify(inputs)]
    elif method == "hard_gumbel":
        out = [F.gumbel_softmax(e.contiguous().view(-1, e.size(-1)), hard=True).view(e.size())
               for e in listify(inputs)]
    else:
        out = [F.softmax(e, -1) for e in listify(inputs)]
    return [delistify(x) for x in out]


def _qed(mol):
    try:
        return Descriptors.qed(mol)
    except Exception:
        return 0.0


def _sa(mol):
    try:
        from rdkit.Chem import RDConfig
        import sys, os
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        return 10.0  # worst case


def _logp(mol):
    try:
        return Descriptors.MolLogP(mol)
    except Exception:
        return -3.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = _parse()

    model_dir = _resolve_model_dir(args.run_dir)
    ckpt = os.path.join(model_dir, f"{args.epoch}-G.ckpt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"Loading checkpoint: {ckpt}")

    state_dict = torch.load(ckpt, map_location="cpu")
    z_dim, g_conv_dim = _infer_generator_dims(state_dict)
    mol_data_dir = _infer_dataset(state_dict, args.mol_data_dir)
    print(f"z_dim={z_dim}  g_conv_dim={g_conv_dim}  dataset={mol_data_dir}")

    # Silence get_GAN_config argparse
    _orig_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    config = get_GAN_config()
    sys.argv = _orig_argv
    config.mol_data_dir = mol_data_dir

    data = SparseMolecularDataset()
    data.load(config.mol_data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(g_conv_dim, z_dim, data.vertexes,
                  data.bond_num_types, data.atom_num_types, config.dropout).to(device)
    G.load_state_dict(state_dict)
    G.eval()

    # ------------------------------------------------------------------ #
    # Sample molecules
    # ------------------------------------------------------------------ #
    print(f"Sampling {args.num_samples} molecules...")
    z = torch.randn(args.num_samples, z_dim).to(device)
    with torch.no_grad():
        e_logits, n_logits = G(z)
        e_hard, n_hard = _postprocess((e_logits, n_logits), args.post_method)
        e_hard = torch.max(e_hard, -1)[1]
        n_hard = torch.max(n_hard, -1)[1]
        mols = [
            data.matrices2mol(n_.cpu().numpy(), e_.cpu().numpy(), strict=True)
            for e_, n_ in zip(e_hard, n_hard)
        ]

    # ------------------------------------------------------------------ #
    # Filter: valid, no fragments, no wildcards
    # ------------------------------------------------------------------ #
    clean = []
    for mol in mols:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if smi and "." not in smi and "*" not in smi:
            clean.append((mol, smi))

    print(f"Clean valid molecules: {len(clean)} / {args.num_samples}")
    if not clean:
        print("No clean molecules found. Try increasing --num_samples or use epoch with non-zero validity.")
        return

    # ------------------------------------------------------------------ #
    # Deduplicate by canonical SMILES
    # ------------------------------------------------------------------ #
    seen = set()
    unique_clean = []
    for mol, smi in clean:
        canon = Chem.MolToSmiles(mol, canonical=True)
        if canon not in seen:
            seen.add(canon)
            unique_clean.append((mol, canon))
    print(f"Unique clean molecules: {len(unique_clean)} / {len(clean)}")
    clean = unique_clean

    # ------------------------------------------------------------------ #
    # Rank by chosen property
    # ------------------------------------------------------------------ #
    if args.rank_by == "qed":
        scored = [(mol, smi, _qed(mol)) for mol, smi in clean]
        scored.sort(key=lambda x: x[2], reverse=True)   # higher QED = better
        prop_label = "QED"
    elif args.rank_by == "sa":
        scored = [(mol, smi, _sa(mol)) for mol, smi in clean]
        scored.sort(key=lambda x: x[2], reverse=False)  # lower SA = better
        prop_label = "SA"
    else:  # logp
        scored = [(mol, smi, _logp(mol)) for mol, smi in clean]
        scored.sort(key=lambda x: abs(x[2] - 2.5))      # closest to Lipinski target
        prop_label = "logP"

    top = scored[: args.top_n]

    # ------------------------------------------------------------------ #
    # Print top SMILES
    # ------------------------------------------------------------------ #
    print(f"\nTop {len(top)} molecules ranked by {prop_label}:")
    for i, (mol, smi, score) in enumerate(top, 1):
        print(f"  {i:2d}. {prop_label}={score:.3f}  SMILES={smi}")

    # ------------------------------------------------------------------ #
    # Draw grid image
    # ------------------------------------------------------------------ #
    top_mols = [m for m, _, _ in top]
    legends = [f"{prop_label}={s:.3f}" for _, _, s in top]

    img = Draw.MolsToGridImage(
        top_mols,
        molsPerRow=args.mols_per_row,
        subImgSize=(args.img_size, args.img_size),
        legends=legends,
        returnPNG=False,
    )
    img.save(args.out)
    print(f"\nSaved grid image: {args.out}")


if __name__ == "__main__":
    main()
