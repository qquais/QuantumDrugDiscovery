import argparse
import logging
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit import RDLogger

from data.sparse_molecular_dataset import SparseMolecularDataset
from models.models import Generator
from utils.args import get_GAN_config
from utils.utils import MolecularMetrics


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MolGAN/QuantumMolGAN generator checkpoint."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to a run root, train dir, or model_dir.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Checkpoint epoch to evaluate.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of molecules to sample.",
    )
    parser.add_argument(
        "--mol_data_dir",
        type=str,
        default=None,
        help="Dataset path. If omitted, infer from checkpoint/logs.",
    )
    parser.add_argument(
        "--post_method",
        type=str,
        default=None,
        choices=["softmax", "soft_gumbel", "hard_gumbel"],
        help="Generator postprocessing method. If omitted, try run logs then default to softmax.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/quantum/metrics",
        help="Directory for evaluation summaries.",
    )
    return parser.parse_args()


def _get_default_config():
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return get_GAN_config()
    finally:
        sys.argv = original_argv


def _resolve_run_paths(run_dir):
    run_dir = os.path.normpath(run_dir)

    if os.path.basename(run_dir) == "model_dir":
        train_dir = os.path.dirname(run_dir)
        model_dir = run_dir
    elif os.path.basename(run_dir) == "train":
        train_dir = run_dir
        model_dir = os.path.join(train_dir, "model_dir")
    elif os.path.isdir(os.path.join(run_dir, "train", "model_dir")):
        train_dir = os.path.join(run_dir, "train")
        model_dir = os.path.join(train_dir, "model_dir")
    elif os.path.isdir(os.path.join(run_dir, "model_dir")):
        train_dir = run_dir
        model_dir = os.path.join(train_dir, "model_dir")
    else:
        raise FileNotFoundError(f"Could not resolve model_dir from: {run_dir}")

    log_dir = os.path.join(train_dir, "log_dir")
    img_dir = os.path.join(train_dir, "img_dir")
    return train_dir, model_dir, log_dir, img_dir


def _find_logger_file(log_dir):
    if not os.path.isdir(log_dir):
        return None
    logger_files = sorted(
        [
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.endswith(".log")
        ]
    )
    return logger_files[0] if logger_files else None


def _extract_config_value(logger_file, key):
    if not logger_file:
        return None

    pattern = re.compile(rf"{re.escape(key)}=([^,)\n]+)")
    with open(logger_file, "r", errors="ignore") as fh:
        for line in fh:
            if "Namespace(" not in line:
                continue
            match = pattern.search(line)
            if match:
                raw = match.group(1).strip()
                if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
                    raw = raw[1:-1]
                return raw
    return None


def _infer_generator_dims(state_dict):
    pattern = re.compile(r"multi_dense_layers\.linear_layers\.(\d+)\.weight$")
    layers = []
    for key, value in state_dict.items():
        match = pattern.match(key)
        if match:
            layers.append((int(match.group(1)), tuple(value.shape)))

    if not layers:
        pattern = re.compile(r"multi_dense_layers\.(\d+)\.weight$")
        for key, value in state_dict.items():
            match = pattern.match(key)
            if match:
                layers.append((int(match.group(1)), tuple(value.shape)))

    if not layers:
        raise RuntimeError("Could not infer generator dense layers from checkpoint.")

    layers.sort(key=lambda item: item[0])
    z_dim = layers[0][1][1]
    g_conv_dim = [shape[0] for _, shape in layers]
    return z_dim, g_conv_dim


def _infer_dataset_path(state_dict, requested_path=None):
    if requested_path:
        return requested_path

    nodes_shape = state_dict["nodes_layer.weight"].shape[0]
    if nodes_shape == 54:
        return "data/gdb9_9nodes.sparsedataset"
    if nodes_shape == 45:
        return "data/qm9_5k.sparsedataset"

    raise RuntimeError(
        f"Could not infer dataset from nodes_layer.weight shape: {nodes_shape}"
    )


def _stats(values):
    arr = np.array([v for v in values if v is not None and not np.isnan(v)])
    if arr.size == 0:
        return None, None, None
    return float(arr.mean()), float(arr.std()), float(arr.min())


def _postprocess(inputs, method, temperature=1.0):
    def listify(x):
        return x if isinstance(x, (list, tuple)) else [x]

    def delistify(x):
        return x if len(x) > 1 else x[0]

    if method == "soft_gumbel":
        softmax = [
            F.gumbel_softmax(
                logits.contiguous().view(-1, logits.size(-1)) / temperature,
                hard=False,
            ).view(logits.size())
            for logits in listify(inputs)
        ]
    elif method == "hard_gumbel":
        softmax = [
            F.gumbel_softmax(
                logits.contiguous().view(-1, logits.size(-1)) / temperature,
                hard=True,
            ).view(logits.size())
            for logits in listify(inputs)
        ]
    else:
        softmax = [F.softmax(logits / temperature, -1) for logits in listify(inputs)]

    return [delistify(x) for x in softmax]


def main():
    args = _parse_args()

    train_dir, model_dir, log_dir, img_dir = _resolve_run_paths(args.run_dir)
    ckpt_path = os.path.join(model_dir, f"{args.epoch}-G.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger_file = _find_logger_file(log_dir)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    z_dim, g_conv_dim = _infer_generator_dims(state_dict)
    mol_data_dir = _infer_dataset_path(state_dict, args.mol_data_dir)
    post_method = (
        args.post_method
        or _extract_config_value(logger_file, "post_method")
        or "softmax"
    )

    config = _get_default_config()
    config.mol_data_dir = mol_data_dir

    data = SparseMolecularDataset()
    data.load(config.mol_data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(
        g_conv_dim,
        z_dim,
        data.vertexes,
        data.bond_num_types,
        data.atom_num_types,
        config.dropout,
    ).to(device)
    generator.load_state_dict(state_dict)
    generator.eval()

    z_np = np.random.normal(0, 1, size=(args.num_samples, z_dim))
    z = torch.from_numpy(z_np).to(device).float()
    with torch.no_grad():
        edges_logits, nodes_logits = generator(z)
        edges_hard, nodes_hard = _postprocess(
            (edges_logits, nodes_logits),
            post_method,
        )
        edges_hard = torch.max(edges_hard, -1)[1]
        nodes_hard = torch.max(nodes_hard, -1)[1]
        mols = [
            data.matrices2mol(
                node_.data.cpu().numpy(),
                edge_.data.cpu().numpy(),
                strict=True,
            )
            for edge_, node_ in zip(edges_hard, nodes_hard)
        ]

    valid_fraction = MolecularMetrics.valid_total_score(mols)
    clean_valid_fraction = float(np.mean(MolecularMetrics.valid_scores(mols)))
    unique_fraction = MolecularMetrics.unique_total_score(mols)
    novelty_scores = MolecularMetrics.novel_scores(mols, data)
    novelty_fraction = float(np.mean(novelty_scores))

    qeds = MolecularMetrics.quantitative_estimation_druglikeness_scores(
        mols, norm=False
    )
    logps = MolecularMetrics.water_octanol_partition_coefficient_scores(
        mols, norm=False
    )
    sas = MolecularMetrics.synthetic_accessibility_score_scores(
        mols, norm=False
    )

    valid_smiles = []
    for mol in mols:
        if mol is None:
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if smiles:
            valid_smiles.append(smiles)

    dot_count = sum(1 for s in valid_smiles if "." in s)
    star_count = sum(1 for s in valid_smiles if "*" in s)

    qed_mean, qed_std, qed_min = _stats(qeds)
    logp_mean, logp_std, logp_min = _stats(logps)
    sa_mean, sa_std, sa_min = _stats(sas)

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(os.path.dirname(train_dir)) or os.path.basename(train_dir)
    out_path = os.path.join(
        args.output_dir,
        f"{run_name}_epoch{args.epoch}_n{args.num_samples}_metrics.txt",
    )

    summary = [
        f"Run: {train_dir}",
        f"Checkpoint: {ckpt_path}",
        f"Dataset: {mol_data_dir}",
        f"Samples: {args.num_samples}",
        f"Post method: {post_method}",
        f"Generator dims: z_dim={z_dim}, g_conv_dim={g_conv_dim}",
        "",
        f"Validity fraction:   {valid_fraction:.3f}",
        f"Clean validity frac: {clean_valid_fraction:.3f}",
        f"Uniqueness fraction: {unique_fraction:.3f}",
        f"Novelty fraction:    {novelty_fraction:.3f}",
        "",
        f"Valid smiles count:  {len(valid_smiles)}",
        f'Contains "." count:  {dot_count}',
        f'Contains "*" count:  {star_count}',
        "",
        f"QED  : mean={qed_mean:.3f}, std={qed_std:.3f}, min={qed_min:.3f}",
        f"logP : mean={logp_mean:.3f}, std={logp_std:.3f}, min={logp_min:.3f}",
        f"SA   : mean={sa_mean:.3f}, std={sa_std:.3f}, min={sa_min:.3f}",
    ]
    summary_text = "\n".join(summary)

    print(summary_text)
    with open(out_path, "w") as fh:
        fh.write(summary_text + "\n")
    print(f"\nSaved metrics summary to {out_path}")


if __name__ == "__main__":
    main()
