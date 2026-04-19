import os
import logging
import numpy as np
import torch

from rdkit import Chem

from utils.args import get_GAN_config
from solver import Solver
from utils.utils import MolecularMetrics


def main():
    # -------------------------------
    # 1. Build config using defaults
    # -------------------------------
    config = get_GAN_config()

    # IMPORTANT: override generator dims to match the January training run.
    # The checkpoint 300-G.ckpt has edges_layer.weight with in_features=128,
    # so the Generator used conv_dims = [128] (one dense layer 8 -> 128).
    config.g_conv_dim = [128]

    # Use the same dataset as training
    config.mol_data_dir = r"data/gdb9_9nodes.sparsedataset"

    # We are ONLY evaluating, never training
    config.mode = "test"

    # Path to your training run (same as in rdkit_eval_epoch300.py)
    config.saving_dir = r"results/GAN/20250126_234326/train"
    config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
    config.log_dir_path = os.path.join(config.saving_dir, "log_dir")
    config.img_dir_path = os.path.join(config.saving_dir, "img_dir")

    # Evaluation parameters
    config.test_epoch = 300
    config.test_sample_size = 5000        # we want detailed props for 5000 samples
    config.post_method = "softmax"

    # -------------------------------
    # 2. Init logger + solver
    # -------------------------------
    logging.basicConfig(level=logging.INFO)
    log = logging

    solver = Solver(config, log)
    device = solver.device
    print("Device:", device)

    # -------------------------------
    # 3. Load generator checkpoint
    # -------------------------------
    ckpt_path = os.path.join(config.model_dir_path, f"{config.test_epoch}-G.ckpt")
    print("\nLoading generator checkpoint:", ckpt_path)

    state = torch.load(ckpt_path, map_location="cpu")
    solver.G.load_state_dict(state)
    print("✓ Generator fully restored from checkpoint.")

    # -------------------------------
    # 4. Generate molecules
    # -------------------------------
    n_samples = config.test_sample_size
    print(f"\nGenerating {n_samples} molecules from epoch {config.test_epoch}...")

    z_np = solver.sample_z(n_samples)
    z = torch.from_numpy(z_np).to(device).float()

    with torch.no_grad():
        edges_logits, nodes_logits = solver.G(z)
        mols = solver.get_gen_mols(
            n_hat=nodes_logits,
            e_hat=edges_logits,
            method=config.post_method,
        )

    print(f"Obtained {len(mols)} molecules (including invalid).")

    # -------------------------------
    # 5. Compute per-molecule properties
    # -------------------------------
    # These functions return lists aligned with `mols`
    qeds = MolecularMetrics.quantitative_estimation_druglikeness_scores(
        mols, norm=False
    )
    logps = MolecularMetrics.water_octanol_partition_coefficient_scores(
        mols, norm=False
    )
    sas = MolecularMetrics.synthetic_accessibility_score_scores(
        mols, norm=False
    )

    # Also extract SMILES for each molecule
    smiles = []
    for m in mols:
        if m is None:
            smiles.append(None)
        else:
            try:
                smiles.append(Chem.MolToSmiles(m))
            except Exception:
                smiles.append(None)

    # -------------------------------
    # 6. Save per-molecule CSV
    # -------------------------------
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = "results/metrics/epoch300_n5000_props.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "qed", "logp", "sa"])
        for s, q, l, a in zip(smiles, qeds, logps, sas):
            # Some entries may be None (invalid molecules or RDKit failure)
            writer.writerow([s, q, l, a])

    print(f"\nSaved per-molecule properties to {csv_path}")

    # Optional: also print quick summary like before
    def stats(x):
        x = np.array(
            [v for v in x if v is not None and not np.isnan(v)]
        )
        if x.size == 0:
            return (None, None, None)
        return (float(x.mean()), float(x.std()), float(x.min()))

    qed_mean, qed_std, qed_min = stats(qeds)
    logp_mean, logp_std, logp_min = stats(logps)
    sa_mean, sa_std, sa_min = stats(sas)

    print("\nQuick stats (valid entries only):")
    print(f"QED  : mean={qed_mean:.3f}, std={qed_std:.3f}, min={qed_min:.3f}")
    print(f"logP : mean={logp_mean:.3f}, std={logp_std:.3f}, min={logp_min:.3f}")
    print(f"SA   : mean={sa_mean:.3f}, std={sa_std:.3f}, min={sa_min:.3f}")


if __name__ == "__main__":
    main()
