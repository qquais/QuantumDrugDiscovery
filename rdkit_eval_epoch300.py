import os
import logging
import numpy as np
import torch

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

    # >>> EDIT THIS to point to your JANUARY run directory <<<
    # This folder must contain "model_dir", "log_dir", "img_dir"
    config.saving_dir = r"results/GAN/20250126_234326/train"

    config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
    config.log_dir_path = os.path.join(config.saving_dir, "log_dir")
    config.img_dir_path = os.path.join(config.saving_dir, "img_dir")

    # Evaluation parameters
    config.test_epoch = 300
    # Use 500 for a quick test; later change to 5000 for final baseline
    # config.test_sample_size = 500
    config.test_sample_size = 5000
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

    # Now that g_conv_dim = [128], Generator architecture matches the checkpoint:
    # - multi_dense_layers: 8 -> 128
    # - edges_layer: Linear(128, 405)
    # - nodes_layer: Linear(128, 54)
    state = torch.load(ckpt_path, map_location="cpu")
    solver.G.load_state_dict(state)
    print("✓ Generator fully restored from checkpoint.")

    # We don't need D or V for RDKit evaluation.

    # -------------------------------
    # 4. Generate molecules
    # -------------------------------
    n_samples = config.test_sample_size
    print(f"\nGenerating {n_samples} molecules from epoch {config.test_epoch}...")

    # sample_z is defined inside Solver and uses the correct z_dim (8)
    z_np = solver.sample_z(n_samples)
    z = torch.from_numpy(z_np).to(device).float()

    with torch.no_grad():
        edges_logits, nodes_logits = solver.G(z)

        # get_gen_mols does post-processing + RDKit conversion using SparseMolecularDataset
        mols = solver.get_gen_mols(
            n_hat=nodes_logits,
            e_hat=edges_logits,
            method=config.post_method,
        )

    print(f"Obtained {len(mols)} molecules (including invalid).")

    # -------------------------------
    # 5. Compute RDKit metrics
    # -------------------------------
    data = solver.data  # training set molecules, used for novelty

    valid_fraction = MolecularMetrics.valid_total_score(mols)
    unique_fraction = MolecularMetrics.unique_total_score(mols)

    qeds = MolecularMetrics.quantitative_estimation_druglikeness_scores(
        mols, norm=False
    )
    logps = MolecularMetrics.water_octanol_partition_coefficient_scores(
        mols, norm=False
    )
    sas = MolecularMetrics.synthetic_accessibility_score_scores(
        mols, norm=False
    )

    novelty_scores = MolecularMetrics.novel_scores(mols, data)
    novelty_fraction = float(np.mean(novelty_scores))

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

    print("\n==== Epoch-300 RDKit Evaluation (GDB9 baseline) ====")
    print(f"Total samples:         {n_samples}")
    print(f"Validity fraction:     {valid_fraction:.3f}")
    print(f"Uniqueness fraction:   {unique_fraction:.3f}")
    print(f"Novelty fraction:      {novelty_fraction:.3f}")

    print(
        f"\nQED  : mean={qed_mean:.3f}, std={qed_std:.3f}, min={qed_min:.3f}"
    )
    print(
        f"logP : mean={logp_mean:.3f}, std={logp_std:.3f}, min={logp_min:.3f}"
    )
    print(
        f"SA   : mean={sa_mean:.3f}, std={sa_std:.3f}, min={sa_min:.3f}"
    )

    # Save summary
    os.makedirs("results/metrics", exist_ok=True)
    out_path = f"results/metrics/epoch{config.test_epoch}_n{n_samples}_metrics.txt"

    with open(out_path, "w") as f:
        f.write("Epoch-300 RDKit Evaluation (GDB9 baseline)\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Validity fraction:   {valid_fraction:.3f}\n")
        f.write(f"Uniqueness fraction: {unique_fraction:.3f}\n")
        f.write(f"Novelty fraction:    {novelty_fraction:.3f}\n\n")
        f.write(
            f"QED  : mean={qed_mean:.3f}, std={qed_std:.3f}, min={qed_min:.3f}\n"
        )
        f.write(
            f"logP : mean={logp_mean:.3f}, std={logp_std:.3f}, min={logp_min:.3f}\n"
        )
        f.write(
            f"SA   : mean={sa_mean:.3f}, std={sa_std:.3f}, min={sa_min:.3f}\n"
        )

    print(f"\nSaved metrics summary to {out_path}")


if __name__ == "__main__":
    main()
