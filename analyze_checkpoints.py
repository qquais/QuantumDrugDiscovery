# analyze_checkpoints.py

import os
from utils.args import get_GAN_config
from solver import Solver

def main():
    """Load and analyze saved checkpoints without retraining."""

    config = get_GAN_config()

    # hard-code paths to your classical run for now
    config.saving_dir = "results/GAN/20250126_234326/train"
    config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
    config.log_dir_path = os.path.join(config.saving_dir, "log_dir")
    config.img_dir_path = os.path.join(config.saving_dir, "img_dir")

    solver = Solver(config)

    # Simple inspection
    solver.list_checkpoints()
    solver.analyze_checkpoint(1)
    solver.analyze_checkpoint(100)
    solver.analyze_checkpoint(300)
    solver.generate_molecule(300)

if __name__ == "__main__":
    main()
