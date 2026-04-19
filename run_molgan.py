import logging
import os

from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix


def _build_paths(config):
    if config.mode == "train":
        existing_model_dir = os.path.join(config.saving_dir, "model_dir")
        if config.resume_epoch is not None and os.path.isdir(existing_model_dir):
            # Resume directly from an existing ".../train" folder.
            run_id = get_date_postfix()
            config.log_dir_path = os.path.join(config.saving_dir, "log_dir")
            config.model_dir_path = existing_model_dir
            config.img_dir_path = os.path.join(config.saving_dir, "img_dir")
            log_name = os.path.join(config.log_dir_path, f"{run_id}_resume_logger.log")
        else:
            run_id = get_date_postfix()
            config.saving_dir = os.path.join(config.saving_dir, run_id)
            config.log_dir_path = os.path.join(config.saving_dir, "train", "log_dir")
            config.model_dir_path = os.path.join(config.saving_dir, "train", "model_dir")
            config.img_dir_path = os.path.join(config.saving_dir, "train", "img_dir")
            log_name = os.path.join(config.log_dir_path, f"{run_id}_logger.log")
    else:
        run_id = get_date_postfix()
        config.log_dir_path = os.path.join(config.saving_dir, "post_test", run_id, "log_dir")
        config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
        config.img_dir_path = os.path.join(config.saving_dir, "post_test", run_id, "img_dir")
        log_name = os.path.join(config.log_dir_path, f"{run_id}_logger.log")

    os.makedirs(config.log_dir_path, exist_ok=True)
    os.makedirs(config.model_dir_path, exist_ok=True)
    os.makedirs(config.img_dir_path, exist_ok=True)
    return log_name


def _resolve_complexity(config):
    if config.complexity == "nr":
        config.g_conv_dim = [128, 256, 512]
    elif config.complexity == "mr":
        config.g_conv_dim = [128]
    elif config.complexity == "hr":
        config.g_conv_dim = [16]
    else:
        raise ValueError("Invalid complexity. Use one of: nr, mr, hr")


def main():
    config = get_GAN_config()

    from torch.backends import cudnn
    cudnn.benchmark = True
    _resolve_complexity(config)

    # Delay RDKit-dependent imports so CLI help works even if RDKit is missing.
    from rdkit import RDLogger
    from solver import Solver
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Ensure optional quantum attribute is always present.
    if not hasattr(config, "gen_circuit"):
        config.gen_circuit = None

    log_path = _build_paths(config)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(config)

    solver = Solver(config, logging)
    solver.train_and_validate()


if __name__ == "__main__":
    main()
