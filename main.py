#!/usr/bin/env python
"""Train one MolGAN variant. Everything is set from the command line.

    # classical baseline, matched architecture, no reward shaping
    python main.py --saving_dir results/runs/classical_gaussian_none_s42 \
        --latent gaussian --z_dim 4 --g_conv_dim "[16]" \
        --reward_preset none --seed 42

    # quantum noise generator with balanced reward shaping
    python main.py --saving_dir results/runs/vqc_ablation_b_s42 \
        --latent vqc --z_dim 4 --qubits 4 --layer 3 --g_conv_dim "[16]" \
        --reward_preset ablation_b --qc_lr 0.04 --seed 42

The run's full configuration is written to <saving_dir>/config.json before
training starts, and every downstream tool reads the model architecture from
that file rather than from hard-coded constants.
"""

import json
import logging
import os
import random
import sys

import numpy as np
import torch
from rdkit import RDLogger
from torch.backends import cudnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qmolgan import rewards as reward_utils
from qmolgan.generate import write_run_config
from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix

RDLogger.logger().setLevel(RDLogger.CRITICAL)


def build_config():
    config = get_GAN_config()

    # A preset sets lambda_wgan and every rw_* weight as one coherent block;
    # explicit flags then override individual fields. Setting the weights
    # without setting lambda_wgan is how 'ablation_b' became a silent no-op in
    # runs that also left lambda_wgan at 1.0.
    explicit = {k: getattr(config, k) for k in
                ('lambda_wgan', 'reward_mode', 'metric', *reward_utils.REWARD_KEYS)
                if getattr(config, k) is not None}
    reward_utils.apply_preset(config, config.reward_preset)
    for key, value in explicit.items():
        setattr(config, key, value)
    if getattr(config, 'metric', None) is None:
        config.metric = 'sas,qed,unique'

    if config.run_name is None:
        config.run_name = f'{config.latent}_{config.reward_preset}_z{config.z_dim}_s{config.seed}'
    return config


def seed_everything(seed):
    """Seed before ANY model or latent construction: nn.Module init and the
    latent sampler's parameter init both draw from the global RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cudnn.benchmark = True
    config = build_config()
    seed_everything(config.seed)

    config.log_dir_path = os.path.join(config.saving_dir, 'train', 'log_dir')
    config.model_dir_path = os.path.join(config.saving_dir, 'train', 'model_dir')
    config.img_dir_path = os.path.join(config.saving_dir, 'train', 'img_dir')
    for d in (config.log_dir_path, config.model_dir_path, config.img_dir_path):
        os.makedirs(d, exist_ok=True)

    # config.json is the contract with every offline tool: dataset, latent
    # source, latent dimension and generator width all come from here.
    run_config = {
        'run_name': config.run_name,
        'dataset': config.mol_data_dir,
        'latent': config.latent,
        'z_dim': config.z_dim,
        'qubits': config.qubits,
        'layers': config.layer,
        'n_freq': config.n_freq,
        'g_conv_dim': list(config.g_conv_dim),
        'd_conv_dim': config.d_conv_dim,
        'dropout': config.dropout,
        'post_method': config.post_method,
        'seed': config.seed,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'n_critic': config.n_critic,
        'g_lr': config.g_lr,
        'd_lr': config.d_lr,
        'qc_lr': config.qc_lr,
        'update_latent': config.update_latent,
        'reward_preset': config.reward_preset,
        'lambda_wgan': config.lambda_wgan,
        'lambda_gp': config.lambda_gp,
        'reward_mode': config.reward_mode,
        'reward_weights': {k: getattr(config, k) for k in reward_utils.REWARD_KEYS},
        'val_n': config.val_n,
        'val_seed': config.val_seed,
        'notes': config.notes,
        'argv': sys.argv,
        'started': get_date_postfix(),
    }
    write_run_config(config.saving_dir, run_config)

    log_path = os.path.join(config.log_dir_path,
                            f'{config.run_name}_{get_date_postfix()}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(json.dumps(run_config, indent=2, default=str))
    print(json.dumps(run_config, indent=2, default=str), flush=True)

    from solver import Solver
    Solver(config, logging).train_and_validate()


if __name__ == '__main__':
    main()
