"""Command-line configuration.

Every knob that distinguishes one experiment from another is a flag here, so
an experiment is fully described by its command line (which `main.py` writes
verbatim into the run's config.json). The v1 configuration lived as
edited-in-place assignments in main.py, which is why its classical and quantum
rows silently used different datasets, different latent dimensions and
different generator widths.
"""

import argparse
import json

from qmolgan.latent import LATENT_KINDS
from qmolgan.rewards import REWARD_PRESETS


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('true', '1', 'yes', 'y')


def json_list(v):
    """Parse '[16]' or '16' or '128,256' into a list of ints."""
    if isinstance(v, (list, tuple)):
        return list(v)
    v = str(v).strip()
    if v.startswith('['):
        return json.loads(v)
    return [int(x) for x in v.split(',') if x.strip()]


def get_GAN_config(argv=None):
    p = argparse.ArgumentParser(
        description='Train a classical or quantum MolGAN variant.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- Experiment identity -------------------------------------------
    p.add_argument('--run_name', type=str, default=None,
                   help='human-readable name; defaults to a name derived from '
                        'latent/preset/seed')
    p.add_argument('--saving_dir', type=str, required=True,
                   help='run root; train/{log,model,img}_dir and config.json go here')
    p.add_argument('--seed', type=int, default=42,
                   help='seeds python/numpy/torch AND the latent initialisation')
    p.add_argument('--notes', type=str, default='',
                   help='free-text note recorded in config.json')

    # ---- Latent source (the axis under study) ---------------------------
    p.add_argument('--latent', type=str, default='gaussian', choices=list(LATENT_KINDS),
                   help='gaussian/uniform/rank2/trig are classical controls; '
                        'vqc and vqc_noent are the quantum variants')
    p.add_argument('--z_dim', type=int, default=8,
                   help='latent dimension; must equal --qubits for VQC latents')
    p.add_argument('--qubits', type=int, default=None,
                   help='VQC qubit count (defaults to --z_dim)')
    p.add_argument('--layer', type=int, default=3, help='VQC variational layers')
    p.add_argument('--n_freq', type=int, default=3,
                   help='frequency count for the classical trig surrogate latent')
    p.add_argument('--update_latent', type=str2bool, default=True,
                   help='train the latent source jointly with the generator')
    p.add_argument('--qc_lr', type=float, default=None,
                   help='learning rate for the latent source (defaults to --g_lr)')

    # ---- Architecture ---------------------------------------------------
    p.add_argument('--g_conv_dim', type=json_list, default=[128],
                   help='generator dense widths, e.g. "[16]" or "[128,256,512]"')
    p.add_argument('--d_conv_dim', type=json.loads,
                   default=[[128, 64], 128, [128, 64]],
                   help='discriminator/value dims as JSON')
    p.add_argument('--dropout', type=float, default=0.)
    p.add_argument('--post_method', type=str, default='softmax',
                   choices=['softmax', 'soft_gumbel', 'hard_gumbel'])
    p.add_argument('--gumbel_temp_start', type=float, default=1.0)
    p.add_argument('--gumbel_temp_end', type=float, default=1.0)

    # ---- Objective ------------------------------------------------------
    p.add_argument('--reward_preset', type=str, default='none',
                   choices=sorted(REWARD_PRESETS),
                   help="sets lambda_wgan and every rw_* weight together; "
                        "'none' is pure WGAN-GP")
    p.add_argument('--lambda_wgan', type=float, default=None,
                   help='override the preset: 1.0 = pure GAN (RL term disabled), '
                        '0.5 = reward shaping')
    p.add_argument('--lambda_gp', type=float, default=10.0)
    p.add_argument('--reward_mode', type=str, default=None,
                   choices=['legacy', 'weighted'])
    p.add_argument('--metric', type=str, default=None,
                   help='legacy multiplicative reward metrics, e.g. "sas,qed,unique"')
    p.add_argument('--enable_rl_loss', type=str2bool, default=True)
    for key, default in (('rw_qed', None), ('rw_sa', None), ('rw_logp', None),
                         ('rw_unique', None), ('rw_novelty', None),
                         ('rw_clean_valid', None), ('rw_fragment_penalty', None)):
        p.add_argument(f'--{key}', type=float, default=default,
                       help=f'override the preset value of {key}')
    p.add_argument('--rw_clip_min', type=float, default=0.0)
    p.add_argument('--rw_clip_max', type=float, default=1.0)

    # ---- Training -------------------------------------------------------
    p.add_argument('--mol_data_dir', type=str, default='data/qm9_5k_py37.sparsedataset')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_epochs', type=int, default=300)
    p.add_argument('--g_lr', type=float, default=1e-3)
    p.add_argument('--d_lr', type=float, default=1e-3)
    p.add_argument('--n_critic', type=int, default=5)
    p.add_argument('--critic_type', type=str, default='D', choices=['D', 'G'])
    p.add_argument('--decay_every_epoch', type=int, default=None)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--resume_epoch', type=int, default=None)
    p.add_argument('--model_save_step', type=int, default=1)

    # ---- In-training validation (monitoring only; never a reported number)
    p.add_argument('--val_n', type=int, default=1000,
                   help='molecules generated per validation pass; fixed across '
                        'epochs so the curve is comparable')
    p.add_argument('--val_seed', type=int, default=777,
                   help='noise seed for in-training validation; deliberately '
                        'disjoint from the protocol selection/report streams')
    p.add_argument('--val_every', type=int, default=1)

    # ---- Test mode ------------------------------------------------------
    p.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    p.add_argument('--test_epoch', type=int, default=None)

    # ---- Misc -----------------------------------------------------------
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--use_tensorboard', type=str2bool, default=False)

    config = p.parse_args(argv)

    if config.qubits is None:
        config.qubits = config.z_dim
    if config.latent in ('vqc', 'vqc_noent') and config.z_dim != config.qubits:
        p.error(f'--z_dim ({config.z_dim}) must equal --qubits ({config.qubits}) '
                'for a VQC latent: the circuit emits one value per wire.')
    return config
