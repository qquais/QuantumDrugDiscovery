"""Reward presets and the weighted multi-objective reward.

Kept out of the solver so that (a) the exact weights used for a run are
serialisable into its config JSON, (b) a weight sweep can enumerate presets
without touching training code, and (c) the unit tests can check the reward
is bounded and monotone without constructing a Solver.

Normalisation note: the reward uses NORMALISED SA and logP (mapped into
[0, 1], higher = better) because a weighted sum needs commensurate terms.
Reported tables always use RAW SA/logP. Conflating the two is what produced
the "SA = 0.410" line in the v1 results table.
"""

from copy import deepcopy

import numpy as np

from utils.utils import MolecularMetrics

# Component weights + fragment penalty. `lambda_wgan` is stored alongside
# because a reward preset is meaningless without knowing how much weight the
# generator loss gives it: at lambda_wgan = 1.0 the RL term is multiplied by
# zero and the preset has no effect whatsoever.
REWARD_PRESETS = {
    # Pure WGAN-GP, no reward shaping. lambda_wgan = 1.0 -> RL term disabled.
    'none': dict(lambda_wgan=1.0, reward_mode='legacy', metric='sas,qed,unique',
                 rw_qed=0.0, rw_sa=0.0, rw_logp=0.0, rw_unique=0.0,
                 rw_novelty=0.0, rw_clean_valid=0.0, rw_fragment_penalty=0.0),

    # Legacy MolGAN multiplicative reward (De Cao & Kipf), for reference.
    'legacy': dict(lambda_wgan=0.5, reward_mode='legacy', metric='sas,qed,unique',
                   rw_qed=0.0, rw_sa=0.0, rw_logp=0.0, rw_unique=0.0,
                   rw_novelty=0.0, rw_clean_valid=0.0, rw_fragment_penalty=0.0),

    # A: property-only. Tests reward without any diversity term.
    'ablation_a': dict(lambda_wgan=0.5, reward_mode='weighted',
                       rw_qed=0.50, rw_sa=0.50, rw_logp=0.00, rw_unique=0.00,
                       rw_novelty=0.00, rw_clean_valid=0.00, rw_fragment_penalty=0.00),

    # B: balanced multi-objective. The paper's headline configuration.
    'ablation_b': dict(lambda_wgan=0.5, reward_mode='weighted',
                       rw_qed=0.35, rw_sa=0.35, rw_logp=0.00, rw_unique=0.15,
                       rw_novelty=0.10, rw_clean_valid=0.05, rw_fragment_penalty=0.20),

    # C: aggressive fragment penalisation.
    'ablation_c': dict(lambda_wgan=0.5, reward_mode='weighted',
                       rw_qed=0.30, rw_sa=0.30, rw_logp=0.00, rw_unique=0.10,
                       rw_novelty=0.10, rw_clean_valid=0.20, rw_fragment_penalty=0.40),

    # D: clean-validity-directed. Follow-up to the v1 analysis's own
    # "future work" item; boosts the structural terms only.
    'ablation_d_clean': dict(lambda_wgan=0.5, reward_mode='weighted',
                             rw_qed=0.25, rw_sa=0.25, rw_logp=0.00, rw_unique=0.15,
                             rw_novelty=0.10, rw_clean_valid=0.25,
                             rw_fragment_penalty=0.35),

    # E: diversity-dominant, to map the far end of the quality-diversity front.
    'ablation_e_diverse': dict(lambda_wgan=0.5, reward_mode='weighted',
                               rw_qed=0.20, rw_sa=0.20, rw_logp=0.00, rw_unique=0.35,
                               rw_novelty=0.20, rw_clean_valid=0.05,
                               rw_fragment_penalty=0.10),
}

REWARD_KEYS = ('rw_qed', 'rw_sa', 'rw_logp', 'rw_unique', 'rw_novelty',
               'rw_clean_valid', 'rw_fragment_penalty')


def get_preset(name):
    """Return a copy of a named preset (so callers cannot mutate the table)."""
    if name not in REWARD_PRESETS:
        raise ValueError(f'unknown reward preset {name!r}; '
                         f'available: {sorted(REWARD_PRESETS)}')
    return deepcopy(REWARD_PRESETS[name])


def apply_preset(config, name):
    """Write a preset's fields onto an argparse-style config object.

    Only fields the preset actually defines are written, and `lambda_wgan` is
    written too — omitting it was how 'ablation_b' silently became a no-op in
    runs that also set lambda_wgan = 1.0.
    """
    preset = get_preset(name)
    for key, value in preset.items():
        setattr(config, key, value)
    setattr(config, 'reward_preset', name)
    return config


def weighted_reward(mols, data_smiles_set, weights, clip=(0.0, 1.0)):
    """Normalised additive reward with a fragment penalty, in [clip_min, clip_max].

    Parameters
    ----------
    mols : list of Mol or None
    data_smiles_set : set of str
        Training SMILES, for the novelty term.
    weights : dict
        Keys from REWARD_KEYS.
    """
    from rdkit import Chem

    qed = MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=False)
    sa = MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
    logp = MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
    unique = MolecularMetrics.unique_scores(mols).astype(np.float32)
    clean_valid = MolecularMetrics.valid_scores(mols).astype(np.float32)
    novelty = np.array(
        [1.0 if (m is not None and MolecularMetrics.valid_lambda(m)
                 and Chem.MolToSmiles(m) not in data_smiles_set) else 0.0
         for m in mols], dtype=np.float32)

    num = (weights['rw_qed'] * qed + weights['rw_sa'] * sa + weights['rw_logp'] * logp
           + weights['rw_unique'] * unique + weights['rw_novelty'] * novelty
           + weights['rw_clean_valid'] * clean_valid)
    den = sum(weights[k] for k in
              ('rw_qed', 'rw_sa', 'rw_logp', 'rw_unique', 'rw_novelty', 'rw_clean_valid'))
    base = num / den if den > 0 else np.zeros_like(num)
    penalised = base - weights['rw_fragment_penalty'] * (1.0 - clean_valid)
    return np.clip(penalised, clip[0], clip[1]).reshape(-1, 1)


def weight_sweep_grid(base='ablation_b', axis='rw_unique',
                      values=(0.0, 0.05, 0.10, 0.15, 0.25, 0.35)):
    """Named one-at-a-time perturbations of a preset, for sensitivity analysis.

    The v1 weights were hand-picked with no sensitivity analysis; this
    enumerates the runs that supply one. Returns a list of
    (run_name, weight_dict) pairs with the remaining component weights
    renormalised so the sweep varies the *balance*, not the total scale.
    """
    preset = get_preset(base)
    others = [k for k in ('rw_qed', 'rw_sa', 'rw_logp', 'rw_unique',
                          'rw_novelty', 'rw_clean_valid') if k != axis]
    other_total = sum(preset[k] for k in others)
    runs = []
    for v in values:
        w = deepcopy(preset)
        w[axis] = float(v)
        remaining = max(1.0 - float(v), 0.0)
        if other_total > 0:
            for k in others:
                w[k] = preset[k] / other_total * remaining
        runs.append((f'{base}__{axis}_{v:g}'.replace('.', 'p'), w))
    return runs
