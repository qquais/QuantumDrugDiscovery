"""Regression tests for the metric definitions.

Each test pins down one of the defects that produced a wrong number in the
v1 pipeline, so the same mistake cannot silently return.

Run with:  python -m pytest tests/ -q      (from the project root)
"""

import os
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmolgan import chem, protocol, rewards
from utils.utils import MolecularMetrics


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


def _single_bond_label(data):
    """Bond-decoder key for a single bond.

    Matched on the enum's integer value rather than its repr: the shipped
    .sparsedataset pickles were written by an older RDKit whose BondType
    objects repr as 'BondType(1)' instead of 'SINGLE'.
    """
    for k, v in data.bond_decoder_m.items():
        if int(v) == int(Chem.BondType.SINGLE):
            return k
    raise AssertionError('no SINGLE bond in the dataset decoder')


# ---------------------------------------------------------------------------
# Clean-validity vs padding
# ---------------------------------------------------------------------------

def test_clean_validity_rejects_fragments_and_wildcards():
    assert chem.is_clean_valid(mol('CC1CCCNC1'))
    assert not chem.is_clean_valid(mol('CC.CC'))          # disconnected
    assert not chem.is_clean_valid(mol('*CC'))            # wildcard
    assert chem.is_valid(mol('CC.CC'))                    # still 'valid'


def test_padded_graph_decodes_to_a_connected_molecule():
    """A 5-atom molecule padded into a 9-vertex graph must not be scored as a
    fragment. Leaving unbonded PAD slots in place turned every sub-9-atom
    molecule into '*.*.<mol>' and drove quantum clean-validity to exactly
    0.000 in the v1 pipeline."""
    from data.sparse_molecular_dataset import SparseMolecularDataset

    dataset_path = 'data/qm9_5k_py37.sparsedataset'
    if not os.path.exists(dataset_path):
        pytest.skip(f'{dataset_path} not available')
    data = SparseMolecularDataset()
    data.load(dataset_path)

    # Propane (3 heavy atoms) inside a 9-vertex graph: nodes 0-2 are carbon,
    # nodes 3-8 are PAD, single bonds 0-1 and 1-2.
    carbon = [k for k, v in data.atom_decoder_m.items() if v == 6][0]
    single = _single_bond_label(data)
    nodes = np.zeros(data.vertexes, dtype=int)
    nodes[:3] = carbon
    edges = np.zeros((data.vertexes, data.vertexes), dtype=int)
    edges[0, 1] = edges[1, 0] = single
    edges[1, 2] = edges[2, 1] = single

    m = data.matrices2mol(nodes, edges, strict=True)
    assert m is not None
    assert Chem.MolToSmiles(m) == 'CCC'
    assert chem.is_clean_valid(m)


def test_bonded_pad_slot_stays_invalid():
    """A PAD atom the model actually bonded to is a real generation error and
    must keep counting against clean-validity; only unbonded PAD slots are
    structural padding."""
    from data.sparse_molecular_dataset import SparseMolecularDataset

    dataset_path = 'data/qm9_5k_py37.sparsedataset'
    if not os.path.exists(dataset_path):
        pytest.skip(f'{dataset_path} not available')
    data = SparseMolecularDataset()
    data.load(dataset_path)

    carbon = [k for k, v in data.atom_decoder_m.items() if v == 6][0]
    pad = [k for k, v in data.atom_decoder_m.items() if v == 0][0]
    single = _single_bond_label(data)
    nodes = np.zeros(data.vertexes, dtype=int)
    nodes[0] = carbon
    nodes[1] = pad
    edges = np.zeros((data.vertexes, data.vertexes), dtype=int)
    edges[0, 1] = edges[1, 0] = single

    m = data.matrices2mol(nodes, edges, strict=True)
    assert m is not None and '*' in Chem.MolToSmiles(m)
    assert not chem.is_clean_valid(m)


# ---------------------------------------------------------------------------
# SA reporting
# ---------------------------------------------------------------------------

def test_reported_sa_is_the_raw_ertl_scale():
    """Raw SA lives on [1, 10]; the normalised reward-space version lives on
    [0, 1]. The v1 results table reported 0.410 as an SA score."""
    mols = [mol(s) for s in ('CC1CCCNC1', 'c1ccccc1', 'CCO')]
    props = chem.raw_properties(mols)
    assert np.all(props['SA'] >= 1.0) and np.all(props['SA'] <= 10.0)

    normalised = MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
    assert np.all(normalised >= 0.0) and np.all(normalised <= 1.0)
    # The two scales are anti-correlated, so they can never be confused for
    # one another by a reader who checks the direction.
    assert np.corrcoef(props['SA'], normalised)[0, 1] < 0


def test_evaluate_reports_raw_properties():
    mols = [mol(s) for s in ('CC1CCCNC1', 'CCO', 'c1ccccc1')]
    out = chem.evaluate_molecules(mols, train_smiles=['CCO'])
    assert 1.0 <= out['SA'] <= 10.0
    assert 0.0 <= out['QED'] <= 1.0


# ---------------------------------------------------------------------------
# Denominators
# ---------------------------------------------------------------------------

def test_validity_denominator_is_all_generated_graphs():
    mols = [mol('CCO'), None, mol('CCO'), None]
    out = chem.evaluate_molecules(mols, train_smiles=[])
    assert out['validity'] == pytest.approx(0.5)
    assert out['n_generated'] == 4


def test_uniqueness_is_measured_among_valid_not_among_all():
    mols = [mol('CCO'), mol('CCO'), mol('CCC'), None]
    out = chem.evaluate_molecules(mols, train_smiles=[])
    assert out['uniqueness'] == pytest.approx(2 / 3)


def test_novelty_uses_the_training_split_only():
    mols = [mol('CCO'), mol('CCC')]
    out = chem.evaluate_molecules(mols, train_smiles=['CCO'])
    assert out['novelty'] == pytest.approx(0.5)


def test_empty_generation_is_nan_not_zero():
    out = chem.evaluate_molecules([None, None], train_smiles=['CCO'])
    assert out['validity'] == 0.0
    assert np.isnan(out['uniqueness'])
    assert np.isnan(out['QED'])


# ---------------------------------------------------------------------------
# Uniqueness is only comparable at a fixed n
# ---------------------------------------------------------------------------

def test_uniqueness_decreases_with_sample_size():
    """The withdrawn '130-fold uniqueness improvement' compared a quantum
    number measured on training batches of 16 with a classical number
    measured at n=5000."""
    rng = np.random.default_rng(0)
    pool = [f'C{i}' for i in rng.integers(0, 50, size=4000)]
    curve = chem.uniqueness_vs_sample_size(pool, sizes=(16, 64, 256, 1000))
    values = [curve[n] for n in (16, 64, 256, 1000)]
    assert all(a >= b - 1e-9 for a, b in zip(values, values[1:])), values
    assert curve[16] > curve[1000]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def test_reward_is_bounded_and_penalises_fragments():
    weights = rewards.get_preset('ablation_b')
    clean = [mol('CC1CCCNC1')] * 4
    dirty = [mol('CC.CC')] * 4
    r_clean = rewards.weighted_reward(clean, set(), weights)
    r_dirty = rewards.weighted_reward(dirty, set(), weights)
    assert np.all((r_clean >= 0) & (r_clean <= 1))
    assert np.all((r_dirty >= 0) & (r_dirty <= 1))
    assert r_clean.mean() > r_dirty.mean()


def test_preset_sets_lambda_wgan_with_the_weights():
    """A reward preset that does not also set lambda_wgan is a no-op whenever
    lambda_wgan is left at 1.0, because the RL term is then multiplied by
    (1 - 1.0) = 0."""
    for name in rewards.REWARD_PRESETS:
        preset = rewards.get_preset(name)
        assert 'lambda_wgan' in preset
    assert rewards.get_preset('none')['lambda_wgan'] == 1.0
    assert rewards.get_preset('ablation_b')['lambda_wgan'] == 0.5


def test_weight_sweep_keeps_component_weights_normalised():
    for _, w in rewards.weight_sweep_grid():
        total = sum(w[k] for k in ('rw_qed', 'rw_sa', 'rw_logp', 'rw_unique',
                                   'rw_novelty', 'rw_clean_valid'))
        assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def test_selection_and_report_streams_are_disjoint():
    for seed in (0, 1, 42, 123, 456, 999):
        assert protocol.select_seed(seed) != protocol.report_seed(seed)
        assert protocol.latent_seed(seed) not in (protocol.select_seed(seed),
                                                  protocol.report_seed(seed))


def test_selection_rule_rejects_degenerate_early_checkpoints():
    """An epoch-2 checkpoint emitting one molecule forever has
    clean_validity = 1.0 but uniqueness_clean ~ 0, and is below the epoch
    floor besides."""
    rows = [{'epoch': 2, 'clean_validity': 1.0, 'uniqueness_clean': 0.001},
            {'epoch': 150, 'clean_validity': 0.6, 'uniqueness_clean': 0.4},
            {'epoch': 299, 'clean_validity': 0.2, 'uniqueness_clean': 0.9}]
    best, scored = protocol.select_best_epoch(rows, n_epochs=300)
    assert best['epoch'] == 150
    assert not [r for r in scored if r['epoch'] == 2][0]['eligible']


def test_paired_difference_reports_pair_count():
    res = protocol.paired_difference([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])
    assert res['n_pairs'] == 3
    assert res['mean_diff'] == pytest.approx(0.1)
    assert res['excludes_zero']


def test_paired_difference_needs_matched_seeds():
    with pytest.raises(ValueError):
        protocol.paired_difference([0.5, 0.6], [0.4])


# ---------------------------------------------------------------------------
# Latent geometry
# ---------------------------------------------------------------------------

def test_vqc_latent_is_rank_deficient_relative_to_gaussian():
    """The Kao circuit draws two random scalars per sample, so its outputs lie
    on a 2-D manifold whatever the qubit count. This is the mechanism behind
    the mode collapse the v1 analysis attributed to 'compressed latent
    space' without measuring it."""
    from qmolgan.latent import make_latent, latent_statistics

    gauss = latent_statistics(make_latent('gaussian', dim=4), n=1024, seed=0)
    vqc = latent_statistics(make_latent('vqc', dim=4, qubits=4, layers=3, seed=0),
                            n=1024, seed=0)
    assert gauss['participation_ratio'] > 3.5
    assert vqc['participation_ratio'] < 2.5
    assert vqc['effective_rank_99pct'] <= 3


def test_vqc_requires_z_dim_to_equal_qubits():
    from qmolgan.latent import make_latent
    with pytest.raises(ValueError):
        make_latent('vqc', dim=8, qubits=4, layers=3)


def test_trig_budget_is_not_smaller_than_the_vqc():
    """The trig surrogate is the classical control for the circuit's function
    class; the comparison is only fair if it is never the under-parameterised
    arm. make_latent's default must satisfy that for every grid width."""
    from qmolgan.latent import make_latent, n_circuit_weights
    for dim in (2, 4, 6, 8):
        trig = make_latent('trig', dim=dim)
        vqc = make_latent('vqc', dim=dim, qubits=dim, layers=3)
        assert trig.describe()['n_params'] >= vqc.describe()['n_params']
        assert vqc.describe()['n_params'] == n_circuit_weights(dim, 3, True)


def test_circuit_weight_count_is_what_the_circuit_reads():
    """Kao's indexing applies the same 2q-1 weights in every layer, so depth
    adds no parameters. The count reported per run must be the number the
    circuit reads, not the length of an over-allocated vector."""
    from qmolgan.latent import make_latent, n_circuit_weights
    assert n_circuit_weights(4, 3, entangle=True) == 7
    assert n_circuit_weights(4, 6, entangle=True) == 7
    assert n_circuit_weights(4, 3, entangle=False) == 4
    assert make_latent('vqc', dim=4, qubits=4, layers=3).weights.numel() == 7
    assert make_latent('vqc_noent', dim=4, qubits=4, layers=3).weights.numel() == 4


def test_legacy_over_allocated_circuit_checkpoint_still_loads(tmp_path):
    """Checkpoints written when the vector was layers*(2q-1) long must load
    into the honest-length parameter, keeping the leading slice the circuit
    actually used."""
    import torch
    from qmolgan.generate import load_latent_state
    from qmolgan.latent import make_latent

    lat = make_latent('vqc', dim=4, qubits=4, layers=3, seed=0)
    legacy = torch.arange(21, dtype=torch.float64)
    torch.save({'weights': legacy}, tmp_path / '5-Z.ckpt')
    assert load_latent_state(lat, str(tmp_path), 5) == 'state_dict'
    assert torch.equal(lat.weights.detach(), legacy[:7])


def test_latent_statistics_can_hand_back_its_sample():
    from qmolgan.latent import make_latent, latent_statistics
    stats, z = latent_statistics(make_latent('uniform', dim=4), n=256, seed=3,
                                 return_sample=True)
    assert z.shape == (256, 4)
    assert np.allclose(z.mean(0), stats['mean'])


# ---------------------------------------------------------------------------
# RNG hygiene
# ---------------------------------------------------------------------------

def test_seeded_sampling_leaves_the_callers_rng_alone():
    """validate() runs once per epoch inside the training loop. If its seeded
    sampling touched the global torch/numpy RNG, every epoch after the first
    would replay identical noise and batch order, whatever --seed was."""
    import torch
    from qmolgan.latent import make_latent, latent_statistics, seeded_rng

    torch.manual_seed(42)
    np.random.seed(42)
    torch.rand(3); np.random.rand(3)
    expected_t = torch.get_rng_state().clone()
    expected_np = np.random.get_state()

    with seeded_rng(777):
        torch.rand(10); np.random.rand(10)
    assert torch.equal(torch.get_rng_state(), expected_t)
    assert np.array_equal(np.random.get_state()[1], expected_np[1])

    latent_statistics(make_latent('vqc', dim=2, qubits=2, layers=1), n=4, seed=777)
    assert torch.equal(torch.get_rng_state(), expected_t)

    # And the seeded draw itself is reproducible regardless of outer state.
    with seeded_rng(777):
        a = torch.rand(4)
    torch.manual_seed(1)
    with seeded_rng(777):
        b = torch.rand(4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Selection fallback, config authority, reference sampling
# ---------------------------------------------------------------------------

def test_all_degenerate_sweep_falls_back_to_the_final_epoch():
    """A run with no clean-valid molecule at any epoch has no uniqueness_clean
    anywhere. The pre-registered rule must then report the final epoch and
    flag the fallback, not quietly pick the floor epoch with score 0."""
    rows = [{'epoch': e, 'clean_validity': 0.0, 'uniqueness_clean': float('nan')}
            for e in (1, 30, 150, 300)]
    best, scored = protocol.select_best_epoch(rows, n_epochs=300)
    assert best['epoch'] == 300
    assert best['selection_fallback'] == 'no_eligible_epoch_final_used'
    assert all(not np.isfinite(r['selection_score']) for r in scored)


def test_config_json_wins_over_a_conflicting_cli_override(tmp_path):
    """ERRATA D5: a hand-typed architecture evaluated the wrong network. The
    recorded config is authoritative; a conflicting flag is an error."""
    import json
    from qmolgan.generate import read_run_config

    cfg = {'dataset': 'd', 'latent': 'vqc', 'z_dim': 4, 'g_conv_dim': [16]}
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    assert read_run_config(str(tmp_path), {'z_dim': 4})['z_dim'] == 4
    assert read_run_config(str(tmp_path), {'qubits': 4})['qubits'] == 4   # fills a gap
    with pytest.raises(ValueError):
        read_run_config(str(tmp_path), {'z_dim': 8})


def test_reference_sample_does_not_depend_on_what_was_generated():
    """SNN / W1 / KL compare every model against the training split. The
    reference sub-sample must be a function of the seed alone, not of how
    much other randomness the evaluation consumed first."""
    train = [f'C{"C" * (i % 7)}O' for i in range(6000)]
    a = chem.reference_sample(train, seed=5)
    rng = np.random.default_rng(99)
    rng.random(1000)                      # unrelated consumption
    b = chem.reference_sample(train, seed=5)
    assert [Chem.MolToSmiles(m) for m in a] == [Chem.MolToSmiles(m) for m in b]
    assert len(a) == 5000
