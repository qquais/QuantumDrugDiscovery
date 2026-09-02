"""Molecule decoding and the canonical evaluation metric suite.

This module is the single source of truth for every number that appears in
a table or figure. Training-time logging, the epoch sweep and the final
evaluation all call `evaluate_molecules` here.

Metric conventions (fixed once, applied everywhere):

* ``validity``      — fraction of *generated* graphs that RDKit sanitises.
* ``clean_validity``— fraction of *generated* graphs that sanitise **and**
                      whose canonical SMILES contains neither '.' (multiple
                      disconnected components) nor '*' (wildcard atom).
* ``uniqueness``    — unique canonical SMILES / valid molecules.
* ``uniqueness_clean`` — unique canonical SMILES / clean-valid molecules.
* ``novelty``       — valid molecules absent from the *training split*
                      (never the full dataset: val/test molecules must be
                      novel by construction or novelty is inflated).
* ``QED``/``logP``/``SA`` — reported as RAW RDKit values over the clean-valid
                      set. SA is on the 1-10 Ertl scale, lower = easier.
                      Normalised [0,1] variants exist only inside the RL
                      reward and are never reported.

All population metrics are undefined-safe: they return ``nan`` rather than
0.0 when the denominator is empty, so an empty model is never silently
scored as "0.0 diversity, which is a number".
"""

import math
import warnings
from collections import Counter

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.utils import MolecularMetrics

RDLogger.logger().setLevel(RDLogger.CRITICAL)

# Reported property keys, in table order.
PROPERTY_KEYS = ('QED', 'logP', 'SA', 'MW')

# Metrics that make up the headline table, in table order.
HEADLINE_KEYS = (
    'validity', 'clean_validity', 'uniqueness', 'uniqueness_clean',
    'novelty', 'novelty_clean', 'QED', 'logP', 'SA',
)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_mol(node_labels, edge_labels, data, strict=True):
    """Graph matrices -> RDKit Mol (or None).

    Delegates to the dataset decoder so that the PAD-atom handling stays in
    exactly one place. See data/sparse_molecular_dataset.py::matrices2mol.
    """
    return data.matrices2mol(node_labels, edge_labels, strict=strict)


def decode_batch(nodes_hard, edges_hard, data, strict=True):
    """Decode a batch of hard label matrices into RDKit Mols."""
    return [decode_mol(n, e, data, strict=strict)
            for n, e in zip(np.asarray(nodes_hard), np.asarray(edges_hard))]


def canonical_smiles(mol):
    """Canonical SMILES, or None if the mol is unusable."""
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        return None
    return smi if smi else None


def is_valid(mol):
    """RDKit-sanitisable and non-empty — the permissive, standard definition."""
    return MolecularMetrics.valid_lambda(mol)


def is_clean_valid(mol):
    """Valid AND a single connected component AND no wildcard atoms."""
    return MolecularMetrics.valid_lambda_special(mol)


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

def _safe(fn, mol, default=float('nan')):
    try:
        return float(fn(mol))
    except Exception:
        return default


def raw_properties(mols):
    """Raw (un-normalised) physicochemical properties for a list of Mols.

    Returns a dict of float arrays aligned with ``mols``. Values are ``nan``
    where the property could not be computed, so downstream ``nanmean`` gives
    an honest mean over what was computable instead of imputing a sentinel.
    """
    qed, logp, sa, mw = [], [], [], []
    for m in mols:
        if m is None:
            qed.append(np.nan); logp.append(np.nan); sa.append(np.nan); mw.append(np.nan)
            continue
        qed.append(_safe(QED.qed, m))
        logp.append(_safe(Crippen.MolLogP, m))
        try:
            sa.append(float(MolecularMetrics._compute_SAS(m)))
        except Exception:
            sa.append(np.nan)
        mw.append(_safe(Descriptors.MolWt, m))
    return {
        'QED': np.array(qed, dtype=float),
        'logP': np.array(logp, dtype=float),
        'SA': np.array(sa, dtype=float),
        'MW': np.array(mw, dtype=float),
    }


# ---------------------------------------------------------------------------
# Fingerprints and diversity
# ---------------------------------------------------------------------------

def morgan_fps(mols, radius=2, n_bits=2048):
    """Morgan bit-vector fingerprints; None entries are dropped."""
    from rdkit.Chem import rdMolDescriptors
    fps = []
    for m in mols:
        if m is None:
            continue
        try:
            fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))
        except Exception:
            continue
    return fps


def internal_diversity(fps, p=1, max_n=3000, rng=None):
    """MOSES-style IntDiv_p = 1 - (mean pairwise Tanimoto^p)^(1/p).

    Sub-samples to ``max_n`` molecules because the pairwise matrix is O(n^2);
    the sub-sample is drawn with the supplied RNG so the number is reproducible.
    """
    if len(fps) < 2:
        return float('nan')
    if len(fps) > max_n:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(fps), max_n, replace=False)
        fps = [fps[i] for i in idx]
    total, count = 0.0, 0
    for i in range(1, len(fps)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]), dtype=float)
        total += float(np.sum(sims ** p))
        count += sims.size
    if count == 0:
        return float('nan')
    return float(1.0 - (total / count) ** (1.0 / p))


def nearest_neighbour_similarity(gen_fps, ref_fps, max_ref=5000, rng=None):
    """SNN: mean over generated molecules of the max Tanimoto to the reference set.

    High SNN means the model is reproducing training chemistry; low SNN paired
    with low validity usually means it is producing nonsense. Reported so that
    'novelty = 1.0' can be interpreted rather than taken at face value.
    """
    if not gen_fps or not ref_fps:
        return float('nan')
    if len(ref_fps) > max_ref:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(ref_fps), max_ref, replace=False)
        ref_fps = [ref_fps[i] for i in idx]
    best = [max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps)) for fp in gen_fps]
    return float(np.mean(best))


def scaffold_stats(mols):
    """Bemis-Murcko scaffold diversity over a list of Mols.

    Returns (n_unique_scaffolds, scaffold_diversity) where the diversity is
    unique scaffolds / molecules. Reported because uniqueness alone is easy to
    fake: a model can look 'unique' by decorating one scaffold 5000 ways.
    """
    scaffs = []
    for m in mols:
        if m is None:
            continue
        try:
            scaffs.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
        except Exception:
            continue
    if not scaffs:
        return 0, float('nan')
    uniq = len(set(scaffs))
    return uniq, float(uniq / len(scaffs))


# ---------------------------------------------------------------------------
# Distribution distances
# ---------------------------------------------------------------------------

def wasserstein1(a, b):
    """1-D Wasserstein-1 distance between two samples (scipy-free)."""
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float('nan')
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), grid, side='right') / a.size
    cb = np.searchsorted(np.sort(b), grid, side='right') / b.size
    return float(np.sum(np.abs(ca - cb)[:-1] * np.diff(grid)))


def kl_divergence(a, b, bins=50, eps=1e-10):
    """KL(P_generated || Q_reference) on a shared histogram grid.

    Reference support defines the grid, so a generator that puts mass outside
    the reference range is penalised rather than silently clipped away.
    """
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float('nan')
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float('nan')
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=False)
    q, _ = np.histogram(b, bins=edges, density=False)
    p = p / max(p.sum(), 1) + eps
    q = q / max(q.sum(), 1) + eps
    return float(np.sum(p * np.log(p / q)))


def frechet_chemnet_distance(gen_smiles, ref_smiles, device='cpu'):
    """FCD via fcd_torch when installed; ``nan`` (with a warning) otherwise.

    FCD is optional because it pulls in a pretrained ChemNet checkpoint that
    is awkward on air-gapped clusters. When it is unavailable the property
    Wasserstein/KL distances below carry the distribution-matching claim, and
    the evaluation JSON records ``fcd_available: false`` so a table can never
    silently omit it without the reader knowing.
    """
    try:
        from fcd_torch import FCD
    except Exception:
        warnings.warn('fcd_torch not installed — FCD reported as nan. '
                      'pip install fcd_torch to enable.')
        return float('nan'), False
    gen = [s for s in gen_smiles if s]
    ref = [s for s in ref_smiles if s]
    if len(gen) < 2 or len(ref) < 2:
        return float('nan'), True
    try:
        return float(FCD(device=device, n_jobs=1)(ref=ref, gen=gen)), True
    except Exception as exc:  # pragma: no cover - depends on optional weights
        warnings.warn(f'FCD computation failed: {exc}')
        return float('nan'), True


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------

def evaluate_molecules(mols, train_smiles, reference_mols=None,
                       compute_fcd=False, seed=0, device='cpu'):
    """Score a list of generated Mols against a training reference.

    Parameters
    ----------
    mols : list of Mol or None
        Exactly the molecules generated — including the failures. The
        denominator of ``validity`` is ``len(mols)``, so callers must not
        pre-filter.
    train_smiles : sequence of str
        Canonical SMILES of the TRAINING split only.
    reference_mols : sequence of Mol, optional
        Reference molecules for distribution distances and SNN. Defaults to
        parsing ``train_smiles`` (sub-sampled).
    compute_fcd : bool
        Attempt FCD (needs fcd_torch).

    Returns
    -------
    dict of metric name -> float / int
    """
    rng = np.random.default_rng(seed)
    n_total = len(mols)
    out = {'n_generated': n_total}
    if n_total == 0:
        return {**out, **{k: float('nan') for k in HEADLINE_KEYS}}

    train_set = set(train_smiles)

    valid_mols, clean_mols = [], []
    valid_smi, clean_smi = [], []
    for m in mols:
        if not is_valid(m):
            continue
        smi = canonical_smiles(m)
        if smi is None:
            continue
        valid_mols.append(m)
        valid_smi.append(smi)
        if is_clean_valid(m):
            clean_mols.append(m)
            clean_smi.append(smi)

    n_valid, n_clean = len(valid_mols), len(clean_mols)
    out['n_valid'] = n_valid
    out['n_clean_valid'] = n_clean
    out['validity'] = n_valid / n_total
    out['clean_validity'] = n_clean / n_total

    out['uniqueness'] = len(set(valid_smi)) / n_valid if n_valid else float('nan')
    out['uniqueness_clean'] = len(set(clean_smi)) / n_clean if n_clean else float('nan')
    out['novelty'] = (sum(s not in train_set for s in valid_smi) / n_valid
                      if n_valid else float('nan'))
    out['novelty_clean'] = (sum(s not in train_set for s in clean_smi) / n_clean
                            if n_clean else float('nan'))

    # Properties are reported over the CLEAN-valid set: averaging QED over
    # fragment soup is what let the v1 pipeline claim drug-likeness for
    # molecules that were mostly wildcards.
    props_clean = raw_properties(clean_mols)
    props_valid = raw_properties(valid_mols)
    for key in PROPERTY_KEYS:
        vals = props_clean[key]
        out[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float('nan')
        out[f'{key}_std'] = float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else float('nan')
        vv = props_valid[key]
        out[f'{key}_over_valid'] = float(np.nanmean(vv)) if np.any(np.isfinite(vv)) else float('nan')

    # Diversity over clean-valid uniques (duplicates would inflate IntDiv's
    # denominator with zero-distance pairs and understate collapse).
    uniq_clean_mols, seen = [], set()
    for m, s in zip(clean_mols, clean_smi):
        if s not in seen:
            seen.add(s)
            uniq_clean_mols.append(m)
    gen_fps = morgan_fps(uniq_clean_mols)
    out['int_div1'] = internal_diversity(gen_fps, p=1, rng=rng)
    out['int_div2'] = internal_diversity(gen_fps, p=2, rng=rng)
    n_scaff, scaff_div = scaffold_stats(clean_mols)
    out['n_scaffolds'] = n_scaff
    out['scaffold_diversity'] = scaff_div

    # Reference-dependent metrics.
    if reference_mols is None:
        ref_smi = list(train_smiles)
        if len(ref_smi) > 5000:
            ref_smi = [ref_smi[i] for i in rng.choice(len(ref_smi), 5000, replace=False)]
        reference_mols = [Chem.MolFromSmiles(s) for s in ref_smi]
        reference_mols = [m for m in reference_mols if m is not None]
    ref_fps = morgan_fps(reference_mols)
    out['snn_to_train'] = nearest_neighbour_similarity(gen_fps, ref_fps, rng=rng)

    ref_props = raw_properties(reference_mols)
    for key in PROPERTY_KEYS:
        out[f'w1_{key}'] = wasserstein1(props_clean[key], ref_props[key])
        out[f'kl_{key}'] = kl_divergence(props_clean[key], ref_props[key])

    if compute_fcd:
        fcd, avail = frechet_chemnet_distance(
            clean_smi, [canonical_smiles(m) for m in reference_mols], device=device)
        out['fcd'] = fcd
        out['fcd_available'] = avail
    else:
        out['fcd'] = float('nan')
        out['fcd_available'] = False

    return out


def uniqueness_vs_sample_size(all_smiles, sizes=(16, 64, 256, 1000, 5000), seed=0):
    """Uniqueness measured at several generated-sample sizes from one pool.

    ``all_smiles`` must be one entry per GENERATED graph, with ``None`` for
    the invalid ones, so that a size-n draw reproduces what "generate n
    molecules and score them" would have given.

    Uniqueness is monotonically non-increasing in n, so it is only comparable
    across models at a FIXED n. This produces the curve that documents that
    (and that explains the withdrawn 73% figure, which was read off training
    batches of 16 and compared against a classical number measured at 5000).
    """
    rng = np.random.default_rng(seed)
    pool = list(all_smiles)
    curve = {}
    for n in sizes:
        if len(pool) < n:
            curve[n] = float('nan')
            continue
        sub = [pool[i] for i in rng.choice(len(pool), n, replace=False)]
        valid = [s for s in sub if s]
        curve[n] = len(set(valid)) / len(valid) if valid else float('nan')
    return curve
