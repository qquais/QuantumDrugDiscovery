"""The pre-registered evaluation protocol.

Every comparability problem found in the v1 audit traces back to the same
root cause: different models evaluated under different conditions --
different datasets, different epochs, different sample sizes, and one row
taken from a training-time log instead of the offline pipeline. This module
fixes all of that in one place, and every reported number must come through
it.

The protocol, stated before any results are looked at:

1. **Identical training conditions.** Classical and quantum variants share the
   dataset, the number of epochs, batch size, n_critic, optimiser and
   learning rates. Only the axis under study differs.

2. **Two disjoint noise streams per run.** Checkpoint selection uses the
   *selection* stream (``N_SELECT`` molecules, seed ``SELECT_SEED_BASE +
   run_seed``). Every reported number uses the *report* stream (``N_REPORT``
   molecules, seed ``REPORT_SEED_BASE + run_seed``). Selecting and reporting
   on the same molecules is what makes a "best epoch" number optimistic.

3. **One fixed selection rule for all models**, declared here and never
   varied per model:

       score(epoch) = clean_validity x uniqueness_clean

   Both factors are measured on the selection stream. The product is used
   because either factor alone is degenerate: an early collapsed checkpoint
   reaches clean_validity ~ 1.0 by emitting one molecule forever, and a late
   unstable one reaches high uniqueness by emitting noise. Epochs below
   ``MIN_SELECTABLE_EPOCH`` are excluded so that a lucky epoch-2 checkpoint
   cannot win a 300-epoch comparison.

4. **A selection-free row too.** Every model additionally reports its FINAL
   epoch on the report stream. If a conclusion only holds at the selected
   epoch and not at the final one, that is stated rather than hidden.

5. **Uniqueness is only ever compared at equal n.** ``N_REPORT`` is fixed at
   5000 for every model in every table. `chem.uniqueness_vs_sample_size`
   documents the dependence for the appendix.
"""

import numpy as np

# Sample sizes. N_SELECT is smaller because it is paid once per epoch across
# a 300-epoch sweep; N_REPORT is paid once per model.
N_SELECT = 1000
N_REPORT = 5000

# Disjoint RNG streams. Kept far apart so that no arithmetic on run seeds can
# make a selection stream collide with a report stream.
SELECT_SEED_BASE = 10_000
REPORT_SEED_BASE = 20_000
LATENT_ANALYSIS_SEED_BASE = 30_000

# An epoch earlier than this is not eligible for selection.
MIN_SELECTABLE_EPOCH_FRACTION = 0.10

SELECTION_RULE = 'argmax_epoch clean_validity * uniqueness_clean (selection stream)'


def select_seed(run_seed):
    return SELECT_SEED_BASE + int(run_seed)


def report_seed(run_seed):
    return REPORT_SEED_BASE + int(run_seed)


def latent_seed(run_seed):
    return LATENT_ANALYSIS_SEED_BASE + int(run_seed)


def selection_score(row):
    """The pre-registered scalar. ``row`` is any mapping with the two keys."""
    cv = row.get('clean_validity', float('nan'))
    uq = row.get('uniqueness_clean', float('nan'))
    if cv is None or uq is None:
        return float('nan')
    cv, uq = float(cv), float(uq)
    if not np.isfinite(cv) or not np.isfinite(uq):
        return 0.0
    return cv * uq


def min_selectable_epoch(n_epochs):
    return max(1, int(round(MIN_SELECTABLE_EPOCH_FRACTION * n_epochs)))


def select_best_epoch(sweep_rows, n_epochs=None):
    """Apply the pre-registered rule to a list of per-epoch metric dicts.

    Returns ``(best_row, scored_rows)``. ``scored_rows`` carries the
    ``selection_score`` and ``eligible`` fields so the choice is auditable
    from the saved CSV without re-deriving anything.
    """
    rows = [dict(r) for r in sweep_rows]
    if not rows:
        raise ValueError('no epochs to select from')
    if n_epochs is None:
        n_epochs = max(int(r['epoch']) for r in rows)
    floor = min_selectable_epoch(n_epochs)

    for r in rows:
        r['selection_score'] = selection_score(r)
        r['eligible'] = int(r['epoch']) >= floor

    eligible = [r for r in rows if r['eligible'] and np.isfinite(r['selection_score'])]
    if not eligible:
        # Degenerate run (e.g. nothing clean-valid anywhere). Fall back to the
        # final epoch and say so, rather than silently picking epoch 1.
        best = max(rows, key=lambda r: int(r['epoch']))
        best = {**best, 'selection_fallback': 'no_eligible_epoch_final_used'}
        return best, rows

    best = max(eligible, key=lambda r: (r['selection_score'], -int(r['epoch'])))
    return dict(best), rows


def protocol_metadata(run_seed, n_epochs):
    """The block stamped into every results JSON so a table can be audited."""
    return {
        'protocol_version': '2.0',
        'selection_rule': SELECTION_RULE,
        'n_select': N_SELECT,
        'n_report': N_REPORT,
        'select_seed': select_seed(run_seed),
        'report_seed': report_seed(run_seed),
        'min_selectable_epoch': min_selectable_epoch(n_epochs),
        'run_seed': int(run_seed),
        'n_epochs': int(n_epochs),
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def mean_std(values):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float('nan'), float('nan'), 0
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0, int(vals.size)


def bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean. Reported alongside mean +- std
    because with 3-5 seeds a std alone invites over-reading."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size < 2:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_difference(values_a, values_b, n_boot=10_000, alpha=0.05, seed=0):
    """Seed-paired difference A - B with a bootstrap CI.

    Runs are paired by seed, so this compares like with like and removes the
    seed-to-seed variance that otherwise swamps a 3-seed comparison. Returns
    a dict with the mean difference, its CI, and whether the CI excludes 0.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError('paired comparison needs one value per seed on both sides')
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if d.size == 0:
        return {'n_pairs': 0, 'mean_diff': float('nan'), 'ci_low': float('nan'),
                'ci_high': float('nan'), 'excludes_zero': False}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    lo, hi = float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
    return {'n_pairs': int(d.size), 'mean_diff': float(d.mean()),
            'ci_low': lo, 'ci_high': hi, 'excludes_zero': bool(lo > 0 or hi < 0)}
