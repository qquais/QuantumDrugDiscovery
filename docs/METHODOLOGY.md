# Methodological requirements and how each is met

The v1 audit ([ERRATA.md](ERRATA.md)) surfaced a set of methodological gaps.
This is the checklist of what each one requires and where it is now handled.
"Run" columns refer to tiers in [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Clean-validity must measure chemistry, not graph size

v1's 0.000 was a decoder padding artifact, and the classical/quantum contrast
it supported was really a dataset difference. See [ERRATA.md §A](ERRATA.md).
Fixed in the decoder; the same checkpoint now scores 0.630 clean-validity. All
models train on the same dataset, so the contrast is meaningful in either
direction.

## 2. Results must be multi-seed with reported variance

Every tier is multi-seed (5 seeds for the T1 headline, 3 for ablations, 2 for
sweeps). `aggregate.py` reports mean ± std, a percentile bootstrap CI, and
`n_seeds` in every row; `to_latex` prints the seed count as a table column so a
single-seed number cannot be typeset as a mean. Conclusions come from
**seed-paired** differences with bootstrap CIs (`configs/pairs.json`, 15
pre-specified contrasts), which removes seed variance instead of being swamped
by it.

## 3. Classical and quantum arms must share training conditions

v1 differed in dataset (GDB9 vs qm9_5k), epochs (300 vs 30 vs 141), latent
dimension (8 vs 4), generator width ([128] vs [16]) and evaluation sample size,
simultaneously. `scripts/build_manifest.py` fixes dataset, epochs, batch size,
`n_critic`, optimiser and learning rates across every run in T1-T6; only the
named factor varies. `qmolgan/protocol.py` fixes n = 5000 for every reported
number.

## 4. Reward shaping and latent source must be separable

Otherwise any observed gain is unattributable. **T1** is a latent × reward
factorial: the reward effect is measured *within* each latent and the quantum
effect *within* each reward. If the reward effect is the same in both, the
paper says so.

## 5. The entanglement / expressibility mechanism must be measured, not asserted

`analyze_latent.py` measures it, with no training required:

* The Kao circuit draws **two** random scalars per sample and encodes them on
  every wire, so its outputs lie on a 2-D manifold whatever the qubit count.
  Measured participation ratio **1.19** vs 3.99 for a 4-D Gaussian; latent-space
  coverage **2.3%** vs 33.5%. This is a measured mechanism for mode collapse,
  replacing v1's hand-waved "compressed latent space".
* Growing 4 → 8 qubits raises effective rank only 3/4 → 3/8: capacity is
  bounded by the *encoding*, not the qubit count.
* Removing the CNOTs drives mean single-qubit von Neumann entropy to exactly
  **0.000** (product state) vs 0.257 with them, so `vqc_noent` (T2) is a clean
  entanglement ablation.

## 6. "Quantum vs classical" needs matched classical controls

Three controls, each matching the VQC on one axis: `uniform` (bounded
support), `rank2` (intrinsic rank 2, no trainable parameters), and `trig` (a
trainable trigonometric polynomial — the *same function class* a Pauli-Z
expectation of this circuit family belongs to, with a matched parameter
budget). Run in **T2** under the identical reward.

## 7. The metric suite must cover diversity and distribution matching

`qmolgan/chem.py` computes, for every reported model: validity,
clean-validity, uniqueness, **uniqueness among clean-valid**, novelty,
**novelty among clean-valid**, raw QED/logP/SA/MW with std, **IntDiv1/IntDiv2**,
**Bemis-Murcko scaffold count and diversity**, **nearest-neighbour Tanimoto
similarity to the training set (SNN)**, **Wasserstein-1 and KL divergence** of
each property against the training distribution, and **FCD** when `fcd_torch`
is installed (with `fcd_available` recorded either way, so a table cannot
silently omit it).

## 8. "100% novelty" must be interpretable

Three fixes: novelty is measured against the **training split only** (v1 used
train+val+test, inflating it), `novelty_clean` restricts it to clean-valid
molecules, and **SNN** is reported alongside so a high novelty figure can be
read against how close the molecules actually are to training chemistry.

## 9. Checkpoint selection must be pre-registered

`qmolgan/protocol.py` fixes the rule (`argmax clean_validity ×
uniqueness_clean`), an epoch floor at 10% of training, and **two disjoint noise
streams** — selection at n = 1000, reporting at n = 5000 — so a model is never
selected and reported on the same molecules. Every model additionally reports
its **final** epoch as a selection-free row.

## 10. Property scales must be stated and consistent

v1's "SA = 0.410" was the normalised reward-space score. See
[ERRATA.md §B](ERRATA.md). Raw values only in reports now, pinned by a test.

## 11. Uniqueness must be compared at fixed n

v1's 73% came from batches of 16. See [ERRATA.md §C](ERRATA.md). Uniqueness at
n = 16 is 0.923 and at n = 1000 is 0.082 for the same checkpoint. Every
`results.json` now carries a `uniqueness_curve` so the dependence is visible.

## 12. Reward weights need a sensitivity analysis

**T6** sweeps `rw_unique` (6 points) and `rw_fragment_penalty` (5 points) one
at a time, renormalising the remaining components so the sweep varies the
*balance* rather than the total scale (`rewards.weight_sweep_grid`).
`make_figures.py::fig_quality_diversity` plots the resulting quality-diversity
front. Two new presets probe the ends: `ablation_d_clean`
(clean-validity-directed) and `ablation_e_diverse` (diversity-dominant).

## 13. Claims must be scoped to what the representation supports

MolGAN's fixed 9-vertex graph **cannot represent** ZINC or ChEMBL molecules, so
the honest move is to scope every claim to QM9-scale 9-heavy-atom chemistry and
add **T7**: the same headline conditions on the full 133k-molecule GDB9 set.
State the representation limit explicitly as a limitation rather than listing
larger datasets as future work.

## 14. Positioning against external baselines

The study's subject is the VQC-vs-classical *noise source* within one fixed
architecture, so the controlled comparison is against matched classical latents
(T1/T2) — a stronger internal control than v1 had. Against external baselines,
report the published QM9 numbers for MolGAN, ORGAN, GraphAF and MOSES-suite
models in a related-work table, computed on the same metrics where the
published numbers allow it, and be explicit that they are quoted rather than
re-run. Do not claim to beat them.

## 15. Reproducibility

The whole pipeline is four commands (`main.py`, `evaluate_run.py`,
`aggregate.py`, `make_figures.py`) over one package (`qmolgan/`). `.gitignore`
commits every small result artifact — `config.json`, `history.csv`,
`epoch_sweep.csv`, `results.json`, generated SMILES, tables, figures — and
ignores only large regenerable files. The experiment grid is version-controlled
data (`configs/manifest.csv`), 19 unit tests pin the metric definitions, and
`scripts/reproduce_errata.py` reproduces the v1 defects from a v1 checkpoint in
one command.

## 16. "Best overall" must be a supported claim

v1's Ablation B improved validity, novelty, QED and SA but *lowered* both
clean-validity and uniqueness relative to the baseline, so "best overall" was
not unqualified. With multi-seed paired contrasts, "best" is only claimable
where the CI excludes zero; otherwise report the quality-diversity front and
let the tradeoff be the finding.

## 17. Molecule figures must show what they claim to show

`evaluate_run.py` writes only **clean-valid** SMILES to `smiles_*.txt`, and the
solver draws only clean-valid molecules to `img_dir`. The v1 grids are
quarantined in `figures/superseded_v1/`.
