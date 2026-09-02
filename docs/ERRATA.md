# Errata: measurement artifacts in the v1 pipeline

An audit of the earlier analysis (`results/manuscript_v1/`) found that three of
its headline numbers were measurement artifacts rather than findings. Each is
reproducible in one command against the v1 checkpoint that produced it:

```bash
python scripts/reproduce_errata.py \
    --model_dir results/quantum/ablation_300/train/model_dir --epoch 113
```

All numbers below are that script's output on 1000 graphs from epoch 113 of the
Quantum + Ablation B run.

---

## A. "Quantum clean-validity = 0.000" — a decoder padding artifact

**Claimed:** the pure quantum and Quantum + Ablation B models reach 0.000
clean-validity, i.e. "every molecule passing RDKit sanitisation is chemically
fragmented" — presented as the strongest evidence for the clean-validity metric
and as a near-total failure mode of quantum generation.

**Actually:** MolGAN graphs have a fixed 9 vertices, but QM9/qm9_5k molecules
average 6.5 heavy atoms, so ~2.5 vertex slots are PAD in a *correct*
generation. The decoder in `data/sparse_molecular_dataset.py` materialised
every PAD slot as an RDKit atom of atomic number 0 — a `*` dummy — and left it
unbonded. Any molecule smaller than 9 heavy atoms therefore canonicalised to
something like `*.*.CC1CC(CO)C1`: both a wildcard and a disconnected fragment,
so clean-validity was 0 by construction.

Decoding the *same generated graphs* both ways:

| decoder | validity | clean-validity | example SMILES |
|---|---|---|---|
| v1 (PAD kept) | 0.845 | **0.000** | `*.*.CC1CC(CO)C1` |
| fixed (unbonded PAD dropped) | 0.845 | **0.630** | `CC1CC(CO)C1` |

Validity is identical; only the wildcard/fragment test changes. The molecules
were chemically fine the whole time.

**The compounding error:** the classical baseline was trained on full GDB9,
where molecules fill nearly all 9 slots, so it suffered far less padding and
scored 0.630 clean-validity. The v1 diagnostic contrast — "classical 0.630 vs
quantum 0.000" — was therefore *a dataset difference read as a chemistry
difference*. Under the fixed decoder and a shared dataset, the contrast
disappears.

**Fixed in:** `data/sparse_molecular_dataset.py::matrices2mol` (drops unbonded
PAD slots only; a PAD slot the model actually bonded to is a real generation
error and still counts as unclean). Pinned by
`tests/test_metrics.py::test_padded_graph_decodes_to_a_connected_molecule` and
`::test_bonded_pad_slot_stays_invalid`.

---

## B. "SA = 0.410" — the normalised reward score reported as an SA score

**Claimed:** Quantum + Ablation B achieves "SA = 0.410", in a table footnoted
"SA: lower is better".

**Actually:** the training loop logged `all_scores(..., norm=True)`, whose SA
entry is `clip((5 - SA) / 3.5, 0, 1)` — a reward-space score in [0,1] where
*higher* is better. The Ertl SA scale runs 1-10 where *lower* is better. The
two run in opposite directions, so 0.410 is not a small SA; it corresponds to a
raw SA of 5 - 0.410 x 3.5 = **3.57**.

At the same checkpoint: raw SA 3.117, normalised SA 0.539.

**Fixed in:** `qmolgan/chem.py` reports raw QED/logP/SA/MW only; normalisation
is confined to `qmolgan/rewards.py`. Pinned by
`tests/test_metrics.py::test_reported_sa_is_the_raw_ertl_scale`.

---

## C. "Uniqueness 0.56% -> 73.0%, a 130-fold improvement" — different sample sizes

**Claimed:** v1's single most prominent result, and the basis for claiming a
4.8x improvement over the 15.25% uniqueness reported by Kao et al.

**Actually:** the 73.0% came from the training loop's score block, which ran
every 10 steps on the *training batch of 16 molecules*. The classical numbers
it was compared against were measured on 5000 generated molecules. Uniqueness
is monotonically non-increasing in the sample size, so the two are not
comparable at all. From one pool of 1000 generated molecules at epoch 113:

| n | 16 | 64 | 256 | 1000 |
|---|---|---|---|---|
| uniqueness | **0.923** | 0.556 | 0.199 | **0.082** |

The left column is where v1 measured the quantum model; the right column is
where it measured the classical one.

Under the honest offline pipeline the same run peaks at uniqueness **0.209**
(epoch 135, n = 500) and reaches **0.171** at the epoch 141 that v1 reported as
0.730. There is no 130-fold effect. Whether reward shaping improves quantum
uniqueness at all is now an open question that the T1 factorial is designed to
answer.

**Fixed in:** `solver.py` validates on a fixed, seeded sample of `--val_n`
(default 1000) molecules per epoch and records `n_sampled` in `history.csv`;
`qmolgan/protocol.py` fixes n = 5000 for every reported number; every
`results.json` carries a `uniqueness_curve`. Pinned by
`tests/test_metrics.py::test_uniqueness_decreases_with_sample_size`.

---

## D. Secondary defects found during the audit

| # | Defect | Where | Effect |
|---|---|---|---|
| D1 | Property means taken over `np.array(v)[np.nonzero(v)]` | v1 `solver.py` score block | every molecule scoring exactly 0 was silently dropped, biasing all reported means upward |
| D2 | Novelty measured against `data.smiles` (train+val+test) | `MolecularMetrics.novel_scores` | held-out molecules counted as novel; novelty inflated. Now measured against the training split only |
| D3 | Quantum circuit weights appended to `molgan_red_weights.csv` and read positionally | v1 `solver.py`, `find_best_epoch.py` | resumed runs appended duplicate rows, so epoch *k* loaded some other epoch's circuit. Weights are now `{epoch}-Z.ckpt` state dicts |
| D4 | Checkpoints saved twice per epoch (train pass and val pass) | v1 `solver.py` | duplicated the CSV rows that caused D3 |
| D5 | Evaluation scripts hardcoded `z_dim`/`g_conv_dim` | v1 `find_best_epoch.py` (`Z_DIM=4, G_CONV_DIM=[16]`) vs the "quantum" metrics file recording `z_dim=8, g_conv_dim=[128,256,512]` | at least one reported quantum row was generated with a mismatched architecture. Architecture now comes from each run's `config.json` |
| D6 | `generate_figures.py` hardcoded the table values | v1 figures | a wrong table produced matching wrong figures with nothing to flag it. Figures are now built only from `results.json` |
| D7 | A reward preset set the `rw_*` weights but not `lambda_wgan` | v1 `main.py` | with `lambda_wgan = 1.0` the RL term is multiplied by zero, so the preset is a silent no-op. Presets now carry `lambda_wgan` |
| D8 | Checkpoint selection rule chosen after seeing results | v1 analysis (epoch 141 for one model, 30 for another) | pre-registered rule + disjoint selection/report noise streams in `qmolgan/protocol.py` |

## What survives from v1

* **The clean-validity metric itself is a good idea** and is kept — it just has
  to be computed after padding removal, which is what makes it measure
  chemistry rather than graph size.
* **The WGAN-GP sign correction** is real and is preserved in the rewritten
  `solver.py`.
* **The reward-shaping ablation design** is sound; it now needs multi-seed
  results and a quantum arm.

## What does not survive

* "Superior drug-likeness and 100% novelty" for the quantum model — measured on
  a different dataset from the classical baseline, with inflated novelty (D2).
* The 130-fold uniqueness recovery, and with it the comparison against Kao et
  al.'s 15.25%.
* The "near-total failure mode in quantum generation" reading of clean-validity.
* Every quantum-advantage statement: the VQC differed from the classical
  baseline in latent dimension, support, intrinsic rank, generator width,
  dataset and epoch count simultaneously.

## The classical baseline is affected too

The v1 classical numbers are not a valid reference point for new work. Its
0.630 clean-validity is also padding-affected (less so, because GDB9 molecules
fill more slots), so **every row of both v1 tables must be regenerated**.
Tiers T1 and T3 in [EXPERIMENTS.md](EXPERIMENTS.md) do that.
