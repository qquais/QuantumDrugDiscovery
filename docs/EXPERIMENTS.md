# What to run on the cluster

Nothing in this file needs to run on your laptop. Everything is CPU-only —
the generator is ~8k parameters and PennyLane's `default.qubit` simulator runs
on CPU, so requesting GPUs only costs queue time.

## 0. One-time setup on the cluster

```bash
git clone <repo> && cd QuantumDrugDiscovery
conda env create -f environment.yml && conda activate molgan-pt
python -m pytest tests/ -q          # 19 tests, ~10 s. Must pass before anything else.

# Datasets (not in git). Copy or regenerate:
#   data/qm9_5k_py37.sparsedataset   — 4,994 molecules, main comparison
#   data/gdb9_9nodes.sparsedataset   — 133k molecules, T7 scale check
```

Sanity-check one epoch before submitting 120 jobs:

```bash
python main.py --saving_dir /tmp/smoke --latent vqc --z_dim 4 --qubits 4 \
    --layer 3 --g_conv_dim '[16]' --reward_preset ablation_b --qc_lr 0.04 \
    --num_epochs 1 --batch_size 256 --val_n 200 --seed 42
```

## 1. Latent-space analysis — run this first, it needs no training

```bash
python analyze_latent.py --out_dir results/latent_analysis
```

About 20 CPU-minutes. This is the cheapest strong result in the project: it
settles the latent-geometry question on its own, before a single GAN is
trained. Reference output from the current code:

| latent | participation ratio | eff. rank @99% | space coverage | mean 1-qubit vN entropy |
|---|---|---|---|---|
| gaussian | 3.99 | 4 / 4 | 33.5% | — |
| uniform | 4.00 | 4 / 4 | 39.5% | — |
| rank2 (classical control) | 1.53 | 4 / 4 | 2.8% | — |
| trig (classical surrogate) | 2.06 | 3 / 4 | 1.6% | — |
| **vqc** | **1.19** | 3 / 4 | **2.3%** | 0.257 |
| vqc_noent | 1.13 | 2 / 4 | 1.3% | **0.000** |

Two facts fall out immediately:

* The Kao circuit draws only two random scalars (z1, z2) per sample and encodes
  them on every wire, so its outputs live on a 2-D manifold whatever the qubit
  count. Growing 4 -> 8 qubits raises the effective rank only from 3/4 to 3/8.
  **Latent capacity is bounded by the encoding, not by the qubit count** — this
  reframes v1's vague "compressed latent space" as a measured, structural fact.
* Removing the CNOTs drives mean single-qubit von Neumann entropy to exactly
  0.000 (product state) while the entangling circuit sits at 0.257. The
  entanglement ablation is therefore clean: any difference between `vqc` and
  `vqc_noent` downstream is attributable to entanglement and nothing else.

## 2. Build the manifest and submit training

```bash
python scripts/build_manifest.py --out configs/manifest.csv
mkdir -p slurm_logs
sbatch --array=1-120%20 scripts/train_array.slurm
```

| tier | runs | core-hours | what it answers |
|---|---|---|---|
| T1 headline | 30 | 200 | latent {gaussian, uniform, vqc} x reward {none, ablation_b}, 5 seeds. Is the effect from the VQC or from the reward? |
| T2 mechanism | 12 | 102 | rank-2 and trigonometric classical controls, plus entanglement on/off |
| T3 reward-classical | 15 | 45 | the original Table I ablations, now with 3 seeds |
| T4 reward-quantum | 12 | 168 | the same ablations on the quantum model (v1 never ran these) |
| T5 circuit shape | 15 | 210 | qubits {2,4,6,8} x layers {1,3,6} |
| T6 weight sensitivity | 28 | 150 | one-at-a-time sweeps of `rw_unique` and `rw_fragment_penalty` |
| T7 scale | 8 | 192 | headline conditions on full GDB9 |
| **total** | **120** | **~1070** | |

Roughly 3 h per classical run and 14 h per quantum run at 300 epochs. Tasks
self-chain across walltime windows and resume from the newest complete
checkpoint quartet, so a 12 h limit is fine.

**If you need to cut scope, cut in this order:** T6 -> T5 -> T7 -> T4. Do not
cut T1 or T2 — they are the two tiers that carry the central claim, and T2 in
particular is what converts "we compared quantum to classical" into "we
isolated which property of the quantum latent matters".

```bash
python scripts/build_manifest.py --tiers T1 T2 --out configs/manifest_core.csv
MANIFEST=configs/manifest_core.csv sbatch --array=1-42%20 scripts/train_array.slurm
```

## 3. Evaluate

Each training task submits its own evaluation job on completion. To (re-)run
everything by hand:

```bash
for d in results/runs/*/; do python evaluate_run.py --run_dir "$d"; done
```

Per run: a selection sweep of 300 checkpoints x 1000 molecules, then two
reports of 5000 molecules. Budget ~1 h for classical, ~4 h for quantum. Add
`--epoch_stride 2` to halve it (and say so in the paper if you do).

Add `--fcd` for Frechet ChemNet Distance; it needs `pip install fcd_torch` and
downloads a ChemNet checkpoint, so do it on a node with network access. Without
it, `results.json` records `fcd_available: false` and the property Wasserstein
and KL distances carry the distribution-matching claim.

## 4. Tables and figures

```bash
python aggregate.py --results_root results/runs --list_conditions   # copy exact condition strings
python aggregate.py --results_root results/runs --out_dir results/tables --pairs configs/pairs.json
python aggregate.py --results_root results/runs --out_dir results/tables --report final
python make_figures.py --results_root results/runs --out_dir figures
```

`configs/pairs.json` holds the seed-paired contrasts the conclusions rest on.
Every contrast reports a mean difference, a bootstrap CI, and the number of
usable seed pairs.

## 5. The order to read results in

1. **T1 first, and read it before writing any claim.** The four cells you need
   are `vqc/none`, `vqc/ablation_b`, `gaussian/none`, `gaussian/ablation_b`.
   The reward effect is (ablation_b - none) *within* each latent; the quantum
   effect is (vqc - gaussian) *within* each reward. If the reward effect is
   similar in both latents, then reward shaping — not the VQC — is the story,
   and the paper should say so plainly. That is a perfectly publishable result
   and a far more defensible one than v1's.
2. **T2 next.** If `uniform` or `rank2` or `trig` reproduces the VQC's
   behaviour, the mechanism is boundedness / rank / function class, not
   quantumness. If `vqc_noent` matches `vqc`, entanglement is not doing the
   work. Either outcome is a real finding; report whichever one you get.
3. **T5** tells you whether the circuit is a meaningful design axis at all.
   Given the latent analysis above, expect the answer to be "barely" — which
   is itself worth stating, because it is the honest version of v1's claim that
   reward design matters as much as architecture.

## 6. Claims to write only if the data supports them

Do not reuse any v1 headline sentence. In particular:

* No "130-fold" / "4.8x over Kao et al." uniqueness claim without a
  fixed-n, multi-seed, paired comparison showing it.
* No "100% novelty" without stating that novelty is measured against the
  training split among clean-valid molecules.
* No "quantum advantage" phrasing unless T1 and T2 jointly isolate an effect
  that survives the matched classical controls.
* No SA number without stating the scale and direction.
* Scope the claims to QM9-scale 9-heavy-atom graphs. MolGAN's fixed 9-vertex
  representation cannot express ZINC or ChEMBL molecules, so the honest move
  is to narrow the claim and add T7 (full GDB9, 133k molecules) rather than
  to imply scale the representation does not support.
