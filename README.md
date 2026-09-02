# Classical and Quantum MolGAN: a controlled study of latent source vs reward shaping

Molecular graph generation on QM9 with a MolGAN/WGAN-GP backbone, comparing a
variational-quantum-circuit noise source against matched classical latent
sources, under identical training and a pre-registered evaluation protocol.

> **Status.** An audit of the earlier pipeline (v1, written up in
> `results/manuscript_v1/`) found that three of its headline numbers were
> measurement artifacts. They are documented and reproduced in
> **[docs/ERRATA.md](docs/ERRATA.md)**. This repository is the rebuilt
> pipeline; **no result from v1 should be reused.**

## Quick start

```bash
conda env create -f environment.yml && conda activate molgan-pt
python -m pytest tests/ -q                 # 19 tests pinning the metric definitions

python analyze_latent.py                   # latent-space analysis, no training needed
python main.py --saving_dir results/runs/demo --latent vqc --z_dim 4 --qubits 4 \
    --layer 3 --g_conv_dim '[16]' --reward_preset ablation_b --qc_lr 0.04 --seed 42
python evaluate_run.py --run_dir results/runs/demo
python aggregate.py --results_root results/runs --out_dir results/tables
python make_figures.py
```

Full cluster instructions, the 120-run experiment grid and its cost table are
in **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**; the methodological
requirements each part of the design satisfies are in
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Datasets

Neither dataset is in git. GDB-9 downloads with the helper script; both are then
preprocessed into the sparse format the pipeline loads.

```bash
cd data && bash download_dataset.sh          # GDB-9 + the NP/SA score tables
python sparse_molecular_dataset.py           # writes *.sparsedataset
```

`data/qm9_5k.smi` (the 5,000-molecule QM9 subset used for the main comparison)
ships with the repository. The preprocessing script's `main` block selects which
dataset to build; edit it to choose between `gdb9_9nodes` and `qm9_5k`.

Two files are produced and referenced throughout:

| file | molecules | used by |
|---|---|---|
| `data/qm9_5k_py37.sparsedataset` | 4,994 | the T1-T6 comparison |
| `data/gdb9_9nodes.sparsedataset` | ~133k | the T7 scale check |

## What the pipeline guarantees

Every defect in v1 came from the same root cause: a metric could be computed
one way during training and a different way during evaluation, and a table
could disagree with both. The structure here removes that possibility.

* **One metric implementation.** `qmolgan/chem.py` is the only place validity,
  clean-validity, uniqueness, novelty and the properties are defined. Training,
  the epoch sweep and the final evaluation all call it.
* **Runs are self-describing.** `main.py` writes `config.json` (dataset, latent
  source, dimensions, generator width, seed, reward weights) before training.
  Offline tools read the architecture from there and never hardcode it.
* **Selection and reporting use disjoint noise streams.** A checkpoint is
  chosen on 1000 molecules from the selection stream and reported on 5000 from
  the report stream, under a rule fixed in `qmolgan/protocol.py` before any
  results are seen. Every model also reports its final epoch.
* **Raw properties only in reports.** QED/logP/SA/MW are reported on the Ertl
  and Crippen scales. Normalisation exists solely inside the RL reward.
* **Figures are built from results.** `make_figures.py` reads `results.json`
  and nothing else; a figure whose inputs are missing is skipped, not defaulted.
* **Nothing is single-seed by accident.** `aggregate.py` prints `n_seeds` in
  every row and every LaTeX table.

## Layout

```
qmolgan/                package shared by training and evaluation
  chem.py               decoding + the canonical metric suite
  latent.py             latent sources: gaussian, uniform, rank2, trig, vqc, vqc_noent
  rewards.py            reward presets and the weighted multi-objective reward
  protocol.py           pre-registered selection rule, noise streams, aggregation stats
  generate.py           run dir + epoch -> reproducible molecule sampling

main.py                 train one variant (fully CLI-driven)
solver.py               WGAN-GP training loop
evaluate_run.py         sweep -> select -> report, under the protocol
aggregate.py            multi-seed tables, bootstrap CIs, seed-paired contrasts
analyze_latent.py       latent geometry and circuit entanglement (no training needed)
make_figures.py         all figures, from results only

configs/manifest.csv    the experiment grid, as version-controlled data
configs/pairs.json      the 15 pre-specified paired contrasts
scripts/                manifest builder, SLURM array drivers, errata reproduction
tests/                  19 regression tests, one per known defect
docs/                   ERRATA, EXPERIMENTS, METHODOLOGY, PAPER_PLAN
solver_legacy.py        the v1 training loop, kept only for reference
```

## The six latent sources

The point of the study is that "quantum vs classical" is not one comparison but
four, and v1 varied all of them at once. Each control matches the VQC on
exactly one axis:

| latent | support | intrinsic rank | trainable | isolates |
|---|---|---|---|---|
| `gaussian` | unbounded | d | no | the MolGAN default |
| `uniform` | [-1,1]^d | d | no | boundedness |
| `rank2` | [-1,1]^d | 2 | no | low-rank stochasticity |
| `trig` | [-1,1]^d | 2 | yes | the circuit's *function class* |
| `vqc` | [-1,1]^q | 2 | yes | the quantum circuit itself |
| `vqc_noent` | [-1,1]^q | 2 | yes | the circuit without entanglement |

`trig` is not a loose analogy: a Pauli-Z expectation of this circuit family is
a finite trigonometric polynomial in the two encoded angles, so it is a
classical model of the same function class with a matched parameter budget.

## Artifact release

Code, configs, the experiment manifest, per-run `config.json` / `history.csv` /
`epoch_sweep.csv` / `results.json`, generated SMILES, tables and figures are all
committed. Model checkpoints are excluded from git (a 300-epoch run writes ~850
files) and should be released as a separate archive alongside the paper.

## Citation of prior work

The quantum noise-generator circuit follows Kao et al., *Exploring the
advantages of quantum generative adversarial networks in generative chemistry*,
JCIM 63:3307-3318, 2023. The backbone follows De Cao & Kipf, *MolGAN*, 2018,
with WGAN-GP from Gulrajani et al., 2017.

## Credits

This repository builds on:

- [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
- [ZhenyueQin/Implementation-MolGAN-PyTorch](https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch)
- [jundeli/quantum-gan](https://github.com/jundeli/quantum-gan)
- [pykao/QuantumMolGAN-PyTorch](https://github.com/pykao/QuantumMolGAN-PyTorch)
