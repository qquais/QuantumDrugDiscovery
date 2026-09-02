# Plan for the next write-up

v1's five contributions were built on three measurement artifacts (see
[ERRATA.md](ERRATA.md)). Rather than patching the same claims, the honest and
more interesting paper is a different one, and the audit points straight at it.

## The problem with v1's framing

v1 asked "does a VQC noise generator beat classical Gaussian noise?" and
answered yes. But the two arms differed in **six** ways at once — dataset,
epoch budget, latent dimension, latent support, latent rank, and generator
width — plus the evaluation differed in sample size and checkpoint choice. No
"quantum advantage" statement can survive that.

## The paper the experiments now support

**Title direction:** something in the shape of *"What Does a Variational
Quantum Circuit Actually Contribute as a GAN Noise Source? A Controlled Study
on Molecular Graph Generation."*

**Thesis:** the reported benefits of a VQC noise generator in molecular GANs
are attributable to properties a classical latent can match — bounded support
and low intrinsic rank — and to reward shaping, and we show this by
constructing the matched controls and measuring the latent geometry directly.

This is a strong contribution whichever way the numbers land. If the VQC *does*
beat all three matched controls, the paper has the first properly-controlled
evidence for it. If it does not, the paper is the negative result the subfield
needs, with a measured mechanism instead of speculation. Either way it is
publishable, and it does not depend on a particular outcome — which is
precisely the property v1 lacked.

## Proposed contributions

**C1 — A measured account of VQC latent geometry.** The Kao-style circuit
draws two random scalars per sample and encodes them on every wire, so its
outputs lie on a 2-D manifold whatever the qubit count. Measured participation
ratio 1.19 vs 3.99 for a 4-D Gaussian; 2.3% vs 33.5% latent-space coverage;
effective rank rises only 3/4 → 3/8 when going from 4 to 8 qubits. **Latent
capacity is set by the encoding, not the qubit count.** This replaces v1's
hand-waved "compressed latent space" with a number, needs no GAN training, and
is a reusable diagnostic for anyone building VQC-based generative models.
(`analyze_latent.py`; already computed.)

**C2 — Matched classical controls that isolate what "quantum" contributes.**
Three controls, each matching one property of the VQC: bounded uniform
(support), rank-2 (intrinsic dimension), and a trigonometric surrogate (the
same function class — a Pauli-Z expectation of this circuit family *is* a
finite trigonometric polynomial in the encoded angles, so this is a model of
the circuit, not an analogy). Plus a clean entanglement ablation: removing the
CNOTs drives mean single-qubit von Neumann entropy to exactly 0.000 while the
entangling circuit sits at 0.257. (Tier T2.)

**C3 — A latent × reward factorial.** Reward shaping and latent source varied
independently under otherwise identical training, 5 seeds, seed-paired
bootstrap contrasts. This is the direct answer to "is it the VQC or the
reward?" (Tier T1.)

**C4 — Clean-validity, correctly defined.** The metric is a genuinely useful
idea; v1's implementation measured graph size. Report it after padding removal,
alongside uniqueness-among-clean-valid, scaffold diversity and SNN, and show
what each catches that standard validity does not. Include the padding trap
itself as a methodological warning — it is not unique to this codebase, and any
fixed-vertex graph generator evaluated on variable-size molecules can hit it.

**C5 — A reproducible evaluation protocol.** Pre-registered selection rule,
disjoint selection/report noise streams, fixed n, multi-seed, plus the
demonstration that uniqueness at n=16 is 0.923 where the same checkpoint gives
0.082 at n=1000. That last figure is worth publishing on its own: it shows how
easily a molecular-GAN diversity claim can be inflated by an order of magnitude
without anyone doing anything deliberately wrong.

## Structural changes from v1

| v1 | new |
|---|---|
| 5 contributions, 4 of them results-dependent | 5 contributions, 2 of which (C1, C5) hold regardless of how training turns out |
| "quantum advantage" framing | "what does the quantum component contribute" framing |
| 1 seed | 5 seeds (headline), 3 (ablations), seed-paired CIs |
| classical on GDB9, quantum on qm9_5k | one dataset for the comparison, GDB9 as a scale check |
| epochs 300 / 30 / 141 | one budget, one pre-registered selection rule, plus a selection-free final-epoch row |
| 7 metrics | 20+, including diversity, scaffold, SNN, distribution distances, FCD |
| "de novo drug discovery" claims | scoped to QM9-scale 9-heavy-atom graphs, with the representation limit stated |

## Things to write into the paper regardless of results

* **State the sample size next to every uniqueness number.** Then add the
  uniqueness-vs-n figure.
* **State the SA scale and direction** every time SA appears.
* **State that novelty is computed against the training split**, among
  clean-valid molecules, and report SNN beside it.
* **State the 9-vertex representation limit** in the limitations section, and
  that this is why ZINC/ChEMBL are out of scope rather than future work.
* **Report the final-epoch row**, not only the selected-epoch row.

## Related work to add

v1's related work was thin on the classical side and did not engage with the
evaluation-methodology literature, which is where its own failures live.

* MOSES (Polykovskiy et al.) and GuacaMol (Brown et al.) for the metric suite
  and for the argument that validity alone is not a quality measure.
* Renz et al., *On failure modes in molecule generation and optimization* —
  directly on point for the uniqueness/sample-size trap.
* For the quantum side, keep Kao et al., Anoshin et al., Mousavi et al., Li et
  al., and add the barren-plateau and expressibility literature (Sim et al. on
  expressibility/entangling capability of PQCs) since C1 is measuring exactly
  those quantities.

## Venue

Any IEEE venue with a quantum-computing or ML track. A controlled negative or
mechanism-focused result lands better at a venue with a methods/reproducibility
angle than at one expecting a benchmark win. Whichever you pick, submit the
code and the `results/` artifacts alongside: the repository is now in a state
where "is everything released?" can be answered "yes, all of it".
