# Experimental / not part of the reported pipeline

Code here is exploratory and is **not** wired into `main.py`, `solver.py` or
any reported result. It is kept because it represents real work and may become
a follow-up, but nothing in this directory has been through the evaluation
protocol in `qmolgan/protocol.py`.

## `q_discriminator.py`

Quantum discriminator variants: a 3-stage hierarchical `HybridModel` and
`KaoQuantumDisc` (9-qubit amplitude embedding + `StronglyEntanglingLayers`,
following Kao et al. 2023). The v1 solver could swap the classical critic for
these via a `--use_quantum_disc` flag.

It is unwired in the rewritten solver on purpose. A quantum discriminator is a
*second* independent variable, and the current question — does the VQC noise
source do anything a matched classical latent does not — has to be settled
before adding another. Wiring it back in means:

1. adding a `--critic {classical,quantum_kao,quantum_hier}` flag,
2. a matched classical critic of the same parameter count as the control,
3. its own tier in `scripts/build_manifest.py` with the same seed budget,
4. checking the WGAN-GP objective still makes sense — the v1 path skipped the
   gradient penalty entirely for the quantum critic and substituted weight
   clamping to +-0.5, which is WGAN, not WGAN-GP, and so is not comparable to
   the classical rows without saying so.
