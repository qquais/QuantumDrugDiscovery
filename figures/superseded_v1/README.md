# Superseded figures (v1)

These figures were produced by the deleted `generate_figures.py`, which carried
the v1 table values as Python literals. Every one of them encodes at least one
of the three measurement artifacts documented in `docs/ERRATA.md`:

* `radar_chart.png`, `ablation_comparison.png`, `quantum_training_curves.png`
  — built from the 0.000 clean-validity, the normalised "SA = 0.410", and the
  batch-of-16 "73% uniqueness".
* `molecules_grid.png`, `classical_molecules_grid.png` — drawn from molecules
  decoded with the padding bug, so structures may show spurious `*` fragments.
* `qed_hist.png`, `logp_hist.png`, `sa_hist.png` — property histograms over the
  un-filtered "valid" set rather than the clean-valid set.

Kept only so the v1 analysis can be reconstructed. **Do not reuse any of
them.** Regenerate everything with `python make_figures.py`.
