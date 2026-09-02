# v1 reference numbers — DO NOT CITE

Small artifacts salvaged from the v1 result trees, kept so the audit in
`docs/ERRATA.md` stays checkable after the bulk checkpoint directories are
deleted.

**Every number in here is affected by at least one defect in ERRATA.md.**
Specifically:

* `metrics_corrected/`, `paper_results/`, `quantum_metrics/` — computed with
  the v1 decoder, so clean-validity is a graph-size measurement, not a
  chemistry one. `quantum_metrics/*epoch30*` also records
  `z_dim=8, g_conv_dim=[128,256,512]`, i.e. the *classical* architecture, while
  v1 reports that row as the quantum model (ERRATA D5).
* `quantum_ablationB_epoch_sweep.csv`, `best_epoch_summary.json`,
  `quantum_results.csv` — a partially-corrected re-sweep run after the padding
  fix but before the protocol existed: selection and reporting share a noise
  stream, novelty is measured against train+val+test, and the sweep and the
  5000-sample evaluation use different sample sizes (which is why epoch 113
  shows uniqueness 0.151 at n=500 and 0.021 at n=5000 for the same checkpoint).

Use `results/runs/*/eval/results.json` for anything that goes in a paper.
