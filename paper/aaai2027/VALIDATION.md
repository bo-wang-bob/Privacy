# VEIL paper validation

Status: validated source, evidence, figure, and compiled PDF artifact.

## Experimental evidence

- The main matrix contains 54 unique runs: three datasets, seeds 42--44, and six
  mechanisms. Every selected run has 64 members and 64 non-members with exactly
  equal label histograms, all six attacks in the fixed order, finite metrics,
  and no audit errors.
- All Caltech101 and DTD formal runs set `require_cuda: true`; formal commands
  logged `cuda:0` or `cuda:1`. Flowers102 runs specify a GPU, and the seven new
  component runs additionally enforce `require_cuda: true`.
- The component study contains seven uniquely matched Flowers102 seed-42 runs.
  Its selector matches the complete formal configuration plus exactly one
  override, preventing earlier development sweeps from entering the figure.
- `run_level.csv` was independently regrouped to reproduce all 18 dataset-method
  aggregates and all 108 attack aggregates. The manuscript's FedAvg, private-
  mechanism, and privacy-accounting tables were then checked against these CSVs
  at their displayed precision.

## Artifact checks

- `python -m pytest -q -ra` reports `31 passed in 27.84s`; none requires a
  dataset or CLIP checkpoint.
- All changed Python modules pass `py_compile`, and `git diff --check` reports no
  whitespace errors.
- `python -m analysis_scripts.check_veil_paper` passes brace/environment,
  citation, AAAI package/command, anonymity, placeholder, three-stage method,
  grouped-table styling, figure, result-data, and reproducibility-checklist
  checks.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error veil.tex` produces the
  eight-page anonymous artifact (main text, references, and checklist) with no
  overfull boxes, unresolved citations, or unresolved references.
- Plotting code asserts every source typography setting is at least 25 points.
  Visual inspection covered the focused three-panel attack heatmap at original
  resolution. `pdffonts` reports embedded CID TrueType Tinos fonts and no Type 3
  fonts.
- Raw datasets, result directories, logs, checkpoints, and saved models remain
  excluded from Git. Only aggregate CSV evidence, the focused paper figure, and
  the compiled paper are retained.

## Interpretation limits

- With at most 64 evaluated non-members, TPR@1%FPR is a zero-false-positive
  operating point and is visibly quantized. Three seeds do not justify strong
  significance claims.
- VEIL is an empirical defense, not a differential-privacy mechanism. The
  reported conservative epsilon upper bounds are large at the selected noise
  settings.
- VEIL has the lowest cross-dataset mean attack TPR among the evaluated FedAvg
  defenses, but it loses utility and does not improve the privacy summaries on
  DTD. The manuscript reports this transfer failure explicitly.
