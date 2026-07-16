# VEIL AAAI 2027 paper artifact

The primary source is `veil.tex`, written as a single-file anonymous submission
using the official AAAI 2027 author kit from <https://aaai.org/authorkit27/>
(retrieved 2026-07-16). The unmodified official `aaai2027.sty` and
`aaai2027.bst` files are included beside the manuscript, and the validated
compiled artifact is `veil.pdf`.

## Evidence contract

- The paper-level comparison uses Flowers102, Caltech101, and DTD; seeds
  42--44; 10 clients with full participation; Dirichlet alpha 0.1; 16 shots per
  class; five rounds; and one local epoch.
- Every formal run uses six attacks, 64 member and 64 non-member candidates,
  exact candidate-label histogram matching, and an audit-specific RNG.
- `configs/veil_multidataset.yaml` sets `require_cuda: true`, so a formal run
  fails rather than silently falling back to CPU.
- `analysis_scripts/veil_paper_results.py` rejects incomplete, unmatched, or
  pre-fairness runs and produces the reviewed CSV evidence in `evidence/`, including
  run-level metrics, seed aggregates, attack profiles, and conservative privacy-
  accounting ranges.
- Raw result directories, datasets, prompts, and model checkpoints remain
  excluded from Git.

## Chart map

| Paper section | Question | Form | Evidence | Output |
|---|---|---|---|---|
| Private-mechanism comparison | How does VEIL compare with DP-FPL and FedASK under every attack? | Three-panel annotated heatmap with one shared scale | `evidence/attack_aggregate.csv` | `figures/private_methods_attack_profile.pdf` |

The manuscript uses grouped tables for FedAvg results, private-mechanism
summaries, and component analysis because exact values are clearer than
additional chart forms.  The one retained diagnostic figure uses at least
25-point source typography, a zero-anchored shared scale, embedded serif fonts,
numeric annotations, and a non-color outline for VEIL.  At its full-width paper
placement, labels remain approximately 9 points or larger.

## Exact training commands

For each `DATASET` in `flowers caltech101 dtd` and each `SEED` in `42 43 44`,
run the following six commands. The YAML already enables the complete six-
attack audit and sets `require_cuda: true`; choose `--gpu 0` or `--gpu 1` for
the available device.

```bash
python main.py --config configs/veil_multidataset.yaml --dataset_name DATASET --seed SEED --gpu 0 --aggregator fedavg --defense none
python main.py --config configs/veil_multidataset.yaml --dataset_name DATASET --seed SEED --gpu 0 --aggregator fedavg --defense prompt_dp
python main.py --config configs/veil_multidataset.yaml --dataset_name DATASET --seed SEED --gpu 0 --aggregator fedavg --defense hamp
python main.py --config configs/veil_multidataset.yaml --dataset_name DATASET --seed SEED --gpu 0 --aggregator fedavg --defense veil
python main.py --config configs/veil_multidataset.yaml --dataset_name DATASET --seed SEED --gpu 0 --aggregator dpfpl --defense none
python main.py --config configs/veil_multidataset.yaml --dataset_name DATASET --seed SEED --gpu 0 --aggregator fedask --defense none
```

The component analysis fixes `DATASET=flowers`, `SEED=42`, `aggregator=fedavg`,
and `defense=veil`. In addition to the unmodified full configuration, its six
single-factor overrides are:

```text
--local_ggeur_anchor_mode sample
--local_ggeur_augments 0
--local_ggeur_original_mode drop
--local_ggeur_upload_clip_norm 0 --local_ggeur_upload_noise_std 0
--local_ggeur_output_temperature 1
--local_ggeur_original_mode class_mean_noise --local_ggeur_original_noise 0.08
```

## Remaining generation steps

After all matrix and ablation runs complete:

```bash
python -m analysis_scripts.veil_paper_results
python -m analysis_scripts.veil_ablation_results
python -m analysis_scripts.plot_veil_paper
python -m analysis_scripts.check_veil_paper
cd paper/aaai2027 && latexmk -pdf -interaction=nonstopmode -halt-on-error veil.tex
```

The generated CSVs and figures are then checked against the values embedded in
the single manuscript source. The final paper contains no placeholder dashes or
template text and compiles without overfull boxes or unresolved references.
