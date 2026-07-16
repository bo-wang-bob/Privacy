# VEIL AAAI 2027 paper artifact

The primary deliverable is `veil.tex`, written as a single-file anonymous
submission using the official AAAI 2027 author kit from
<https://aaai.org/authorkit27/> (retrieved 2026-07-16). The unmodified official
`aaai2027.sty` and `aaai2027.bst` files are included beside the manuscript.
Per request, no LaTeX distribution is installed and the paper is not compiled
inside this repository.

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
| FedAvg defenses | Which defense has the best privacy--utility position per dataset? | Three-panel scatter with x/y uncertainty | `evidence/aggregate.csv` | `figures/fedavg_privacy_utility.pdf` |
| Private mechanisms | Which attacks remain strongest under DP-FPL and FedASK? | Three-panel grouped bars | `evidence/attack_aggregate.csv` | `figures/private_methods_attack_profile.pdf` |
| Cross-dataset surface | Does an aggregate hide attack-specific failures? | Annotated heatmap | `evidence/attack_aggregate.csv` | `figures/attack_defense_heatmap.pdf` |
| Ablation | Which VEIL components drive privacy and utility? | Aligned bars, no dual axis | validated Flowers102 ablation runs | `figures/veil_ablation.pdf` |

All plotting defaults in `analysis_scripts/plot_veil_paper.py` use at least
28-point source typography, a zero baseline for absolute comparisons, explicit
colors, and non-color marker/hatch encodings. PDF fonts are exported as TrueType
outlines (`pdf.fonttype=42`) and are intended to remain above the AAAI 9-point
minimum after placement.

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
```

The generated CSVs and figures are then checked against the values embedded in
the single manuscript source. The final paper must contain no placeholder
dashes or template text.
