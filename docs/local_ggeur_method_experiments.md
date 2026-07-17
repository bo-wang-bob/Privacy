# Method

> Historical development log. Its tables include a retired, unpublished
> composite diagnostic and must not be used as current paper evidence. The
> validated five-attack results are in `paper/aaai2027/veil.pdf`.

## Local-GGEUR for Membership-Private Federated Prompt Learning

We study membership inference defenses for federated CLIP soft-prompt tuning.
The defense is designed around a simple principle: a client should update the
prompt from its local distribution, not from raw member samples.  We therefore
replace direct image-level training with a local feature-distribution training
procedure, called Local-GGEUR.

Each client keeps the CLIP image encoder frozen and extracts normalized image
embeddings for its local training set.  For every local class `c`, the client
computes a private feature bank with mean `μ_c` and centered features `Z_c`.
No class mean, covariance, prototype, feature, or generated sample is shared
with the server or other clients.  The server only receives the usual prompt
update.

Local-GGEUR has two stages.

1. Local geometric feature generation.  For a sample label `c`, the client
   draws a Gaussian vector `ε` and generates a covariance-shaped perturbation

   ```text
   β = ε Z_c / sqrt(n_c - 1),  ε ~ N(0, I).
   ```

   This is the low-rank empirical-covariance form of the GGEUR-style
   eigenvector/eigenvalue perturbation.  The generated training feature is
   `normalize(μ_c + s β)`, where `s` controls geometry strength.  The default
   anchor is the class mean rather than the original sample, so the generated
   feature is distribution-centered instead of instance-centered.

2. Private replacement of original samples.  Raw sample embeddings are not
   directly used as prompt-training inputs.  The default replacement maps each
   original feature to `normalize(μ_c + η)`, where `η` is Gaussian feature noise.
   This branch preserves local class support while blurring instance-specific
   feature directions.

The prompt is optimized on the union of private replacements and generated
features.  A small prediction-entropy regularizer can be enabled to reduce
confidence separation exploited by low-FPR membership attacks:

```text
L = CE(f_prompt(x_private), y) - λ H(p_prompt(x_private)).
```

The implementation exposes direct ablations:

- no geometric generation: `local_ggeur_augments=0`;
- no private replacement branch: `local_ggeur_original_mode=drop`;
- instance-centered generation: `local_ggeur_anchor_mode=sample`;
- entropy strength sweep: `local_ggeur_entropy_weight`;
- early-only entropy: `local_ggeur_entropy_rounds`;
- late geometric annealing: `local_ggeur_late_start_round` and
  `local_ggeur_late_augments`;
- upload smoothing: `local_ggeur_upload_clip_norm` and
  `local_ggeur_upload_noise_std`;
- deployment/query calibration: `local_ggeur_output_temperature`, applied only
  to the final released model, not to local training or protocol observations.

# Experimental Results

## Setup

Unless noted otherwise, experiments use CIFAR100 with 20 clients, full
participation, Dirichlet `α=0.1`, `fpl_shots=16`, FedAvg prompt aggregation,
target client 0, and GPU execution.  The primary privacy metric is
`TPR@1%FPR`; lower is better for defenses.  Clean utility is measured by final
test accuracy in `training_metrics.csv`.

The deterministic few-shot split with seed 42 first selects 16 examples per
class, giving 1600 training examples, then partitions them across clients with
Dirichlet `α=0.1`.  Client 0, the audit target, has 53 training examples:

| Client | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Samples | 53 | 54 | 72 | 102 | 107 | 84 | 78 | 56 | 73 | 87 | 84 | 88 | 83 | 66 | 109 | 84 | 63 | 106 | 65 | 86 |

The no-defense reference from the same setting reaches 67.08% accuracy.  Its
strongest attack is FedMIA joint with `TPR@1%FPR=0.6981`.

## Main Local-GGEUR Result

The recommended broad multi-attack configuration is Local-GGEUR with balanced
feature training and slightly stronger upload smoothing:

```text
local_ggeur_augments = 3
local_ggeur_geometry_scale = 0.60
local_ggeur_anchor_mode = class_mean
local_ggeur_original_mode = class_mean_noise
local_ggeur_original_noise = 0.08
local_ggeur_entropy_weight = 0.00
local_ggeur_upload_clip_norm = 0.50
local_ggeur_upload_noise_std = 0.07
```

| Attack | No defense TPR@1%FPR | Local-GGEUR+Smooth TPR@1%FPR | AUC |
|---|---:|---:|---:|
| FedMIA loss | 0.5094 | 0.1321 | 0.6736 |
| FedMIA cosine | 0.6038 | 0.2642 | 0.7426 |
| FedMIA joint | 0.6981 | 0.1698 | 0.6972 |
| Canary | 0.3750 | 0.0000 | 0.4102 |
| Nasr passive | 0.1852 | 0.1852 | 0.6713 |
| RMIA | 0.0377 | 0.0000 | 0.4104 |
| YOQO | 0.0000 | 0.0000 | 0.5000 |
| Quantile-MIA | n/a | 0.0000 | 0.4652 |

This setting reduces the strongest observed attack, FedMIA joint, from 0.6981
to 0.1698, drives Canary, RMIA, YOQO, and Quantile-MIA to the zero-FP floor,
and keeps Nasr passive at the no-defense low-FPR value.  Its final accuracy is
64.48%, a 2.60-point drop from the no-defense run.  A FedMIA-first operating
point with `local_ggeur_upload_noise_std=0.05` gives lower FedMIA joint
(`0.1321`) and 64.30% accuracy, but has weaker Canary/RMIA/YOQO/Quantile
coverage.

The default setting was also repeated with seed 43 for the multi-attack suite:

| Seed | FedMIA loss | FedMIA cosine | FedMIA joint | Nasr | RMIA | Quantile | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.1321 | 0.2642 | 0.1698 | 0.1852 | 0.0000 | 0.0000 | 0.6448 |
| 43 | 0.0156 | 0.0156 | 0.0938 | 0.0000 | 0.0000 | 0.0625 | 0.6370 |

The same balanced feature training without upload smoothing remains a useful
pure data-replacement baseline:

| Attack | Balanced no-smoothing TPR@1%FPR | AUC |
|---|---:|---:|
| FedMIA loss | 0.3774 | 0.7810 |
| FedMIA cosine | 0.2264 | 0.6797 |
| FedMIA joint | 0.3396 | 0.7264 |
| Canary | 0.2188 | 0.6855 |
| Nasr passive | 0.1852 | 0.5949 |
| RMIA | 0.1132 | 0.6209 |
| YOQO | 0.0000 | 0.5781 |
| Quantile-MIA | 0.0566 | 0.5814 |

For the narrow FedMIA/Canary threat model, the entropy-regularized variant is
stronger:

```text
local_ggeur_augments = 2
local_ggeur_geometry_scale = 0.45
local_ggeur_original_noise = 0.05
local_ggeur_entropy_weight = 0.05
```

| Attack | No defense TPR@1%FPR | Entropy Local-GGEUR TPR@1%FPR | Entropy AUC |
|---|---:|---:|---:|
| FedMIA loss | 0.5094 | 0.0566 | 0.7730 |
| FedMIA cosine | 0.6038 | 0.2264 | 0.7420 |
| FedMIA joint | 0.6981 | 0.2642 | 0.7798 |
| Canary | 0.3750 | 0.1562 | 0.6465 |
| Nasr passive | 0.1852 | 0.2222 | 0.6262 |
| RMIA | 0.0377 | 0.3019 | 0.6616 |
| YOQO | 0.0000 | 0.2500 | 0.6250 |
| Quantile-MIA | n/a | 0.0566 | 0.5313 |

This version keeps accuracy at 66.46%, but it is not the default because RMIA
and YOQO rebound relative to the balanced setting.

## Ablation Study

| Variant | FedMIA loss | FedMIA cosine | FedMIA joint | Accuracy |
|---|---:|---:|---:|---:|
| No defense | 0.5094 | 0.6038 | 0.6981 | 0.6708 |
| Local-GGEUR, no entropy | 0.2264 | 0.3019 | 0.4151 | 0.6692 |
| Local-GGEUR, entropy 0.05 | 0.0566 | 0.2264 | 0.2642 | 0.6646 |
| Local-GGEUR, entropy 0.10 | 0.4906 | 0.2453 | 0.4151 | 0.6586 |
| Local-GGEUR, balanced no entropy | 0.3774 | 0.2264 | 0.3396 | 0.6576 |
| + upload smoothing, noise 0.02 | 0.3585 | 0.2075 | 0.2642 | 0.6480 |
| + upload smoothing, noise 0.04 | 0.3774 | 0.1321 | 0.3774 | 0.6456 |
| + upload smoothing, noise 0.05 | 0.1321 | 0.1698 | 0.1321 | 0.6430 |
| + upload smoothing, noise 0.055 | 0.0566 | 0.4340 | 0.2453 | 0.6374 |
| + upload smoothing, noise 0.06 | 0.2264 | 0.2830 | 0.2264 | 0.6410 |
| + upload smoothing, noise 0.07 | 0.1321 | 0.2642 | 0.1698 | 0.6448 |
| + softer clipping, clip 0.4/noise 0.05 | 0.2830 | 0.0943 | 0.2453 | 0.6540 |
| + stronger clipping, clip 0.3/noise 0.05 | 0.2453 | 0.2453 | 0.3396 | 0.6468 |
| + calibrate round observations | 0.2642 | 0.3396 | 0.1132 | 0.6430 |
| Local-GGEUR, aug=1 | 0.3396 | 0.3019 | 0.3774 | 0.6594 |
| Early entropy, first 3 rounds | 0.2453 | 0.4340 | 0.3962 | 0.6494 |
| Late aug anneal, rounds 7--9 use aug=0 | 0.2075 | 0.1321 | 0.3585 | 0.5740 |
| Posterior margin cap | 0.3774 | 0.2264 | 0.3396 | 0.6576 |
| Class-balanced local sampling | 0.3019 | 0.2264 | 0.4717 | 0.6452 |
| Class-mean original replacement | 0.3962 | 0.1887 | 0.3774 | 0.6660 |
| No private replacement branch | 0.4151 | 0.3396 | 0.5283 | 0.6684 |
| No geometric generation | 0.3208 | 0.3019 | 0.3585 | 0.6252 |
| No geometric generation + upload smoothing | 0.2453 | 0.0943 | 0.1509 | 0.5674 |

The ablation supports three observations.

First, replacing raw member features with local class-mean noisy features is
the largest privacy contributor.  Even without entropy, FedMIA joint drops from
0.6981 to 0.4151 with negligible accuracy loss.

Second, the geometric generation stage matters for the final privacy-utility
balance.  Removing it lowers accuracy from 0.6646 to 0.6252 and worsens FedMIA
joint from 0.2642 to 0.3585.

Third, entropy regularization has a non-monotonic effect.  A moderate value
(`λ=0.05`) strongly suppresses FedMIA confidence evidence; a larger value
(`λ=0.10`) hurts accuracy and causes the confidence attack to rebound.

## Cross-Attack Tradeoffs

The existing attacks do not move together.  Increasing entropy suppresses
FedMIA confidence but can increase Nasr passive and RMIA/YOQO.  Lightweight
upload smoothing is the most effective cross-attack addition: increasing the
upload noise from 0.02 to 0.05 reduces FedMIA joint from `0.3396` to `0.1321`
and RMIA from `0.1132` to `0.0566`, while keeping Nasr passive at `0.1852`.
The tradeoff is lower accuracy (`0.6430`) and a Quantile-MIA increase from
`0.0566` to `0.1132`.  The intermediate `0.04` setting is not Pareto-optimal:
it lowers Quantile-MIA to `0.0377` but leaves FedMIA joint at `0.3774` and
raises Nasr passive to `0.2222`.  Reducing the
number of generated geometric features (`aug=1`) brings RMIA back to the
no-defense low-FPR value (`0.0377`) and lowers Nasr passive to `0.1481`, but it
raises Quantile-MIA to `0.2075`.  Early-only entropy improves FedMIA-loss but
worsens FedMIA-cosine, RMIA, and Quantile-MIA.  Late geometric annealing
improves the individual FedMIA loss/cosine signals, but it sharply hurts Nasr
passive, RMIA, Quantile-MIA, and clean accuracy.  Posterior margin capping is
neutral in this setting because the relevant logits are already low-margin.
Removing geometric generation entirely gives the strongest query/reference
suppression (`RMIA=YOQO=Quantile=0`) but drops accuracy to 62.52%.

The current default therefore prioritizes Local-GGEUR with upload smoothing as
the most defensible paper setting for the protocol-plus-query threat model.  The
pure data-replacement variant remains the cleaner mechanism-only ablation.  The
upload-noise sweep is non-monotonic: `noise_std=0.055` suppresses FedMIA loss
to `0.0566` and RMIA to `0.0189`, but FedMIA cosine rebounds to `0.4340`;
`noise_std=0.06` drives RMIA to `0.0000`, Nasr passive to `0.1481`, and
Quantile-MIA to `0.0755`, but FedMIA joint remains higher than the FedMIA-first
`0.05` setting at `0.2264`.  `noise_std=0.07` keeps RMIA and Quantile-MIA at
`0.0000`, drives Canary and YOQO to `0.0000`, improves FedMIA joint to
`0.1698`, and gives the best broad-defense operating point.  Increasing the
final output temperature from 4 to 8 leaves all
ranking-based low-FPR metrics unchanged in this setting.  Applying the output
calibration to round-level observations improves FedMIA joint to `0.1132`, RMIA
to `0.0189`, and Quantile-MIA to `0.0000`, but it raises FedMIA loss/cosine and
Nasr passive; it is therefore not used as the default under the broad
multi-attack criterion.  Disabling geometric generation while keeping upload
smoothing reduces FedMIA cosine and Quantile-MIA but collapses accuracy to
56.74% and raises Nasr/RMIA, so it is retained only as an ablation showing why
geometry is needed for utility.

The clip-strength sweep confirms that the current `clip_norm=0.5` is the best
privacy-first setting among the tested values.  The `clip_norm=0.4` trial
improves accuracy to 65.40% and lowers FedMIA cosine to `0.0943`, but FedMIA
loss/joint rebound to `0.2830/0.2453`, RMIA rises to `0.1132`, and Quantile-MIA
rises to `0.1887`.  The stronger `clip_norm=0.3` trial slightly improves
accuracy to 64.68%, but FedMIA joint rebounds to `0.3396` and Nasr passive rises
to `0.2222`.  Neither clipping variant is used as the paper default.

## Current Limitation

The current best setting is not uniformly dominant across all attacks.  Upload
smoothing substantially reduces FedMIA and drives Canary/RMIA/YOQO/Quantile-MIA
to the low-FPR floor, while clean accuracy falls to 64.48%.  The main remaining
tradeoff is between the FedMIA-first `noise_std=0.05` setting and the broad
`noise_std=0.07` setting: the former gives lower FedMIA joint, while the latter
gives stronger cross-attack coverage.
