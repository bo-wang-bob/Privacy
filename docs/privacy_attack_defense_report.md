# Privacy attack and defense reproduction report

> Historical reproduction log. Its tables include a retired, unpublished
> composite diagnostic and must not be used as current paper evidence. The
> validated five-attack results are in `paper/aaai2027/veil.pdf`.

Date: 2026-07-15

## Scope

This report summarizes the current CIFAR100 FedAvg soft-prompt reproduction run.
The primary metric is `TPR@1%FPR`; AUC is treated as diagnostic only.  With 16
to 32 nonmember evaluation samples, 1% FPR means a zero-false-positive threshold,
so a single high-scoring nonmember can move `TPR@1%FPR` to zero.

All full Local-GGEUR runs used the GPU-enabled `pfedba` environment.  PyTorch
reported CUDA available with two NVIDIA GeForce RTX 4090 devices.  The main
effect config is `configs/fedavg_privacy_effect.yaml`: CIFAR100, 20 clients,
Dirichlet `alpha=0.1`, 10 FedAvg rounds, 5 local epochs, `fpl_shots=16`, seed
42, and `gpu: 1`.  Earlier quick defense-screening tables used the smaller
4-client/4-shot configuration noted by their result directories.

## Attack adaptation status

The low-FPR attack adaptations now use `TPR@1%FPR` as the summary
`primary_score`.

| Attack | Main adaptation/fix | Baseline TPR@1%FPR | AUC |
|---|---:|---:|---:|
| Nasr passive | Low-dimensional shrinkage head over passive update/loss features | 0.12500 | 0.80859 |
| Nasr active | 3 ascent/train probe cycles with trajectory recovery features | 0.25000 | 0.65625 |
| FedMIA loss | Correct upload sign, upper-3-sigma null filtering | 0.34375 | 0.70801 |
| FedMIA cosine | Correct upload sign, upper-3-sigma null filtering | 0.28125 | 0.61328 |
| Transfer representation | Frozen CLIP teacher/shared prompt student adaptation | 0.12500 | 0.73047 |
| PIPRA | Output-free shadow prompt adaptation | 0.12500 | 0.63086 |
| RMIA | Low-FPR gamma set to 2.0 | 0.25000 | 0.74902 |
| IMIA | Prompt-only imitative models | 0.25000 | 0.78125 |
| Quantile MIA | Auxiliary nonmember quantile regression | 0.06250 | 0.67578 |
| YOQO | Offline OUT-model optimization, hard-label target query | 0.25000 | 0.62500 |
| Canary | Prompt-only IN/OUT surrogate canaries | 0.31250 | 0.62109 |
| PromptMIA | Shared CoOp prompt-token probe with signed projected update | 0.00000 | 0.61719 |
| CodePoison | Training-time synthetic membership encoding | 0.40625 | 0.59180 |

PromptMIA remains the weakest low-FPR adaptation.  It gets nonzero
`TPR@1%FPR=0.0625` when run alone with 8 adversarial keys, but the unified
multi-attack baseline has one nonmember above all members, giving zero TPR at
the zero-FP threshold.  This should be reported as an architectural limitation
of adapting a keyed visual prompt-pool attack to a shared CoOp text prompt.

Baseline result directories:

- `results/cifar100_fedavg_multi_attack_none_20260715_152104`
- `results/cifar100_fedavg_codepoison_none_20260715_152842`
- `results/cifar100_fedavg_promptmia_none_20260715_152658`

## Defense effects

Representative non-training-time attacks were run for each defense:
`fedmia_loss`, `nasr_active`, `rmia`, `canary`, and `promptmia`.

| Defense | FedMIA loss | Nasr active | RMIA | Canary | PromptMIA | Final acc. |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.34375 | 0.25000 | 0.25000 | 0.31250 | 0.00000 | 0.6434 |
| cofedmid | 0.09375 | 0.62500 | 0.00000 | 0.12500 | 0.00000 | 0.6604 |
| prompt_dp | 0.00000 | 0.00000 | 0.00000 | 0.18750 | 0.00000 | 0.6518 |
| mist | 0.21875 | 0.00000 | 0.06250 | 0.43750 | 0.06250 | 0.6646 |
| soft | 0.21875 | 0.12500 | 0.15625 | 0.06250 | 0.00000 | 0.6458 |
| hamp | 0.09375 | 0.00000 | 0.00000 | 0.06250 | 0.00000 | 0.6626 |

CodePoison was run separately because it changes the training trajectory.

| Defense | CodePoison TPR@1%FPR | AUC | Final acc. |
|---|---:|---:|---:|
| none | 0.40625 | 0.59180 | 0.6442 |
| cofedmid | 0.53125 | 0.87988 | 0.6368 |
| prompt_dp | 0.15625 | 0.53125 | 0.6384 |
| mist | 0.00000 | 0.50781 | 0.6598 |
| soft | 0.12500 | 0.52441 | 0.6138 |
| hamp | 0.40625 | 0.66895 | 0.6492 |

## Defense flow validation

- CoFedMID now uses a balanced class-guided partition.  For CIFAR100, 4
  clients, and 50 classes per client in the first round, the union covers all
  100 classes and the maximum pairwise overlap is 17 classes.  This fixes the
  previous contiguous-slice behavior where two clients could receive identical
  class subsets.
- Prompt-DP uses per-sample gradients over trainable prompt parameters only,
  clips each sample, and adds Gaussian noise before the optimizer step.  The
  representative run clipped 94.1% of samples; the CodePoison run clipped
  96.5%.  The reported epsilon is a conservative full-participation Gaussian
  composition bound, not a tight subsampled accountant.
- MIST performs local training followed by cross-difference refinement against
  peer client submodels.  The representative run recorded mean
  `mist_cross_difference=0.1488`.
- SOFT performs a warm-up round, then selects low-loss samples using a
  validation-loss threshold and applies the repository's image-domain
  obfuscation adaptation.  The representative run selected 64.9% of samples;
  CodePoison selected 57.8%.
- HAMP trains with high-entropy soft labels and entropy regularization, then
  applies a label-preserving low-confidence output mapping at evaluation/query
  time.  The representative run recorded mean entropy 3.37.

## Main conclusions

Prompt-DP and HAMP are the strongest broad defenses in this fixed-seed matrix.
Prompt-DP drives FedMIA, Nasr active, RMIA, and PromptMIA to zero at 1% FPR, but
Canary remains at 0.1875 and the current DP noise is not a strong epsilon
setting.  HAMP drives Nasr active, RMIA, and PromptMIA to zero and reduces
FedMIA/Canary to 0.09375/0.0625.

CoFedMID is effective for FedMIA/RMIA/PromptMIA but worsens Nasr active and
CodePoison in this setup.  MIST suppresses Nasr active and CodePoison, but
Canary increases.  SOFT helps Canary, PromptMIA, and CodePoison, but leaves
FedMIA/RMIA nonzero and has the largest runtime overhead when active prompt
probes are included.

## 20-client non-IID no-defense sweep

The follow-up optimization changed the main effect config to 20 clients with
Dirichlet `alpha=0.1`, full participation, and `fpl_shots=16`.  The larger
client pool gives FedMIA and reference/query attacks more non-target protocol
messages, while the non-IID split increases membership separability.

| Dataset / setting | Target | Attacks | Best TPR@1%FPR | Notes |
|---|---:|---|---:|---|
| CIFAR100, shots=16 | client 0 | FedMIA, RMIA, Canary, YOQO, Nasr passive | 0.69811 | Best current no-defense result, from FedMIA joint loss/cosine evidence |
| Flowers, shots=16 | client 0 | FedMIA, Canary, Nasr passive | 0.15094 | Weaker than CIFAR100 |
| Caltech101, shots=16 | client 0 | FedMIA, Nasr passive | 0.41509 | FedMIA weaker than CIFAR100; Nasr passive reaches 0.37037 |
| DTD, shots=16 | client 0 | FedMIA, Nasr passive | 0.03571 | Weaker than CIFAR100 |
| TinyImageNet-200, shots=16 | client 0 | FedMIA | 0.09375 | Weaker than CIFAR100 |
| CIFAR100, shots=32 | client 0 | FedMIA | 0.12500 | More shots reduced FedMIA low-FPR separation |
| CIFAR100, shots=8 | client 0 | FedMIA joint | 0.14815 | Smaller few-shot subset also reduced low-FPR separation |
| CIFAR100, shots=16 | client 3 | FedMIA | 0.10938 | Larger target client was less vulnerable |
| CIFAR100, shots=16 | client 1 | FedMIA joint | 0.12963 | Low-sample target did not improve over client 0 |
| CIFAR100, shots=16 | client 7 | FedMIA joint | 0.05357 | Most class-concentrated small client was not vulnerable |
| CIFAR100, shots=16 | client 18 | FedMIA joint | 0.18750 | Concentrated 65-sample target remained below client 0 |

For the best CIFAR100 setting (`shots=16`), the deterministic Dirichlet split
with seed 42 assigns these training sample counts by client:

| Client | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Samples | 53 | 54 | 72 | 102 | 107 | 84 | 78 | 56 | 73 | 87 | 84 | 88 | 83 | 66 | 109 | 84 | 63 | 106 | 65 | 86 |

Detailed 20-client CIFAR100 results:

| Attack | TPR@1%FPR | AUC | Samples |
|---|---:|---:|---:|
| FedMIA joint | 0.69811 | 0.86114 | 117 |
| FedMIA cosine | 0.60377 | 0.86350 | 117 |
| FedMIA loss | 0.50943 | 0.84611 | 117 |
| Canary | 0.37500 | 0.72168 | 64 |
| Nasr passive | 0.18519 | 0.70370 | 59 |
| RMIA | 0.03774 | 0.76445 | 85 |
| YOQO | 0.00000 | 0.62500 | 64 |

Current best no-defense attack success is therefore `TPR@1%FPR=0.69811` with
CIFAR100, 20 clients, Dirichlet `alpha=0.1`, `fpl_shots=16`, target client 0,
and `fedmia_joint`.  The joint score is a fixed, non-oracle combination of
FedMIA confidence and update-cosine evidence: early-3-round confidence z-score
and late-3-round cosine z-score.  FedMIA loss remains best with mean round
aggregation when reported alone.

Result directories:

- `results/cifar100_fedavg_multi_attack_none_20260715_173059`
- `results/flowers_fedavg_multi_attack_none_20260715_174912`
- `results/caltech101_fedavg_multi_attack_none_20260715_180906`
- `results/cifar100_fedavg_multi_attack_none_20260715_181337`
- `results/cifar100_fedavg_multi_attack_none_20260715_181916`
- `results/dtd_fedavg_multi_attack_none_20260715_182636`
- `results/tiny-imagenet-200_fedavg_multi_attack_none_20260715_182935`
- `results/cifar100_fedavg_multi_attack_none_20260715_184338`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_185333`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_190705`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_191713`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_191925`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_192400`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_192946`
- `results/cifar100_fedavg_fedmia_joint_none_20260715_193315`

## Local-GGEUR defense sweep

Local-GGEUR implements the new distribution-level defense: each client keeps
class-wise CLIP feature geometry local, trains on class-mean anchored geometric
features, and replaces original member features with noisy class-mean features.

The strongest no-defense reference remains CIFAR100, 20 clients, Dirichlet
`alpha=0.1`, `fpl_shots=16`, target client 0.  Under the same setting, the
current recommended cross-attack Local-GGEUR configuration is:

- `local_ggeur_augments=3`
- `local_ggeur_geometry_scale=0.60`
- `local_ggeur_anchor_mode=class_mean`
- `local_ggeur_original_mode=class_mean_noise`
- `local_ggeur_original_noise=0.08`
- `local_ggeur_entropy_weight=0.00`
- `local_ggeur_output_temperature=4.0`
- `local_ggeur_upload_clip_norm=0.50`
- `local_ggeur_upload_noise_std=0.07`

| Attack | No defense TPR@1%FPR | Local-GGEUR+Smooth TPR@1%FPR | AUC | Notes |
|---|---:|---:|---:|---|
| FedMIA loss | 0.50943 | 0.13208 | 0.67364 | Confidence evidence reduced |
| FedMIA cosine | 0.60377 | 0.26415 | 0.74263 | Update-alignment evidence reduced |
| FedMIA joint | 0.69811 | 0.16981 | 0.69723 | Dominant no-defense attack reduced by 75.7% |
| Canary | 0.37500 | 0.00000 | 0.41016 | Zero at the low-FPR operating point |
| Nasr passive | 0.18519 | 0.18519 | 0.67130 | No low-FPR rebound |
| RMIA | 0.03774 | 0.00000 | 0.41038 | Zero at the low-FPR operating point |
| YOQO | 0.00000 | 0.00000 | 0.50000 | Matches no-defense low-FPR floor |
| Quantile-MIA | n/a | 0.00000 | 0.46521 | Zero at the low-FPR operating point |

Upload smoothing trades some utility for much stronger protocol-side and query
privacy: final accuracy is `0.6448` versus `0.6708` without defense.  The
FedMIA-first `noise_std=0.05` setting reaches `0.6430` accuracy and lowers
FedMIA joint further to `0.13208`, but the broad `noise_std=0.07` setting is
preferred for the main paper table because it drives Canary, RMIA, YOQO, and
Quantile-MIA to the low-FPR floor.  The pure balanced data-replacement baseline, without
upload smoothing, reaches `0.6576` accuracy and gives
FedMIA-joint/Canary/RMIA/YOQO/Quantile of
`0.33962/0.21875/0.11321/0.00000/0.05660`.

Seed robustness for the default `noise_std=0.07` setting:

| Seed | FedMIA loss | FedMIA cosine | FedMIA joint | Nasr | RMIA | Quantile | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.13208 | 0.26415 | 0.16981 | 0.18519 | 0.00000 | 0.00000 | 0.6448 |
| 43 | 0.01562 | 0.01562 | 0.09375 | 0.00000 | 0.00000 | 0.06250 | 0.6370 |

For a narrower FedMIA/Canary setting, entropy regularization gives stronger
suppression:

| Attack | No defense TPR@1%FPR | Entropy Local-GGEUR TPR@1%FPR | Entropy AUC |
|---|---:|---:|---:|
| FedMIA loss | 0.50943 | 0.05660 | 0.77299 |
| FedMIA cosine | 0.60377 | 0.22642 | 0.74204 |
| FedMIA joint | 0.69811 | 0.26415 | 0.77978 |
| Canary | 0.37500 | 0.15625 | 0.64648 |
| Nasr passive | 0.18519 | 0.22222 | 0.62616 |
| RMIA | 0.03774 | 0.30189 | 0.66156 |
| YOQO | 0.00000 | 0.25000 | 0.62500 |
| Quantile-MIA | n/a | 0.05660 | 0.53125 |

This entropy setting keeps final accuracy at `0.6646`, but it is not the
default because RMIA and YOQO rebound.

Ablations:

| Variant | FedMIA loss | FedMIA cosine | FedMIA joint | Accuracy |
|---|---:|---:|---:|---:|
| No defense | 0.50943 | 0.60377 | 0.69811 | 0.6708 |
| Local-GGEUR, no entropy | 0.22642 | 0.30189 | 0.41509 | 0.6692 |
| Local-GGEUR, entropy `0.05` | 0.05660 | 0.22642 | 0.26415 | 0.6646 |
| Local-GGEUR, entropy `0.10` | 0.49057 | 0.24528 | 0.41509 | 0.6586 |
| Local-GGEUR, entropy `0.02` | 0.26415 | 0.00000 | 0.49057 | 0.6558 |
| Local-GGEUR, balanced no entropy | 0.37736 | 0.22642 | 0.33962 | 0.6576 |
| + upload smoothing, noise `0.02` | 0.35849 | 0.20755 | 0.26415 | 0.6480 |
| + upload smoothing, noise `0.04` | 0.37736 | 0.13208 | 0.37736 | 0.6456 |
| + upload smoothing, noise `0.05` | 0.13208 | 0.16981 | 0.13208 | 0.6430 |
| + upload smoothing, noise `0.055` | 0.05660 | 0.43396 | 0.24528 | 0.6374 |
| + upload smoothing, noise `0.06` | 0.22642 | 0.28302 | 0.22642 | 0.6410 |
| + upload smoothing, noise `0.07` | 0.13208 | 0.26415 | 0.16981 | 0.6448 |
| + softer clipping, clip `0.4`/noise `0.05` | 0.28302 | 0.09434 | 0.24528 | 0.6540 |
| + stronger clipping, clip `0.3`/noise `0.05` | 0.24528 | 0.24528 | 0.33962 | 0.6468 |
| + calibrate round observations | 0.26415 | 0.33962 | 0.11321 | 0.6430 |
| Local-GGEUR, `aug=1` | 0.33962 | 0.30189 | 0.37736 | 0.6594 |
| Early entropy, first 3 rounds | 0.24528 | 0.43396 | 0.39623 | 0.6494 |
| Late aug anneal, rounds 7--9 use aug=0 | 0.20755 | 0.13208 | 0.35849 | 0.5740 |
| Posterior margin cap | 0.37736 | 0.22642 | 0.33962 | 0.6576 |
| Class-balanced local sampling | 0.30189 | 0.22642 | 0.47170 | 0.6452 |
| Class-mean original replacement | 0.39623 | 0.18868 | 0.37736 | 0.6660 |
| No private replacement branch | 0.41509 | 0.33962 | 0.52830 | 0.6684 |
| No geometric generation | 0.32075 | 0.30189 | 0.35849 | 0.6252 |
| No geometric generation + upload smoothing | 0.24528 | 0.09434 | 0.15094 | 0.5674 |

Additional reference/query attack ablations:

| Variant | RMIA | YOQO | Quantile-MIA | Accuracy |
|---|---:|---:|---:|---:|
| Local-GGEUR, entropy `0.05` | 0.30189 | 0.25000 | 0.05660 | 0.6646 |
| Local-GGEUR, balanced no entropy | 0.11321 | 0.00000 | 0.05660 | 0.6576 |
| + upload smoothing, noise `0.02` | 0.01887 | n/a | 0.07547 | 0.6480 |
| + upload smoothing, noise `0.04` | 0.09434 | n/a | 0.03774 | 0.6456 |
| + upload smoothing, noise `0.05` | 0.05660 | 0.03125 | 0.11321 | 0.6430 |
| + upload smoothing, noise `0.055` | 0.01887 | n/a | 0.09434 | 0.6374 |
| + upload smoothing, noise `0.06` | 0.00000 | n/a | 0.07547 | 0.6410 |
| + upload smoothing, noise `0.07` | 0.00000 | 0.00000 | 0.00000 | 0.6448 |
| + softer clipping, clip `0.4`/noise `0.05` | 0.11321 | n/a | 0.18868 | 0.6540 |
| + stronger clipping, clip `0.3`/noise `0.05` | 0.03774 | n/a | 0.07547 | 0.6468 |
| + calibrate round observations | 0.01887 | n/a | 0.00000 | 0.6430 |
| Local-GGEUR, `aug=1` | 0.03774 | n/a | 0.20755 | 0.6594 |
| Early entropy, first 3 rounds | 0.26415 | n/a | 0.20755 | 0.6494 |
| Late aug anneal, rounds 7--9 use aug=0 | 0.24528 | n/a | 0.26415 | 0.5740 |
| Posterior margin cap | 0.11321 | n/a | 0.05660 | 0.6576 |
| Class-balanced local sampling | 0.28302 | n/a | 0.16981 | 0.6452 |
| Class-mean original replacement | 0.16981 | n/a | 0.11321 | 0.6660 |
| No geometric generation | 0.00000 | 0.00000 | 0.00000 | 0.6252 |
| No geometric generation + upload smoothing | 0.22642 | n/a | 0.05660 | 0.5674 |

The upload-noise and clipping sweeps motivate the selected broad default.
Raising `noise_std` to `0.055` suppresses FedMIA loss and RMIA, but FedMIA
cosine rebounds to `0.43396`.  Raising it to `0.06` drives RMIA to zero and
lowers Canary/Nasr/Quantile, but FedMIA joint remains higher than the
FedMIA-first `0.05` setting at `0.22642`.  Raising it to `0.07` gives the best
broad setting: Canary, RMIA, YOQO, and Quantile-MIA all reach zero at 1% FPR
while FedMIA joint remains far below no defense.  Changing only
`local_ggeur_output_temperature` from 4 to 8 leaves the ranking-based low-FPR
metrics unchanged.  Applying output calibration to round-level observations
improves FedMIA joint, RMIA, and Quantile-MIA, but FedMIA loss/cosine and Nasr
passive rebound, so this is retained as a tradeoff
ablation rather than the default.  The `clip_norm=0.4` trial is utility-biased:
it reaches `0.6540` accuracy and lowers FedMIA cosine to `0.09434`, but FedMIA
loss/joint rebound to `0.28302/0.24528`, RMIA rises to `0.11321`, and
Quantile-MIA rises to `0.18868`.  The stronger `clip_norm=0.3` trial recovers
only 0.38 points of clean accuracy relative to the current default, while FedMIA
joint rebounds to `0.33962` and Nasr passive rises to `0.22222`.

Result directories:

- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_002031`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_002451`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_002851`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_003316`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_003723`
- `results/cifar100_fedavg_canary_local_ggeur_20260716_004158`
- `results/cifar100_fedavg_nasr_passive_local_ggeur_20260716_005454`
- `results/cifar100_fedavg_nasr_passive_local_ggeur_20260716_005855`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_010554`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_011006`
- `results/cifar100_fedavg_canary_local_ggeur_20260716_011504`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_013912`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_015828`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_021519`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_021920`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_022408`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_022830`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_030024`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_030519`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_031049`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_031717`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_032116`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_032606`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_033015`
- `results/cifar100_fedavg_canary_local_ggeur_20260716_033721`
- `results/cifar100_fedavg_yoqo_local_ggeur_20260716_034954`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_102624`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_103215`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_103954`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_104702`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_105112`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_105508`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_105922`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_110834`
- `results/cifar100_fedavg_canary_local_ggeur_20260716_111408`
- `results/cifar100_fedavg_yoqo_local_ggeur_20260716_112646`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_114725`
- `results/cifar100_fedavg_canary_local_ggeur_20260716_120714`
- `results/cifar100_fedavg_yoqo_local_ggeur_20260716_124141`
- `results/cifar100_fedavg_multi_attack_local_ggeur_20260716_132733`
