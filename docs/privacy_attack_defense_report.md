# Privacy attack and defense reproduction report

Date: 2026-07-15

## Scope

This report summarizes the current CIFAR100 FedAvg soft-prompt reproduction run.
The primary metric is `TPR@1%FPR`; AUC is treated as diagnostic only.  With 16
to 32 nonmember evaluation samples, 1% FPR means a zero-false-positive threshold,
so a single high-scoring nonmember can move `TPR@1%FPR` to zero.

All runs used the GPU-enabled `pfedba` environment.  PyTorch reported CUDA
available with two NVIDIA GeForce RTX 4090 devices.  The experiment config was
`configs/fedavg_privacy_effect.yaml`: CIFAR100, 4 clients, 10 FedAvg rounds,
5 local epochs, 4-shot FPL subset, seed 42, and `gpu: 1`.

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
| CIFAR100, shots=16 | client 0 | FedMIA, RMIA, Canary, YOQO, Nasr passive | 0.50943 | Best current no-defense result, from FedMIA loss |
| Flowers, shots=16 | client 0 | FedMIA, Canary, Nasr passive | 0.15094 | Weaker than CIFAR100 |
| Caltech101, shots=16 | client 0 | FedMIA, Nasr passive | 0.41509 | FedMIA weaker than CIFAR100; Nasr passive reaches 0.37037 |
| CIFAR100, shots=32 | client 0 | FedMIA | 0.12500 | More shots reduced FedMIA low-FPR separation |
| CIFAR100, shots=16 | client 3 | FedMIA | 0.10938 | Larger target client was less vulnerable |

Detailed 20-client CIFAR100 results:

| Attack | TPR@1%FPR | AUC | Samples |
|---|---:|---:|---:|
| FedMIA loss | 0.50943 | 0.84611 | 117 |
| FedMIA cosine | 0.32075 | 0.83933 | 117 |
| Canary | 0.37500 | 0.72168 | 64 |
| Nasr passive | 0.18519 | 0.70370 | 59 |
| RMIA | 0.03774 | 0.76445 | 85 |
| YOQO | 0.00000 | 0.62500 | 64 |

Current best no-defense attack success is therefore `TPR@1%FPR=0.50943` with
CIFAR100, 20 clients, Dirichlet `alpha=0.1`, `fpl_shots=16`, target client 0,
and FedMIA loss scoring.

Result directories:

- `results/cifar100_fedavg_multi_attack_none_20260715_173059`
- `results/flowers_fedavg_multi_attack_none_20260715_174912`
- `results/caltech101_fedavg_multi_attack_none_20260715_180906`
- `results/cifar100_fedavg_multi_attack_none_20260715_181337`
- `results/cifar100_fedavg_multi_attack_none_20260715_181916`
