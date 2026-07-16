# Method

## Problem Setting

We consider membership inference attacks against federated CLIP soft-prompt
tuning.  The CLIP image and text encoders are frozen; clients only optimize the
trainable prompt parameters and the server aggregates prompt updates with
FedAvg.  The attacker observes protocol messages and released prompt
checkpoints, and may additionally query the final released model.  The defense
must therefore reduce both update-side membership evidence and final-model
confidence evidence without sharing extra client statistics.

## Local-GGEUR

The proposed defense, Local-GGEUR, follows one design rule: raw member samples
should not directly act as prompt-training anchors.  Each client instead trains
on (i) private replacements of its original samples and (ii) synthetic features
sampled from a locally fitted class-conditional feature distribution.  All
distribution fitting is local; no feature, class mean, covariance factor, or
generated feature is transmitted to the server.

For client \(k\), let \(D_k=\{(x_i,y_i)\}\) be its private data and let
\(\phi(x)\in\mathbb{R}^d\) be the frozen CLIP image embedding.  For every local
class \(c\), the client builds a feature bank

\[
F_{k,c}=\{\mathrm{norm}(\phi(x_i)): y_i=c\},
\]

with class mean

\[
\mu_{k,c}=\frac{1}{|F_{k,c}|}\sum_{z\in F_{k,c}}z
\]

and centered feature matrix

\[
Z_{k,c}=[z-\mu_{k,c}:z\in F_{k,c}].
\]

Following the GGEUR-style geometric sampling idea, Local-GGEUR samples a
low-rank covariance-shaped perturbation without explicitly decomposing the
high-dimensional covariance matrix:

\[
\beta = \epsilon Z_{k,c}/\sqrt{|F_{k,c}|-1},\qquad \epsilon\sim\mathcal{N}(0,I).
\]

The generated feature is

\[
\tilde z = \mathrm{norm}(\mu_{k,c} + s\beta),
\]

where \(s\) controls geometry strength.  The default anchor is the class mean,
not an individual training sample.  If a class has too few local examples, the
client falls back to small isotropic feature noise.

## Private Replacement of Original Samples

Local-GGEUR also keeps a privatized branch for the original labels, but the
original feature vector is not used directly.  The default replacement maps a
member feature to a noisy class-mean feature:

\[
z^{\mathrm{priv}} = \mathrm{norm}(\mu_{k,c}+\eta),\qquad
\eta\sim\mathcal{N}(0,\sigma^2 I).
\]

This preserves local class support while removing instance-specific feature
directions.  We evaluate several alternatives: dropping the original branch,
using the clean class mean, mixing the original feature toward the class mean,
adding direct feature noise, changing the number of geometric samples, and
changing the geometric anchor.

## Prompt Optimization on Feature Logits

For a feature \(z\), the prompt model produces class logits by comparing \(z\)
with the CLIP text features induced by the current prompt.  Each local batch is
replaced by a feature batch containing one privatized original feature and
\(m\) generated geometric features per label.  The local objective is

\[
\mathcal{L}_k =
\frac{1}{|\mathcal{B}_k|}
\sum_{(z,y)\in\mathcal{B}_k}
\mathrm{CE}(g_\theta(z),y)
-\lambda H(\mathrm{softmax}(g_\theta(z))).
\]

The entropy term is optional.  In the final default, \(\lambda=0\), because
entropy strongly reduces confidence attacks but can increase reference/query
attacks.

## Upload Smoothing

After local prompt optimization, Local-GGEUR optionally smooths the prompt delta
before upload.  For client \(k\), let

\[
\Delta_k=\theta_k^{\mathrm{local}}-\theta^{\mathrm{global}}.
\]

The client applies L2 clipping and Gaussian perturbation:

\[
\bar{\Delta}_k =
\Delta_k \cdot \min(1, C/\|\Delta_k\|_2) + \xi,
\qquad \xi\sim\mathcal{N}(0,(\rho C)^2 I).
\]

The uploaded prompt is \(\theta^{\mathrm{global}}+\bar{\Delta}_k\).  Only
trainable prompt parameters are optimized and aggregated.  CLIP encoder weights
remain frozen and local feature statistics remain private.

The selected cross-attack default is:

```text
geometric samples per private original: 3
geometry scale: 0.60
geometric anchor: class mean
original replacement: noisy class mean
original feature noise: 0.08
entropy weight: 0.00
upload clip norm: 0.50
upload noise std: 0.07
final output temperature: 4.0
```

# Experimental Results

## Setup

Experiments use CIFAR100 with 20 clients, FedAvg, full client participation,
Dirichlet non-IID partitioning with \(\alpha=0.1\), `fpl_shots=16`, 10 global
rounds, 5 local epochs, seed 42, and GPU execution.  The audit target is client
0.  The few-shot subset contains 1600 training examples, partitioned across the
20 clients as follows:

| Client | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Samples | 53 | 54 | 72 | 102 | 107 | 84 | 78 | 56 | 73 | 87 | 84 | 88 | 83 | 66 | 109 | 84 | 63 | 106 | 65 | 86 |

The primary privacy metric is \( \mathrm{TPR}@1\%\mathrm{FPR} \).  Lower is
better.  Clean utility is final test accuracy.

## Main Results

The no-defense model reaches 67.08% accuracy.  Its strongest observed attack is
FedMIA joint with \( \mathrm{TPR}@1\%\mathrm{FPR}=0.6981 \).  The broad
Local-GGEUR default with upload smoothing reduces the dominant attack to 0.1698
and reaches 64.48% accuracy.  A FedMIA-first operating point with slightly lower
upload noise reduces FedMIA joint further to 0.1321, but the broad default drives
Canary, RMIA, YOQO, and Quantile-MIA to the low-FPR floor.

| Attack | No defense | Local-GGEUR | AUC under Local-GGEUR |
|---|---:|---:|---:|
| FedMIA loss | 0.5094 | 0.1321 | 0.6736 |
| FedMIA cosine | 0.6038 | 0.2642 | 0.7426 |
| FedMIA joint | 0.6981 | 0.1698 | 0.6972 |
| Canary | 0.3750 | 0.0000 | 0.4102 |
| Nasr passive | 0.1852 | 0.1852 | 0.6713 |
| RMIA | 0.0377 | 0.0000 | 0.4104 |
| YOQO | 0.0000 | 0.0000 | 0.5000 |
| Quantile-MIA | n/a | 0.0000 | 0.4652 |

Local-GGEUR reduces the strongest no-defense protocol attack by 75.7% relative
to the baseline low-FPR TPR.  Canary is reduced from 0.3750 to 0.0000.  Nasr
passive remains at the no-defense low-FPR value, and RMIA, YOQO, and
Quantile-MIA are driven to zero at the low-FPR operating point.

## Seed Robustness

We repeated the default Local-GGEUR multi-attack evaluation with seed 43 while
keeping the same CIFAR100/FedAvg/20-client protocol.  The second seed confirms
that the update-side and reference/query attacks remain strongly suppressed.

| Seed | FedMIA loss | FedMIA cosine | FedMIA joint | Nasr | RMIA | Quantile | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.1321 | 0.2642 | 0.1698 | 0.1852 | 0.0000 | 0.0000 | 0.6448 |
| 43 | 0.0156 | 0.0156 | 0.0938 | 0.0000 | 0.0000 | 0.0625 | 0.6370 |

## Component Ablations

| Variant | FedMIA loss | FedMIA cosine | FedMIA joint | Accuracy |
|---|---:|---:|---:|---:|
| No defense | 0.5094 | 0.6038 | 0.6981 | 0.6708 |
| Local-GGEUR, no entropy | 0.2264 | 0.3019 | 0.4151 | 0.6692 |
| Local-GGEUR, entropy 0.05 | 0.0566 | 0.2264 | 0.2642 | 0.6646 |
| Local-GGEUR, balanced no entropy | 0.3774 | 0.2264 | 0.3396 | 0.6576 |
| + upload smoothing, noise 0.05 | 0.1321 | 0.1698 | 0.1321 | 0.6430 |
| No private replacement branch | 0.4151 | 0.3396 | 0.5283 | 0.6684 |
| No geometric generation | 0.3208 | 0.3019 | 0.3585 | 0.6252 |
| No geometric generation + upload smoothing | 0.2453 | 0.0943 | 0.1509 | 0.5674 |

The original-sample replacement branch is essential: dropping it leaves FedMIA
joint at 0.5283.  The geometry branch is essential for utility: disabling it and
keeping upload smoothing gives a competitive FedMIA joint value of 0.1509, but
accuracy collapses to 56.74%.  The best default therefore keeps both private
replacement and geometric generation.

## Smoothing and Calibration Sweeps

| Variant | FedMIA loss | FedMIA cosine | FedMIA joint | Nasr | RMIA | Quantile | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Upload noise 0.02 | 0.3585 | 0.2075 | 0.2642 | 0.1852 | 0.0189 | 0.0755 | 0.6480 |
| Upload noise 0.04 | 0.3774 | 0.1321 | 0.3774 | 0.2222 | 0.0943 | 0.0377 | 0.6456 |
| Upload noise 0.05 | 0.1321 | 0.1698 | 0.1321 | 0.1852 | 0.0566 | 0.1132 | 0.6430 |
| Upload noise 0.055 | 0.0566 | 0.4340 | 0.2453 | 0.1481 | 0.0189 | 0.0943 | 0.6374 |
| Upload noise 0.06 | 0.2264 | 0.2830 | 0.2264 | 0.1481 | 0.0000 | 0.0755 | 0.6410 |
| Upload noise 0.07 | 0.1321 | 0.2642 | 0.1698 | 0.1852 | 0.0000 | 0.0000 | 0.6448 |
| Clip 0.4, noise 0.05 | 0.2830 | 0.0943 | 0.2453 | 0.1852 | 0.1132 | 0.1887 | 0.6540 |
| Clip 0.3, noise 0.05 | 0.2453 | 0.2453 | 0.3396 | 0.2222 | 0.0377 | 0.0755 | 0.6468 |
| Output temperature 8.0 | 0.1321 | 0.1698 | 0.1321 | 0.1852 | 0.0566 | 0.1132 | 0.6430 |
| Calibrate round observations | 0.2642 | 0.3396 | 0.1132 | 0.2593 | 0.0189 | 0.0000 | 0.6430 |

The smoothing sweep is non-monotonic.  Increasing upload noise from 0.05 to
0.06 drives RMIA to zero and lowers Nasr/Quantile, but FedMIA joint increases
from 0.1321 to 0.2264.  The 0.055 setting suppresses FedMIA loss and RMIA but
causes FedMIA cosine to rebound.  The 0.07 setting gives the best broad
coverage: Canary, RMIA, YOQO, and Quantile-MIA all reach zero at 1% FPR, while
FedMIA joint remains far below no defense.  Reducing the clipping threshold
improves utility in some cases but weakens the dominant joint attack.  Increasing
only the final output temperature from 4 to 8 leaves ranking-based low-FPR
metrics unchanged.

Applying output calibration to round-level observations improves FedMIA joint,
RMIA, and Quantile-MIA, but it raises FedMIA loss/cosine and Nasr passive.
Thus, calibration is useful as a release-side tradeoff but is not the selected
default under the broad multi-attack criterion.

## Cross-Attack Tradeoff

No tested setting strictly dominates on every attack.  The selected default is
the best privacy-first configuration for the protocol-plus-query threat model
because it improves the largest set of attacks while keeping accuracy above
64%.  The FedMIA-first setting gives a lower FedMIA joint value, but the broad
default gives stronger Canary, RMIA, YOQO, and Quantile-MIA protection.
Removing geometric generation improves some attack signals but damages clean
accuracy.  These results support the final design choice: use local
distributional feature generation for utility, noisy class-mean replacement for
membership obfuscation, and lightweight upload smoothing for update-side
privacy.
