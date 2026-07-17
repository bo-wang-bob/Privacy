# DP-FPL 与 FedASK 隐私攻击实验记录

> 注意：本页的早期结果使用顺序抽取的非成员候选，没有严格匹配成员与非成员
> 的类别直方图。Flowers102 的最终公平比较应以
> `docs/flowers_fair_method_comparison.md` 中的五 seed 标签匹配实验为准。

日期：2026-07-16

## 目标

评估当前仓库中已有 membership privacy attacks 在 DP-FPL 与 FedASK 上的攻击效果，并尽量对齐以下论文及其官方仓库：

- DP-FPL: *Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models*，官方仓库：https://github.com/coderanik/Privacy-Preserving-Paper1
- FedASK: *Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix*，官方仓库：https://github.com/FLEECERmw/PrivacyFedLLM

本轮实验使用 membership privacy attacks。论文报告的被动审计包括 `fedmia_loss`、`fedmia_cosine`、`nasr_passive`、`rmia`、`quantile_mia`。第二批为其余合规隐私攻击：`nasr_active`、`transfer_representation`、`pipra`、`imia`、`yoqo`、`canary`、`promptmia`。

## 官方实现核对

### DP-FPL

官方 DP-FPL 仓库以 notebook 为主。关键公开设置与实现要点：

- Flowers102，10 clients，5 rounds，local epoch 1，batch size 16。
- prompt length 16，low-rank rank 4。
- prompt 由 global prompt 与 local prompt 组成。
- local prompt 做低秩分解 `U @ V + residual`。
- local prompt 梯度与 global prompt 更新分别加噪。

当前仓库对应实现：

- `trainmodel/custom_clip.py` 中 `parameterization=dpfpl` 使用 `global_ctx + local_ctx`。
- `users/user.py::_train_dpfpl` 中对 local prompt 做随机低秩分解并保留 residual。
- `aggregator/dpfpl_aggregator.py` 中服务器只聚合 global prompt gradient，并保留每个客户端的 local prompt 状态。
- `main.py::_load_local_clip` 使用 `local_files_only=True` 加载本地 CLIP。

差异：

- 官方 notebook 使用 CPU 和自动下载模型/数据；本仓库按要求使用本地 CLIP 缓存与 GPU。
- 官方 notebook 的 DP 逻辑较简化；本仓库额外记录 conservative RDP accounting 和协议可见消息。

### FedASK

官方 FedASK 仓库是 LLM LoRA 框架，不是 CLIP prompt 框架。关键实现要点：

- `policy_fedask.py` 中 Stage 1：`Y_i = W_B_i @ (W_A_i @ Omega)`。
- 服务端对聚合 `Y` 做 QR 得到 `Q`。
- Stage 2：`P_i = W_A_i^T @ (W_B_i^T @ Q)`。
- 服务端对聚合 `P` 做 SVD，重构 LoRA `A/B`。
- 官方配置中 `num_clients=10`、`para_niid_deg=0.1`、`lora_rank=16`、`s_sketch_dim=16`、`target_grad_clip_persample=1.0`。

当前仓库对应实现：

- `trainmodel/custom_clip.py` 中 `parameterization=fedask` 使用 `base_ctx + scaling * (B @ A)`。
- `users/user.py::_train_fedask` 对完整 prompt 梯度做裁剪/加噪后更新 `B`。
- `aggregator/fedask_aggregator.py` 实现两阶段 sketch、QR、SVD，并把 `stage1_y` 和 `stage2_p` 作为协议消息暴露给审计器。

差异：

- 官方仓库面向 LLM LoRA；当前仓库是 CLIP soft prompt，因此本实验采用结构等价的 prompt 低秩 adapter 映射。
- 为贴近官方 `lora_rank=16 / s_sketch_dim=16`，新增 `configs/fedask_official_repo_aligned.yaml`，设置 `rank=16, oversampling=0, n_ctx=16`。该配置下最终 `rank_based_width_condition_met=true`。

## 实验设置

审计视角统一为 `protocol_plus_released_prompts`。额外防御为 `none`；DP-FPL/FedASK 自身的裁剪与噪声机制仍启用。

### A. 仓库默认论文配置

配置来源：`configs/fedprompt_privacy.yaml`

| 项 | 值 |
|---|---:|
| dataset | Caltech101 |
| clients / sampled | 10 / 10 |
| Dirichlet alpha | 0.1 |
| few-shot | 16 per class |
| train samples per client | `[151, 212, 189, 170, 136, 196, 158, 103, 206, 95]` |
| test samples per client | `[1039, 352, 808, 536, 396, 534, 612, 350, 505, 515]` |
| rounds | 20 |
| local epochs | 2 |
| batch size | 16 |
| prompt length | 32 |
| GPU | 1 |

DP-FPL 参数：rank 4，local/global clip norm 1.0，local/global noise multiplier 1.0。

FedASK 参数：rank 4，oversampling 2，local_steps 2，clip norm 1.0，noise multiplier 1.0。该配置下最终 `rank_based_width_condition_met=false`，因为观测到的 client-B subspace rank 为 32，诊断要求 oversampling 至少 28；不过最终重构误差仍为 `1.56e-6`。

### B. 官方仓库风格映射配置

新增配置：

- `configs/dpfpl_official_repo_aligned.yaml`
- `configs/fedask_official_repo_aligned.yaml`

| 项 | DP-FPL official-style | FedASK official-style |
|---|---:|---:|
| dataset | Flowers102 | Flowers102 |
| clients / sampled | 10 / 10 | 10 / 10 |
| Dirichlet alpha | 0.1 | 0.1 |
| few-shot | 16 per class | 16 per class |
| train samples per client | `[151, 212, 189, 186, 136, 196, 158, 103, 206, 95]` | same |
| test samples per client | `[437, 636, 635, 499, 667, 779, 541, 501, 745, 709]` | same |
| rounds | 5 | 5 |
| local epochs | 1 | 1 |
| batch size | 16 | 16 |
| prompt length | 16 | 16 |
| GPU | 1 | 1 |
| rank | 4 | 16 |
| sketch dim | n/a | 16 |

DP-FPL official-style 采用官方 simulated notebook 中的噪声量级映射：local noise multiplier 0.3，global noise multiplier 0.1。

FedASK official-style 采用官方配置中的 rank/sketch 映射：`rank=16, oversampling=0`，即 sketch dim 为 16。

## 结果

指标解释：

- `TPR@1%FPR`：低误报约束下的攻击召回，越高表示隐私泄露越明显。
- `AUC`：攻击排序能力，0.5 约等于随机。
- `n`：该攻击实际用于评估的样本数。

### A. Caltech101 默认配置，无额外防御

结果目录：

- `results/caltech101_dpfpl_multi_attack_none_20260716_153301`
- `results/caltech101_fedask_multi_attack_none_20260716_153935`

| Method | Attack | TPR@1%FPR | TPR@10%FPR | AUC | n |
|---|---|---:|---:|---:|---:|
| DP-FPL | fedmia_loss | 0.0000 | 0.0000 | 0.5000 | 64 |
| DP-FPL | fedmia_cosine | 0.2188 | 0.3125 | 0.4834 | 64 |
| DP-FPL | nasr_passive | 0.2500 | 0.3125 | 0.7070 | 32 |
| DP-FPL | rmia | 0.0000 | 0.0000 | 0.5000 | 48 |
| DP-FPL | quantile_mia | 0.0000 | 0.0000 | 0.1465 | 48 |
| FedASK | fedmia_loss | 0.0000 | 0.0000 | 0.5000 | 64 |
| FedASK | fedmia_cosine | 0.1562 | 0.2812 | 0.5703 | 64 |
| FedASK | nasr_passive | 0.5000 | 0.5000 | 0.6250 | 32 |
| FedASK | rmia | 0.0000 | 0.0000 | 0.5000 | 48 |
| FedASK | quantile_mia | 0.0312 | 0.0312 | 0.2461 | 48 |

主任务最终测试：

| Method | Round | Loss | Accuracy | Test samples |
|---|---:|---:|---:|---:|
| DP-FPL | 19 | 0.7019 | 0.8330 | 5647 |
| FedASK | 19 | 0.9741 | 0.8275 | 5647 |

方法诊断：

| Method | DP epsilon upper bound | Reconstruction error | Width condition |
|---|---:|---:|---|
| DP-FPL local/global | 51.51 / 31.51 | n/a | n/a |
| FedASK | 51.51 | 1.56e-6 | false |

### B. Official-style Flowers102 映射配置，无额外防御

结果目录：

- `results/flowers_dpfpl_multi_attack_none_20260716_154944`
- `results/flowers_fedask_multi_attack_none_20260716_155232`
- `results/flowers_dpfpl_multi_attack_none_20260716_160329`
- `results/flowers_fedask_multi_attack_none_20260716_161133`

| Method | Attack | TPR@1%FPR | TPR@10%FPR | AUC | n |
|---|---|---:|---:|---:|---:|
| DP-FPL | fedmia_loss | 0.0000 | 0.0000 | 0.5000 | 64 |
| DP-FPL | fedmia_cosine | 0.0000 | 0.0938 | 0.5820 | 64 |
| DP-FPL | nasr_passive | 0.1875 | 0.1875 | 0.4766 | 32 |
| DP-FPL | rmia | 0.0000 | 0.0000 | 0.5000 | 48 |
| DP-FPL | quantile_mia | 0.0625 | 0.2188 | 0.4883 | 48 |
| FedASK | fedmia_loss | 0.0000 | 0.0000 | 0.5000 | 64 |
| FedASK | fedmia_cosine | 0.0000 | 0.1250 | 0.6924 | 64 |
| FedASK | nasr_passive | 0.1875 | 0.1875 | 0.4805 | 32 |
| FedASK | rmia | 0.0000 | 0.0000 | 0.5000 | 48 |
| FedASK | quantile_mia | 0.0625 | 0.1250 | 0.5059 | 48 |

主任务最终测试：

| Method | Round | Loss | Accuracy | Test samples |
|---|---:|---:|---:|---:|
| DP-FPL | 4 | 1.6227 | 0.6108 | 6149 |
| FedASK | 4 | 1.6305 | 0.6097 | 6149 |

方法诊断：

| Method | DP epsilon upper bound | Reconstruction error | Width condition |
|---|---:|---:|---|
| DP-FPL local/global | 122.62 / 511.51 | n/a | n/a |
| FedASK | 61.51 | 2.24e-6 | true |

第二批扩展攻击结果：

| Method | Attack | TPR@1%FPR | TPR@10%FPR | AUC | n |
|---|---|---:|---:|---:|---:|
| DP-FPL | nasr_active | 0.2500 | 0.2500 | 0.4375 | 8 |
| DP-FPL | transfer_representation | 0.0000 | 0.1250 | 0.5273 | 32 |
| DP-FPL | pipra | 0.0000 | 0.1875 | 0.5254 | 48 |
| DP-FPL | imia | 0.3438 | 0.3750 | 0.5859 | 48 |
| DP-FPL | yoqo | 0.0000 | 0.0000 | 0.5000 | 32 |
| DP-FPL | canary | 0.0000 | 0.0000 | 0.5000 | 32 |
| DP-FPL | promptmia | 0.1250 | 0.3125 | 0.6445 | 32 |
| FedASK | nasr_active | 0.5000 | 0.5000 | 0.6250 | 8 |
| FedASK | transfer_representation | 0.0625 | 0.2500 | 0.5000 | 32 |
| FedASK | pipra | 0.0312 | 0.0625 | 0.3594 | 48 |
| FedASK | imia | 0.0625 | 0.0625 | 0.5020 | 48 |
| FedASK | yoqo | 0.0000 | 0.0000 | 0.5000 | 32 |
| FedASK | canary | 0.0000 | 0.0000 | 0.5000 | 32 |
| FedASK | promptmia | 0.0000 | 0.0000 | 0.5000 | 32 |

扩展攻击对应方法诊断：

| Method | DP epsilon upper bound | Active probe steps | Reconstruction error | Width condition |
|---|---:|---:|---:|---|
| DP-FPL local/global | 1900.40 / 511.51 | 80 | n/a | n/a |
| FedASK | 141.51 | 80 | 2.95e-6 | true |

## 初步结论

1. 在更贴近 DP-FPL 官方 notebook 的 Flowers102 / 5 round / prompt length 16 设置下，第一批被动攻击在 `TPR@1%FPR` 上整体较弱；最高为 `nasr_passive` 与 `quantile_mia` 的 0.1875 与 0.0625。

2. 在 official-style FedASK 映射下，FedMIA-II 的 AUC 达到 0.6924，说明存在一定排序信号；但在严格低误报约束下 `TPR@1%FPR` 仍为 0。这表示攻击者能部分排序成员/非成员，但不能在 1% FPR 下稳定抓到成员。

3. 扩展攻击中，DP-FPL 对 `imia` 和 `promptmia` 有更明显响应：`imia` 达到 `TPR@1%FPR=0.3438/AUC=0.5859`，`promptmia` 达到 `TPR@1%FPR=0.125/AUC=0.6445`。FedASK 中除 `nasr_active` 外，其余扩展攻击基本接近随机或低 TPR。

4. `nasr_active` 在 FedASK 上达到 `TPR@1%FPR=0.5/AUC=0.625`，但只有 8 个评估样本，且它包含额外 isolated probe 更新，解释时应弱化为提示性结果，而不是稳健结论。

5. Caltech101 默认 20 轮配置下，FedASK 的 `nasr_passive` 达到 `TPR@1%FPR=0.5`，高于 DP-FPL 的 0.25；但该 FedASK 默认配置的 sketch width 诊断未满足。该结果可作为仓库默认配置现象，不应作为 FedASK 官方风格主结论。

6. official-style FedASK 将 rank/sketch dim 调整为 16 后，重构误差仍很低且 width condition 满足，结果更适合作为论文实现对齐版本。

## 验证与 caveats

- 已用 GPU 运行，未切到 CPU。PyTorch 检测到 2 张 RTX 4090；运行命令显式使用 `gpu: 1`。
- CLIP 加载路径使用本地缓存，代码中设置 `local_files_only=True`。
- 本轮没有运行 `codepoison`。`nasr_active` 与 `promptmia` 属于 isolated probe/query 适配，不进入正式联邦训练轨迹，但会增加 privacy accounting 中的 active probe steps。
- 目前每个实验只跑了 seed 42。由于审计样本数较小，尤其 `nasr_active` 只有 8 个样本，建议后续至少跑 3 个 seed 后报告均值和标准差。
- FedASK 官方仓库是 LLM LoRA 实现；当前仓库是 CLIP soft prompt。这里的“对齐”是算法结构映射，不是逐行复现同一模型/数据任务。
- DP-FPL official-style 使用官方 simulated notebook 的噪声量级映射；若要强调 formal DP budget，应另跑 epsilon-targeted 配置。

## 复现实验命令

```bash
micromamba run -n pfedba python main.py \
  --config configs/fedprompt_privacy.yaml \
  --dataset_name caltech101 \
  --gpu 1 \
  --seed 42 \
  --target_client_id 0 \
  --aggregator dpfpl \
  --defense none \
  --audit_attacks fedmia_loss,fedmia_cosine,nasr_passive,rmia,quantile_mia

micromamba run -n pfedba python main.py \
  --config configs/fedprompt_privacy.yaml \
  --dataset_name caltech101 \
  --gpu 1 \
  --seed 42 \
  --target_client_id 0 \
  --aggregator fedask \
  --defense none \
  --audit_attacks fedmia_loss,fedmia_cosine,nasr_passive,rmia,quantile_mia

micromamba run -n pfedba python main.py \
  --config configs/dpfpl_official_repo_aligned.yaml

micromamba run -n pfedba python main.py \
  --config configs/fedask_official_repo_aligned.yaml

micromamba run -n pfedba python main.py \
  --config configs/dpfpl_official_repo_aligned.yaml \
  --audit_attacks nasr_active,transfer_representation,pipra,imia,yoqo,canary,promptmia

micromamba run -n pfedba python main.py \
  --config configs/fedask_official_repo_aligned.yaml \
  --audit_attacks nasr_active,transfer_representation,pipra,imia,yoqo,canary,promptmia
```
