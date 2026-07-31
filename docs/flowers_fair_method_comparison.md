# Flowers102 同场景公平比较与 Local-GGEUR 优化

> 论文阶段本方法正式更名为 **VEIL**（Variance-Echoed Instance-Less Prompt
> Learning）；`local_ggeur` 是同一实现的历史配置名。本文记录的是五 seed
> 开发阶段证据。原 VEIL 论文及其结果聚合脚本已从仓库移除，本文仅作为
> 历史实验记录保留。

日期：2026-07-16

## 最终结论

在五个独立 seed 的 Flowers102 同场景实验中，优化后的 Local-GGEUR 同时取得：

- 平均准确率 `0.6136`，高于 DP-FPL 的 `0.5628` 和 FedASK 的 `0.5420`；
- 每次运行五种论文攻击中最强攻击的 `TPR@1%FPR` 均值 `0.0500`，低于
  DP-FPL 和 FedASK 的 `0.0688`；
- 五攻击简单平均 TPR `0.0150`，低于 DP-FPL 的 `0.0206` 和 FedASK 的
  `0.0219`。

因此，以任务准确率、worst-attack TPR 和五攻击平均 TPR 衡量，Local-GGEUR
在当前场景的五 seed 经验均值上均优于 DP-FPL 与 FedASK。它不是在每个 seed、
每个攻击单项上都严格占优，五个 seed 也不足以支撑强统计显著性声明。

无防御 FedAvg 的平均准确率为 `0.6621`，且本批有限候选审计得到更低的
worst-attack TPR `0.0312`。因此不能宣称 Local-GGEUR 优于无防御 FedAvg，
也不能把这些实验解释为已经证明形式化隐私保证；DP-FPL 的优势仍包括其
差分隐私机制能够提供的理论保证。

## 公平比较口径

所有方法都使用 `configs/flowers_fair_privacy_comparison.yaml`，只通过
`--aggregator` 和 `--defense` 切换方法。公共条件为：

| Setting | Value |
|---|---:|
| Dataset | Flowers102 |
| Clients / sampled | 10 / 10 |
| Dirichlet alpha | 0.1 |
| Few-shot | 16 per class，训练集共 1632 个样本 |
| Global rounds | 5 |
| Local epochs | 1 |
| Batch size | 16 |
| Prompt length | 16 |
| Target client | 0 |
| Seeds | 42, 43, 44, 45, 46 |

训练和 CLIP 前向使用两张 NVIDIA RTX 4090 并行执行；数据加载与小型指标汇总
使用主机侧流水线。CLIP 始终通过 `local_files_only=True` 从本地缓存加载。

方法内部参数属于各算法定义，因此保留论文及官方仓库的更新规则：DP-FPL 使用
rank 4、每轮 1 个私有步骤、local/global noise multiplier `0.3/0.1`；FedASK
使用 rank/sketch dimension 16、每轮 10 个私有步骤、noise multiplier 1.0。
Local-GGEUR 使用 FedAvg prompt 主干。公共数据、轮次和审计预算完全一致，
算法内部步骤数和隐私机制不被强行改写为相同。

## 消除审计偏差

初始实验按数据集顺序抽取非成员。seed 43 中，64 个成员覆盖多个类别，而 64 个
非成员中类别 7 占 53 个；Nasr 主要选择 loss 与 prompt-gradient norm，因而可能
把类别差异误当作成员信号。该批旧结果不用于最终结论。

最终审计设置 `audit.match_candidate_labels=true`，对成员和非成员使用完全相同的
标签直方图。程序从目标客户端及其他客户端测试集构造匹配非成员集合；若任一
类别数量不足则直接报错。审计摘要的 `candidate_sampling` 保存两组直方图，
测试验证二者严格相等。

候选 DataLoader 还使用独立且固定的 `torch.Generator`。即使审计关闭 shuffle，
候选提取也不会推进训练使用的全局 PyTorch RNG，从而避免不同审计时机改变后续
训练批次顺序。正式五 seed 结果均来自加入该隔离后的重新运行。

统一攻击集合为：`fedmia_loss`、`fedmia_cosine`、`nasr_passive`、`rmia`、
`quantile_mia`。每组先取 64 个成员和 64 个非成员，
校准比例 0.5；FedMIA 使用 128 个候选，Nasr 最终评估 64 个候选，
RMIA/Quantile-MIA 最终评估 96 个候选。

## 优化后的 Local-GGEUR

开发搜索使用 seed 42/43；seed 44--46 用于固定配置后的外部复验。最终配置为：

```yaml
local_ggeur_augments: 3
local_ggeur_geometry_scale: 0.60
local_ggeur_anchor_mode: class_mean
local_ggeur_original_mode: class_mean
local_ggeur_original_noise: 0.0
local_ggeur_mean_noise_std: 0.0
local_ggeur_entropy_weight: 0.0
local_ggeur_upload_clip_norm: 0.5
local_ggeur_upload_noise_std: 0.11
```

相较此前 CIFAR100 推荐点，本配置将原样本分支改为直接类均值替换，并把上传
噪声从 0.07 提高到 0.11。实现同时记录上传裁剪前范数；Flowers102 上该范数
通常远低于 0.5，保护主要来自本地几何替代和上传噪声，而非大量裁剪。

## 五 seed 汇总

下表均为 `mean +/- sample standard deviation`。`Worst TPR` 是每次运行五种攻击
中最大的 `TPR@1%FPR`，用于避免一种较强攻击被跨攻击平均数掩盖。

| Method | Accuracy | Worst TPR@1%FPR | Mean TPR@1%FPR | FedMIA-I | FedMIA-II | Nasr | RMIA | Quantile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DP-FPL | 0.5628 +/- 0.0757 | 0.0688 +/- 0.0324 | 0.0206 +/- 0.0110 | 0.0000 +/- 0.0000 | 0.0219 +/- 0.0324 | 0.0437 +/- 0.0419 | 0.0000 +/- 0.0000 | 0.0375 +/- 0.0360 |
| FedASK | 0.5420 +/- 0.1032 | 0.0688 +/- 0.0407 | 0.0219 +/- 0.0121 | 0.0000 +/- 0.0000 | 0.0281 +/- 0.0232 | 0.0375 +/- 0.0407 | 0.0000 +/- 0.0000 | 0.0437 +/- 0.0487 |
| FedAvg, no defense | 0.6621 +/- 0.0034 | 0.0312 +/- 0.0292 | 0.0131 +/- 0.0124 | 0.0063 +/- 0.0086 | 0.0031 +/- 0.0070 | 0.0250 +/- 0.0261 | 0.0156 +/- 0.0349 | 0.0156 +/- 0.0156 |
| Local-GGEUR | 0.6136 +/- 0.0090 | 0.0500 +/- 0.0204 | 0.0150 +/- 0.0087 | 0.0063 +/- 0.0086 | 0.0094 +/- 0.0140 | 0.0125 +/- 0.0171 | 0.0094 +/- 0.0210 | 0.0375 +/- 0.0360 |

相对 DP-FPL，Local-GGEUR 平均准确率高 5.08 个百分点，worst-attack TPR
降低 27.3%，五攻击平均 TPR 降低 27.3%。相对 FedASK，准确率高 7.16 个
百分点，worst-attack TPR 降低 27.3%，五攻击平均 TPR 降低 31.4%。

逐 seed 主指标：

| Seed | Method | Accuracy | Worst TPR | Mean TPR |
|---:|---|---:|---:|---:|
| 42 | DP-FPL | 0.6108 | 0.0625 | 0.0187 |
| 42 | FedASK | 0.6097 | 0.0625 | 0.0312 |
| 42 | FedAvg | 0.6575 | 0.0781 | 0.0312 |
| 42 | Local-GGEUR | 0.6216 | 0.0312 | 0.0063 |
| 43 | DP-FPL | 0.6212 | 0.0938 | 0.0281 |
| 43 | FedASK | 0.6089 | 0.0938 | 0.0344 |
| 43 | FedAvg | 0.6621 | 0.0312 | 0.0156 |
| 43 | Local-GGEUR | 0.6190 | 0.0625 | 0.0125 |
| 44 | DP-FPL | 0.4336 | 0.0781 | 0.0312 |
| 44 | FedASK | 0.3623 | 0.0312 | 0.0125 |
| 44 | FedAvg | 0.6632 | 0.0312 | 0.0156 |
| 44 | Local-GGEUR | 0.6198 | 0.0781 | 0.0187 |
| 45 | DP-FPL | 0.5853 | 0.0938 | 0.0219 |
| 45 | FedASK | 0.5755 | 0.1250 | 0.0250 |
| 45 | FedAvg | 0.6608 | 0.0156 | 0.0031 |
| 45 | Local-GGEUR | 0.6045 | 0.0469 | 0.0281 |
| 46 | DP-FPL | 0.5629 | 0.0156 | 0.0031 |
| 46 | FedASK | 0.5536 | 0.0312 | 0.0063 |
| 46 | FedAvg | 0.6668 | 0.0000 | 0.0000 |
| 46 | Local-GGEUR | 0.6032 | 0.0312 | 0.0094 |

Local-GGEUR 的任务准确率标准差明显小于两种私有训练方法；其聚合 worst-attack
TPR 也更低、更稳定。不过单个 seed 仍有反例，例如 seed 44 的 FedASK 隐私
指标低于 Local-GGEUR，因此结论限定为五 seed 聚合效果。

## 优化消融

- 上传噪声 0.11 是当前最佳折中；提高到 0.15/0.20 会使 Nasr、RMIA 或
  Quantile-MIA 反弹。
- 把 upload clip norm 降到 0.05 会裁剪约 54% 更新，但没有稳定降低 Nasr。
- 类均值锚点噪声 0.005 不改变主要隐私指标；0.015 会明显降低准确率。
- `class_mean_noise` 的样本级噪声 0.08/0.15 未优于直接 `class_mean`。
- 几何增强数从 3 提到 6 增加计算量但未改善最坏攻击。
- 类均衡抽样和完全移除原样本分支均未优于最终配置。

## 复核命令

```bash
# 修改 seed 为 42--46；可在两张 GPU 上并行
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 0 --aggregator dpfpl --defense none
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 1 --aggregator fedask --defense none
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 0 --aggregator fedavg --defense none
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 1 --aggregator fedavg --defense local_ggeur

# 对结果目录进行公共配置一致性校验并输出汇总表
python -m utils.fair_comparison RUN_DIR_1 RUN_DIR_2 RUN_DIR_3 RUN_DIR_4
python -m utils.fair_comparison --aggregate-seeds RUN_DIRS...
```

`results/` 未纳入 Git。正式五 seed 结果可通过各目录 `run_config.yaml` 中的
`seed`、`match_candidate_labels=true` 和方法配置核验。
