# Flowers102 同场景公平比较与 Local-GGEUR 优化

日期：2026-07-16

## 比较口径

本实验使用 `configs/flowers_fair_privacy_comparison.yaml` 作为唯一公共配置，
只通过 `--aggregator` 和 `--defense` 切换方法。公共条件为：Flowers102、
10 个客户端且全参与、Dirichlet `alpha=0.1`、每类 16-shot、5 个通信轮次、
1 个本地 epoch、batch size 16、prompt length 16，以及同一个目标客户端 0。
模型训练与 CLIP 前向分别在两张 NVIDIA RTX 4090 上并行完成；数据加载与小型
指标汇总使用主机侧流水线。

审计统一使用 `protocol_plus_released_prompts`，包含 `fedmia_loss`、
`fedmia_cosine`、`fedmia_joint`、`nasr_passive`、`rmia` 和
`quantile_mia`。每组先取 64 个成员和 64 个非成员，校准比例为 0.5；因此
FedMIA 使用 128 个候选，Nasr 的最终评估集为 64，RMIA/Quantile-MIA 为
96。相较早期每组 32 的实验，这一设置降低了低 FPR 指标的离散误差。

三个 seed 的客户端训练/测试样本数如下。每个 seed 内四种方法使用完全相同
的划分；跨 seed 改变划分用于检验稳健性。

| Seed | Train samples per client | Test samples per client |
|---:|---|---|
| 42 | `[151,212,189,186,136,196,158,103,206,95]` | `[437,636,635,499,667,779,541,501,745,709]` |
| 43 | `[140,190,150,176,163,183,118,198,125,189]` | `[591,693,505,577,519,804,556,868,481,555]` |
| 44 | `[153,187,181,101,166,142,175,153,153,221]` | `[680,514,619,428,880,587,520,701,533,687]` |

方法内部参数保留论文/官方仓库映射：DP-FPL 为 rank 4、每轮 1 个私有步骤、
local/global noise multiplier `0.3/0.1`；FedASK 为 rank/sketch dimension 16、
每轮 10 个私有步骤、noise multiplier 1.0。Local-GGEUR 使用 FedAvg prompt
主干；这些是方法定义的一部分，而非公共场景变量。

## 优化后的 Local-GGEUR

在 seed 42 上进行开发搜索后固定如下配置，再用 seed 43/44 独立检验：

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
噪声从 0.07 提高到 0.11。实现同时新增上传裁剪前范数统计，确认 Flowers102
上平均原始上传范数约为 `0.056--0.077`；默认 0.5 阈值基本不触发裁剪，主要
保护来自上传噪声和本地几何替代。

## 三 seed 结果

下表前三个数值列为 mean +/- sample standard deviation，六个单项攻击列为
三 seed 均值。`Worst TPR` 是每次运行六个攻击中最大的 `TPR@1%FPR`，
`Mean TPR` 是六个攻击的平均值；两者越低越好。

| Method | Accuracy | Worst TPR@1%FPR | Mean TPR@1%FPR | FedMIA loss | FedMIA cosine | FedMIA joint | Nasr | RMIA | Quantile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DP-FPL | 0.5545 +/- 0.1053 | 0.1042 +/- 0.0651 | 0.0408 +/- 0.0278 | 0.0000 | 0.0625 | 0.0885 | 0.0729 | 0.0000 | 0.0208 |
| FedASK | 0.5270 +/- 0.1425 | 0.0677 +/- 0.0631 | 0.0286 +/- 0.0256 | 0.0000 | 0.0365 | 0.0573 | 0.0625 | 0.0000 | 0.0156 |
| FedAvg, no defense | 0.6632 +/- 0.0039 | 0.2083 +/- 0.1329 | 0.0964 +/- 0.0994 | 0.1406 | 0.1042 | 0.1042 | 0.1562 | 0.0052 | 0.0677 |
| Local-GGEUR | **0.6229 +/- 0.0016** | 0.1354 +/- 0.1542 | 0.0321 +/- 0.0376 | 0.0417 | **0.0000** | **0.0052** | 0.1354 | 0.0052 | **0.0052** |

逐 seed 的主指标如下：

| Seed | Method | Accuracy | Worst TPR | Mean TPR |
|---:|---|---:|---:|---:|
| 42 | DP-FPL | 0.6107 | 0.0312 | 0.0104 |
| 42 | FedASK | 0.6097 | 0.0312 | 0.0078 |
| 42 | Local-GGEUR | 0.6221 | 0.0625 | 0.0104 |
| 43 | DP-FPL | 0.6198 | 0.1250 | 0.0469 |
| 43 | FedASK | 0.6089 | 0.1406 | 0.0573 |
| 43 | Local-GGEUR | 0.6247 | 0.3125 | 0.0755 |
| 44 | DP-FPL | 0.4331 | 0.1562 | 0.0651 |
| 44 | FedASK | 0.3625 | 0.0312 | 0.0208 |
| 44 | Local-GGEUR | 0.6219 | 0.0312 | 0.0104 |

## 结论与边界

优化后的 Local-GGEUR 在效用与稳定性上明显优于两种 DP 训练方式：平均准确率
比 DP-FPL 高 6.84 个百分点、比 FedASK 高 9.59 个百分点，且三个 seed 的
标准差只有 0.16 个百分点。相对无防御 FedAvg，它牺牲 4.03 个百分点准确率，
把平均攻击 TPR 从 0.0964 降到 0.0321，降低 66.7%。

隐私方面，Local-GGEUR 的六攻击平均 TPR 低于 DP-FPL (`0.0321 < 0.0408`)，
与 FedASK 接近但略高 (`0.0321` 对 `0.0286`)；FedMIA cosine/joint 和
Quantile-MIA 的均值优于两种方法。另一方面，seed 43 的 Nasr 尾部使
Local-GGEUR 的 worst-attack 均值仍高于 DP-FPL/FedASK。因此当前证据支持
“更好的整体隐私-效用折中和明显更稳定的任务效用”，但不支持“所有 seed、
所有攻击单项都严格占优”。后续优化应优先针对公开 prompt 梯度的 Nasr 尾部，
不能通过更换审计视角或仅做输出温度校准来规避该问题。

## 失败消融

- 上传噪声 0.07 提高到 0.11 后，seed 42 的五个非 Nasr 攻击全部降为 0，
  准确率仍为 0.6221，因此选择 0.11。
- 把上传 clip norm 降到 0.05 会裁剪约 54% 更新，但没有降低 Nasr。
- 每轮类均值锚点噪声 0.005 不改变主隐私指标；0.015 把准确率降到 0.6079，
  仍未降低 Nasr，因此推荐值保持 0。
- `class_mean_noise` 的样本级噪声 0.08/0.15 在 seed 43 上使 Nasr 升到
  0.3750/0.3125，没有优于直接 `class_mean` 的 0.3125。
- 每个样本的几何增强数从 3 提到 6 未改善 seed 43，且计算开销显著增加。

## 可复核命令

```bash
# 单 seed 四方法；修改 --seed 为 43/44
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 0 --aggregator dpfpl --defense none
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 1 --aggregator fedask --defense none
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 0 --aggregator fedavg --defense none
python main.py --config configs/flowers_fair_privacy_comparison.yaml \
  --seed 42 --gpu 1 --aggregator fedavg --defense local_ggeur

# 校验公共配置完全一致并生成表格
python -m utils.fair_comparison RUN_DIR_1 RUN_DIR_2 RUN_DIR_3 RUN_DIR_4
python -m utils.fair_comparison --aggregate-seeds RUN_DIRS...
```

正式三 seed 结果目录未纳入 Git，符合仓库不提交 `results/` 的约束。对应目录
时间戳分别为 seed 42: `171513/171955/172426`，seed 43:
`173512/173938`，seed 44: `174954/175425`；完整目录可由上述命令或
`run_config.yaml` 中的 seed 核验。
