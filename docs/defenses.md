# 联邦 soft-prompt 成员隐私防御

仓库支持六种彼此独立的防御。一次实验只能选择一个 `defense.name`，不会在后台组合其他防御。

| 运行名 | 方法 | 当前联邦 prompt 适配 |
|---|---|---|
| `cofedmid` | [CoFedMID](https://www.usenix.org/conference/usenixsecurity26/presentation/bai)，USENIX Security 2026 | 覆盖全类别且平衡两两重叠的动态类别子集、EXP3 样本回收、回收样本置信度正则、加权聚合中性 prompt 噪声 |
| `prompt_dp` | [Differentially Private Prompt Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html)，NeurIPS 2023 | 逐样本 prompt 梯度裁剪和高斯噪声，冻结 CLIP 参数不参与隐私优化 |
| `mist` | [MIST](https://www.usenix.org/conference/usenixsecurity24/presentation/li-jiacheng)，USENIX Security 2024 | 将客户端数据分区视为 MIST 子空间，先本地训练，再以其他客户端预测作为反事实目标做 cross-difference 更新 |
| `soft` | [SOFT](https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-kaiyuan)，USENIX Security 2025 | 第一轮 warm-up；随后用客户端验证损失均值选择低损失高风险样本，并以视觉翻转和噪声替代文本 paraphrase |
| `hamp` | [HAMP](https://www.ndss-symposium.org/wp-content/uploads/2024-14-paper.pdf)，NDSS 2024 | 高熵软标签、预测熵正则，以及可微且保持 `argmax` 的温度输出映射 |
| `veil`（兼容旧名 `local_ggeur`、`mirage`） | VEIL，本仓库防御方案 | 客户端不共享统计量；只在本地 CLIP embedding 空间估计类均值/协方差，用几何回声替代个体锚点后训练 prompt |

这些实现是针对“冻结 CLIP、只训练共享 CoOp prompt”的场景适配。SOFT 原论文处理文本，因此本仓库使用保持图像语义的视觉混淆；HAMP 原论文测试阶段使用随机低置信度分数重排，本仓库使用可微温度映射，使主动梯度攻击仍能正常运行并得到分数，而不是因输出不可导而失败。

## 运行模式

攻击与防御都通过命令行单独选择：

```powershell
# 一个攻击 + 一个防御
python main.py --config configs/fedprompt_privacy_quick.yaml --attack fedmia_loss --defense cofedmid

# 仅攻击
python main.py --config configs/fedprompt_privacy_quick.yaml --attack pipra --defense none

# 仅防御，不运行成员推理审计
python main.py --config configs/fedprompt_privacy_quick.yaml --attack none --defense prompt_dp

# 普通联邦 prompt 训练，不运行攻击和防御
python main.py --config configs/fedprompt_privacy_quick.yaml --attack none --defense none
```

`--attack` 支持 `privacy_attacks/auditor.py` 中列出的全部攻击。旧的 `--audit_attacks a,b,c` 参数仍保留用于兼容批量审计，但新实验建议使用单数形式 `--attack`。

## 常用防御参数

参数放在 YAML 的 `defense:` 节点中；命令行 `--defense` 只覆盖方法名。

### CoFedMID

- `cofedmid_max_classes`、`cofedmid_min_classes`：每个客户端从前期到后期分配的类别数。省略时按照类别数与客户端数自动计算。
- `cofedmid_intervals`：EXP3 难度区间数。
- `cofedmid_recycle_ratio`：每批最多回收的排除样本比例。
- `cofedmid_entropy_weight`：回收样本置信度正则强度。
- `cofedmid_exp3_gamma`：EXP3 探索强度。
- `cofedmid_noise_std`：上传 prompt 的高斯扰动标准差。
- `cofedmid_perturb_ratio`：从 prompt 尾部开始扰动的参数比例。FedAvg 下扰动按样本权重严格抵消；FedASK 只扰动 `B`，保持客户端 `A` 固定。DP-FPL 的全局更新随后还会经过裁剪，因此不宣称扰动严格中性。

类别分配遵循论文 class-guided partition 的两个约束：防御联盟联合覆盖完整类别空间，且客户端子集之间的两两重叠尽量平衡。默认 CIFAR100、4 客户端、首轮 50 类/客户端时，最大两两重叠为 17 类；这避免了连续切片导致两个客户端类别子集完全相同。

### Prompt-DP

- `dp_max_grad_norm`：逐样本 prompt 梯度裁剪阈值。
- `dp_noise_multiplier`：噪声标准差与裁剪阈值的比例。
- `dp_delta`：隐私会计中的 δ。

`defense_summary.json` 中的 `epsilon_upper_bound` 使用不声明子采样放大的保守高斯组合上界。它适合比较配置，但不会虚报更小的抽样 DP 预算；主动攻击对客户端发起的额外私有更新查询也计入组合次数。

### MIST

- `mist_cross_steps`：每轮本地训练后的 cross-difference 更新步数。
- `mist_cross_weight`：反事实预测差异损失权重。

MIST 至少需要每轮选择两个客户端。

### SOFT

- `soft_obfuscation_strength`：原图到混淆图的插值比例。
- `soft_noise_std`：混淆图上的高斯噪声标准差。

### HAMP

- `hamp_true_probability`：高熵软标签分配给真实类别的概率。
- `hamp_entropy_weight`：训练阶段熵正则强度。
- `hamp_output_temperature`：审计和部署查询的低置信度温度，必须不小于 1。

### VEIL（原 Local-GGEUR）

VEIL（Variance-Echoed Instance-Less Prompt Learning）是面向联邦提示学习成员推理防御的本地数据替代训练方案。旧实验名 `local_ggeur`、`mirage` 与正式名称 `veil` 使用同一实现。它不上传客户端类均值或协方差：每个客户端只在本地冻结 CLIP image encoder 后，按类别构建 feature bank。对类别 `c`，设本地特征均值为 `μ_c`，中心化特征矩阵为 `Z_c`。实现中使用低秩协方差因子生成扰动：

```text
β = ε Z_c / sqrt(n_c - 1),  ε ~ N(0, I)
```

这等价于从本地经验协方差采样，避免显式求高维协方差特征分解。默认增强样本为 `normalize(μ_c + scale · β)`，即分布级样本参与训练，单个原始样本不作为训练锚点。实现的后备默认将原样本分支替换为 `normalize(μ_c + noise)`；VEIL 论文正式配置使用无噪声的 `normalize(μ_c)`，带噪原型作为独立消融。

- `local_ggeur_augments`：每个 batch 样本标签生成多少个本地几何增强特征。设为 0 可做“仅原样本私有化”消融。
- `local_ggeur_geometry_scale`：几何扰动强度。
- `local_ggeur_anchor_mode`：`class_mean` 或 `sample`。默认 `class_mean`，避免单个原样本直接作为增强锚点；`sample` 用于论文消融。
- `local_ggeur_original_mode`：`drop`、`class_mean`、`class_mean_noise`、`mean_mix`、`blur` 或 `noise`。默认 `class_mean_noise`。
- `local_ggeur_original_noise`：原样本私有化分支的 feature 噪声标准差。
- `local_ggeur_mean_noise_std`：每个客户端、每轮对各类别几何均值锚点加入一次的本地 feature 噪声；默认 0，实验调优时用于削弱单个成员对类别中心的影响。
- `local_ggeur_mean_mix`：`mean_mix`/`blur` 模式下向类均值收缩的比例。
- `local_ggeur_fallback_std`：单样本类别无法估计协方差时的各向同性 fallback 噪声。
- `local_ggeur_entropy_weight`：可选预测熵正则；默认 0，避免无必要地牺牲任务效用。
- `local_ggeur_entropy_rounds`：可选早期熵正则轮数；为 `null` 时只由 `local_ggeur_entropy_weight` 决定是否全程启用。
- `local_ggeur_late_start_round`、`local_ggeur_late_augments`：可选后期几何增强退火；例如 `7/0` 表示第 7 轮起关闭几何增强。当前实验显示该策略会显著伤害效用，不作为推荐默认值。
- `local_ggeur_output_temperature`：最终发布/部署模型的查询温度，必须不小于 1。该校准只作用于最终查询输出，不改变本地训练、联邦聚合或轮次协议观察。
- `local_ggeur_output_margin`：可选 top-1/top-2 logit 间隔上限；默认关闭。
- `local_ggeur_calibrate_observations`：是否把输出校准也作用到轮次审计观察；默认关闭。
- `local_ggeur_class_balanced`：是否按本地类别均匀采样类代表特征；实验中会放大更新侧攻击信号，默认关闭。
- `local_ggeur_upload_clip_norm`：对本地训练后的 prompt delta 做 L2 裁剪。默认 `0.5`。
- `local_ggeur_upload_noise_std`：对裁剪后的 prompt delta 加高斯噪声，噪声标准差为该值乘以裁剪阈值。VEIL 论文正式配置使用 `0.11`；早期 Local-GGEUR 配置使用 `0.07`。

当前推荐的跨攻击默认值是：

```yaml
local_ggeur_augments: 3
local_ggeur_geometry_scale: 0.6
local_ggeur_anchor_mode: class_mean
local_ggeur_original_mode: class_mean
local_ggeur_original_noise: 0.0
local_ggeur_entropy_weight: 0.0
local_ggeur_output_temperature: 4.0
local_ggeur_upload_clip_norm: 0.5
local_ggeur_upload_noise_std: 0.11
```

如果只关注 FedMIA confidence/loss 侧信号，可使用更强的熵正则配置：

```yaml
local_ggeur_augments: 2
local_ggeur_geometry_scale: 0.45
local_ggeur_original_noise: 0.05
local_ggeur_entropy_weight: 0.05
```

该配置对 FedMIA loss 更强，但会使 Nasr/RMIA/YOQO 反弹，因此不是默认跨攻击配置。

推荐消融：

- 完整 VEIL：`anchor_mode=class_mean`，`original_mode=class_mean`，`augments=3`，并使用上传平滑和最终输出温度 4。
- 无几何增强：`local_ggeur_augments=0`。
- 无原样本私有化分支：`local_ggeur_original_mode=drop`。
- 个体锚点消融：`local_ggeur_anchor_mode=sample`。
- 无输出温度消融：`local_ggeur_output_temperature=1.0`。

## 输出

结果目录名包含攻击和防御，例如：

```text
results/cifar100_fedmia_loss_cofedmid_YYYYMMDD_HHMMSS/
```

主要文件：

- `training_metrics.csv`：干净任务损失与准确率；
- `defense_summary.json`：防御名、客户端优化步数、选择率、熵或 cross-difference 等运行统计；
- `privacy_audit/summary.json`：攻击 AUC、低 FPR TPR、所用防御和错误信息；
- `privacy_audit/predictions.csv`：逐候选成员分数；
- `final_prompt.pt`：最终可训练 prompt 参数。

比较防御前后效果时，应固定数据划分、随机种子、目标客户端和攻击参数，分别运行 `--defense none` 与目标防御，并同时报告攻击指标和 `training_metrics.csv` 中的任务效用。
