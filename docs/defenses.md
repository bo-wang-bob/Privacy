# 联邦 soft-prompt 成员隐私防御

仓库支持五种彼此独立的防御。一次实验只能选择一个 `defense.name`，不会在后台组合其他防御。

| 运行名 | 方法 | 当前联邦 prompt 适配 |
|---|---|---|
| `cofedmid` | [CoFedMID](https://www.usenix.org/conference/usenixsecurity26/presentation/bai)，USENIX Security 2026 | 动态类别子集、EXP3 样本回收、回收样本置信度正则、加权聚合中性 prompt 噪声 |
| `prompt_dp` | [Differentially Private Prompt Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html)，NeurIPS 2023 | 逐样本 prompt 梯度裁剪和高斯噪声，冻结 CLIP 参数不参与隐私优化 |
| `mist` | [MIST](https://www.usenix.org/conference/usenixsecurity24/presentation/li-jiacheng)，USENIX Security 2024 | 将客户端数据分区视为 MIST 子空间，先本地训练，再以其他客户端预测作为反事实目标做 cross-difference 更新 |
| `soft` | [SOFT](https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-kaiyuan)，USENIX Security 2025 | 第一轮 warm-up；随后用客户端验证损失均值选择低损失高风险样本，并以视觉翻转和噪声替代文本 paraphrase |
| `hamp` | [HAMP](https://www.ndss-symposium.org/wp-content/uploads/2024-14-paper.pdf)，NDSS 2024 | 高熵软标签、预测熵正则，以及可微且保持 `argmax` 的温度输出映射 |

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
