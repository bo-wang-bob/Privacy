# 联邦成员隐私防御

仓库支持多种彼此独立的防御。一次实验只能选择一个 `defense.name`，不会在后台组合其他防御。

| 运行名 | 方法 | 当前联邦 prompt 适配 |
|---|---|---|
| `cofedmid` | [CoFedMID](https://www.usenix.org/conference/usenixsecurity26/presentation/bai)，USENIX Security 2026 | 覆盖全类别且平衡两两重叠的动态类别子集、EXP3 样本回收、回收样本置信度正则、加权聚合中性 prompt 噪声 |
| `prompt_dp` | [Differentially Private Prompt Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html)，NeurIPS 2023 | 逐样本 prompt 梯度裁剪和高斯噪声，冻结 CLIP 参数不参与隐私优化 |
| `record_dp` | 客户端侧记录级 DP-SGD | Poisson 记录采样、完整可训练参数联合梯度裁剪、sampled-Gaussian RDP 会计；支持 ResNet18 FedAvg 与 BERT Adapter one-batch FedSGD |
| `mist` | [MIST](https://www.usenix.org/conference/usenixsecurity24/presentation/li-jiacheng)，USENIX Security 2024 | 将客户端数据分区视为 MIST 子空间，先本地训练，再以其他客户端预测作为反事实目标做 cross-difference 更新 |
| `soft` | [SOFT](https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-kaiyuan)，USENIX Security 2025 | 第一轮 warm-up；随后用客户端验证损失均值选择低损失高风险样本，并以视觉翻转和噪声替代文本 paraphrase |
| `hamp` | [HAMP](https://www.ndss-symposium.org/wp-content/uploads/2024-14-paper.pdf)，NDSS 2024 | 高熵软标签、预测熵正则，以及可微且保持 `argmax` 的温度输出映射 |
| `perturb` | FedMIA Perturb 基线 | 裁剪客户端可训练 prompt delta，并在上传前加入高斯噪声 |
| `sparse` | FedMIA Sparse 基线 | 上传前按绝对值保留最大的 prompt delta 元素，其余置零 |
| `mixup` | FedMIA Mixup 基线 | 本地 prompt 训练使用 Beta 系数混合图像和标签损失 |
| `sampling` | FedMIA Data Sampling 基线 | 每个本地 batch 无放回抽取固定比例样本参与训练 |
| `data_aug` | FedMIA Data Aug 基线 | 对已经预处理的 CLIP 张量做翻转、平移和颜色扰动 |
| `data_aug_sampling` | FedMIA Data Aug + Sampling 基线 | 在同一本地训练分支中先抽样再增强 |
| `iclr` | ICLR（暂定名，第一阶段） | 从上一轮全局模型、客户端本地上传和真实聚合权重反演其他客户端聚合模型，并按逐样本损失差降序排名 |

这些实现是针对“冻结 CLIP、只训练共享 CoOp prompt”的场景适配。SOFT 原论文处理文本，因此本仓库使用保持图像语义的视觉混淆；HAMP 原论文测试阶段使用随机低置信度分数重排，本仓库使用可微温度映射，使主动梯度攻击仍能正常运行并得到分数，而不是因输出不可导而失败。

## 常用防御参数

参数放在 YAML 的 `defense:` 节点中；命令行 `--defense` 只覆盖方法名。

### ICLR（第一阶段）

当前实现使用服务器实际记录的线性聚合权重。客户端对象自然保留上一轮本地模型；下一轮用新全局模型覆盖它之前，客户端即时读取该模型并计算：

```text
theta_-k^t = (theta_global^(t+1) - w_k^t * theta_k^t) / (1 - w_k^t)
```

客户端下一次参与训练时，对该次本地更新实际消费的 batch 流计算：

```text
s_i = L(x_i; theta_-k^t) - L(x_i; theta_k^t)
```

两套参考参数只在当前客户端打分期间临时存在，打分完成后立即释放，不在聚合后为每个客户端保存完整模型副本。`iclr_ranked_positions` 是上述 batch 流拼接后的降序位置，而不是原始数据集的永久索引；`iclr_ranked_scores` 和 `iclr_ranked_labels` 分别保存对应的有序分数与标签。首轮没有历史模型对，因此不打分。若采用部分客户端参与，只有连续两轮均被选中的客户端能够执行这种精确反演。现阶段只生成排名并保存在客户端运行态，同时将汇总统计写入 `defense_summary.json`；不会根据排名筛样本、改变损失或改变原有训练顺序。该实现支持预计算特征的 CLIP-MLP/CLIP-Adapter，也支持直接编码原始图像的 CLIP-LoRA，并要求每轮至少聚合两个客户端。LoRA 使用一个全局共享的冻结 CLIP 主干，但每个客户端模型独立持有自己的 `lora_A/lora_B`；下一轮 ICLR 在全局参数覆盖之前直接读取该客户端模型中的上次上传状态。这些因子不会形成服务器端客户端状态映射，也不会写入结果目录。

BERT Adapter 使用 `iclr_analysis_timing: post_round`：在每个配置的分析轮完成聚合后，
由真实 one-batch 上传梯度构造对应客户端 step state，再与其他客户端梯度形成的聚合
step state 比较。
默认 `iclr_analysis_interval: 50`，因此 500 轮任务分析第 50、100、…、500 轮，并将
逐客户端统计写入 `iclr_round_metrics.csv`，逐样本分数写入 `iclr_round_samples.csv`，
轮次索引写入 `iclr_series.json`。这个后聚合模式不需要 CLIP 辅助特征统计，
所以 BERT 配置使用 `iclr_feature_statistics: false`。CLIP 原有的下一轮打分协议不变。
BERT 的同轮 ProjRes 结果会额外保存成员的 `batch_position` 和 `local_sample_index`，并与
ICLR 记录做四键严格连接。`privacy_audit/iclr_projres_samples.csv` 保存逐样本配对；
`iclr_projres_relationship.csv/json` 保存 Pearson/Spearman、类别控制 Spearman、ICLR
高低分组的 ProjRes 分数，以及双方 Top-K 的重合率和富集倍数。ProjRes 命中率不再复用
固定残差阈值预测，而是以同轮共享的 160 个非成员连续分数分别在 10% 和 1% FPR 下复算，
并报告 ICLR 高低分组的命中率差与比值；不再统计 0.1% FPR。

为了验证该分数能否衡量样本特异性，ICLR 会为训练 batch 携带相对于客户端训练集的稳定本地索引，并在训练期间为每个样本累计 `mean`、`last`、`max`、标准差和观测次数。审计完成后，这些索引会与 `low_fpr_full` 候选池中的成员索引严格连接，并针对每种攻击分别计算：

- Pearson 和 Spearman 相关性；
- 类别内秩中心化 Spearman 及逐类别宏平均 Spearman，用于控制类别分布泄漏；
- ICLR 最高/最低分位组的平均攻击分数差；
- ICLR Top-K 与攻击 Top-K 的重合率和相对随机期望的富集倍数；
- 在 `FPR=0.1/0.01/0.001` 下，全部成员、ICLR 高分组和低分组被攻击命中的比例及其差值。

`defense.iclr_validation_top_fraction` 控制高分组和低分组比例，默认 `0.2`。验证只比较能够严格对齐且至少获得一次 ICLR 观测的成员样本；它衡量的是 ICLR 分数与“成员样本被攻击识别的难易程度”之间的关联，不能单独证明因果关系或防御有效性。`candidate_sampling: low_fpr_full` 和 `balanced_global_holdout` 都会保存目标客户端成员的稳定本地索引，可用于该验证；缺少本地索引的采样方式会在验证 JSON 中标记为不可用。

验证产物位于任务的 `privacy_audit/`：

- `iclr_attack_samples.csv`：攻击、候选索引、客户端本地索引、类别、攻击分数及 ICLR 聚合分数；
- `iclr_attack_relationship.csv`：每个攻击、客户端和 ICLR 聚合方式的关联指标；
- `iclr_attack_relationship.json`：方法定义、覆盖范围和完整指标；
- `summary.json` 中的 `iclr_validation`：状态、覆盖规模和产物入口。

### FedMIA 比较基线

这六个基线只允许在共享模型的集中式 `FedAvg` 协议下单独运行。所有更新操作仅处理 `requires_grad=True` 的 prompt 参数，冻结的 CLIP 权重不会训练、稀疏化或加噪。

- `perturb_clip_norm`：上传前 prompt delta 的全局 L2 裁剪阈值；`perturb_noise_std`：裁剪后加入的高斯噪声绝对标准差。FedMIA 报告的噪声标准差范围为 0.01–0.5。
- `sparse_ratio`：按绝对值从小到大置零的比例，取值 `[0, 1)`；论文考察 0.1–0.99。
- `mixup_alpha`：对称 Beta 分布的参数，必须为正。
- `sampling_ratio`：每批保留的数据比例，取值 `(0, 1]`；论文考察 0.1–1.0。
- `data_aug_strength`：随机平移幅度；`data_aug_flip_probability`：水平翻转概率；`data_aug_color_jitter`：亮度与对比度扰动强度。

`data_aug` 没有重新调用图像处理器，而是在已归一化的 NCHW CLIP 输入上实施确定性可复现的张量变换，因此不需要数据集或模型特定的反归一化逻辑。

### CoFedMID

- `cofedmid_max_classes`、`cofedmid_min_classes`：每个客户端从前期到后期分配的类别数。省略时按照类别数与客户端数自动计算。
- `cofedmid_intervals`：EXP3 难度区间数。
- `cofedmid_recycle_ratio`：每批最多回收的排除样本比例。
- `cofedmid_entropy_weight`：回收样本置信度正则强度。
- `cofedmid_exp3_gamma`：EXP3 探索强度。
- `cofedmid_noise_std`：上传 prompt 的高斯扰动标准差。
- `cofedmid_perturb_ratio`：从 prompt 尾部开始扰动的参数比例。FedAvg 下扰动按样本权重严格抵消。

类别分配遵循论文 class-guided partition 的两个约束：防御联盟联合覆盖完整类别空间，且客户端子集之间的两两重叠尽量平衡。默认 CIFAR100、4 客户端、首轮 50 类/客户端时，最大两两重叠为 17 类；这避免了连续切片导致两个客户端类别子集完全相同。

### Prompt-DP

- `dp_max_grad_norm`：逐样本 prompt 梯度裁剪阈值。
- `dp_noise_multiplier`：噪声标准差与裁剪阈值的比例。
- `dp_delta`：隐私会计中的 δ。

`defense_summary.json` 中的 `epsilon_upper_bound` 使用不声明子采样放大的保守高斯组合上界。它适合比较配置，但不会虚报更小的抽样 DP 预算；主动攻击对客户端发起的额外私有更新查询也计入组合次数。

### Record-DP

`record_dp` 保护客户端训练集中的单条记录。图像任务的一条记录是一张图像，文本
任务的一条记录是一条完整序列，而不是 token。每一步独立以 `q=B/n_i` 对客户端
`i` 的本地记录做 Poisson 采样，对每条记录在全部可训练参数上的联合梯度裁剪到
`max_grad_norm`，对裁剪梯度和加入标准差为
`noise_multiplier * max_grad_norm` 的高斯噪声，再除以固定 expected batch size。
microbatch 只改变计算方式；所有 microbatch 累加后仅添加一次噪声。

主要参数：

- `target_epsilon` 与数值 `noise_multiplier` 二选一。选择目标 epsilon 时，运行前按
  最坏客户端计划自动反推一个共享 noise multiplier。
- `delta`：目标近似 DP 参数。
- `grad_sample_backend`：`vmap`、`loop` 或 `auto`。ResNet18 默认使用分块
  `vmap`；共享 Transformer Adapter 使用通用逐记录参考实现。
- `microbatch_size`：一次保留的逐记录计算块大小，不是新的 DP batch。
- `reproducible_noise`：仅测试可设为 `true`；此时输出会明确标记
  `formal_dp_enabled: false`。

每个客户端跨轮顺序组合；不同客户端数据互不相交，因此发布的记录级预算取逐客户端
epsilon 最大值，而不是把所有客户端相加。会计结果、实际/计划步数和公开采样率
写入 `defense_summary.json`。默认不发布真实 Poisson batch 大小、空 batch 比例或
裁剪比例，因为这些数据相关诊断量本身没有加噪。显式设置
`release_private_diagnostics: true` 可用于封闭研究环境，但会令
`formal_dp_enabled` 变为 `false`。成员推理审计文件包含实验真值，不属于可公开的
DP 机制输出。

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
