# 联邦成员隐私防御

仓库支持多种彼此独立的防御。一次实验只能选择一个 `defense.name`，不会在后台组合其他防御。

| 运行名 | 方法 | 当前适配 |
|---|---|---|
| `cofedmid` | [CoFedMID](https://www.usenix.org/conference/usenixsecurity26/presentation/bai)，USENIX Security 2026 | 六个 PEFT 模型的动态类别分配、EXP3 回收、软目标正则、聚合中性上传扰动；默认全客户端联盟 |
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
| `www` | WWW（原 ICLR） | Poisson 样本采样、损失差升序及固定尾部积分权重、逐样本裁剪、add/remove 采样 RDP 核算 |

历史通用防御主要针对“冻结 CLIP、只训练共享 CoOp prompt”的场景适配；正式模型的可用范围以 `configs/experiment_catalog.yaml` 为准。CoFedMID 已单独适配当前六个 PEFT 模型的 one-batch FedSGD。SOFT 原论文处理文本，因此本仓库使用保持图像语义的视觉混淆；HAMP 原论文测试阶段使用随机低置信度分数重排，本仓库使用可微温度映射。

## 常用防御参数

参数放在 YAML 的 `defense:` 节点中；命令行 `--defense` 只覆盖方法名。

### WWW

原 ICLR 已统一更名为 WWW：配置、代码和新产物使用 `www` / `www_*`，旧名称不再作为防御入口。历史 `results/` 及其中的 `iclr_*` 文件不改写；历史观测型 ICLR 不能当作 WWW 的防御结果。

WWW 支持 CLIP-MLP、CLIP-Adapter、CLIP-LoRA、BERT Adapter 和 BERT-LoRA，要求线性 FedSGD/FedAvg 且每轮至少两个客户端。统一入口保留各模型既有的 IID、few-shot、学习率、轮数及 one-batch 等权 FedSGD；batch 大小作为 Poisson 采样的期望值。默认每一轮执行防御，与攻击审计和任务评估频次无关。

**样本排序与尾部。** 在当前全局参数覆盖客户端状态前，用上一轮防御后的上传与实际聚合权重重建参考模型：

```text
theta_-k = (theta_global - w_k * theta_k) / (1 - w_k)
M_i = loss(theta_-k, z_i) - loss(theta_k, z_i)
```

与普通 `record_dp` 共用 Poisson 采样器：每条本地记录独立以固定概率 `q=b/N` 进入当前 batch，其中 `b=min(configured_batch_size,N)` 在训练前确定。每步重新抽取，实际样本数 `n` 可以小于或大于 `b`，也可以为 0；空抽样不会重抽，仍执行一次纯噪声更新并计入预算。样本本地索引随 batch 保留，以便真实 Batch 攻击精确配对。

对真实 batch 按 `M_i` 稳定升序排序。默认 `defense.www_tail_fraction=0.8`，尾部宽度在每个客户端训练前固定为 `m=ceil(0.8*b)` 个长度为 C 的区间，实际落入尾部的样本数为 `min(n,m)`，仍是损失差最大的样本。同分保持原顺序。这里 80% 以**固定期望 batch 大小**计算，不能每次按随机的 `n` 重算尾部宽度，否则不能直接沿用论文的 C 增删敏感度界。比如期望 batch=32 时尾部宽度固定为 26；实际抽到 35 条时仍取最后 26 条，抽到 20 条时全部使用完整尾部最右侧的 20 个区间。期望 batch=16 时尾部宽度为 13。

首轮没有历史参考模型，统一以 C 裁剪并加噪；部分参与时，缺少紧邻上一轮上传的客户端也使用这一回退规则。

**INO-SGD 自适应裁剪。** 使用 [Tian 等，INO-SGD](https://arxiv.org/abs/2605.07930) 的 Algorithm 1、Definition 3.2 和 Appendix C.2.3。WWW 将论文默认的 loss 降序替换为上述 `M_i` 升序；其它排序的适用性见 Appendix C.2.2。设所有样本初始阈值为 `C`，累计阈值 `c_j=jC`，固定尾部长度 `gamma=ceil(0.8b)C`，其中 b 是期望 batch 大小，n 是实际抽样大小。尾部重要性函数为翻转的 Beta CDF：

```text
f_tail(u) = I_(1-u/gamma)(alpha, beta)
f_B(c) = 1                           if c <= nC - gamma
         f_tail(c - (nC - gamma))    otherwise
rho_j = integral[f_B(c), c_(j-1), c_j] / C
g_bar_i = g_i / max(1, ||g_i||_2 / C)
g_private = (sum_i rho_i * g_bar_i + Normal(0, sigma_sum^2 I)) / b
```

梯度范数联合覆盖所有可训练参数，冻结主干不参与。采用“先裁剪到 C，再乘 rho”的论文算法，故 `rho_i*C` 是最终贡献范数上界；并非直接把原始梯度裁剪到 `rho_i*C`，两者对小梯度的处理不同。默认 `alpha=beta=1`，尾部函数为线性下降。当实际 batch 小于尾部宽度时，依照 Appendix C.2.3 使用尾部函数最右侧的区间。比如 `b=10` 且实际 batch 有 10 条样本时，前两条权重为 `1`，最后八条权重依次为 `0.9375、0.8125、0.6875、0.5625、0.4375、0.3125、0.1875、0.0625`；`C=8` 时尾部贡献范数上界依次为 `7.5、6.5、5.5、4.5、3.5、2.5、1.5、0.5`。

**全程隐私预算。** 默认 `defense.target_epsilon=3` 为整个训练任务预算，`max_grad_norm=8` 为基准阈值，`delta=1e-5`。WWW 与普通 `record_dp` 使用相同的 `add_remove` 邻接、客户端采样率、计划更新步数、Poisson sampled-Gaussian RDP 校准函数和固定期望 batch 归一化：

```text
Delta_sum = C
sigma_sum = noise_multiplier * C
RDP_i(a,T_i) = T_i * poisson_sampled_gaussian_rdp(noise_multiplier, q_i, a)
epsilon_i = min_a [RDP_i(a,T_i) + log(1/delta)/(a-1)]
epsilon_run = max_i epsilon_i
```

固定长度、右对齐的 INO 尾部使增加或删除一条样本引起的总变化（包括其他样本权重变化）被 C 控制。采样率、期望 batch 大小和尾部宽度是训练前固定的公开机制参数；邻接比较中不随记录的增删重新校准这些参数。不同客户端数据互不相交，各客户端跨步顺序组合，整体取最大值。FedAvg 还计入每轮全部计划 Poisson 更新，空 batch 也计步。

自动噪声校准覆盖完整计划。相同 epsilon、delta、C、客户端规模、期望 batch 和轮数下，WWW 与普通 `record_dp` 的噪声尺度一致。默认各任务独立使用系统随机种子，采样分布相同但实际抽样名单不保证相同；仅封闭调试时可共同设置 `defense.reproducible_noise=true`，此时采样及噪声种子规则也一致，并标记 `formal_dp_enabled=false`。

超过校准步数会在更新前报错；手动指定不足以满足预算的噪声会被拒绝。当前不支持隔离式主动客户端探测及含 BatchNorm 运行统计的模型。早期 WWW 的打乱采样、replace_one 与无放大核算属于旧协议，已有结果不改写，不能直接混入当前实验。

常用参数集中在 `configs/experiment_catalog.yaml`：

| 配置键（`defense.` 下） | 默认值 | 用途 |
|---|---:|---|
| `target_epsilon` | `3.0` | 全程隐私预算 |
| `max_grad_norm` | `8.0` | 初始联合 L2 裁剪阈值 |
| `delta` | `1e-5` | DP 的 delta |
| `noise_multiplier` | `auto` | 相对于 `C` 的噪声倍数 |
| `www_tail_fraction` | `0.8` | 固定期望 batch 的尾部宽度比例，向上取整 |
| `www_beta_alpha`, `www_beta_beta` | `1.0`, `1.0` | 尾部函数形状 |
| `reproducible_dp_noise` | `false` | 仅调试时固定噪声种子 |
| `release_private_diagnostics` | `false` | 是否导出未私有化的分数诊断 |

```bash
python scripts/run_privacy_experiments.py --models clip_mlp --defenses www
python scripts/run_privacy_experiments.py --models bert_lora --defenses www \
  --set defense.target_epsilon=5 --set defense.max_grad_norm=4
```

以 BERT-Adapter/CoLA 比较 WWW 与普通样本级 DP，统一全程预算 8 和裁剪阈值 8：

```bash
python scripts/run_privacy_experiments.py --models bert_adapter --datasets cola \
  --defenses www,record_dp --attacks all \
  --set defense.target_epsilon=8 --set defense.max_grad_norm=8
```

**计算后端。** WWW 默认 `defense.grad_sample_backend=auto`、`defense.microbatch_size=4`，在现有 PEFT 模型上使用分块 batched VJP 求逐记录梯度。每块仍逐记录计算联合 L2 范数、裁剪并乘 INO 权重，汇总整个真实 Poisson batch 后才加一次噪声、执行一次 optimizer step。计算块不参与尾部排序，也不改变期望 batch、隐私会计和客户端等权 FedSGD。可设 `defense.grad_sample_backend=loop` 使用原逐记录实现；启用 Transformer gradient checkpointing 时 `auto` 也会选择 `loop`。性能基准与审计加速参数见 [FedSGD 计算优化](fedsgd_performance.md)。

**输出与审计。** `defense_summary.json` 保存算法、目标/实际 epsilon、delta、噪声尺度和每客户端采样率、期望 batch 大小、固定尾部宽度和计划/实际步数。随机噪声默认使用不写入配置的系统随机种子。形式化保证的范围是防御后上传与模型；审计成员标签、私有信号、原始训练数据和可选诊断均为本地研究资料，不属于受保护发布内容。打开固定噪声或未私有化诊断时，汇总会明确设置 `formal_dp_enabled=false`。

`www_ranked_positions`、`www_tail_mask`、`www_importance_weights`、`www_effective_clip_norms` 及本地索引只保留在客户端运行态。默认不导出损失差/特征统计；可用 `release_private_diagnostics=true` 开启原有攻击相关性验证。额外设置 `www_analysis_timing=post_round` 与 `www_analysis_interval=50` 可保留周期后聚合诊断及 WWW-ProjRes 配对 CSV；这些选项仅影响诊断，不会降低防御频次或改变用于训练的上一轮参考模型。

ProjRes 成员仍是当轮真实 Poisson batch，非成员仍为严格标签匹配的 10 倍候选。WWW 的 `projres.max_candidates/min_nonmembers/max_nonmembers` 均为 0，表示按实际 batch 动态构造完整候选；不把大于期望大小的抽样截断。空 batch 没有真实成员，六种真实 Batch 攻击会跳过该轮并在 `privacy_audit/summary.json` 的 `exact_batch_skipped_rounds` 中记录 `empty_poisson_batch`；固定候选攻击继续执行。由于每个上传参数坐标都加了噪声，WWW 下取消无噪声 batch 秩上限，并记录 `paper_fedsgd_exact=false`、`attacked_parameter_perturbed=true`；这不意味着使用了非真实 batch 候选。真实数据上的防御效果仍需通过实验衡量。

### FedMIA 比较基线

这六个基线只允许在共享模型的集中式 `FedAvg` 协议下单独运行。所有更新操作仅处理 `requires_grad=True` 的 prompt 参数，冻结的 CLIP 权重不会训练、稀疏化或加噪。

- `perturb_clip_norm`：上传前 prompt delta 的全局 L2 裁剪阈值；`perturb_noise_std`：裁剪后加入的高斯噪声绝对标准差。FedMIA 报告的噪声标准差范围为 0.01–0.5。
- `sparse_ratio`：按绝对值从小到大置零的比例，取值 `[0, 1)`；论文考察 0.1–0.99。
- `mixup_alpha`：对称 Beta 分布的参数，必须为正。
- `sampling_ratio`：每批保留的数据比例，取值 `(0, 1]`；论文考察 0.1–1.0。
- `data_aug_strength`：随机平移幅度；`data_aug_flip_probability`：水平翻转概率；`data_aug_color_jitter`：亮度与对比度扰动强度。

`data_aug` 没有重新调用图像处理器，而是在已归一化的 NCHW CLIP 输入上实施确定性可复现的张量变换，因此不需要数据集或模型特定的反归一化逻辑。

### CoFedMID

统一入口支持 CLIP-MLP、CLIP-Adapter、CLIP-LoRA、BERT Adapter、BERT LoRA 和 GPT2 Adapter。选择 `cofedmid` 时，默认所有客户端组成防御联盟，三个模块全部开启；要求每轮全员参与。保留各模型的 IID/few-shot、学习率、轮数、batch size 上限、一次 SGD step 和等权聚合。

```bash
python scripts/run_privacy_experiments.py --models clip_mlp --defenses cofedmid
python scripts/run_privacy_experiments.py --models clip_mlp --defenses none,cofedmid
```

第二条命令让基线与防御共同预留独立验证集。可用 `--dry-run --max-runs 1` 检查最终参数。完整默认值集中在 catalog 的 `defense_overrides.cofedmid.common`，无需复制模型 YAML。

| 参数 | 默认值与含义 |
|---|---|
| `cofedmid_clients` | `all`；也支持至少两个不重复的客户端编号列表，例如 `[0, 1]` |
| `cofedmid_partition` / `cofedmid_compensation` / `cofedmid_perturbation` | 均为 `true`；分别控制类别分配、回收正则和上传噪声 |
| `cofedmid_max_class_ratio` / `cofedmid_min_class_ratio` | `0.5` / `0.2`；类别数先向上取整，再随通信轮数线性衰减并四舍五入 |
| `cofedmid_coverage` | `strict`；最低类别数至少为 `ceil(总类别数/联盟人数)`，保证联合覆盖；`maximize` 允许并记录覆盖不足 |
| `cofedmid_max_classes` / `cofedmid_min_classes` | 可显式指定类别数，优先于比例；仍受严格覆盖下限约束 |
| `cofedmid_init_round` / `cofedmid_intervals` | `10` / `10`；首次回收发生在第 11 轮，用已完成第 10 轮的全局模型初始化固定难度区间 |
| `cofedmid_recycle_ratio` | `0.05`；每轮回收池最多为**完整本地训练集**的 5%，向下取整 |
| `cofedmid_exp3_gamma` / `cofedmid_exp3_learning_rate` / `cofedmid_reward_history` | `0.2` / `0.3` / `20`；探索、对数权重更新步长与历史奖励窗口 |
| `cofedmid_entropy_weight` | `0.005`；全部实际 batch 样本使用 CE，回收样本额外使用 `KL(q||p) - μH(p)`，按整个 batch 求均值 |
| `cofedmid_noise_std` / `cofedmid_noise_space` | `0.01` / `parameter`；FedSGD 中按 `-δ/lr` 转成上传梯度扰动；也支持显式 `gradient` 单位 |
| `cofedmid_perturb_ratio` | `0.2`；按模型可训练参数顺序拼接后，扰动统一向量的末尾 20% |
| `cofedmid_reproducible_noise` | `false`；默认用未记录的私有随机状态生成上传噪声；`true` 仅供可复现机制检查 |
| `cofedmid_validation_fraction` | `0.1`；从原独立 evaluation 分区按类预留约 10%，其余用于任务评估及审计非成员 |

类别分配使用多项式规模的平衡贪心，尽量降低两两重叠，支持 30 客户端。每轮从分配类别样本与回收样本的并集中，无放回抽取至多一个 batch。池不足时采用实际较小的 N；空池明确报错。评分采用 `eval/no_grad`，不算训练暴露。由于 one-batch 始终只执行一步，必须通过暴露统计实测保护效果，不能直接套用原论文完整 local epoch 的效用和攻击结果。

EXP3 每轮重新计算全本地样本损失，用 min-max 归一化后映射到固定边界；根据独立验证集上全局训练前模型与本地训练后模型的损失差更新。软目标真实类概率为停止梯度的当前预测，其余类均分剩余概率。二分类时该 KL 项梯度为零，熵正则仍有效。

上传扰动为每联盟成员一个高斯标量，经真实聚合权重投影后，在同一尾部掩码上满足 `sum(w_k * δ_k) = 0`。只修改可训练参数的上传；LoRA A/B 分别处理，冻结主干不参与。聚合与所有攻击读取同一份防御后消息。浮点抵消残差写入指标。`0.01` 是工程起点，尚未通过真实任务调参；CoFedMID 不提供形式 DP 保证。

候选协议保持五种固定攻击的完整原训练集 M/M，以及六种真实 Batch 攻击的 N/10N 标签匹配。验证集不会进入任一非成员池。未被训练选中过的原训练成员仍是固定候选成员；实际训练/回收/评分次数单独保存。ProjRes 继续观察真实上传；若其攻击层被扰动，则取消无噪声 batch 梯度的秩上限，并通过 `paper_fedsgd_exact`、`attacked_parameter_perturbed` 和 `batch_rank_bound` 标明条件。

同一 sweep 包含 CoFedMID 时，所有对照自动采用相同的验证集预留比例和种子；清单 hash 可核对实际划分。独立运行 `none` 时若要与 CoFedMID 对照，需显式设置 `--set defense.cofedmid_validation_fraction=0.1`。历史未预留结果不能直接作为此协议下的基线。

新增产物位于同一训练任务目录：`defense_validation_split.json`、`cofedmid_round_metrics.csv`、`cofedmid_noise_metrics.csv` 和 `cofedmid_sample_exposure.pt`。这些含本地选择/奖励的文件是离线实验诊断，攻击信号不读取它们；公开上传消息也不包含 EXP3 状态或噪声随机种子。

这是六个 PEFT 模型的适配实现；ResNet18/FedAvg 的原论文复现、IFL 非成员、SeqMIA 和全局模型攻击评估尚未接入。论文、作者代码差异与设计依据见 [实现计划](cofedmid_implementation_plan.md)。

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
- `grad_sample_backend`：`batched`、`vmap`、`loop` 或 `auto`。ResNet18 保留分块
  `vmap`；BERT Adapter 默认 `auto` 使用真实前向图上的 batched VJP，兼容共享主干与客户端 PEFT 参数。
- `microbatch_size`：一次保留的逐记录计算块大小，不是新的 DP batch，BERT Adapter 默认 4。
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

一次训练任务占用一个一级结果目录，多攻击共享训练，例如：

```text
results/YYYY-MM-DD_HH-MM-SS-ffffff_clip_mlp_caltech101_fedsgd_cofedmid_seed42_target0_RUNID/
```

主要文件：

- `training_metrics.csv`：干净任务损失与准确率；
- `defense_summary.json`：防御名、客户端优化步数、选择率、熵或 cross-difference 等运行统计；
- `privacy_audit/summary.json`：攻击 AUC、低 FPR TPR、所用防御和错误信息；
- `privacy_audit/predictions.csv`：逐候选成员分数；
- `final_prompt.pt`：最终可训练 prompt 参数。

比较防御前后效果时，应固定数据划分、随机种子、目标客户端和攻击参数；CoFedMID 使用统一入口 `--defenses none,cofedmid`，并同时报告攻击指标和 `training_metrics.csv` 中的任务效用。
