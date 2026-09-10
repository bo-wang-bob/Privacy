# 联邦成员隐私防御

## FedSGD / FedAvg 适配说明

六种 PEFT 模型可在统一入口用 `--methods` 切换。WWW、CoFedMID 和 BERT Adapter
Record-DP 已接入 FedAvg 多 batch/多 epoch 本地训练；`local_client_dp` 仍只支持
FedSGD。FedAvg 对原始完整客户端训练集做成员评价，不能把防御选择的最后一个
batch 当作整个上传的成员集合；CoFedMID 的选择/回收曝光另存，Record-DP 按
多步 Poisson 训练校准预算。下文 one-batch 默认值描述 FedSGD；FedAvg 的损失、
参考状态、上传扰动和攻击适配详见 [联邦方法文档](federated_methods.md)。

仓库支持多种彼此独立的防御。一次实验只能选择一个 `defense.name`，不会在后台组合其他防御。

| 运行名 | 方法 | 当前适配 |
|---|---|---|
| `cofedmid` | [CoFedMID](https://www.usenix.org/conference/usenixsecurity26/presentation/bai)，USENIX Security 2026 | 六个 PEFT 模型的动态类别分配、EXP3 回收、软目标正则、聚合中性上传扰动；默认全客户端联盟 |
| `prompt_dp` | [Differentially Private Prompt Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html)，NeurIPS 2023 | 逐样本 prompt 梯度裁剪和高斯噪声，冻结 CLIP 参数不参与隐私优化 |
| `record_dp` | 客户端侧记录级 DP-SGD | Poisson 记录采样、完整可训练参数联合梯度裁剪、sampled-Gaussian RDP 会计；支持 ResNet18 FedAvg 与 BERT Adapter FedSGD/FedAvg |
| `mist` | [MIST](https://www.usenix.org/conference/usenixsecurity24/presentation/li-jiacheng)，USENIX Security 2024 | 将客户端数据分区视为 MIST 子空间，先本地训练，再以其他客户端预测作为反事实目标做 cross-difference 更新 |
| `soft` | [SOFT](https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-kaiyuan)，USENIX Security 2025 | 第一轮 warm-up；随后用客户端验证损失均值选择低损失高风险样本，并以视觉翻转和噪声替代文本 paraphrase |
| `hamp` | [HAMP](https://www.ndss-symposium.org/wp-content/uploads/2024-14-paper.pdf)，NDSS 2024 | 高熵软标签、预测熵正则，以及可微且保持 `argmax` 的温度输出映射 |
| `perturb` | FedMIA Perturb 基线 | 裁剪客户端可训练 prompt delta，并在上传前加入高斯噪声 |
| `sparse` | FedMIA Sparse 基线 | 上传前按绝对值保留最大的 prompt delta 元素，其余置零 |
| `mixup` | FedMIA Mixup 基线 | 本地 prompt 训练使用 Beta 系数混合图像和标签损失 |
| `sampling` | FedMIA Data Sampling 基线 | 每个本地 batch 无放回抽取固定比例样本参与训练 |
| `data_aug` | FedMIA Data Aug 基线 | 对已经预处理的 CLIP 张量做翻转、平移和颜色扰动 |
| `data_aug_sampling` | FedMIA Data Aug + Sampling 基线 | 在同一本地训练分支中先抽样再增强 |
| `www` | WWW（原 ICLR） | 打乱后分批、CE 加风险加权的真实类概率差异正则；记录风险、损失及梯度范数，不裁剪、不加噪、不提供 DP 保证 |

历史通用防御主要针对“冻结 CLIP、只训练共享 CoOp prompt”的场景适配；正式模型的可用范围以 `configs/experiment_catalog.yaml` 为准。CoFedMID 已单独适配当前六个 PEFT 模型的 one-batch FedSGD。SOFT 原论文处理文本，因此本仓库使用保持图像语义的视觉混淆；HAMP 原论文测试阶段使用随机低置信度分数重排，本仓库使用可微温度映射。

## 常用防御参数

参数放在 YAML 的 `defense:` 节点中；命令行 `--defense` 只覆盖方法名。

### WWW

原 ICLR 已统一更名为 WWW：配置、代码和新产物使用 `www` / `www_*`，旧名称不再作为防御入口。历史 `results/` 及其中的 `iclr_*` 文件不改写；历史观测型 ICLR 不能当作 WWW 的防御结果。

当前 WWW 使用风险控制损失，已移除梯度裁剪、INO 积分缩放、高斯加噪和隐私预算校准。它是经验性防御，不提供差分隐私保证。历史带噪及裁剪版本属于不同方法，分析时必须依据任务配置和 `defense_summary.json` 中的 `mechanism` 区分；当前为 `risk_controlled_loss`。

WWW 支持 CLIP-MLP、CLIP-Adapter、CLIP-LoRA、BERT Adapter 和 BERT-LoRA，要求线性 FedSGD/FedAvg 且每轮至少两个客户端。统一入口保留各模型既有的 IID、few-shot、学习率、轮数及 one-batch 等权 FedSGD。当前不使用 Poisson，改用普通打乱后分批采样，每轮只取下一批；遍历完客户端数据后重新打乱。最后一个不足 batch size 的批次保留，梯度按该批实际样本数求均值。默认每一轮执行防御，与攻击审计和任务评估频次无关。

**样本排序与尾部。** 在当前全局参数覆盖客户端状态前，用上一轮防御后的上传与实际聚合权重重建参考模型：

```text
theta_-k = (theta_global - w_k * theta_k) / (1 - w_k)
M_i = loss(theta_-k, z_i) - loss(theta_k, z_i)
```

每个客户端使用与普通 FedSGD 相同的打乱种子规则；一次完整遍历中，每条本地记录只出现一次。样本本地索引随 batch 保留，以便真实 Batch 攻击及诊断精确配对。正常非空训练集不会产生空 batch。普通 `record_dp` 继续使用 Poisson 采样。

对真实 batch 按 `M_i` 稳定升序排序，同分保持 batch 原顺序。默认 `defense.www_tail_fraction=0.8`、`www_tail_basis=actual_batch`，尾部包含最高风险的 `m=ceil(0.8*n)` 条。低风险的 `n-m` 条只用交叉熵，尾部额外施加预测差异正则。完整 batch=32 时为 6 条仅用 CE、26 条加正则；最后一个 batch 若有 10 条，则为 2 条仅用 CE、8 条加正则。由于取整，低风险部分是约 20%；n<5 时可能全批都在尾部。`www_tail_basis=expected_batch` 可显式使用历史固定尾部宽度 `m=ceil(0.8*min(batch_size,N))`，短 batch 选中最多 m 条，权重按固定宽度的最后若干位置右对齐。

首轮没有历史参考模型，所有风险权重为 0，仅用交叉熵；部分参与时，缺少紧邻上一轮上传的客户端也使用这一回退规则。所有轮次均不裁剪、不加噪。

**风险控制损失。** 借用 [MIST](https://www.usenix.org/conference/usenixsecurity24/presentation/li-jiacheng) 的 cross-difference 思路，约束当前模型和其他客户端参考模型对真实类别的预测差异。设高风险尾部内部升序名次为 `j=1,...,m`：

```text
r_i = 0                 if low-risk or reference unavailable
      (j - 0.5) / m     if high-risk
q_i = stop_gradient(exp(-loss(theta_-k_previous, z_i)))
p_i = exp(-CE(theta_current, z_i))
R_i = lambda * stop_gradient(r_i) * abs(p_i - q_i)
L_batch = mean_i(CE(theta_current, z_i) + R_i)
g_upload = gradient(L_batch)
```

默认 `lambda=1.0`，配置键为 `defense.www_regularization_weight`，必须有限且非负；设为 0 时退化为普通 CE 更新。实际 batch=10 时，最后八条的风险正则权重依次为 `0.0625、0.1875、0.3125、0.4375、0.5625、0.6875、0.8125、0.9375`。风险越高，预测差异惩罚越强。风险排序和教师概率均固定于上一轮参考状态，不通过它们反向传播；学生概率来自当前训练前向图，必须保留梯度。

当前实现是 MIST 思路对 one-batch FedSGD 的适配：CE 和正则合成一次更新，没有额外的第二个优化步骤。`theta_-k` 是排除当前客户端上一轮直接贡献后得到的**参数聚合参考**，并非其他客户端预测的平均值；共享全局训练历史仍可能包含目标客户端的数据影响，因此它不是严格从未见过该样本的 leave-one-out 模型。

正则梯度与该样本的 CE 梯度共线，其有符号系数为 `a_i=1-lambda*r_i*p_i*sign(p_i-q_i)`。`p_i>q_i` 时正则抵消部分 CE 梯度，`p_i<q_i` 时增强 CE 梯度；系数足够大时还可能反向，日志会记录这一现象。默认 lambda=1 时 a_i 非负，不能据此假定会主动降低每条高风险样本的置信度。该机制的效用和攻击抵抗能力需要用新实验测量。

**隐私标记与兼容性。** WWW 不再使用 epsilon、delta、邻接定义、隐私会计、裁剪阈值或 Beta 积分参数。为兼容旧配置及 `--defenses record_dp,www` 的共享参数，WWW 配置校验会将 `target_epsilon`、`delta`、`adjacency`、`accountant`、`max_grad_norm`、`www_beta_alpha`、`www_beta_beta` 统一设为 `null`，将 `noise_multiplier` 设为 `0.0`。旧覆盖值不会重新启用裁剪或加噪。普通 `record_dp` 仍使用预算和裁剪阈值校准噪声及累计预算。不要把 WWW 的无预算理解成 epsilon=0。

WWW 的打乱顺序由实验 seed 和客户端编号决定，与普通 FedSGD 一致；`reproducible_noise` / `reproducible_dp_noise` 在 WWW 中不再控制任何采样或噪声。显式传入旧的 `sampling: poisson` 会被拒绝，避免旧实验配置悄然切换协议。历史运行中的进程不会热更新，也无法补录此前没有保存的范数。

超过配置的训练步数会在更新前报错。当前不支持隔离式主动客户端探测及含 BatchNorm 的模型。已有实验结果不改写，历史协议不得直接混入当前实验。

常用参数集中在 `configs/experiment_catalog.yaml`：

| 配置键（`defense.` 下） | 默认值 | 用途 |
|---|---:|---|
| `www_regularization_weight` | `1.0` | 风险正则系数 lambda；有限且非负，0 为 CE 消融 |
| `max_grad_norm` | `null` | 不裁剪；旧覆盖值被清空 |
| `target_epsilon`, `delta`, `adjacency`, `accountant` | `null` | 不适用；旧配置中的覆盖值被清空 |
| `noise_multiplier` | `0.0` | 固定为零，不生成梯度噪声 |
| `sampling` | `shuffled_batches` | 打乱后分批，FedSGD 每轮取下一批 |
| `www_tail_fraction` | `0.8` | 高风险尾部比例，向上取整 |
| `www_tail_basis` | `actual_batch` | 按实际 batch 划分；`expected_batch` 为历史固定宽度 |
| `www_record_diagnostics` | `false` | 显式开启后，每个实际训练 batch 记录风险、损失与范数；需要额外逐样本求导 |
| `www_beta_alpha`, `www_beta_beta` | `null`, `null` | 不再使用 INO 权重；旧覆盖值被清空 |
| `release_private_diagnostics` | `false` | 是否额外启用旧版分数、特征和攻击相关性诊断；不控制新的范数记录 |

```bash
python scripts/run_privacy_experiments.py --models clip_mlp --defenses www
python scripts/run_privacy_experiments.py --models bert_lora --defenses www \
  --set defense.www_regularization_weight=3
```

以 BERT-Adapter/CoLA 比较 WWW 与普通样本级 DP，裁剪阈值 8 和预算 8 仅应用于普通 DP，WWW 使用默认 lambda=1：

```bash
python scripts/run_privacy_experiments.py --models bert_adapter --datasets cola \
  --defenses www,record_dp --attacks all \
  --set defense.target_epsilon=8 --set defense.max_grad_norm=8
```

**计算后端。** WWW 对当前真实 batch 做一次训练前向，对平均组合损失执行一次 backward 和 optimizer step；训练本身不需要逐样本梯度。范数诊断默认关闭；显式开启后，`defense.grad_sample_backend=auto`、`defense.microbatch_size=4` 使用分块 batched VJP，在同一个完整 batch 前向图上计算逐样本 CE 梯度范数。正则和总损失范数由有符号系数精确换算；额外启用 code-poison 损失时，总范数单独求导测量。诊断不更换 dropout、不改写上传梯度；块大小控制逐样本梯度存储，不切分训练前向图。显式 `loop` 或启用 Transformer gradient checkpointing 时的 `auto` 使用循环求导。性能说明见 [FedSGD 计算优化](fedsgd_performance.md)。

**输出与审计。** `defense_summary.json` 保存风险损失公式、正则系数、教师定义、采样与归一化方式、计划/实际步数及诊断文件路径。兼容字段 `privacy_accounting` 明确记录 `mechanism=risk_controlled_loss`、`clipping_enabled=false`、`formal_dp_enabled=false`、`client_upload_is_private=false`、`noise_enabled=false`，epsilon/delta/accountant 为 `null`，噪声尺度为 0。控制台显示 `risk-controlled loss`、正则系数、裁剪关闭和 `Epsilon/Delta: N/A`；审计不会为 WWW 输出基于 epsilon 的 DP 攻击理论上界。

使用 `--set defense.www_record_diagnostics=true` 开启诊断后，在每个任务目录下生成 `www_diagnostics/`：

- `sample_gradients.csv`：每次实际训练访问一行，以 `(client_id, local_sample_index)` 标识样本，另附一基通信轮次和客户端更新序号。保存风险分数、升序排名、参考轮次、标签、尾部标记、`risk_weight`、`regularization_weight`、当前与教师真实类概率、带符号及绝对概率差、CE/正则/额外损失/总损失、`ce_gradient_factor`。范数列为 `raw_grad_norm`（CE）、`regularizer_grad_norm`、`total_grad_norm`，联合覆盖全部可训练参数；`normalized_contribution_norm` 是总损失范数除以实际 batch 大小。字段不再表示裁剪前后范数。
- `batch_summary.csv`：每个客户端、batch 分别输出 `all`、`low_risk`、`high_risk` 分组；无参考时使用 `warmup`。包括损失及预测均值、范数均值/中位数/P90/P99/最大值、CE 方向反转数量，以及风险与三种范数的 Pearson／Spearman 相关系数。相关系数不跨客户端或轮次混算；少于 3 条、分数/范数恒定或风险不可用时留空。`all` 与分组行是同一批样本的不同汇总视图，不能相加。
- `summary.json`：`schema_version=2`，记录行数、状态、字段定义和文件路径。首轮及其他缺少参考的 batch 标记 `risk_available=0`，风险分数、排名、教师概率/差异及相关系数留空，风险权重和正则损失为 0。重复访问同一样本会产生不同训练访问记录。

范数诊断增加逐样本求导开销，关闭 `www_record_diagnostics` 可省去这部分计算而不改变训练。保留诊断时按批传输少量标量到 CPU 并流式写盘，每批 flush，不保存全训练期逐样本梯度。正常结束或 Python 异常时关闭文件并记录状态；强制结束进程前已 flush 的 CSV 可用于分析。旧 `release_private_diagnostics` / `www_analysis_interval` 不影响新记录的频次。额外设置 `www_analysis_timing=post_round` 与 `www_analysis_interval=50` 仍可保留旧版周期诊断；新记录始终对应训练前实际使用的风险。

ProjRes 成员仍是当轮真实 batch，非成员仍为严格标签匹配的 10 倍候选。统一入口按普通 batch 配置候选上限，batch=32 时为 `32/320/320`；最后一个短 batch 使用其真实 n/10n 候选。当前真实类概率正则的逐样本梯度与 CE 共线，保留 batch 的梯度秩上限，记录 `attacked_parameter_perturbed=false`。由于训练修改了损失，仍记录 `paper_fedsgd_exact=false`；这不意味着使用了非真实 batch 候选，也不能据此声称防住了 ProjRes。

无噪声时，LoRA 初始化等情况可能使被攻击层的上传恰好为零。此时仅跳过当轮 ProjRes，记录 `zero_observed_update` 及参数名，其余攻击继续运行；不生成虚构的 ProjRes 分数，也不让该退化情形中止训练。

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
