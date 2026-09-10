本次核查发现 **3 处已复现的实现错误**：固定候选池损失未统一关闭 Dropout、DP ProjRes 错用无噪声秩上限、NaN 分数被 ROC 计算接受并可能成为满分。候选池划分、默认上传符号与学习率转换、攻击调度及当前 ProjRes 层选择，在本次检查范围内未发现额外的确定错误。

核查日期：2026-09-09；代码版本：`9eac173`。范围为统一入口的 7 个正式模型、11 种注册攻击、FedSGD/FedAvg 以及正式防御的攻击接口。本次只新增核查文档和合成复现脚本，未修复生产代码、重跑真实实验或修改 `results/`。以下公式描述当前实现，不表示全部是原论文的严格复现。

**已确认的问题，按修复优先级排列**

1. **[P1] 固定候选池的损失、置信度受训练模式及攻击组合影响。**

   位置：`privacy_attacks/auditor.py:4078`、`:4112`、`:3092`。固定池路径加载参数后直接前向；`_candidate_outputs` 的 `no_grad()` 只关闭求导，不关闭 Dropout。流式梯度路径的 `eval()` 在这些前向之后才执行。共享 BERT 骨干在客户端训练退出后仍可处于训练模式。

   当前 BERT-LoRA 基线保留分类头 Dropout=0.1：第 10/20/30/40 等轮只运行固定池攻击时，`loss_series`、`fedmia_loss`、`fta` 使用带随机掩码的输出；第 50 等轮先运行真实 batch 攻击，会提前切到 eval，固定池输出又变为确定性的。FedAvg 同样存在仅固定池攻击到期的轮次。单独选择损失攻击也会触发。BERT/GPT2 Adapter 若显式开启分类头 Dropout，也受影响；默认 Dropout=0 的模型不能据此认定其分数已经变化。

   使用随机初始化的小型真实 BERT 计算图、LoRA Dropout=0、分类头 Dropout=0.1，在完全相同候选与上传下复现：最大负 CE 差为 `0.0030450821`；3 次前向启用了分类头 Dropout，并消耗全局随机数状态。同轮增加真实 batch `grad_cosine` 后，固定池损失变为 eval 结果。数值大小仅是这个小模型的证据，不是对真实实验偏差的估计。审计额外消耗训练所用 RNG 也可能改变后续训练轨迹。

   建议：统一在审计前向前进入 eval，明确恢复共享模型模式/状态的责任；测试同一上传在不同攻击组合、不同信号存储模式下的公共信号一致性，并验证审计不额外消耗 Dropout 随机数。

2. **[P2] BERT Adapter 的 Record-DP/Local-Client-DP ProjRes 仍按无噪声 batch 截断秩。**

   位置：`privacy_attacks/auditor.py:2508`、`:2529`；噪声来源见 `privacy_defenses/controller.py:1413`、`:1479`。`parameter_perturbed` 只识别 CoFedMID，忽略两种正式 DP 防御。FedSGD 因而仍设置 `max_rank=hidden_vector_count`，并错误记录 `attacked_parameter_perturbed=false`。

   裁剪后的无噪声梯度仍可以受 batch 秩限制，但逐坐标高斯噪声一般破坏该限制。如果有效 token 总数小于攻击矩阵的数值秩，当前实现会删掉真实上传的方向。长序列时上限可能不生效，因此不能说所有 DP 轮次的分数都已改变。FedAvg 已统一取消该秩上限，不受这一截断错误影响。

   合成复现：一个 rank-1 梯度叠加全坐标噪声后，上传矩阵数值秩为 8；两种 DP 分支都只使用 rank=1。当前两个候选分数为 `[-0.4441491, -1.0113627]`，完整上传行空间的负残差约为 `[-5.7e-16, -1.2e-15]`。这证明实现改变了被观察的子空间，不能据此推断真实攻击率改变的方向。

   建议：DP 噪声覆盖攻击参数时设置 `batch_rank_bound=null`、`attacked_parameter_perturbed=true`。如果另行研究带噪截断 SVD，应将它作为显式、单独标注的攻击变体，不能把截断理由写成理论 batch 秩上限。

3. **[P2] 全 NaN 分数可输出可报告的 AUC=1、TPR=100%。**

   位置：`privacy_attacks/metrics.py:7`、`:17`、`:22`，以及 `privacy_attacks/base.py:46`。ROC 在排序前未检查分数是否有限；`NaN != NaN` 又将 NaN 错当作不同阈值。在当前成员先拼接、非成员后拼接的布局下，稳定排序会保留这个顺序，产生虚假的完美 ROC。

   复现：100 个成员、100 个非成员，200 个分数全部是 NaN，仍得到 `auc=1.0`、可报告的 `primary_score=1.0`，且 `score_degenerate=false`。这不是小样本可报告性检查能拦截的问题。它影响共享指标出口；不表示现有真实结果已经出现 NaN。

   建议：在评分进入 ROC 前检查所有分数有限、标签合法；发现非有限值明确使该攻击失败并记录数量及轮次。不要用零替换或静默丢弃这些候选，否则会改变候选协议。

**共享调用链与成员定义**

统一入口 `scripts/run_privacy_experiments.py` 根据 catalog 合并模型、方法、防御配置；`utils/federated_protocol.py` 解析成员协议。`users/user.py` 执行训练并保留真实 batch/梯度，防御控制器处理上传，`aggregator/fedavg_aggregator.py` 构造协议消息，`servers/serverbase.py:1072` 将它们交给共享 `MembershipAuditor`。模型适配器负责前向和 ProjRes 表示；评分函数集中在 `fedmia_baselines.py`、`fedmia.py`、`update_attacks.py` 和 `projres_mlp.py`。

| 适用范围 | 成员 | 非成员 | 时间处理 |
|---|---|---|---|
| PEFT FedSGD 的 6 种单轮上传攻击 | 当轮实际 batch 的 N 条记录 | 从未训练的全局 evaluation 池，严格按标签直方图抽取 10N 条 | 每个审计轮独立构造候选、独立评分 |
| PEFT FedSGD 的 5 种固定池攻击 | 目标客户端原始完整训练集 M 条记录 | 同一个固定 evaluation 子集，默认 M 条，类别尽力匹配 | 在固定候选上累计所需历史信号 |
| PEFT FedAvg 的全部 11 种攻击 | 目标客户端原始完整训练集 M 条记录 | 固定 evaluation 子集，默认 M 条，类别尽力匹配 | 6 种仍逐轮独立评分，5 种保留时序/跨客户端定义 |
| ResNet18 `fedmia_loss` 基线 | 目标客户端完整训练集，论文默认 5000 条 | 1000 条独立测试样本，加另外 9 个客户端各 1000 条训练样本 | 固定混合候选、跨客户端与跨轮评分 |

FedAvg 多个 epoch 不重复计数。CoFedMID 回收与 Poisson 采样不改变固定池中的原始训练集身份；固定池成员并不意味着当轮实际访问过。ResNet18 的其他客户端训练样本只属于“目标客户端非成员”，与 PEFT 的“从未训练非成员”不同，不能直接合并作同协议比较。

**11 种攻击的当前实现**

记 `theta_t` 为轮初模型，`theta_{k,t}^+` 为根据目标客户端公开上传重建的更新后模型；`L` 为真实标签 CE，`p_y` 为真实类概率。`u_{k,t}` 在 FedSGD 为公开上传梯度，在 FedAvg 为 `-delta/lr_t` 累计更新代理。所有最终分数都以越大越像成员为方向。

| 攻击 | 分数/执行方式 | FedSGD 候选组 | 实现位置与核查结论 |
|---|---|---|---|
| `blackbox_loss` | `-L(theta^+; x,y)` | 单轮 batch | `fedmia_baselines.py:59`；标准负 CE，统一更新路径逐轮独立评价 |
| `loss_series` | 多个已审计轮的 `-L(theta^+; x,y)` 均值 | 固定池 | 同上；时间均值实现正确，受问题 1 影响 |
| `grad_cosine` | `cos(grad L(theta_t;x,y), u_{k,t})` | 单轮 batch | `auditor.py:3287`；覆盖全部可训练参数，轮初逐样本梯度 |
| `avg_cosine` | 上述真实标签梯度余弦的跨轮均值 | 固定池 | `fedmia_baselines.py:59`；使用相同候选跨轮平均 |
| `fedmia_loss` | 每轮用其他客户端的同候选负 CE 拟合 null，移除上侧 3-sigma 异常值，计算目标观测的高斯 CDF，再跨轮聚合 | 固定池 | `fedmia.py:36`；默认 upper/mean，损失信号受问题 1 影响 |
| `fedmia_cosine` | 用上述梯度余弦代替负 CE，其余 null/CDF 处理相同 | 固定池 | 同上；各客户端比较同一候选的轮初梯度与各自公开上传 |
| `gradient_diff` | `2<u,g_all> - ||g_all||²`，`g_all=sum_c grad L(theta_t;x,c)` | 单轮 batch | `auditor.py:3385`；等价于 `||u||²-||u-g_all||²`，不是仅真实标签梯度 |
| `score_diff` | `L_pre-L_post` | 单轮 batch | `update_attacks.py:99`；同一候选的轮前/轮后真实标签损失 |
| `score_ratio` | `-(L_post+c)/(L_pre+c)`，默认 `c=1e-6` | 单轮 batch | 同上；负比值保持统一 ROC 方向 |
| `fta` | 默认对 `p_y(theta^+)` 与实际通信轮编号作 OLS，取斜率；可选对负 CE 作 OLS | 固定池 | `update_attacks.py:153`；仅 1 个检查点时用该轮 post-pre，受问题 1 影响 |
| `projres` | 对攻击矩阵做 SVD 取行空间，将候选层输入投影后取负原始 L1 残差 | 单轮 batch | `auditor.py:2400` → `projres_mlp.py:57`；模型差异见下表，DP 下受问题 2 影响 |

所有这些评分还共用问题 3 的指标出口。表中“未发现错误”限于本次静态检查和小模型验证，不等于完整真实数据有效性证明。

四个 FedMIA 基线的单轮/多轮与目标客户端信息划分，与作者公开说明一致；Loss-Series 在这里是官方比较中的时间均值基线，不是另训练一个时序攻击网络。[FedMIA 作者仓库](https://github.com/Liar-Mask/FedMIA)、[官方攻击代码](https://raw.githubusercontent.com/Liar-Mask/FedMIA/main/mia_attack_auto.py)。FTA 的线性斜率核心与原文一致；本仓库只取目标客户端上传后检查点，单检查点 post-pre 属于显式适配，不能与完整历史斜率混同。[FTA 原文，第 4 节](https://www.usenix.org/system/files/usenixsecurity24-chang.pdf)。本次未能通过 OpenReview 的访问验证，未把 Gradient-Diff 的原论文逐式一致性列为已验证项；代码公式与本仓库定义、相关测试一致。

**各模型如何提供攻击信号**

除 ResNet18 外，六个正式模型均开放全部 11 种攻击；损失类攻击用各模型真实分类前向，常规梯度攻击用全部 `requires_grad` 参数。`get_audit_key_parameter()` 不代表常规余弦攻击只取一个参数，它与 ProjRes 的单矩阵攻击面需要区分。

| 模型 | 常规梯度攻击的参数范围 | 默认 ProjRes 矩阵 | ProjRes 候选表示 |
|---|---|---|---|
| CLIP-MLP | 两层 MLP 的 weight/bias；CLIP 冻结，可复用图像特征缓存 | `classifier.0.weight` | 第一层输入的冻结 CLIP 图像向量 |
| CLIP-Adapter，当前 transformer 版 | 每个视觉 block 后 Adapter 的 down/up weight/bias；其余 CLIP 冻结 | 最后视觉 Adapter 的 `down.weight`；ViT-B/32 为第 11 层 | 进入该 down 投影的 CLS |
| CLIP-LoRA | 图像、文本全部已启用 Q/K/V 的 `lora_A/lora_B`；基线 rank=32 | 最后已训练视觉 Query 的 `lora_A` | 进入该 Query 投影的 CLS |
| BERT Adapter | 各层 Adapter down/up weight/bias，以及分类头 | 最后 Adapter 的 `down.weight` | 进入 down 投影的 CLS |
| BERT-LoRA | 全部已启用 Query/Value 的 `lora_A/lora_B`，以及分类头；基线 rank=16 | 最后已训练 Query 的 `lora_A` | 进入该 Query 投影的 CLS |
| GPT2 Adapter | 各层 Adapter down/up weight/bias，以及分类头 | 最后 Adapter 的 `down.weight` | 最后一个 attention-active token 的输入表示 |
| ResNet18 | 正式入口只运行 FedMIA-Loss，不提取攻击用逐样本梯度 | 不支持 | 对各客户端上传模型做分类前向；本实现使用 GroupNorm |

旧 `clip_adapter.variant=feature` 仍支持相同公共攻击：梯度参数为已启用的末端 Adapter，ProjRes 攻击 `adapter.net.0.weight`，输入为冻结 CLIP 图像特征。这是兼容路径，不应与当前逐层 Adapter 合并解释。

ProjRes 始终让成员与非成员使用同一目标客户端公开端点：FedSGD 为 `base-lr*uploaded_gradient`，FedAvg 为 `base+uploaded_delta`；提取后恢复参数。普通无防御 MLP/旧 feature Adapter 的攻击输入冻结，因此前后表示相同。Transformer Adapter/LoRA 的层输入会随训练变化，当前使用更新后输入匹配本轮上传属于经验性适配，代码已标记 `paper_fedsgd_exact=false`，本次不把这一已明确选择的协议列为 bug。

LoRA 攻击的是 A 因子上传，不是先合成 `BA` 后攻击。`lora_A` 的行数为 rank（CLIP 默认 32，BERT 默认 16），小 rank 会限制可恢复子空间。rank=32 与 batch=32 相等不保证上传满秩，更不保证攻击成功。首轮 B 零初始化可令 A 上传为零；代码仅跳过 ProjRes 并保存原因，这一处理符合当前协议。首层 Q/K/V 的恒定 CLS 组合也已有拒绝检查。

**防御接口与解释边界**

| 防御 | 攻击实际看到的信号 | 本次结论 |
|---|---|---|
| none | 真实 FedSGD 梯度或 FedAvg delta | 符号与学习率换算未发现重复转换 |
| WWW | 带风险正则的实际训练上传；候选梯度仍用攻击原始 CE/all-label 目标 | 允许训练目标与候选攻击目标不同；当前真实类概率正则使逐记录梯度与 CE 共线，但不保证防住 ProjRes |
| CoFedMID | 已做协作扰动的公开上传，原始训练/回收身份另存 | 聚合前扰动和攻击信号顺序正确；攻击层被噪声覆盖时取消无噪声秩上限 |
| BERT Adapter Record-DP | 裁剪并加噪后的梯度或多步 delta；空 Poisson batch 的单轮攻击跳过 | ProjRes 的 DP 秩处理需修复；固定池仍是原始训练集身份 |
| BERT Adapter Local-Client-DP | 完整上传裁剪加噪后的梯度；仅 FedSGD | ProjRes 的 DP 秩处理需修复 |
| ResNet18 Record-DP | FedAvg 上传的客户端模型，供 FedMIA-Loss 前向 | 不经过 ProjRes；本次未重新证明隐私会计或 DP 理论 ROC 界 |

默认采样与汇总的另外几条限制：

- 固定池只有类别“尽力匹配”，不是始终精确匹配；现有代码保存直方图和 TV。类分布偏移可能成为攻击信号，需要在真实结果分析时另行控制。
- 单轮真实 batch 的非成员确实按标签严格 1:10 匹配；当某类 holdout 容量不足时会报错，不会静默借用其他客户端训练样本。
- 当前正式 summary 对 6 种单轮上传攻击取最后一个成功观测轮；不是挑最佳轮，也不是把各轮候选拼起来提高 FPR 分辨率。逐轮分数的指标另存 `attack_round_metrics.csv`。若末轮 ProjRes 为零上传，会回退到更早的有效观测，解释最终结果时要读取其 `communication_round`。
- 低 FPR 可报告性按每次评分的非成员数判断。完整 CLIP batch 为 320 个非成员，FPR 步长为 0.3125%；BERT/GPT2 为 160 个，步长为 0.625%。这些数量可报告 1% FPR 指标，不能解析 0.1% FPR；多轮观测不自动提高分辨率。
- `docs/attack_mapping.md` 中仍有“FedMIA 每轮采集”“候选 prompt 梯度”等旧说明；应以当前模型配置和 catalog 的方法覆盖为准。仓库保留的 Nasr/RMIA/YOQO 等研究文件没有注册为正式攻击。

**默认审计频率**

| 模型/方法 | 6 种单轮上传攻击 | 5 种固定池攻击 |
|---|---|---|
| 三个 CLIP，FedSGD | 每 10 轮 | 每 10 轮 |
| 三个 CLIP，FedAvg | 每 50 轮 | 每 10 轮 |
| BERT Adapter/LoRA，两种方法 | 每 50 轮 | 每 10 轮 |
| GPT2 Adapter，两种方法 | 每 50 轮 | 每 50 轮 |
| ResNet18，FedAvg | 不适用 | FedMIA-Loss 每 10 轮 |

间隔按已完成轮次计数，最终轮也会纳入。单轮函数中的内部零基 `round` 与产物里的 `communication_round` 相差 1。公共调度器只计算本轮到期攻击所需的信号族。

**验证与复现**

两批直接相关测试合计 **298 passed、14 skipped**。第一批覆盖评分函数、上传梯度流式计算、真实小 Transformer 的 ProjRes 表示、CLIP/FedAvg 多步候选协议；第二批覆盖统一入口、CoFedMID、Record-DP、GPT2 调度及结果格式。跳过项来自梯度优化测试的 CUDA 条件；本次没有 CUDA 数值路径和真实大模型训练的验证。既有测试通过并不能覆盖上面 3 个新发现。

复现脚本：[review_attack_implementations_20260909.py](../analysis_scripts/review_attack_implementations_20260909.py)。脚本直接断言并打印当前错误行为，使用随机初始化小 BERT 和合成矩阵，不依赖 checkpoint 或数据下载；修复后其中的“错误行为”断言应失败，再将其改成对应回归测试。

```bash
/root/.local/share/mamba/envs/pfedba/bin/python analysis_scripts/review_attack_implementations_20260909.py
```

本次测试命令：

```bash
/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q tests/test_update_attacks.py tests/test_privacy_attacks.py tests/test_projres_mlp.py tests/test_projres_post_state.py tests/test_projres_last_token.py tests/test_fedsgd_gradient_optimization.py tests/test_transformer_lora.py tests/test_clip_transformer_adapter.py tests/test_clip_peft_fedsgd.py
/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q tests/test_privacy_experiment_runner.py tests/test_cofedmid.py tests/test_record_dp.py tests/test_result_formatting.py tests/test_gpt2_adapter_schedule.py
```
