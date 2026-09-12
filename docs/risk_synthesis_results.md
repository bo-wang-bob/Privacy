# 本地分布与风险指导生成：部分替换方案结果

2026-09-12。**本文结果属于旧的部分替换方案。** 用户随后要求全部训练位置替换、风险只控制原始编码保留量，并选择语义重试失败时保留最佳虚拟候选。当前默认及运行命令见[全部替换方案](risk_synthesis_all_replacement.md)；以下确认收益与耗时不能归属于新版本。旧方案在 CLIP transformer Adapter / CIFAR100 的三个确认种子上观察到整体防御收益，风险排序相对打乱风险的额外低 FPR 收益不稳定；尚未完成的旧迁移生成任务已停止或取消后续启动。

完整设计见[方案](local_risk_guided_synthesis_plan.md)，包含失败实验、消融和后续修订的记录见[研究日志](risk_synthesis_research_log.md)。代码已保存在本地 main；按照用户最新决定，暂不推送远端。

## 实际生成的是什么

生成对象是可进入 CLIP 视觉 Transformer 的 **Patch+Position token**，不是 RGB 图片。ViT-B/32 每张图片使用 49×768 个数，统计时排除公共 CLS，训练时拼回 CLS。这样，生成输入仍经过全部视觉 Adapter/LoRA，LoRA 文本分支也正常求导。

每客户端从自己的原始训练集提取冻结输入编码，每类拟合均值和低秩协方差，再向该客户端合并类内协方差收缩。统计不使用其他客户端的特征，不使用 evaluation 数据，也不上传给服务器。当前全局每类 100 张、10 个 IID 客户端意味着本地每类仅 10 张；这里估计的是经验二阶几何，并非已识别完整真实概率密度。

对原始记录 i，复用 WWW 上一轮 own/other 模型损失差，转成 batch 内尾部秩权重 r_i。以 0.25r_i 的概率请求替换，每批最多 floor(0.25B) 个位置，生成：

\[
\widetilde h_i=(1-r_i)h_i+r_i\mu_{c,-i}+0.1L_c\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).
\]

\(\mu_{c,-i}\) 是排除自身的本地同类均值，\(L_cL_c^\top\) 是收缩后的低秩协方差。固定原始 CLIP 教师检查类别语义，最多尝试两次，失败则回退原图。首轮全部使用原图；之后在同样大小的混合 batch 上使用普通 CE，不附加 WWW 正则，不增加 optimizer step。

实现复用 Nguyen 论文的输入 embedding 介入位置，以及 Ma 论文的几何建模思路；这是本仓库对视觉 PEFT 的适配。局部统计、风险模型和虚拟输入仍依赖原始记录，因此不提供形式差分隐私保证，审计成员仍为原始完整客户端训练集。

## 确认实验结果

固定协议：100 轮 FedAvg，每轮一个完整本地 epoch，batch 32，学习率 0.01，10 个 IID 客户端，sample-count 聚合；目标客户端 0。种子 43、44、45 使用事前锁定的方案和预留原始记录分区，每个种子的四组共享实际候选身份：none、WWW、真实风险生成、打乱风险生成。全局训练每类 100 张，每个目标候选池为 1,000 个原始成员对 1,000 个 evaluation 非成员，每类 10:10。

“最大攻击”在各模型、各种子的全部 11 种注册攻击中分别取最大值，不固定为无防御时最强的攻击。下表是三个种子结果的描述性均值，不是种子总体的置信区间。

| 方案 | Accuracy ↑ | 最大攻击 AUC ↓ | 最大 TPR@1%FPR ↓ |
| --- | ---: | ---: | ---: |
| 无防御 | 81.09% | 0.6806 | 38.20% |
| WWW | 81.32% | 0.6834 | 36.40% |
| 打乱风险生成 | 81.94% | 0.6640 | 31.77% |
| 真实风险生成 | **82.08%** | **0.6541** | 31.80% |

真实风险生成相对无防御，最大 AUC 在三个种子分别下降 **0.021079、0.027823、0.0305785**，平均下降 **0.0264935**；最大 TPR@1%FPR 平均下降 **6.4 个百分点**，准确率平均提高 **0.99 个百分点**。事前规定的四项标准全部满足：平均最大 AUC 至少下降 0.02、每个种子最大 AUC 下降、平均最大 TPR 下降、每个种子准确率损失不超过 2 个百分点。

这不是每种攻击都下降：33 个逐种子/攻击 AUC 对照中，31 个下降，seed 43 的 Grad-Cosine 上升 0.001814，seed 44 的 Gradient-Diff 上升 0.001781。允许攻击分数翻转后重新计算的最大 AUC 没有出现反向结论。类条件 AUC 检查也保留改善方向。

风险排序本身的证据较弱：相对打乱风险，最大 AUC 三个种子均下降，平均下降 0.0098355；最大低 FPR TPR 的差值却为 −1.7、+7.1、−5.3 个百分点，均值近于 0。探索种子 42 的完整 100 轮结果还出现打乱风险优于真实风险的反例。两种策略的请求总数一致，但语义筛选后的接受数不同，不能声称实际替换预算完全相等，也不能将整体防御收益全部归功于风险排序。

同一批确认任务的实际端到端耗时如下，包含初始化、100 轮训练、评估和全部配置的攻击审计：

| 方案 | 平均耗时 | 三种子范围 | 对匹配无防御任务的平均耗时比 |
| --- | ---: | ---: | ---: |
| 无防御 | 29.4 分钟 | 28.5–31.0 分钟 | 1.00× |
| WWW | 66.9 分钟 | 64.5–71.8 分钟 | 2.28× |
| 真实风险生成 | 79.3 分钟 | 74.4–83.1 分钟 | 2.70× |
| 打乱风险生成 | 77.8 分钟 | 73.8–82.0 分钟 | 2.65× |

这些是当前机器并行运行时的实测值，不是其他硬件或数据集的时间保证。以真实风险 seed 43 为例，风险排序累计约 1,952 秒、生成处理约 796 秒、候选语义筛选约 128 秒；筛选包含在生成处理中，不能重复相加。本地几何一次初始化约 18 秒，主要额外开销来自训练中的重复风险查询与生成处理，而非初始分布拟合。来源为十二个已完成任务的队列时间戳与 `performance_summary.json`，核验结果保存在 `analysis_scripts/risk_synthesis_confirmation_timing_20260912/`。

## 核验与证据边界

- 已从原始预测独立复算 132 个正式 AUC 和三个可报告 FPR 水平，核对实际配置、候选来源、类别比例与六百万条生成访问记录。原始结果未被覆盖。
- 每种子 2,000 次配对候选重采样，风险生成减无防御的最大 AUC 区间均低于 0。该区间条件于已训练模型，不度量训练种子或总体不确定性。
- 原始记录编号不重叠，但 CIFAR100 源数据存在少量完全重复像素。使用不依赖攻击分数的排除与类别配平规则后，最大 AUC 平均仍下降 0.026221。此补充分析不改变训练或主分析，不能据此宣称消除了近重复或预训练暴露。
- 单目标的低 FPR 分辨率有限。补充筛选后种子 43/44 仅有 998/999 个非成员，TPR@0.1%FPR 留空；不能合并重复种子候选来扩充独立非成员数。
- 原 LoRA/CIFAR100 与 Adapter/Food101 各三组、seed 43 的方案未完成全部比较。用户改动后停止旧生成任务，保留已完成 LoRA 基线及正在运行的 Food101 基线，不能据此得出旧方案的迁移效果结论。

可核查的本地产物：

| 内容 | 路径 |
| --- | --- |
| 冻结确认计划 | `analysis_scripts/risk_synthesis_confirmation_plan_20260912.json` |
| 完整确认结果、逐攻击 CSV、来源核验 | `analysis_scripts/risk_synthesis_confirmation_report_20260912/` |
| 重复像素检查 | `analysis_scripts/risk_synthesis_confirmation_exact_image_identity_20260912.json` |
| 排除重复的补充分析 | `analysis_scripts/risk_synthesis_confirmation_duplicate_sensitivity_20260912/` |
| 未结束的适用范围计划及状态 | `analysis_scripts/risk_synthesis_transfer_plan_20260912.json`、`risk_synthesis_transfer_lane{0,1}_20260912.json` |

上述实验产物保留在本地，其中部分被 Git 忽略。正式核验工具为 [`analyze_risk_synthesis.py`](../scripts/analyze_risk_synthesis.py)、[`summarize_synthesis_confirmation.py`](../scripts/summarize_synthesis_confirmation.py)、[`paired_synthesis_uncertainty.py`](../scripts/paired_synthesis_uncertainty.py) 和 [`confirmation_duplicate_sensitivity.py`](../scripts/confirmation_duplicate_sensitivity.py)。

## 复现旧部分替换方案

通过上述确认的是范数下限 **0.1**、均匀类别中心的部分替换方案。当前 catalog 已改为全部替换，因此复现旧方案须显式恢复请求策略与 warmup。以下命令新建任务，使用常规默认数据分区：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_privacy_experiments.py \
  --models clip_adapter --datasets cifar100 --methods fedavg \
  --defenses risk_synthesis --attacks all --seeds 43 \
  --set defense.synthesis.replacement_policy=risk_probability \
  --set defense.synthesis.replacement_fraction=0.25 \
  --set defense.synthesis.warmup_rounds=1 \
  --set defense.synthesis.norm_ratio_min=0.1
```

默认已固定全局每类 100 张、100 轮与上述训练参数。先附加 `--dry-run --max-runs 1` 核对最终配置。切换 LoRA 使用 `--models clip_lora`；Food101 使用 `--datasets food101`，这只表示实现支持，不代表当前已有相同防御效果证据。

要复现本次确认分区，须在相同命令上附加下列两项，并分别运行 `--seeds 43,44,45`；直接使用常规默认分区不能复现确认结果：

```bash
--set confirmation_split_manifest=analysis_scripts/risk_synthesis_confirmation_data_20260912/split.json \
--set confirmation_split_sha256=00941791be9727022bb09c4b9d55f8b06d05a1cc3489a99753cbdb393151c178
```

none/WWW 对照也必须使用同一分区与种子；生成参数覆盖只加给 `risk_synthesis` 任务。完整实际命令保存在冻结计划及各任务的 `run_config.yaml`。

每个生成任务内的 `risk_synthesis/client_*_distribution.pt` 保存客户端各类均值和低秩因子，`client_*_source_codes.pt` 保存原始编码，暴露 CSV 与摘要记录请求、接受、回退和原始记录身份。这些是本地研究诊断，不属于联邦上传内容。

## 从结果推导的下一步

1. **按用户改动验证全部替换版本。** 这是一次明确的协议修改，不依据旧迁移中间分数调参，也不把旧确认的收益继承给新版本。
2. **优先研究风险与低 FPR 攻击的错位。** 真实风险与打乱风险低 FPR 效果不一致，说明 WWW 的批内秩不能直接当作通用泄漏概率。下一版应保留独立攻击评价、请求及接受预算对照，并在新的确认数据上检查风险改进，不能以代理风险下降作为成功标准。
3. **保留几何机制的反例。** 现有均值与经验协方差采样仍来自原始输入的仿射空间；它们不会自动消除所有子空间线索。只有实际攻击暴露了明确瓶颈时，才检验同类多锚点或独立噪声先验等扩展，并继续约束语义和效用。这些是待检验方向，不是已验证功能。
