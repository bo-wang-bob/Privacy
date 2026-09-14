# 多个替身共同训练

最新全局类别几何扩展见 [首轮全局类别分布](risk_synthesis_global_geometry.md)。本文件以下的 v7 类别本地几何定义及冻结研究仍作为历史对照保留；本分支 catalog 已默认首轮全局共享并以全局同类几何生成。

2026-09-13。用户澄清要增加的是“每次访问一个原始样本，生成多个替身并共同训练”。此前的两个候选择一实验只验证候选筛选，不能作为共同训练多替身的效果证据。

用户随后明确要求删除本地合并类内几何。当前v7只使用本地当前类别的协方差，已移除合并协方差的构建、缓存与采样，以及 `pooled_rank`、`shrinkage` 两个参数。旧的混合几何多视图计划尚未开始训练，等待进程已取消；原计划和已有真实结果保留，不冒充新方案结果。

## 训练定义

每条原始记录使用K个独立生成且彼此不同的虚拟patch+position编码，保留原类别标签：

\[
\tilde h_{i,v}=(1-r_i)h_i+r_i\mu_{-i}+0.1L_c\epsilon_{i,v},\qquad
\mathcal L=\frac1B\sum_{i=1}^B\frac1K\sum_{v=1}^K\mathrm{CE}(f_\theta(\tilde h_{i,v}),y_i).
\]

每个原始batch只计算一次风险/历史排序及打乱分配。同一原始记录的K个视图使用相同风险、中心和几何，独立抽取噪声。每个视图分别沿用最多两次语义检查；所有候选语义失败仍保留语义最佳的有效虚拟候选，不能回退原图。共同训练视图数K与语义候选尝试上限attempts是不同概念：K=2、attempts=2会训练两个视图，最多生成四个候选。

其中 `L_c=U_c sqrt(Lambda_c)` 只从该客户端类别c的编码残差计算，保留最多 `class_rank=5` 个方向；噪声为 `0.1*L_c*epsilon`，没有0.5混合系数，也不借用其他类别的变化方向。风险混合中心仍排除自身，协方差仍使用该类全部原始记录。若该类没有可用变化方向、无法生成改变后的有效编码，则按已有全部替换约束报错，不回退原图或其他类别几何。

全部视图必须先通过有限值、范数、相对原始编码改变以及视图间不重复检查，才开始反向传播。为控制激活显存，逐视图前向和累积除以K的梯度，然后执行**一次optimizer step**。原始batch大小、本地epoch、优化器状态更新次数、样本数聚合权重不变；训练前向/反向工作量增加，不能称训练预算相同。直接平均虚拟编码再前向不等价于本方案。

所有位置全部替换，首轮和零风险记录也加入几何噪声。可组合既有历史因子和候选选择，但第一批效果实验只比较类别几何全替换K=1与K=2，保持 `risk_history=none`、`candidate_selection=first_semantic`，隔离共同训练多个替身的作用。统计与编码仍只在客户端保存，无形式DP保证。

## 参数与入口

`defense.synthesis.views_per_record=1`为单视图默认，专用入口默认K=2；两者均使用新的类别几何。多视图要求全替换策略。旧配置若显式包含已删除的 `pooled_rank` 或 `shrinkage` 会报错，复现旧混合几何需使用原代码版本。专用轻量入口实际仍调用唯一批量入口 `scripts/run_privacy_experiments.py`：

```bash
PY=/root/.local/share/mamba/envs/pfedba/bin/python
$PY scripts/run_synthesis_multiview.py --dry-run
$PY scripts/run_synthesis_multiview.py --gpus 0
```

默认CLIP transformer Adapter/CIFAR100、100张原始图像/类、10个IID客户端、FedAvg100轮、每轮一个完整本地epoch、全部11种攻击、seed43、K=2。LoRA可用 `--models clip_lora`；K可用 `--set defense.synthesis.views_per_record=3`，但冻结实验不做K搜索。去掉两个合并几何参数，保留类别秩5、噪声强度0.1及原语义检查参数，不增加新的混合权重。

当前最新实现已整合到 `/root/Privacy` 的 `main`，上述命令统一从正式目录执行。本文的类别本地几何实验定义属于历史v7对照，复现须使用 `research/risk-synthesis-multiview` 对应提交及其原冻结计划；当前效果验证使用 `scripts/run_global_synthesis_validation.py`。

## 记录和审计

- `synthetic_exposure.csv`：每次原始记录优化访问一行，只有optimizer成功后才写入。附K、语义合格视图数、总尝试数。已有范数、语义差、所选尝试等单视图字段明确取**语义最差的训练视图**，由 `representative_view_index` 标明，不能把这一行当作所有视图的均值。
- `synthetic_views.csv`：每个实际训练视图一行，包含 `view_index`、`loss_weight=1/K`、完整生成/语义检查指标。分析多个视图的平均语义失败率及邻近度必须读取该文件。
- `counts`与 `source_exposure.pt` 的 `synthetic_steps` 继续统计原始访问；`view_counts`和 `synthetic_views` 计数统计训练视图。历史E在成功优化后只提交一次原始访问，重复epoch仍按原始记录轮内平均。
- `candidate_choices.csv` 若开启候选选择，额外按 `view_index` 区分各视图自己的尝试。候选数、训练视图数、原始成员数必须分开。
- 成员推理候选仍为原始客户端训练集。K=2不会把1000个成员变成2000个，也不增加独立非成员数量或低FPR分辨率。

`scripts/verify_synthesis_multiview.py` 独立重放视图与原始记录的对应、归一化权重、语义/候选决策及计数；`scripts/verify_synthesis_history.py` 同时支持原始记录级历史核验。日志核验不等于重算所有真实编码或梯度；梯度归一化另由带动量和权重衰减、含短batch的数值对照验证。

## 效果验证

类别几何修改后的146项相关测试通过（13.25秒），包括修改其他类别的全部编码后、目标类别的采样严格不变，并拒绝已删除的配置参数。还包括Adapter/LoRA三轮、两个local epoch、全部11种攻击；每组成员/非成员仍各9条，126次原始访问对应252个训练视图，历史仅按126次原始访问累计。无效或重复的后续视图会在优化前终止；每个视图语义失败独立保留最佳有效候选；伪造视图权重会被独立核验器拒绝。测试证明实现行为，不能作为真实数据防御有效性结论。

冻结实验入口 `scripts/run_synthesis_multiview_study.py` 改用新目录 `analysis_scripts/synthesis_class_only_multiview_study_20260913`，预先指定seed43/44/45各一个类别几何K=1和K=2百轮任务，共6项。不能直接用旧的混合几何K=1作为多视图独立收益的对照。每种子分别比较类别几何K=2与K=1、两者与无防御，以及类别几何K=1与旧混合几何K=1。来源及这些种子已被研究，本次不是未接触数据确认。整体标准仍为最大AUC下降至少0.02、最大TPR@1%下降、准确率损失不超过2个百分点；多视图新增价值要求相对类别几何K=1最大AUC和最大TPR都下降、准确率损失不超过2个百分点。固定100轮，报告全部攻击和种子，不依据seed43结果改K。

为避免与已运行队列争用GPU，新工作进程先核验旧研究任务和进程，等其全部完成、工作进程退出后再领取新任务。等待期间不修改旧源码、结果或计划。完整结果还需复算正式攻击、配对候选区间、重复图像敏感性、每原始记录的视图归一化暴露分布，并人工核验报告。当前尚无该多视图方案的真实数据效果结论。
