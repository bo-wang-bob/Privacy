# 生成视图训练的 FedSGD 支持

2026-09-27：CLIP transformer Adapter / CLIP LoRA 的 `risk_synthesis` direct 模式支持 FedSGD 和 FedAvg。历史筛选模式仍只支持 FedAvg。生成公式、全局类别统计、风险排序、噪声幅度、视图数量和统计清理方式均沿用配置。

## 训练与上传

FedSGD 每客户端每参与轮取一个打乱的原始 batch，跨轮接着取，遍历完再打乱，保留短 batch。每条原始记录生成 K 个 token 视图，损失为 `sum_v mean_i CE(view_v_i, y_i) / K`。各视图分别前向/反向、梯度累计，整批只执行一次 optimizer step；上传优化前捕获的真实累计梯度，不将模型差值再次除以学习率。默认服务器等权聚合后执行 `theta - lr * mean(gradient)`。

首轮风险为0，但仍生成并加噪；后续风险参考沿用上一参与轮 own/other 模型。类别统计仍在首轮训练前读取所有客户端完整原始训练集，不随 FedSGD 改成 batch 统计。统计没有 DP，现有攻击不包含统计上传视图；“不是本轮 batch 成员”不代表没有参与类别统计。

FedAvg 保持每轮完整本地 epoch、按本地训练样本数聚合。FedSGD 必须 `local_epochs=1`，本地优化器为无动量、无 weight decay、无优化器裁剪的 SGD。K 不扩大 batch 的原始记录数、客户端聚合权重或候选数量。

## 攻击口径

- 六种单轮攻击：成员是本轮原始来源 batch，记录 `current_round_original_source_batch`。对原图评分，上传信号来自生成视图；保存原始本地 ID 和 `exact_batch_candidate_selection.pt`。默认完整 batch 为32成员/320个独立非成员，短 batch 按实际数量缩放，K 不乘进成员数。
- 五种固定候选攻击仍评价原始客户端训练集身份。
- FedSGD ProjRes 使用真实上传梯度和原图表示，标记 `paper_fedsgd_exact=false`。原图候选 token 数不能作为生成视图梯度的秩上界，因此 `batch_rank_bound=null`；没有额外扰动上传参数，`attacked_parameter_perturbed=false`。零上传仍只跳过该轮 ProjRes。
- 摘要记录联邦方法、单轮成员口径与统计来源。核验器支持 FedSGD 的不均匀访问次数、统计清理凭据和确认分区逐轮 batch 来源映射。普通无防御 FedSGD 同步记录本地 ID，保持原有随机 batch 顺序，便于配对核验。

## 运行

2026-09-28起，新入口默认 `defense.synthesis.risk_tail_fraction=1.0`：有参考状态时，所有原始batch记录均获得非零、按排序递增的风险权重；首轮仍为0。batch32时权重从0.015625到0.984375。该比例不会把所有权重设为1，也不会改变噪声幅度。此前0.5噪声的结果使用0.8；复现须加 `--set defense.synthesis.risk_tail_fraction=0.8`。新比例的效果需要重新运行评估。

沿用最新0.5噪声、K=2、全局完整秩几何、seed43、确认数据分区和全部11攻击，先检查计划：

```bash
python scripts/run_synthesis_center_ablation.py --methods fedsgd --variants risk --dry-run
```

运行 Adapter：

```bash
python scripts/run_synthesis_center_ablation.py --methods fedsgd --variants risk --models clip_adapter --gpus 0
```

LoRA 使用 `--models clip_lora`。省略模型选择会在单卡上串行运行两种模型。纯来源、纯类中心、打乱风险以及本地/全局中心消融同样支持 `--methods fedsgd`。原脚本不传方法时继续使用 FedAvg。

包含匹配无防御对照：

```bash
python scripts/run_global_synthesis_validation.py --methods fedsgd --set defense.synthesis.noise_scale=0.5 --set defense.synthesis.mixing_mode=risk
```

以上 Adapter/LoRA 的 FedSGD 默认1000轮、等权；FedAvg 默认100轮、按样本数聚合。轮数由模型基线和 catalog 方法覆盖共同确定，无防御对照与生成方案使用相同默认预算。显式 `--rounds` / `--aggregation-weighting` 可覆盖。唯一批量入口也支持 `--methods fedsgd,fedavg --defenses none,risk_synthesis`，通用入口的视图/幅度仍取 catalog 配置。

相同轮数不等于相同训练预算：每客户端1000张、batch32时，FedAvg每轮32次本地更新，FedSGD每轮1次。最新 FedAvg 结果不能作为 FedSGD 的匹配基线，也不能据此预测新方法的隐私或准确率。

`scripts/analyze_risk_synthesis.py` 可核验各方法结果；联合候选重采样脚本 `paired_synthesis_uncertainty.py` 仍限所有攻击共享固定候选的 FedAvg 研究，FedSGD 的 batch 与完整训练集候选不可混成同一重采样池。
