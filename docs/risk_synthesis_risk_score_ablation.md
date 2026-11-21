# 固定0.2噪声的风险分数对照

入口：`scripts/run_synthesis_risk_score_ablation.py`。默认 CLIP Transformer Adapter、CIFAR100、FedSGD 1000轮、K=2、seed43、客户端0、全部11种攻击。10个IID客户端全部参与，每类100张训练样本，每客户端每轮一个原始batch，batch=32，学习率0.01，等权梯度聚合。

```bash
cd /root/Privacy
micromamba run -n pfedba python scripts/run_synthesis_risk_score_ablation.py --gpus 1
```

加 `--dry-run` 只核对配置；`--smoke` 默认5轮、逐轮评估、无攻击；`--variants absolute,shuffled` 可补跑指定组。每次调用仅使用一张GPU、一个串行队列，实际调度始终交给 `run_privacy_experiments.py`。失败停止后续组，保留日志及失败原因，并按当前统计保留协议清理四类大缓存。

## 唯一比较因素

在当前原始batch上，用上一参训轮的自身模型与其他客户端聚合参考模型计算

`d_i = CE(theta_other, z_i) - CE(theta_own, z_i)`。

这仍是本轮重新计算的损失差，没有缓存逐样本历史分数；参考模型定义沿用当前方案。新增 `defense.synthesis.risk_score` 只作用于direct生成，缺省仍为 `signed_loss_gap`，独立WWW防御不变。

| 组 | risk_score | mode | 分配方式 |
| --- | --- | --- | --- |
| signed | signed_loss_gap | risk | 按 d 稳定升序排序 |
| absolute | absolute_loss_gap | risk | 按 abs(d) 稳定升序排序 |
| shuffled | signed_loss_gap | shuffled_risk | 对signed得到的权重做批内随机排列 |

三组均固定 `risk_tail_fraction=1`。有参考时，batch大小为B，升序第j个样本的权重为 `(j-0.5)/B`。因此权重集合和均值一致，负分数不会直接截断为零；绝对差可能把较大的负差排到高风险端。首轮/缺少参考时仍全部为零。同一原始样本的K个视图共用一次权重分配，打乱和几何噪声使用独立随机数流。

训练编码为 `(1-r_i) h_i + r_i mu_global_class + 0.2 L_class epsilon`。0.2是特征几何噪声系数；全局含自身类别中心、完整数值秩协方差、直接生成、K视图CE平均后一次更新均保持一致。不存在额外上传梯度噪声。

## 保存及判读

新结果写入 `results/synthesis_risk_score_ablation_<时间戳>/{signed,absolute,shuffled}`；根目录的 `study_plan.json` 保存全部分组配置。正式启动前，程序核对除risk_score、mode和结果目录之外配置完全一致。已有结果不会覆盖。

显式配置risk_score的新运行在逐样本及逐视图CSV中增加 `signed_loss_gap`、`risk_ranking_score`；无参考时留空。摘要记录分数定义、变换与来源。核验器检查分数变换、稳定排序对应的逐条风险、批内权重集合和跨视图一致性。它验证保存的分数与权重之间的一致性，不重新执行历史参考模型前向。未含risk_score的旧配置与旧核验口径保持兼容。

先比较同预算准确率，再比较ProjRes在相同审计轮次、完整batch上的AUC和TPR@1%FPR，并检查其他攻击是否变强。沿用用户标准：相对同配置无防御对照最多下降5个百分点，再在合格组中比较隐私。当前seed43无防御80.37%对应75.37%下限；单seed结果用于选择下一步复验方向，不能替代多seed确认。

验证：122项CPU/协议回归通过，包含双模型、1/2/3视图、全部11种攻击、真实上传梯度、首轮与短batch、三组对照及失败清理；19项真实CUDA生成测试通过。
