# 统计缓存的自动清理

当前新启动的 `risk_synthesis` direct 任务默认使用 `defense.synthesis.statistics_retention: cleanup_on_success`。训练、最终模型保存、攻击审计、诊断关闭和性能摘要写入成功后，程序先保存精简核验凭据，再删除本任务的大统计缓存。历史结果目录不会被扫描、修改或清理。

## 清理流程

1. 确认任务状态为 completed、审计没有记录错误、没有未提交的训练视图；失败任务保留现场。
2. 在大文件仍存在时，核验全局统计的哈希、接收回执、类别计数与均值，并沿每类八个固定探针方向检查聚合协方差。此检查不是完整高维矩阵逐元素重算，也不是候选有效性或教师语义检查。
3. 核对本地类别身份、原始样本累计计数、多视图累计计数，保存 `statistics_receipt.json`。该文件保留各客户端原始记录的标签、类别计数、数值秩、生成秩、源编码摘要；全局类别额外保留特征值、样本权重计数；记录待删文件的大小、SHA256和清理前核验结论。均值向量、协方差因子和逐样本高维编码不保留。
4. 原子写入凭据及 summary 中的 prepared 标记，释放运行中的 CPU 内存映射与 GPU 缓存引用。
5. 仅删除本任务、预先确定客户端对应的四类大文件，并把清理结果标记为 cleaned，记录删除文件名、释放字节数与清理耗时。

| 文件 | 成功后的策略 |
|---|---|
| `client_*_source_codes.pt` | 删除 |
| `client_*_distribution.pt` | 删除 |
| `client_*_moment_upload.pt` | 删除 |
| `global_distribution.pt` | 删除 |
| `statistics_receipt.json`、`synthesis_summary.json` | 保留 |
| `source_exposure.pt`、客户端全局统计接收回执 | 保留 |
| `synthetic_exposure.csv`、`synthetic_views.csv` | 保留 |
| 模型、配置、训练指标、攻击结果、日志及其他文件 | 保留 |

这项机制不改变生成、风险排序、视图数或训练损失，也不新增候选筛选。统计仍需在运行期间落盘，因此运行时仍需足够磁盘空间；它解决的是成功实验连续累积的磁盘占用，不能挽救已经写满磁盘的任务。清理耗时位于现有 `performance_summary.json` 的 run 计时之后，单独记录在 `synthesis_summary.json.statistics_storage.elapsed_seconds` 中；批量入口的任务总耗时包含它。

按此前完整2视图任务的实测文件量，预计每任务释放约5.76 GiB，保留约0.32 GiB。保留量随视图CSV长度、模型和审计产物变化，不保证所有任务都固定为这个数值。

## 清理后的分析

`scripts/analyze_risk_synthesis.py`、direct/多视图/历史路由核验入口及全局几何核验入口已支持精简格式。分析仍能读取原始标签、校验访问与视图计数、重算攻击指标，并核对保留凭据的哈希。

全局几何返回 `status: verified_before_cleanup`、`covariance_recomputed_now: false`、`full_geometry_replay_available: false`。历史大文件哈希置于 `historical_source_hashes`，不会当成仍可读取的文件来源。清理后不能重放完整几何生成过程；如需重放，使用 keep 模式或重新生成统计。没有合法清理标记却缺少大文件的旧结果仍按缺失文件处理，不会默认为已经核验。

核验或凭据写入失败时不删除大文件，记录 retained_error；删除过程中失败时记录 cleanup_incomplete，剩余文件保留，分析可使用先前已写好的精简凭据。清理错误不掩盖已经完成的训练结果。该机制不提供针对历史目录的批量删除入口。

## 使用方式

原有正式训练命令无需追加参数，2/4/8视图均自动使用成功后清理。需要长期保留完整统计时，显式覆盖：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
  --models clip_adapter --gpus 0 --defenses risk_synthesis \
  --set defense.synthesis.statistics_retention=keep
```

此选项只控制文件生命周期。旧筛选协议默认 keep；cleanup_on_success 目前只支持 direct。已经启动的进程不会因代码修改自动启用新机制。
