# 统计缓存的自动清理

自2026-10-02起，新启动的 `risk_synthesis` direct 任务默认使用 `defense.synthesis.statistics_retention: cleanup_on_exit`：成功时保留原有完整核验后清理；失败、中断或统计核验报错时，也清理本次运行拥有的四类大统计文件。配置、训练与攻击指标、模型、日志和逐条CSV保留。自动清理不扫描历史目录。

| 策略 | 成功且核验通过 | 训练失败或核验异常 |
|---|---|---|
| `cleanup_on_exit`（新默认） | 保存完整核验凭据后清理 | 保存未核验清理凭据后清理 |
| `cleanup_on_success`（兼容旧配置） | 保存完整核验凭据后清理 | 保留统计 |
| `keep` | 保留统计 | 保留统计 |

## 清理流程

1. 成功路径确认任务状态为 completed、审计没有记录错误、没有未提交的训练视图。
2. 在大文件仍存在时，核验全局统计的哈希、接收回执、类别计数与均值，并沿每类八个固定探针方向检查聚合协方差。此检查不是完整高维矩阵逐元素重算，也不是候选有效性或教师语义检查。
3. 核对本地类别身份、原始样本累计计数、多视图累计计数，保存 `statistics_receipt.json`。该文件保留各客户端原始记录的标签、类别计数、数值秩、生成秩、源编码摘要；全局类别额外保留特征值、样本权重计数；记录待删文件的大小、SHA256和清理前核验结论。均值向量、协方差因子和逐样本高维编码不保留。
4. 原子写入凭据及 summary 中的 prepared 标记，释放运行中的 CPU 内存映射与 GPU 缓存引用。
5. 仅删除本任务、预先确定客户端对应的四类大文件，并把清理结果标记为 cleaned，记录删除文件名、释放字节数与清理耗时。

| 文件 | 默认终止后的策略 |
|---|---|
| `client_*_source_codes.pt` | 删除 |
| `client_*_distribution.pt` | 删除 |
| `client_*_moment_upload.pt` | 删除 |
| `global_distribution.pt` | 删除 |
| `statistics_receipt.json`、`synthesis_summary.json` | 保留 |
| `statistics_owner.json`、`statistics_cleanup_manifest.json`、`statistics_cleanup.json` | 保留（如果产生） |
| `source_exposure.pt`、客户端全局统计接收回执 | 保留 |
| `synthetic_exposure.csv`、`synthetic_views.csv` | 保留 |
| 模型、配置、训练指标、攻击结果、日志及其他文件 | 保留 |

这项机制不改变生成、风险排序、视图数或训练损失，也不新增候选筛选。统计仍需在运行期间落盘，因此运行时仍需足够磁盘空间。成功核验路径的清理耗时记录在 `synthesis_summary.json.statistics_storage.elapsed_seconds`；失败清理记录在独立凭据中，批量入口总耗时包含清理。

按此前完整2视图任务的实测文件量，预计每任务释放约5.76 GiB，保留约0.32 GiB。保留量随视图CSV长度、模型和审计产物变化，不保证所有任务都固定为这个数值。

## 清理后的分析

`scripts/analyze_risk_synthesis.py`、direct/多视图/历史路由核验入口及全局几何核验入口已支持精简格式。分析仍能读取原始标签、校验访问与视图计数、重算攻击指标，并核对保留凭据的哈希。

全局几何返回 `status: verified_before_cleanup`、`covariance_recomputed_now: false`、`full_geometry_replay_available: false`。历史大文件哈希置于 `historical_source_hashes`，不会当成仍可读取的文件来源。清理后不能重放完整几何生成过程；如需重放，使用 keep 模式或重新生成统计。没有合法清理标记却缺少大文件的旧结果仍按缺失文件处理，不会默认为已经核验。

新默认下，核验报错仍会在原摘要中保留 `retained_error` 和具体错误，再执行下述未核验清理。`cleanup_on_success` 显式旧配置继续按原规则保留。已有合法 `statistics_receipt.json` 的成功任务不降级为未核验。

## 失败和信号退出

- 初始化时先写 `statistics_owner.json`，记录本次创建的目录身份、客户端ID、进程身份与保留策略。普通异常和初始化中断在进程内释放映射/缓存后清理。
- 唯一批量入口在子进程 `wait()` 结束后补做清理，覆盖子进程SIGBUS/SIGKILL等无法执行Python finally的情况；原始退出码和日志保留。存活的训练进程不能被父进程清理器提前清理。
- 对失败或损坏的 `.pt` 文件只读取字节计算SHA256，不调用 `torch.load`。先持久化精确文件清单和prepared回执，再删除四类允许的文件。拒绝符号链接、目录身份变化、文件内容变化、未登记客户端和路径越界。
- `statistics_cleanup_manifest.json` 保存待删文件大小、分配块数、SHA256及保留证据哈希；`statistics_cleanup.json` 保存原因、实际删除量和状态 `cleaned_unverified`，失败可留下 `cleanup_incomplete` 并按原清单重试。原有摘要、日志、指标和CSV不改写。
- `cleaned_unverified` 只表示释放存储，绝不表示训练、统计或防御效果通过核验。分析器仍可独立复算攻击指标，但标记生成统计不可核验、排除完整核验配对；直接几何核验器明确报错。删除后无法重放已丢弃的几何。

如果连清理清单/凭据也无法写入，则不开始删除；已有清单后删除失败会保留可重试凭据。若启动器和子进程同时被强杀或机器断电，进程内/父进程清理均无法执行，需用显式历史清理入口恢复。单任务进程遭SIGKILL且没有批量启动器时同样如此。

## 指定历史实验清理

历史清理必须显式列出任务目录，先保存计划，再单独执行。它不递归选择全部results，不改变旧配置的保留策略，不伪造历史核验结论。

```bash
python scripts/cleanup_synthesis_statistics.py \
  --runs results/具体实验目录1 results/具体实验目录2 \
  --plan /tmp/synthesis_cleanup_plan.json

# 核对计划中的目录、文件和大小后执行：
python scripts/cleanup_synthesis_statistics.py --apply /tmp/synthesis_cleanup_plan.json
```

工具检查是否有进程打开目标目录中的文件，删除前再次核对文件身份、内容哈希和配置哈希。保留原有摘要，即使它仍记录历史清理错误；新清理状态以独立回执为准。

## 使用方式

原有正式训练命令无需追加参数，2/4/8视图均默认使用退出后清理。需要长期保留完整统计时，显式覆盖：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
  --models clip_adapter --gpus 0 --defenses risk_synthesis \
  --set defense.synthesis.statistics_retention=keep
```

此选项只控制文件生命周期。旧筛选协议默认keep；两种自动清理策略只支持direct。已经启动的进程不会因代码修改自动启用新机制。
