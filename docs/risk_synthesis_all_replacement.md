# 全部替换、风险控制保留量

2026-09-12，依据用户的新要求及“语义重试失败时保留最佳虚拟候选”的选择，将 `risk_synthesis` 默认改为 `replacement_policy: all`。目前支持 CLIP transformer Adapter/LoRA + FedAvg。此前部分替换方案的确认结果不能作为本版本的有效性证据。

## 当前流程

每客户端仍从原始本地训练集拟合各类 Patch+Position token 的均值和低秩协方差，并向该客户端合并类内协方差收缩；统计不共享。当前每类全局 100 张、10 个 IID 客户端，即本地每类 10 张。

每次训练访问的每个位置都使用：

\[
\widetilde h_i=(1-r_i)h_i+r_i\mu_{c,-i}+0.1L_c\epsilon,
\qquad\epsilon\sim\mathcal N(0,I).
\]

- 风险只控制线性生成中心中的原始编码系数 `1-r`。例如 r=0.2 保留 80% 原始编码系数，r=0.9 保留 10%；这不是对剩余隐私信息量的估计。
- 所有位置均生成，不使用概率抽样或 batch 替换数量上限。`replacement_fraction=1` 在本模式中是协议校验值，不会再乘以 r。
- 从首轮开始替换，`warmup_rounds=0`。没有上一轮 own/other 参考时 r=0，输入为原编码加几何噪声，仍须与原编码实际不同；不伪造首轮风险分数。
- 后续复用 WWW 损失差与 80% 尾部秩权重；低风险 r=0 的记录也会加入几何噪声。`shuffled_risk` 对照只打乱 r 与记录的对应，仍替换所有位置。
- 类别中心继续使用均匀权重。当前全部替换模式要求正噪声，且不接受 MixUp、历史风险加权中心或原图 warmup；旧实验选项仍可在历史概率替换模式中使用。

每个生成候选先检查有限值、实际改变和范数比 0.1～2.0，再由固定原始 CLIP 教师计算真实类余弦 margin。最多尝试两次；第一次达到原 margin−0.02 即使用该候选。若重试后仍未达到语义门槛，按用户选择，使用所有有效尝试中 **margin 最高的虚拟候选**，同时记录 `quality_passed=0`、`reason=best_semantic_candidate`。这一候选可能来自第一次尝试，不默认采用最后一次。

不回退原图，也不缩小 r 来通过语义检查。如果两次都无法产生有限、范数合格且确实改变的编码，该 batch 在优化前报错；不会把无效值或未改变的输入记为替换成功。每个本地类别仍需至少三条原始记录。

替换后的 token 经过当前全部视觉 PEFT 层，使用普通 CE；LoRA 文本分支继续训练。原始 batch 数量、大小、本地 epoch、优化器步骤和 sample-count 聚合保持不变。审计成员始终为原始完整客户端训练集，虚拟编码不充当非成员；本方案无形式 DP 保证。

## 诊断与兼容性

新实现标记为 `local_token_geometry_v4_all_replacement`。每次成功训练访问应满足：

- `requested=accepted=1`、`fallback=0`，从第一轮起全部计入 `synthetic_steps`，`real_steps=0`。
- `original_distance>0`；`retained_original_fraction=1-used_risk`。
- `selected_attempt` 标明实际使用哪次候选，`quality_passed` 区分“已替换”与“语义达标”。`quality_failed` 单独汇总，不与原图回退混淆。
- 范数和 margin 记录实际选用的候选。旧版本的同名诊断记录最后一次尝试；分析器通过 `measurement_scope` 区分。

旧的已保存、非空 synthesis 配置缺少 `replacement_policy` 时，仍按历史 `risk_probability` 解释，保留原 warmup、随机请求和原图回退行为。统一入口的新默认配置显式为 `all`。历史 `results/` 不补写、不改写。

## 运行

稳定参数已写入 catalog，新方案无需额外 `--set`：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_privacy_experiments.py \
  --models clip_adapter --datasets cifar100 --methods fedavg \
  --defenses risk_synthesis --attacks all
```

默认 100 轮、一个完整本地 epoch、10 个 IID 客户端、每类全局 100 张、batch 32、学习率 0.01。LoRA 使用 `--models clip_lora`。可先追加 `--dry-run --max-runs 1`；输出必须包含 `synthesis=policy:all warmup_rounds:0 semantic_failure:best_generated_candidate`。

旧的范数下限 0.1 部分替换对照，须在 risk_synthesis 任务上同时显式设置：

```bash
--set defense.synthesis.replacement_policy=risk_probability \
--set defense.synthesis.replacement_fraction=0.25 \
--set defense.synthesis.warmup_rounds=1
```

完整历史配置以各任务 `run_config.yaml` 为准；上述对照继续使用新 catalog 的范数下限 0.1，如需初版 0.5 再显式覆盖。

## 验证状态

已通过核心与新增全部替换测试共 43 项，覆盖首轮/零风险/单条短 batch 的实际改变、全位置请求、语义最优候选来自第一次或第二次、几何重试、禁止原图回退、历史配置兼容，以及两个模型的两轮多 epoch FedAvg、全部 11 种攻击与原始成员身份。独立分析器核对逐记录替换计数及保留系数，并拒绝原始距离为零的伪替换记录。

入口、确认来源、汇总和历史敏感性分析另有 86 项回归通过，合计 129 项；新增默认参数断言也通过。真实 CIFAR100 的 Adapter/LoRA 各两轮执行检查均已完成并独立核验：

| 模型 | 原始位置访问 | 实际替换 | 原图回退 | 语义未达标但仍使用最佳虚拟候选 |
| --- | ---: | ---: | ---: | ---: |
| Adapter | 20,000 | 20,000 | 0 | 1,163 |
| LoRA | 20,000 | 20,000 | 0 | 1,017 |

两个模型第一轮各 10,000 次访问全部替换，均有 2 次语义未达标。每个原始记录均有两次 synthetic_steps、零 real_steps、一次风险读取；十个客户端的原始输入编码在两模型间一致。22 个正式攻击输出、原始成员身份和流式诊断通过独立分析器核验，证据为 `analysis_scripts/risk_synthesis_all_pilot_validation_20260912.json` 及 `risk_synthesis_all_{adapter,lora}_pilot_verified_20260912/`。

这些是两轮执行检查，没有相同两轮预算的无防御对照，不能证明防御有效。旧部分替换的 LoRA 任务已因本次用户改动停止；无防御 Food101 基线于 17:41 成功完成，旧队列随后因源码指纹变化停止，没有启动旧部分替换任务。旧 Adapter 三种子效果及实测耗时只属于旧方案，见[历史结果说明](risk_synthesis_results.md)。

## 完整预算效果对照

2026-09-12 17:45 启动 Adapter/CIFAR100 的真实风险与打乱风险各 100 轮，seed=43、目标客户端 0。计划为 `analysis_scripts/risk_synthesis_all_study_plan_20260912.json`，SHA256 `4eb407a68e8be2c82fa7a1ff867016267f11950a30c345c002fd8200efadb0b3`。两组仍经唯一统一训练入口运行，实际解析配置已逐项核对；每个新任务预期有 1,000,000 次原始训练位置访问。

复用相同种子、原始来源分区、训练/审计预算的已完成无防御基线，并附旧部分替换组作为整体协议对照。共享模型、训练器、聚合器、数据与攻击源码均未变化；差异仅为新防御实现、默认值及干运行显示。所有非防御实际配置完全相同，已有结果文件指纹冻结，最终还需核对原始候选身份。

这次继续使用此前评估过的预留来源分区，是探索性效果检查，不是全新的独立确认。判据固定为：相对无防御，11 种攻击中的最大 AUC 至少降低 0.02，最大 TPR@1%FPR 下降，准确率降低不超过 2 个百分点。真实风险与打乱风险另行比较；旧部分替换与新版属于同时改变多项行为的整体协议比较。

状态为 `risk_synthesis_all_study_{risk,shuffled_risk}_execution_20260912.json`。`analysis_scripts/verify_synthesis_all_study_20260912.py` 等待这两个既有进程完成，再独立复算全部攻击、检查全替换计数、复算重复像素敏感性并进行固定模型下的候选配对重采样。正式效果报告保存到 `analysis_scripts/risk_synthesis_all_study_verified_20260912/`；任务未完成时不据中途准确率判定防御有效。
