# 补充实验：本地类中心与全局类中心

入口：`scripts/run_synthesis_center_source_ablation.py`。

## 要回答的问题

在相同的全局协方差、0.5噪声幅度和训练协议下，将参考均值从全局同类均值改为本地同类均值，是否改善准确率与成员隐私的权衡？

已有的 `source/class_center/risk` 实验只比较了生成位置，三组参考中心都是全局中心，不能回答这个问题。`source` 是原图编码，不是本地类中心。

## 设计：补齐两个本地组

| 混合方式 | 已有全局组 | 新增本地组 | 对比回答什么 |
|---|---|---|---|
| 风险混合 | `risk` | `risk_local` | 当前风险方案是否受益于全局中心 |
| 纯类中心 | `class_center` | `class_center_local` | 不含显式原图混合项时，中心来源的作用 |

每种模型新增2项，Adapter和LoRA共4项。默认只补本地组；已有同配置的全局组作为配对对照。`source` 中心系数为0，无需再按中心来源重复运行。

两个维度独立：`--variants risk,class_center` 选择混合方式，`--centers local,global` 选择中心范围。默认值分别为 `risk,class_center` 和 `local`。

记 `h_i` 为原图输入patch+position token，`mu_{k,c}` 为客户端k的同类均值，`mu_c` 为全局同类均值，`L_c` 为全局同类协方差因子：

```text
risk_global:         (1-r_i) h_i + r_i mu_c     + 0.5 L_c epsilon
risk_local:          (1-r_i) h_i + r_i mu_{k,c} + 0.5 L_c epsilon
class_center_global:              mu_c         + 0.5 L_c epsilon
class_center_local:               mu_{k,c}     + 0.5 L_c epsilon
```

### 必须保持一致的条件

- **只换均值，协方差仍全部使用全局同类协方差**，保留全部数值有效方向。若同时切换为本地协方差，会改变噪声方向、秩和总能量，无法把结果归因于中心来源。该实验也不检验“完全不共享统计”的本地协议。
- **两种均值都包含样本自身**。旧 `center_source=local_class` 排除自身，不能直接用它与现有包含自身的全局中心作单因素对比。因此新增 `center_source=local_class_mean`，仅支持direct模式；旧配置行为保留。
- 均值在首次训练前从原始完整训练集拟合并固定，不是当前mini-batch均值。默认IID划分下，本地每类10张、全局每类100张。因此原始样本在两种均值中的权重分别是1/10和1/100；这提示可能存在隐私差异，但不直接给出攻击强弱结论。
- 0.5是噪声幅度，不是50%的记录加噪。每次原始访问均生成K=2视图，按视图平均CE后更新一次；攻击成员仍是原始客户端训练集。
- 保持风险计算规则相同。训练轨迹改变后，后续风险数值可以自然不同；本实验不强制重放全局组的风险序列。纯类中心配对可帮助观察不使用风险混合系数时的中心效应。

## 运行命令

在 `/root/Privacy` 下，两个终端分别运行：

```bash
# Adapter，GPU0；先风险混合，再纯类中心，共2项
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_synthesis_center_source_ablation.py \
  --models clip_adapter --gpus 0
```

```bash
# LoRA，GPU1；先风险混合，再纯类中心，共2项
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_synthesis_center_source_ablation.py \
  --models clip_lora --gpus 1
```

如果只用一张卡，不传参数即可在GPU0串行运行默认4项：

```bash
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_synthesis_center_source_ablation.py
```

固定预设沿用已有全局实验：CIFAR100、seed43、目标客户端0、10个IID客户端、每类100张训练图像、batch32、FedAvg100轮、每轮1个完整本地epoch、按样本数聚合、K=2、全部11攻击、准确率每5轮评估。训练/evaluation划分清单和SHA256沿用 `run_global_synthesis_validation.py`。

- 只检查配置：末尾加 `--dry-run`，不创建结果目录、不训练。
- 5轮无攻击检查：末尾加 `--smoke`。这种结果不与已有100轮结果直接比较，也不评价隐私。
- 先只补当前风险方案：末尾加 `--variants risk`，每模型1项。
- 完整重跑两种中心：末尾加 `--centers local,global`，每模型4项。
- 在新种子上验证：`--centers local,global --seeds 44,45`，每模型8项。新种子的本地组须配同种子的全局组，不能只与seed43比较。

其他统一入口参数可透传，但改变训练预算、数据划分、视图数等设置后，应配套重跑全局组。入口拒绝改变0.5噪声、全局完整秩协方差及本次指定的生成/风险协议；中心和混合方式须通过专门选项设置。

实际执行仍交给唯一批量调度器 `scripts/run_privacy_experiments.py`。每次调用单GPU、`--jobs 1`，一组失败则停止后续组。

## 与哪些已有结果配对

默认seed43补跑使用以下已有结果，分别匹配该目录内同模型的 `risk/` 和 `class_center/`：

| 模型 | 已有全局实验根目录 |
|---|---|
| Adapter | `results/synthesis_center_ablation_20260916_181617_203365/` |
| LoRA | `results/synthesis_center_ablation_20260916_181629_236141/` |

已按实际主入口补齐默认值并按调度GPU核对，四个配对的完整运行配置只差 `defense.synthesis.center_source` 和输出目录。新的全局条件仍使用 `global_class`，生成公式和随机数调用顺序与旧全局条件一致。

新结果保存为：

```text
results/synthesis_center_source_ablation_<时间戳>/
  study_plan.json
  risk_local/
  class_center_local/
  # 指定 --centers local,global 时另有 risk_global/、class_center_global/
```

各子目录保存独立任务、统一入口manifest和summary。`study_plan.json` 保存公式、调度参数与配置，并显式记录 `covariance_source=global_same_class`、`center_includes_source=true`。

## 如何判断是否值得用全局中心

分别在同模型、同混合方式内比较本地和全局，不用 `risk_local vs class_center_global` 代替单因素对照：

1. 看最终准确率和学习曲线；不只挑某一组的最佳轮次。
2. 比较同名攻击的AUC、TPR@1%FPR，并汇总全部11攻击中的最大值及对应攻击。保持成员/非成员身份、审计轮次、分数方向和时序汇总口径一致；低FPR阈值仍由独立非成员确定。
3. 如果全局中心准确率更高且攻击成功率更低，则支持当前条件下全局中心占优。如果本地准确率提高但攻击也更强，则是效用与隐私的取舍，不能笼统说某种中心更好。若差异接近波动，则暂不能认定中心来源有稳定收益。
4. `risk` 与 `class_center` 两个配对方向一致，证据更完整；若方向不同，说明中心效应依赖混合方式，应分开报告。

先用seed43低成本补齐对照，有明确差异后再做新种子匹配验证。单种子不能证明普遍优势；当前11攻击也没有覆盖统计传输视图，不能据此给出整个协议的DP保证。

## 实现与核验口径

`local_class_mean` 直接读取已有本地同类均值，不新增统计上传。该枚举只开放给direct模式，所有旧枚举和缺省行为不变；本入口继续使用v15混合机制。

本地组摘要为 `generation_center=local_same_class_mean`、`center_includes_source=true`、`geometry_source=global_same_class`。全局组维持 `global_same_class_mean`。核验器检查中心定义，并兼容统计清理后的凭据和历史局部记录；不声称重新读取已删除的协方差。

291项CPU/协议测试和19项真实CUDA测试通过，覆盖中心公式、单样本类别、相同全局噪声与随机数流、旧排除自身行为、补跑配置匹配、失败停止、双模型1/2/3视图和全部11攻击、清理前后凭据及CPU/CUDA一致性。正式CIFAR100训练尚未启动，使用上述命令开始。
