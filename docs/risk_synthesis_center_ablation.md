# 0.5噪声下的生成位置消融

入口：`scripts/run_synthesis_center_ablation.py`。

本地/全局类中心补充对照另见 [中心来源消融](risk_synthesis_center_source_ablation.md)：保持全局协方差与0.5噪声，只改变包含自身的均值来源。

## 补充：打乱风险对照

新增可选组 `shuffled_risk`，沿用0.5噪声、100轮、K=2、seed43和全部11攻击。当前已完成的 `risk` 组可作为匹配对照，本次只需补跑打乱组。

公式为 `(1-r_perm)h + r_perm*mu_c + 0.5L_c*epsilon`。每个客户端的每个原始batch计算当前风险后，用独立随机数流打乱风险与样本的对应关系，保留该batch完整风险权重集合、零风险数量和平均原图系数。图片、标签和对应的同类中心不变；同一样本的两个视图共用一次打乱结果，视图噪声仍分别抽样。首轮风险全为0，与真实风险组保持相同初始化。随机排列可以存在未移动的样本，不强制每条记录换权重。

在 `/root/Privacy` 下，两个终端分别执行：

```bash
# Adapter：只补打乱风险组，GPU0
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_synthesis_center_ablation.py \
  --variants shuffled_risk --models clip_adapter --gpus 0
```

```bash
# LoRA：只补打乱风险组，GPU1
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_synthesis_center_ablation.py \
  --variants shuffled_risk --models clip_lora --gpus 1
```

每个命令只生成1项任务。结果保存在新时间戳目录的 `shuffled_risk/` 下，可与已完成的同模型 `risk/` 对比。已按实际主入口补齐默认值核对，两组完整运行配置仅差 `defense.synthesis.mode` 和输出目录。

- 只核对命令：末尾加 `--dry-run`。
- 5轮无攻击检查：末尾加 `--smoke`。
- 新种子同时跑真实/打乱风险：`--variants risk,shuffled_risk --seeds 44,45`，每模型4项任务。
- 原三组加打乱组全部重跑：`--variants source,class_center,risk,shuffled_risk`。不传 `--variants` 仍保持原三组默认值。

底层配置使用 `mixing_mode=risk` 与 `mode=shuffled_risk`；通过 `--variants` 选择，入口拒绝手动覆盖这两个控制字段。摘要版本仍为v15，核验器额外检查每个batch的 `risk` 与 `used_risk` 是同一集合，各视图沿用同一次分配。训练算法复用已有打乱逻辑。

判读时优先比较同模型/同seed的 `risk vs shuffled_risk`：两者权重分布相同，差别在于风险与样本的对应关系。需要同时看准确率、逐攻击AUC与低FPR指标；单种子差异仍不能证明稳定的排序收益。

## 三组定义

固定 `noise_scale=0.5`，只改变加噪前的位置：

| 组名 | 训练输入的 patch token | 原图系数 | 第一轮 |
|---|---|---|---|
| `source`：纯目标图像 | $h_i+0.5L_c\epsilon$ | 1 | 原图编码加噪 |
| `class_center`：纯类中心 | $\mu_c+0.5L_c\epsilon$ | 0 | 类中心加噪 |
| `risk`：当前方案 | $(1-r_i)h_i+r_i\mu_c+0.5L_c\epsilon$ | $1-r_i$ | $r_i=0$，原图编码加噪 |

- “纯”指混合位置的两个端点，三组都加噪。`h_i` 是图像的输入 patch+position token，CLS token 保持原模型定义；这不是直接对像素加噪。
- `mu_c` 是首轮聚合后固定的全局同类均值，包含当前原始样本；`L_c=U_c sqrt(Lambda_c)` 保留同类协方差全部数值有效方向。三组共用同一种统计构造，噪声协方差都是 `0.25 Sigma_c`，不重归一化。
- 0.5是噪声幅度系数，不表示只给50%的样本加噪。每次访问的每个视图独立抽样一次，全部参与训练。
- 纯类中心组从第一轮就将混合系数固定为1，不受“首轮无风险参考”的初始化影响。当前方案则沿用首轮风险为0的行为。

## 固定的比较条件

默认 Adapter、LoRA 各三组，共6项；CIFAR100、seed43、目标客户端0、10个IID客户端、全局每类100张训练图像、batch32、FedAvg100轮、每轮1个完整本地epoch、按样本数聚合、K=2。

沿用刚才0.5实验的训练/evaluation划分清单及SHA256。每个原始batch对两个视图的CE取平均后只更新一次；攻击仍以原始客户端训练集定义成员。两视图不扩大成员/非成员数量。正式运行开启全部11种攻击，沿用FedAvg每10/50轮的分组审计间隔，准确率每5轮评估。

三组均保留现有风险排序计算和记录，在有上一轮参考时执行；`source`/`class_center` 的风险数值不参与混合系数。这样本次比较只改变生成位置，不同时删除风险计算流程。统计共享、教师关闭、候选不筛选、GPU生成和成功后统计清理均沿用当前定义。

脚本在启动前解析所有组的最终配置，检查同一模型/数据集/种子/目标的配置只相差 `mixing_mode`、真实/打乱风险的 `mode` 和结果目录；不兼容组合、非0.5噪声或非全局完整秩几何会在写目录前报错。

## 运行

在 `/root/Privacy` 下，使用已有环境：

```bash
PRIVACY_PY=/root/.local/share/mamba/envs/pfedba/bin/python

# 查看默认6项计划，不创建结果目录
"$PRIVACY_PY" scripts/run_synthesis_center_ablation.py --dry-run

# 默认6项完整实验，GPU0串行
"$PRIVACY_PY" scripts/run_synthesis_center_ablation.py
```

使用两张卡时，在两个终端分别运行，各自串行跑该模型的三组：

```bash
# 终端1：Adapter / GPU0
python scripts/run_synthesis_center_ablation.py --models clip_adapter --gpus 0

# 终端2：LoRA / GPU1
python scripts/run_synthesis_center_ablation.py --models clip_lora --gpus 1
```

短检查和补跑：

```bash
# 三组5轮、每轮评估、无攻击；仅观察早期训练表现
python scripts/run_synthesis_center_ablation.py --smoke

# 只跑Adapter的纯类中心组
python scripts/run_synthesis_center_ablation.py --models clip_adapter --variants class_center

# 三种子匹配对照：每个模型9项
python scripts/run_synthesis_center_ablation.py --models clip_adapter --seeds 43,44,45
```

`python` 指上面的pfedba环境解释器。`--models`、`--gpus`、`--seeds`、`--rounds`、`--attacks`、`--set` 等沿用统一入口，覆盖会统一应用到所有组。`--smoke` 是默认值预设，后续显式参数仍可覆盖。`--max-runs` 按每组截取相同的任务列表。

每次调用单GPU、`--jobs 1`，实际训练始终交给唯一批量调度器 `scripts/run_privacy_experiments.py`。任一组失败会停止后续组；修复后可用 `--variants` 在新目录补跑。

## 结果位置与判读

每次创建独立目录：

```text
results/synthesis_center_ablation_<时间戳>/
  study_plan.json
  source/
  class_center/
  risk/
```

三个子目录各保存统一入口的 `experiment_manifest_*.json`、`experiment_summary_*.csv` 和独立任务目录。`study_plan.json` 保存三组公式、最终配置和实际调度参数。指定 `--results-root` 可改变上级目录。

建议先看准确率曲线与最后一轮准确率，再按攻击逐项比较AUC、TPR@1%FPR，并比较11种攻击中各组的最大值。不要只挑下降的攻击。`source vs risk` 检查风险混合是否优于原图附近加噪；`class_center vs risk` 检查保留原图成分的效用/隐私代价；两个端点直接比较则观察位置选择的整体影响。

5轮无攻击结果只能评价早期效用，无法评价隐私收益；单种子不足以判断稳定性。类中心仍使用原始训练数据统计，现有11种攻击没有覆盖统计传输视图，不能据此宣称整个协议满足DP或没有成员信息。

## 配置与核验

新增可选 `defense.synthesis.mixing_mode: source | class_center | risk`，显式启用时标记 `local_token_geometry_v15_direct_mixing`；缺省保留此前v13/v14行为。新入口固定0.5，不改写历史配置或实验结果。

`synthesis_summary.json` 的 `generation_location_mode` 表示实际位置模式，`mixing_coefficient_source` 区分风险/常数0/常数1。原有 `generation_center` 仍表示参考类中心的定义，不单独表示实际混合位置。

CSV的 `risk` 继续保存原始风险；历史字段名 `used_risk` 在v15明确表示“实际类中心混合系数”，`retained_original_fraction=1-used_risk`。纯类中心首轮允许 `risk=0, used_risk=1`。核验器检查固定端点和各视图系数，记录统计清理后的凭据核验范围，不声称重新生成已删除的高维统计。

曝光分布导出在v15中区分 `zero_risk_fraction`（原始风险为零）和 `zero_mixing_fraction`（实际类中心系数为零），用 `zero_risk_basis` 标明口径；二者及保留系数统计均沿用排除首轮无参考访问的口径。

验证：295项CPU/协议回归和13项真实CUDA测试通过，覆盖公式、首轮端点、双模型1/2/3视图、原始成员身份、全部11种攻击、统计清理和历史版本兼容。当前新增三组尚未启动真实数据完整实验。
