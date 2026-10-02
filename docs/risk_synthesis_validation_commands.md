# 当前全局均值方案：Adapter / LoRA 验证命令

2026-09-27：direct 生成支持 `--methods fedsgd`，验证及消融脚本按模型基线和 catalog 方法配置使用默认值：Adapter/LoRA FedSGD1000轮等权，FedAvg100轮样本数加权。原脚本不传方法仍为 FedAvg。见 [FedSGD 协议与运行命令](risk_synthesis_fedsgd.md)。

2026-09-16新增用户要求的 **0.5噪声、纯目标图像/纯类中心/风险混合三组消融**：使用 `scripts/run_synthesis_center_ablation.py`，默认每模型三组100轮，也支持 `--smoke` 五轮检查。完整公式与命令见 [生成位置消融](risk_synthesis_center_ablation.md)。下文保留原全局验证入口的配置。

新版已整合到 `/root/Privacy` 的 `main`，采用包含自身的全局类别均值、全局同类别几何噪声、风险控制原始编码占比，以及每次访问的多替身直接共同训练。当前v13直接加入 \(L\epsilon\)，删除额外噪声系数；不做有效性或教师语义检查，不重试、不择优。所有命令统一从正式目录运行。

## 固定的初始验证协议

专用脚本 `scripts/run_global_synthesis_validation.py` 仍调用唯一批量入口 `scripts/run_privacy_experiments.py`，不复制模型 YAML 或实现另一套训练流程。

| 项目 | 设置 |
|---|---|
| 模型 | CLIP transformer Adapter、CLIP LoRA |
| 数据 | CIFAR100，全局每类100张训练图片，再划分到10个IID客户端 |
| 对照 | `none` 与 `risk_synthesis` |
| 随机种子 | 43（先做单种子初步验证） |
| 训练 | FedAvg100轮，每轮完整本地epoch为1，原始batch大小32 |
| 聚合 | 按原始本地样本数加权 |
| 生成 | 包含自身的全局类别均值，全局同类协方差；每次原始访问训练2个替身 |
| 当前其余生成参数 | 沿用catalog；噪声为 \(L\epsilon\)，无额外系数，使用全部数值有效方向；每视图只抽样一次 |
| 风险与筛选 | 当前损失差风险、不启用历史因子、生成即训练 |
| 审计 | 全部11种攻击，目标客户端0；保持FedAvg既有审计间隔 |
| 正式候选 | 目标客户端1000个原始成员与1000个独立evaluation非成员 |

共有4次独立训练，每个模型2次；多个攻击共享同一次训练。学习率和PEFT结构沿用各模型基线。原始成员数不会乘替身数。

训练与evaluation使用已存在的隔离清单 `analysis_scripts/risk_synthesis_confirmation_data_20260912/split.json`，脚本固定其SHA256；同模型、同seed的两组使用相同分区协议。该来源与部分种子已经用于此前研究，因此本批不能称为未接触数据的独立确认。evaluation/test不随训练的每类限额截断。

## 两张GPU分别运行

终端一：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
  --models clip_adapter --gpus 0
```

终端二：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
  --models clip_lora --gpus 1
```

每个命令在所选GPU上串行执行2个任务。两个终端可并行运行。该脚本要求一个GPU及 `--jobs 1`，不对既有统一调度器做并行改造；不要向此脚本传 `--gpus 0,1 --jobs 2`。

按当前要求，默认只运行seed43，无需额外传入种子参数。单种子结果用于初步判断效果，不能判断跨种子稳定性。后续仅补充其余种子可显式使用 `--seeds 44,45`；重复执行原命令会新建任务并重跑seed43，不自动跳过历史结果。

## 先检查配置或只用一张GPU

只检查两种模型的全部4项配置，不创建结果目录、不启动训练：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py --dry-run
```

使用一张GPU依次运行两种模型：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py --gpus 0
```

以上是三种运行选择，不需要重复执行所有训练命令。每次实际执行都会创建新的任务；不会续训、覆盖或自动复用旧结果。

## 结果与判断方式

原始实验输出直接保存到 `/root/Privacy/results/`，每次训练一个独立一级目录。每个任务保存配置、日志、训练指标、正式审计和生成诊断；入口结束时输出 `EXPERIMENT OVERVIEW` 和批次汇总产物。全局分布及逐客户端回执位于任务内的 `risk_synthesis/`。

先分别在每种模型的seed43内比较新方案与无防御，核对实际数据身份、候选池和完成状态。跨种子变化需后续增加种子后再评估。至少同时报告Accuracy、全部攻击AUC和可报告的TPR@1%FPR、全部原始访问生成情况、训练视图计数及耗时；不只挑选下降的攻击或最好的种子。

沿用前序研究的事前效用/隐私判据作初筛：每种模型的seed43最大攻击AUC降低至少0.02、最大TPR@1%FPR降低、Accuracy损失不超过2个百分点。报告两种模型是否分别满足，并保留逐攻击结果；这个判据不等于统计显著性证明。当前两组对照只能评价整体方法，不能独立证明风险排序、全局中心或多视图各自的增益。

共享类别统计不受DP保护，现有11种攻击没有专门纳入这些统计的攻击视图。即便本批通过上述判据，也只能在已测威胁视图下报告经验效果。

本文命令已做干运行和协议核对；创建脚本时没有启动本批真实训练。

## 历史v11范数门槛修复后的补跑记录

2026-09-14 的旧 v10 Adapter 和 LoRA 防御任务均因 `invalid_norm_ratio` 退出。v11 删除该门槛，范数仅用于诊断。已完成的无防御任务保留，无需重复训练。以下两个命令分别新建防御任务，默认仍是seed43、100轮、每次两个替身：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
  --models clip_adapter --gpus 0 --defenses risk_synthesis
```

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
  --models clip_lora --gpus 1 --defenses risk_synthesis
```

两张卡可分别执行；只用一张卡则传 `--gpus 0 --defenses risk_synthesis`，默认依次运行两个模型。可先追加 `--dry-run` 检查。新任务从头训练，不续写失败目录；比较时使用各模型此前同分区的完整无防御基线，不把失败防御任务的初始准确率当最终效果。

## 当前v13运行方式

统一入口已默认直接生成并训练，`semantic_filter=false`、`candidate_selection=direct`。删除重试次数、语义容差、最小本地类别候选数和 `noise_scale` 设置。当前默认 `class_rank=all`，噪声直接为 \(L\epsilon\)，使用全部有效方向；无需额外传参。显式 `--set defense.synthesis.class_rank=5` 仅用于整数秩对照，其噪声仍无额外系数。以上命令现在启动v13任务。

相对之前默认0.1系数，固定几何时噪声幅度增至10倍、期望能量增至100倍。后续用户要求尝试0.5噪声，现可显式设置 `--set defense.synthesis.noise_scale=0.5`，新任务记录为v14幅度对照；省略该字段仍为v13。显式旧筛选模式保留原噪声参数，历史结果按原实现版本核验。已有v13五轮结果，尚无v13完整效果实验。固定五轮、2视图、关闭攻击的试验命令见 [0.5噪声对照](risk_synthesis_noise_half.md)。

继续使用两个视图：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py --models clip_adapter --gpus 0 --defenses risk_synthesis
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py --models clip_lora --gpus 1 --defenses risk_synthesis
```

两条命令分别放在对应终端运行。每条原始记录需要更多共同训练候选时，加 `--set defense.synthesis.views_per_record=4`，即每次生成4个、全部训练；候选数不再由attempts控制。新建任务保留历史结果和无防御基线；不续写现有任务。

CUDA训练现在默认启用设备端几何生成与有界类别因子缓存，原命令无需增加参数。只影响新启动任务；正在运行的进程继续使用已加载实现。说明与局部计时见 [GPU生成优化](risk_synthesis_gpu_generation.md)。

新启动的direct任务还默认 `statistics_retention=cleanup_on_success`：训练与审计成功后保留精简核验凭据，自动删除本任务的四类大统计缓存；模型、指标、攻击结果和CSV保留。失败与历史任务不清理。需要保留完整统计时追加 `--set defense.synthesis.statistics_retention=keep`。见 [统计缓存清理](risk_synthesis_statistics_retention.md)。

## 每次原始访问分别生成2、4、8个替身

以下共6次独立训练：Adapter和LoRA各3次，均为seed43、CIFAR100、FedAvg100轮、每类100张、原始batch32、全部11种攻击。全部方向和GPU生成优化默认启用；此次只跑risk_synthesis，使用已有匹配的无防御基线进行比较。2/4/8表示每条原始记录每次访问的替身数，全部参加训练，各视图损失平均后每原始batch更新一次参数。此前6组合的干运行和已完成效果实验属于v12；当前执行这些命令会新建无额外噪声系数的v13任务。

终端一，GPU0依次运行Adapter：

```bash
cd /root/Privacy
for views in 2 4 8; do
  /root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
    --models clip_adapter --gpus 0 --defenses risk_synthesis \
    --set "defense.synthesis.views_per_record=${views}"
done
```

终端二，GPU1依次运行LoRA：

```bash
cd /root/Privacy
for views in 2 4 8; do
  /root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py \
    --models clip_lora --gpus 1 --defenses risk_synthesis \
    --set "defense.synthesis.views_per_record=${views}"
done
```

两个终端可以并行执行，每张卡内部依次运行3个任务。输出各自新建在results下，配置记录views_per_record，历史结果不覆盖。只检查命令时，在python调用末尾追加`--dry-run`。单卡可在任一循环中使用`--models clip_adapter,clip_lora --gpus 0`，依次执行全部6项。
