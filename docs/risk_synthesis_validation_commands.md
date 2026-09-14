# 当前全局均值方案：Adapter / LoRA 验证命令

新版已整合到 `/root/Privacy` 的 `main`，采用包含自身的全局类别均值、全局同类别几何噪声、风险控制原始编码占比，以及每次访问的多替身直接共同训练。当前v12不做有效性或教师语义检查，不重试、不择优。所有命令统一从正式目录运行。

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
| 当前其余生成参数 | 沿用catalog；噪声幅度0.1、生成秩上限5；每视图只抽样一次 |
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

## 当前v12运行方式

统一入口已默认直接生成并训练，`semantic_filter=false`、`candidate_selection=direct`。删除重试次数、语义容差和最小本地类别候选数设置。以上命令现在启动v12任务；含显式旧筛选选项的历史配置继续对应旧机制，不会自动变成v12。已经启动的v11进程不受文件修改影响。

继续使用两个视图：

```bash
cd /root/Privacy
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py --models clip_adapter --gpus 0 --defenses risk_synthesis
/root/.local/share/mamba/envs/pfedba/bin/python scripts/run_global_synthesis_validation.py --models clip_lora --gpus 1 --defenses risk_synthesis
```

两条命令分别放在对应终端运行。每条原始记录需要更多共同训练候选时，加 `--set defense.synthesis.views_per_record=4`，即每次生成4个、全部训练；候选数不再由attempts控制。新建任务保留历史结果和无防御基线；不续写现有任务。
