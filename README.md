# Federated PEFT Membership-Inference Benchmark

本仓库用于研究联邦学习与参数高效微调（PEFT）中的成员隐私泄漏。当前维护的实验
覆盖 ResNet18、CLIP-MLP、双侧 CLIP-Adapter、CLIP-LoRA、BERT-Base Adapter、
BERT-Base LoRA 和 GPT2-Large Adapter，并在统一训练任务中完成模型训练、成员推理审计、低 FPR
指标计算、候选集留档和结果汇总。

项目只研究成员隐私，不包含后门触发器、恶意客户端投毒、攻击成功率（ASR）或
SEISMOGRAPH 防御。仓库中可能保留早期研究攻击的实现文件，但公共审计器目前只
注册本文档列出的 11 种攻击；未注册实现不能通过配置或命令行用于新实验。

## 当前实验协议

| 模型 | 可训练部分 | 联邦方法 | 每轮客户端训练 | 默认轮数 | 默认学习率 | 默认攻击 |
| --- | --- | --- | --- | ---: | ---: | --- |
| ResNet18 / CIFAR100 | 完整 CIFAR ResNet18（GroupNorm） | FedAvg | 1 个完整 local epoch | 300 | 0.1，逐轮乘 0.99 | FedMIA-Loss |
| CLIP-MLP | 冻结 CLIP 图像编码器后的两层 MLP | FedSGD | 1 个 mini-batch / 1 次 SGD step | 150 | 0.1 | 11 种攻击 |
| CLIP-Adapter | 图像、文本两侧瓶颈 Adapter | FedSGD | 1 个 mini-batch / 1 次 SGD step | 300 | 0.001 | 11 种攻击 |
| CLIP-LoRA | 图像、文本注意力 Q/K/V 的 LoRA 因子 | FedSGD | 1 个 mini-batch / 1 次 SGD step | 300 | 0.0002 | 11 种攻击 |
| BERT-Base Adapter | 各 Transformer block 的 Adapter 和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.005 | 11 种攻击 |
| BERT-Base LoRA | 各层注意力 Query/Value 的 LoRA 因子和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.01 | 11 种攻击 |
| GPT2-Large Adapter | 各 Transformer block 的 Adapter 和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.001 | 11 种攻击 |

当前正式 sweep 的共同约定：

- CLIP 三种模型使用 10 个客户端、batch size 32、IID 划分和参与客户端等权 FedSGD；CLIP-MLP 与 CLIP-Adapter 使用每类 16 张训练图像，CLIP-LoRA 使用完整训练集。
- BERT/GPT2 使用 30 个客户端、IID 划分和 one-batch 等权 FedSGD。
- FedSGD 客户端上传各自的可训练参数梯度；服务器先等权聚合梯度，再执行一次
  `global = base - learning_rate * mean_gradient` 并下发新的全局模型。
- CLIP 普通任务每 5 个已完成通信轮评估一次；多轮攻击通常每 10 轮审计。
- BERT 普通任务、六种真实 Batch 攻击与 ICLR 每 50 轮评估，五种固定候选攻击每
  10 轮评估；GPT2 普通任务与全部攻击每 50 轮评估。调度始终包含最后一轮。
- 所有聚合和保存操作只处理 `requires_grad=True` 的参数，冻结主干不上传。

`main.py` 仍保留通用 CLIP prompt 和 PromptFL 兼容入口，但当前重点维护和批量
复现的是上表中的七类模型。

## 成员推理攻击

公共注册表位于
[`privacy_attacks/auditor.py`](privacy_attacks/auditor.py)，当前支持：

| 运行名 | 信号 | 时间/客户端信息 |
| --- | --- | --- |
| `blackbox_loss` | 候选样本负交叉熵 | 单轮、目标客户端 |
| `loss_series` | 候选样本负交叉熵 | 多轮时间平均、目标客户端 |
| `grad_cosine` | 候选梯度与客户端上传更新的余弦相似度 | 单轮、目标客户端 |
| `avg_cosine` | 梯度余弦 | 多轮时间平均、目标客户端 |
| `fedmia_loss` | 损失/置信度的单尾 CDF 分数 | 多轮、其他客户端构成 null 分布 |
| `fedmia_cosine` | 梯度余弦的单尾 CDF 分数 | 多轮、其他客户端构成 null 分布 |
| `gradient_diff` | 客户端梯度与候选梯度的差异 | 更新敏感攻击 |
| `score_diff` | 客户端更新前后的损失差 | 更新敏感攻击 |
| `score_ratio` | 客户端更新前后的损失比 | 更新敏感攻击 |
| `fta` | 多个 FL 快照上置信度或损失的变化斜率 | 多轮攻击 |
| `projres` | 候选表示相对真实上传更新子空间的投影残差 | 严格依赖被攻击层和上传协议 |

所有输出分数统一为“越大越像成员”。完整论文与公式映射见
[`docs/attack_mapping.md`](docs/attack_mapping.md)。

### 各模型的攻击支持范围

| 模型 | 默认通用攻击 | ProjRes | 说明 |
| --- | --- | --- | --- |
| ResNet18 / CIFAR100 | `fedmia_loss` | 不支持 | 论文复现入口；同时输出同一固定候选的逐样本 ICLR 评分 |
| CLIP-MLP | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种；ProjRes 攻击 `classifier.0.weight` |
| CLIP-Adapter | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种，均在同一训练任务内运行 |
| CLIP-LoRA | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种，均在同一训练任务内运行 |
| BERT-Base Adapter | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种 |
| BERT-Base LoRA | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种；ProjRes 攻击首个 Query `lora_A`，使用有效 token 均值表示 |
| GPT2-Large Adapter | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种 |

“代码已注册/可配置”和“正式 sweep 默认启用”是两个不同概念。当前 11 个注册名
均列在上一表；批量复现实验应以上表的默认集合为准。CLIP-MLP、CLIP-Adapter、
CLIP-LoRA、BERT 和 GPT2 现使用相同的攻击分组与候选定义。CLIP-MLP 另保留独立
ProjRes 入口用于单轮诊断，但统一 sweep 已将 ProjRes 作为第 11 种攻击运行。

## 各攻击的成员与非成员定义

仓库存在两种成员问题：

- **真实 Batch（exact-batch）**：判断样本是否出现在目标客户端产生当轮上传的
  真实 mini-batch 中。每个审计轮重新构造候选集，不跨轮拼接成员。
- **固定候选（fixed-candidate）**：判断样本是否属于目标客户端历史训练集。训练
  开始前固定成员/非成员候选，所有审计轮复用同一集合。

同一个攻击名在不同模型上可能使用不同成员问题。逐攻击的当前划分如下：

| 攻击 | CLIP-MLP / CLIP-Adapter / CLIP-LoRA / BERT / GPT2 | 成员 | 非成员 |
| --- | --- | --- | --- |
| `blackbox_loss` | 真实 Batch | 当轮上传 batch | 按成员标签直方图抽取的独立 evaluation 样本 |
| `loss_series` | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立 evaluation 样本 |
| `grad_cosine` | 真实 Batch | 同 `blackbox_loss` | 同 `blackbox_loss` |
| `avg_cosine` | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立 evaluation 样本 |
| `fedmia_loss` | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立 evaluation 样本；其他客户端只用于构造 FedMIA null 分布 |
| `fedmia_cosine` | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立 evaluation 样本；其他客户端只用于构造 FedMIA null 分布 |
| `gradient_diff` | 真实 Batch | 当轮上传 batch | 标签匹配的独立 evaluation 样本 |
| `score_diff` | 真实 Batch | 同 `gradient_diff` | 同 `gradient_diff` |
| `score_ratio` | 真实 Batch | 同 `gradient_diff` | 同 `gradient_diff` |
| `fta` | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立 evaluation 样本 |
| `projres` | 真实 Batch | 产生被观察更新的当轮真实 batch | 从独立 evaluation 池按标签抽取、与该 batch 互斥的样本 |

### 各模型的候选规模与抽样方式

| 模型/视图 | 成员与非成员来源 | 默认规模 |
| --- | --- | --- |
| ResNet18 / CIFAR100 FedMIA-Loss | 目标客户端 0 的完整训练集；非成员由完整独立测试集和其余 9 个客户端训练集等来源构成 | 5000/10000；10 个非成员来源各 1000 |
| CLIP-MLP / CLIP-Adapter / CLIP-LoRA 的 5 种固定候选攻击 | 目标客户端完整训练集，对全局独立 evaluation 样本；按类别尽力匹配，不足类别由其他类别确定性补足 | `M/M`，`M` 为目标客户端训练集大小 |
| CLIP-MLP / CLIP-Adapter / CLIP-LoRA 的 6 种真实 Batch 攻击 | 当轮真实 batch，对严格按其标签直方图抽取的独立 evaluation 非成员 | `N/10N`；完整 batch 为 32/320 |
| BERT Adapter/LoRA 与 GPT2 的 5 种固定候选攻击 | 目标客户端完整训练集，对全局独立 evaluation 样本；按类别尽力匹配，不足类别由其他类别确定性补足 | `M/M`，`M` 为目标客户端训练集大小 |
| BERT Adapter/LoRA 与 GPT2 的 6 种真实 Batch 攻击 | 当轮真实 batch，对严格按其标签直方图抽取的独立 evaluation 非成员 | `N/10N`；完整 batch 为 16/160 |

这里的 6 种真实 Batch 攻击是 `blackbox_loss`、`grad_cosine`、
`gradient_diff`、`score_diff`、`score_ratio` 和 `projres`；5 种固定候选攻击是
`loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine` 和 `fta`。
CLIP-MLP 的独立 ProjRes 诊断入口可使用更大的互斥非成员池；统一 exact-batch
路径只从从未训练的独立 evaluation 池抽取非成员，正式 sweep 以统一路径为准。

ProjRes 只输出连续的负 L1 投影残差并按分数排序，通过 ROC/AUC 或指定 FPR 的
TPR 评价；不再使用跨模型、跨轮次不可校准的固定残差阈值生成成员判断。

CLIP-MLP、CLIP-Adapter、CLIP-LoRA、BERT 和 GPT2 固定候选攻击的经验 FPR 分辨率由目标
客户端训练集大小决定。32/320 或 16/160 的真实 batch 视图只正式报告
可由样本量解析的低 FPR 指标。`summary.json` 会保存成员/非成员
数量、实际 FPR 分辨率、标签匹配方式和候选来源，不应在样本量不足时把插值得到的
极低 FPR 数值当作稳定结论。

## 仓库结构

```text
.
├── main.py                 # CLIP/prompt 训练、配置校验与审计入口
├── aggregator/             # FedAvg、FedSGD、PromptFL 聚合
├── trainmodel/             # CLIP-MLP、Adapter、LoRA、文本 Adapter
├── users/                  # 客户端本地训练与协议消息
├── servers/                # 联邦训练循环
├── privacy_attacks/        # 统一审计器、攻击与 ProjRes
├── privacy_defenses/       # 隐私防御和 ICLR 分析
├── configs/                # 统一能力目录与 7 份模型基线
├── scripts/                # 唯一批量入口及专项校准/诊断工具
├── analysis_scripts/       # CSV/Markdown/PNG 等结果分析工具
├── docs/                   # 协议、公式和实现说明
└── tests/                  # 轻量与完整回归测试
```

## 环境与数据

推荐使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 实验需要安装与本机 CUDA 匹配的 PyTorch。GPT2-Large 正式任务要求 CUDA；
BERT 允许 CPU 小规模调试。

### CLIP 权重

CLIP 通过 `local_files_only=True` 加载 `openai/clip-vit-base-patch32`，正式运行时
不会自动下载。请提前准备本地权重，并在配置中设置：

```yaml
cache_dir: ./checkpoints/clip-vit-base-patch32
```

### 数据集

统一 CLIP sweep 默认运行：

- Caltech101
- Oxford-IIIT Pets
- Flowers102
- Food101
- CIFAR100

图像数据使用本地 `data_root` 且 `download=False`。其他已接入图像数据集及目录
约定见 [`utils/data_loader.py`](utils/data_loader.py)。
默认 `data_root: ./data` 时，CIFAR100 文件应位于
`./data/CIFAR100/data/cifar-100-python/`；运行脚本不会覆盖或自动下载数据。

文本任务支持 SST-5、CoLA 和 IMDB。准备 Hugging Face 模型和 SST-5 的辅助脚本：

```bash
python scripts/download_hf_sst5_models.py
```

CoLA 和 IMDB 分别从 `data/huggingface/cola`、`data/huggingface/imdb` 等本地路径
读取；evaluation split 从不参与任何客户端训练。

## 快速开始

唯一批量入口是 [`scripts/run_privacy_experiments.py`](scripts/run_privacy_experiments.py)。
它按“模型 × 数据集 × 防御 × seed × 目标客户端”展开训练任务；同一命令选择的
多种攻击保留在一个任务中，共享一次训练，不会为每种攻击重复训练。

先查看完整兼容矩阵，再 dry-run：

```bash
python scripts/run_privacy_experiments.py --list
python scripts/run_privacy_experiments.py --dry-run --max-runs 1
```

不指定模型时保持原 CLIP 统一入口的默认范围：CLIP-MLP、CLIP-Adapter、
CLIP-LoRA 和五个图像数据集。常用组合如下：

```bash
# 单模型、单数据集、多个攻击
python scripts/run_privacy_experiments.py \
  --models clip_adapter --datasets caltech101 \
  --attacks blackbox_loss,projres --defenses none

# 多模型、多数据集、多防御；每种防御生成独立训练任务
python scripts/run_privacy_experiments.py \
  --models bert_adapter,bert_lora --datasets sst5,cola \
  --attacks all --defenses none,iclr

# 运行仓库全部模型以及每个模型正式支持的全部攻击/防御
python scripts/run_privacy_experiments.py \
  --models all --attacks all --defenses all --gpus 0,1 --jobs 2

# 纯训练，不执行成员推理审计
python scripts/run_privacy_experiments.py \
  --models clip_mlp --datasets flowers --attacks none

# Dirichlet 非 IID；只传 alpha 时自动切换到 dirichlet
python scripts/run_privacy_experiments.py \
  --models clip_lora --dirichlet-alpha 0.1 --dry-run
```

`--attacks all` 会按模型解析支持集：ResNet18 只选择 `fedmia_loss`，六种 PEFT
模型选择全部 11 种攻击。显式指定不兼容攻击时，该模型不会生成任务并打印原因；
`--defenses all` 同样只展开模型正式支持的防御。可用 `--seeds 1,2,3`、
`--target-clients 0,1` 和可重复的 `--set path=value` 扩展或覆盖最终配置。

配置分为两层：[`configs/experiment_catalog.yaml`](configs/experiment_catalog.yaml)
维护能力矩阵、防御覆盖和别名；[`configs/models/`](configs/models/) 下 7 个 YAML
维护模型训练与默认候选协议。每个实际任务仍会把完整解析结果保存为自己的
`run_config.yaml`。

ResNet18 基线保持完整 CIFAR100、随机等量 IID、10 客户端全参与、300 轮
FedAvg 和每轮 `0.99` 学习率衰减。CLIP-MLP/Adapter 为 16-shot one-batch 等权
FedSGD，CLIP-LoRA 使用完整训练集；BERT Adapter/LoRA 与 GPT2 Adapter 使用文本
one-batch 等权 FedSGD。

独立严格 ProjRes 诊断仍可直接调用分析工具：

```bash
python scripts/validate_projres_mlp_real.py \
  --config configs/models/clip_mlp.yaml --output /tmp/projres_strict.json
```

公式和威胁模型边界见 [`docs/projres_mlp_strict.md`](docs/projres_mlp_strict.md)。

## 审计调度

审计器按攻击所需信号调度计算，不会无条件生成全部梯度和前向结果：

- 三种 CLIP 模型的六种 exact-batch 攻击每 10 轮使用该轮真实上传
  batch 独立评估；五种固定候选攻击也每 10 轮统计一次。
- BERT 的 `loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine` 和 `fta`
  每 10 轮统计一次，六种真实 Batch 攻击每 50 轮统计一次；GPT2 的全部配置攻击
  仍每 50 轮统计一次。逐轮结果保存在 `attack_round_metrics.csv`。

只选择部分攻击时，调度器会跳过其他信号族。余弦和 Gradient-Diff 的逐样本梯度
通常是主要耗时来源；大幅缩小非成员池虽然能加速，但会直接降低低 FPR 的统计
分辨率。

## 多客户端池化审计

设置 `audit.audit_client_ids: all` 或多个客户端 ID 时，审计器先在每个客户端内
独立执行攻击，再合并校准后的分数，避免把客户端身份本身当作成员证据。当前可
池化路径支持除 ProjRes 外的 10 种通用攻击；统一 exact-batch 协议要求单个目标客户端。

候选协议包括：

- `legacy`：客户端内类别匹配的固定候选集；
- `fedmia_mix`：目标客户端训练成员与独立测试/其他客户端训练来源混合；
- `low_fpr_full`：冻结 CLIP 特征模型的低 FPR 完整候选池；
- `balanced_holdout`：目标客户端训练集与其独立测试集按类别精确 1:1；
- `balanced_global_holdout`：目标训练成员与全局独立 evaluation 非成员按固定比例抽样；要求完整成员池时，类别不足会由其他类别确定性补足，并记录实际标签 TV 距离。

当前统一 CLIP-MLP、CLIP-Adapter、CLIP-LoRA、BERT 和 GPT2 sweep 均使用
`balanced_global_holdout`。

## 隐私防御与 ICLR 分析

仓库保留更新扰动、稀疏化、Mixup、采样、数据增强、CoFedMID、Prompt-DP、
MIST、SOFT、HAMP 和 ICLR 等实现。当前 CLIP 三模型
sweep 只把 `none` 与观测型 `iclr` 作为正式可选项；不要把通用 prompt 防御配置
直接套到 MLP、Adapter 或 LoRA。

BERT 配置默认执行不修改训练的 ICLR 排名分析；GPT2 默认不执行。ICLR 输出写入
同一任务目录的 `defense_summary.json` 及对应逐轮 CSV，不会额外创建训练任务。
防御与威胁模型说明见 [`docs/defenses.md`](docs/defenses.md)。

### 记录级 DP-SGD

ResNet18/CIFAR-100 FedAvg 与 BERT-Base Adapter/SST-5 FedSGD 都提供客户端侧
记录级 DP 配置。每条图像或完整文本序列的联合梯度先做 L2 裁剪，再对每个逻辑
batch 的裁剪梯度和加入一次高斯噪声；客户端上传本身就是 DP 输出，因此适用于
服务器能观察单客户端消息的当前威胁模型。

```bash
# 只打印最终任务
python scripts/run_privacy_experiments.py \
  --models resnet18,bert_adapter --defenses record_dp --dry-run

# 正式运行
python scripts/run_privacy_experiments.py \
  --models resnet18,bert_adapter --defenses record_dp
```

默认配置以 `target_epsilon: 3.0`、`delta: 1e-5` 为目标，根据每个客户端的
Poisson 采样率和计划 DP 步数自动校准共享 noise multiplier。实际逐客户端步数、
采样率、累计 epsilon 及其最大值写入 `defense_summary.json`。正式隐私运行必须保持
`reproducible_noise: false`；确定性噪声只用于测试，且会被标记为不具备正式 DP。
`privacy_audit/summary.json` 还会为每种攻击写入
`TPR <= min(1, exp(epsilon) * FPR + delta)` 在已报告 FPR 点上的理论上界；这些攻击
文件包含成员真值，只作为封闭实验审计数据，不属于 DP 发布物。

#### BERT Record-DP × ProjRes 隐私预算 sweep

统一入口可固定 BERT-Base Adapter、SST-5、500 轮训练、目标客户端 0、
`delta=1e-5`、裁剪阈值 1.0，以及每 50 轮一次的 exact-batch ProjRes；同一 seed
下可只改变目标 epsilon 及由此自动校准的噪声强度。无 DP 基线需显式选择
`--defenses none`，避免把不同防御意外混在一个任务中。

```bash
# 核对一个预算；移除 --dry-run 后正式执行
python scripts/run_privacy_experiments.py \
  --models bert_adapter --datasets sst5 --attacks projres \
  --defenses record_dp --seeds 42,43,44 \
  --set defense.target_epsilon=3 --dry-run

# 多预算使用同一个终极入口逐次生成互相独立的任务
for epsilon in 1 3 5 8; do
  python scripts/run_privacy_experiments.py \
    --models bert_adapter --datasets sst5 --attacks projres \
    --defenses record_dp --seeds 42 --set defense.target_epsilon=$epsilon
done
```

正式实验可用 `--require-cuda` 强制检查 CUDA。
Record-DP 结果目录包含 `record_dp_eps<预算>`，实际 epsilon 和噪声强度记录在
`defense_summary.json`，ProjRes 指标记录在 `privacy_audit/summary.json` 和
`attack_round_metrics.csv`。对每个预算应同时比较下游准确率、AUC、
`TPR@0.1FPR` 和 `TPR@0.01FPR`；真实 batch 只有约 10 倍非成员，不适合把
`TPR@0.001FPR` 当作主要结论。

#### BERT 本地客户端级 DP × ProjRes

`local_client_dp` 是与 Record-DP 分开的基准：每个客户端先计算普通 batch-mean
梯度，再把 Adapter 和分类头的完整联合梯度裁剪到 `max_update_norm=S`，最后在
客户端本地给每个坐标加入标准差 `noise_multiplier * S` 的高斯噪声。服务器和
ProjRes 看到的都是这条已加噪上传。当前正式实现限定为 30 个固定客户端全部参与、
每轮一个 batch 的 FedSGD，并按 500 次完整高斯机制进行 RDP 组合。

```bash
# 核对客户端级预算与裁剪阈值；移除 --dry-run 后正式运行
python scripts/run_privacy_experiments.py \
  --models bert_adapter --datasets sst5 --attacks projres \
  --defenses local_client_dp --seeds 42 \
  --set defense.target_epsilon=3 --set defense.max_update_norm=1 --dry-run
```

这里的邻接关系是“一个固定客户端槽位的数据贡献存在或不存在”；没有数据时该槽位
仍须发送零贡献加同分布噪声。因此保证覆盖客户端数据贡献和最终模型，但不隐藏网络
参与元数据。同一个数值的客户端级 epsilon 与记录级 epsilon 不是同一个隐私单位，
不应解释成同等强度；客户端级机制在当前 500 轮全参与设置下通常需要大得多的
noise multiplier。实际校准值、每坐标噪声标准差和逐客户端累计 epsilon 写入
`defense_summary.json`。

正式选择 `max_update_norm=S` 前，可以运行无隐私校准。它使用相同的 30 客户端、
IID 划分、batch size 16、学习率和 one-batch FedSGD，逐轮读取聚合前真实上传的
batch-mean 联合梯度范数；观察过程不修改训练。输出明确标记为非隐私分析数据，并
保存到 `analysis_scripts/`，不会写入或覆盖 `results/`。

```bash
# 只验证完整 500 轮实验计划，不加载模型
python scripts/calibrate_bert_local_client_dp_clipping.py --dry-run

# 两轮流程检查；仍训练全部 30 个客户端，只汇总其中 6 个
python scripts/calibrate_bert_local_client_dp_clipping.py \
  --rounds 2 --client-ids 0,1,2,3,4,5

# 完整校准：500 轮 × 30 个客户端
python scripts/calibrate_bert_local_client_dp_clipping.py --gpu 0
```

主要产物为 `client_batch_gradient_norms.csv`、`summary_by_round.csv`、
`summary_by_phase.csv`、`recommended_s.csv`、`clipping_grid.csv`、PNG 图和
中文 `report.md`。`recommended_s.csv` 分别将总体 `P50/P75/P90` 标成
aggressive、balanced、conservative，默认建议先以 `P75` 为中心做正式固定预算
消融。若统计使用的是敏感训练集，公开这些阈值本身不能宣称零隐私成本；正式论文
应优先使用公开代理数据或预注册候选网格。

## 结果目录与文件

`results/` 下每个一级目录就是一次独立训练任务：

```text
results/
└── <时间>_<模型>_<数据集>_<方法>_<防御>_seed<种子>_target<客户端>_<hash>/
    ├── run_config.yaml
    ├── run.log
    ├── training_metrics.csv
    ├── training_health.json
    ├── federated_method_summary.json
    ├── defense_summary.json
    ├── final_mlp.pt | final_clip_adapter.pt
    │   | final_clip_lora.pt | final_transformer_adapter.pt
    │   | final_transformer_lora.pt
    └── privacy_audit/
        ├── summary.json
        ├── predictions.csv
        ├── signals.pt
        ├── candidate_selection.pt
        ├── exact_batch_candidate_selection.pt
        ├── attack_round_metrics.csv
        ├── iclr_candidate_round_scores.csv
        ├── iclr_candidate_scores.csv
        ├── iclr_candidate_relationship.json
        └── projres_strict.json       # 仅独立 ProjRes 路径
```

文件按实际启用功能生成，并非每次任务都有全部文件：

- `run_config.yaml` 是判断实验协议的唯一可靠入口，包含所有合并后的实际参数。
- `summary.json` 保存 AUC、低 FPR TPR、候选规模、FPR 分辨率和逐客户端指标。
- `predictions.csv` 保存逐候选分数；`signals.pt` 保存配置允许的跨轮信号。
- `candidate_selection.pt` 固定历史候选池；`exact_batch_candidate_selection.pt`
  保存每个真实 batch 审计轮的成员/非成员索引。
- `attack_round_metrics.csv` 保存周期攻击结果。
- ResNet18/CIFAR100 FedMIA 任务的 `iclr_candidate_round_scores.csv` 保存每个审计轮、
  每个候选的目标/其余客户端损失及 ICLR 分数；`iclr_candidate_scores.csv` 保存
  跨轮均值、方差、末轮分数并按 `sample_index` 连接最终 FedMIA-Loss 分数；关系
  与成员推理指标汇总在 `iclr_candidate_relationship.json`。
- CLIP-MLP、CLIP-Adapter、CLIP-LoRA、BERT 和 GPT2 的统一 ProjRes 与其他攻击共享上述
  输出，不生成独立 ProjRes 目录。

统一入口还会在 `results/` 根目录生成本次调用的
`experiment_manifest_<时间>.json` 与 `experiment_summary_<时间>.csv`。

仓库不会为数据分析生成 HTML。已有 `results/` 内容属于实验数据；重新运行会创建
新的时间戳目录，不会复用或覆盖旧任务。

## 分析历史结果时的注意事项

分析任何任务前先读取其 `run_config.yaml`。历史结果与当前协议可能存在以下差异：

- 较早实验可能使用 Dirichlet `alpha=0.1`，不能默认视为 IID。
- 历史 CLIP-MLP 可能按本地样本数加权并执行完整 local epoch FedAvg；当前为每类 16-shot、one-batch、参与客户端等权 FedSGD。
- 历史 CLIP-Adapter 可能遍历完整 local epoch 并使用 FedAvg；当前为 one-batch
  等权 FedSGD。
- 非 IID 结果可能受到成员/非成员标签分布不匹配影响。报告攻击有效性时，应同时
  检查类别直方图、按类别指标以及类内/类别条件 AUC。
- “客户端分布泄漏”不等同于“同类别内具体样本的 record-level membership”。

不要把不同成员定义、候选规模或聚合协议的结果直接合并为同一基线。

## 测试与验证

日常快速核心回归：

```bash
python -m pytest -q
```

完整本地测试：

```bash
python -m pytest -q tests
```

配置或 sweep 修改后至少执行：

```bash
python scripts/run_privacy_experiments.py --dry-run --max-runs 1
git diff --check
```

小范围修改优先运行直接相关的测试文件。共享审计器、聚合核心、候选协议或发布前
高风险变更再运行完整套件。

## 延伸文档

- [`docs/attack_mapping.md`](docs/attack_mapping.md)：论文、公式与运行名映射
- [`docs/federated_methods.md`](docs/federated_methods.md)：FedAvg/FedSGD、聚合和协议消息
- [`docs/projres_mlp_strict.md`](docs/projres_mlp_strict.md)：统一与独立 CLIP ProjRes
- [`docs/defenses.md`](docs/defenses.md)：防御与 ICLR 分析
