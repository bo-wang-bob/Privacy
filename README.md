# Federated PEFT Membership-Inference Benchmark

本仓库用于研究联邦学习与参数高效微调（PEFT）中的成员隐私泄漏。当前维护的实验
覆盖 ResNet18、CLIP-MLP、双侧 CLIP Adapter、CLIP-LoRA、BERT-Base Adapter 和
GPT2-Large Adapter，并在统一训练任务中完成模型训练、成员推理审计、低 FPR
指标计算、候选集留档和结果汇总。

项目只研究成员隐私，不包含后门触发器、恶意客户端投毒、攻击成功率（ASR）或
SEISMOGRAPH 防御。仓库中可能保留早期研究攻击的实现文件，但公共审计器目前只
注册本文档列出的 11 种攻击；未注册实现不能通过配置或命令行用于新实验。

## 当前实验协议

| 模型 | 可训练部分 | 联邦方法 | 每轮客户端训练 | 默认轮数 | 默认学习率 | 默认攻击 |
| --- | --- | --- | --- | ---: | ---: | --- |
| ResNet18 / CIFAR100 | 完整 CIFAR ResNet18（GroupNorm） | FedAvg | 1 个完整 local epoch | 300 | 0.1，逐轮乘 0.99 | FedMIA-Loss |
| CLIP-MLP | 冻结 CLIP 图像编码器后的两层 MLP | FedAvg | 1 个完整 local epoch | 150 | 0.1 | 10 种通用攻击 |
| Visual Adapter | 图像、文本两侧瓶颈 Adapter | FedSGD | 1 个 mini-batch / 1 次 SGD step | 300 | 0.001 | 10 种通用攻击 + ProjRes |
| CLIP-LoRA | 图像、文本注意力 Q/K/V 的 LoRA 因子 | FedSGD | 1 个 mini-batch / 1 次 SGD step | 300 | 0.0002 | 10 种通用攻击 + ProjRes |
| BERT-Base Adapter | 各 Transformer block 的 Adapter 和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.005 | 10 种通用攻击 + ProjRes |
| GPT2-Large Adapter | 各 Transformer block 的 Adapter 和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.001 | 10 种通用攻击 + ProjRes |

当前正式 sweep 的共同约定：

- CLIP 三种模型使用 10 个客户端、batch size 32、IID 划分和参与客户端等权聚合。
- BERT/GPT2 使用 30 个客户端、IID 划分和 one-batch 等权 FedSGD。
- FedSGD 客户端上传各自的可训练参数梯度；服务器先等权聚合梯度，再执行一次
  `global = base - learning_rate * mean_gradient` 并下发新的全局模型。
- CLIP 普通任务每 5 个已完成通信轮评估一次；多轮攻击通常每 10 轮审计。
- BERT 普通任务、六种真实 Batch 攻击与 ICLR 每 50 轮评估，五种固定候选攻击每
  10 轮评估；GPT2 普通任务与全部攻击每 50 轮评估。调度始终包含最后一轮。
- 所有聚合和保存操作只处理 `requires_grad=True` 的参数，冻结主干不上传。

`main.py` 仍保留通用 CLIP prompt 和 PromptFL 兼容入口，但当前重点维护和批量
复现的是上表中的六类模型。

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
| CLIP-MLP | 除 `projres` 外的 10 种 | 独立严格入口 | 统一 sweep 不运行 ProjRes |
| Visual Adapter | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种，均在同一训练任务内运行 |
| CLIP-LoRA | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种，均在同一训练任务内运行 |
| BERT-Base Adapter | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种 |
| GPT2-Large Adapter | 除 `projres` 外的 10 种 | 统一审计器，默认启用 | 共 11 种 |

“代码已注册/可配置”和“正式 sweep 默认启用”是两个不同概念。当前 11 个注册名
均列在上一表；批量复现实验应以上表的默认集合为准。Visual Adapter、CLIP-LoRA、
BERT 和 GPT2 现使用相同的攻击分组与候选定义；CLIP-MLP 的独立 ProjRes 用于严格
复现，不应误写成统一 MLP sweep 的第 11 种攻击。

## 各攻击的成员与非成员定义

仓库存在两种成员问题：

- **真实 Batch（exact-batch）**：判断样本是否出现在目标客户端产生当轮上传的
  真实 mini-batch 中。每个审计轮重新构造候选集，不跨轮拼接成员。
- **固定候选（fixed-candidate）**：判断样本是否属于目标客户端历史训练集。训练
  开始前固定成员/非成员候选，所有审计轮复用同一集合。

同一个攻击名在不同模型上可能使用不同成员问题。逐攻击的当前划分如下：

| 攻击 | CLIP-MLP | Visual Adapter / CLIP-LoRA / BERT / GPT2 | 成员 | 非成员 |
| --- | --- | --- | --- | --- |
| `blackbox_loss` | 固定候选 | 真实 Batch | 固定时为目标客户端训练样本；真实 Batch 时为当轮上传 batch | 固定时为从未训练的独立测试/evaluation 样本；真实 Batch 时为按当轮成员标签直方图抽取的独立 evaluation 样本 |
| `loss_series` | 固定候选 | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立测试/evaluation 样本 |
| `grad_cosine` | 固定候选 | 真实 Batch | 同 `blackbox_loss` | 同 `blackbox_loss` |
| `avg_cosine` | 固定候选 | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立测试/evaluation 样本 |
| `fedmia_loss` | 固定候选 | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立测试/evaluation 样本；其他客户端只用于构造 FedMIA null 分布，不会自动成为目标客户端的成员 |
| `fedmia_cosine` | 固定候选 | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立测试/evaluation 样本；其他客户端只用于构造 FedMIA null 分布，不会自动成为目标客户端的成员 |
| `gradient_diff` | 固定候选 | 真实 Batch | 固定时为目标客户端训练样本；真实 Batch 时为当轮上传 batch | 固定时为独立测试样本；真实 Batch 时为标签匹配的独立 evaluation 样本 |
| `score_diff` | 固定候选 | 真实 Batch | 同 `gradient_diff` | 同 `gradient_diff` |
| `score_ratio` | 固定候选 | 真实 Batch | 同 `gradient_diff` | 同 `gradient_diff` |
| `fta` | 固定候选 | 固定候选 | 目标客户端历史训练样本 | 从未训练的独立测试/evaluation 样本 |
| `projres` | 仅独立严格路径；成员是产生所观察上传的真实 batch | 统一真实 Batch | 产生被观察更新的当轮真实 batch | 与该 batch 互斥、从未用于该目标上传的非成员；统一路径从独立 evaluation 池按标签抽取 |

### 各模型的候选规模与抽样方式

| 模型/视图 | 成员与非成员来源 | 默认规模 |
| --- | --- | --- |
| ResNet18 / CIFAR100 FedMIA-Loss | 目标客户端 0 的完整训练集；非成员由完整独立测试集和其余 9 个客户端训练集等来源构成 | 5000/10000；10 个非成员来源各 1000 |
| CLIP-MLP 10 种通用攻击 | 每个目标客户端的训练样本，对该客户端从未训练的独立测试样本；按类别精确配对 | 1:1，最多 5000/5000 |
| Visual Adapter / CLIP-LoRA 的 5 种固定候选攻击 | 目标客户端完整训练集，对全局独立 evaluation 样本；按类别尽力匹配，不足类别由其他类别确定性补足 | `M/M`，`M` 为目标客户端训练集大小 |
| Visual Adapter / CLIP-LoRA 的 6 种真实 Batch 攻击 | 当轮真实 batch，对严格按其标签直方图抽取的独立 evaluation 非成员 | `N/10N`；完整 batch 为 32/320 |
| BERT/GPT2 的 5 种固定候选攻击 | 目标客户端完整训练集，对全局独立 evaluation 样本；按类别尽力匹配，不足类别由其他类别确定性补足 | `M/M`，`M` 为目标客户端训练集大小 |
| BERT/GPT2 的 6 种真实 Batch 攻击 | 当轮真实 batch，对严格按其标签直方图抽取的独立 evaluation 非成员 | `N/10N`；完整 batch 为 16/160 |
| CLIP-MLP 独立 ProjRes | 所观察上传对应的真实 batch，对互斥非成员池 | 成员不超过配置的 batch/候选上限；非成员默认 1,000--20,000 |

这里的 6 种真实 Batch 攻击是 `blackbox_loss`、`grad_cosine`、
`gradient_diff`、`score_diff`、`score_ratio` 和 `projres`；5 种固定候选攻击是
`loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine` 和 `fta`。
CLIP-MLP 独立 ProjRes 的互斥非成员池可包含所有客户端测试样本和其他客户端训练样本；
“非成员”在这里严格指未参与所观察的目标客户端上传，并不等价于从未参与过全局
训练。统一 exact-batch 路径则只从从未训练的独立 evaluation 池抽取非成员。

ProjRes 只输出连续的负 L1 投影残差并按分数排序，通过 ROC/AUC 或指定 FPR 的
TPR 评价；不再使用跨模型、跨轮次不可校准的固定残差阈值生成成员判断。

Visual Adapter、CLIP-LoRA、BERT 和 GPT2 固定候选攻击的经验 FPR 分辨率由目标
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
├── configs/                # 单任务配置与 sweep 配置
├── scripts/                # CLIP/FedLLM 启动脚本
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

### ResNet18 / CIFAR100 FedMIA-Loss 论文配置

稳定参数集中在
[`configs/resnet18_cifar100_fedmia_loss.yaml`](configs/resnet18_cifar100_fedmia_loss.yaml)：
完整 CIFAR100、随机等量 IID、10 个客户端全参与、每客户端 5000 个训练样本、
batch size 100、1 个本地 epoch、300 个通信轮、SGD momentum 0.9、weight decay
0.0005。模型使用 CIFAR 3x3 stem 和 32 组 GroupNorm，不使用数据增强。

FedMIA-Loss 每 10 个已完成通信轮审计一次。成员为目标客户端 0 的全部 5000 个
训练样本；10000 个非成员由完整独立测试集抽 1000 个，再从其余 9 个客户端的
训练集各抽 1000 个。每轮以其余客户端的负交叉熵构造三西格玛过滤后的高斯 null，
计算目标客户端单尾 CDF，最后对 30 个审计轮的 CDF 分数取均值。

同一审计轮还对全部 15000 个候选逐样本计算只读 ICLR 分数
`L(x; theta_-k) - L(x; theta_k)`：`theta_k` 是目标客户端该轮上传的 post-local
模型，`theta_-k` 由服务器发布的聚合模型按目标客户端实际 FedAvg 权重反解得到。
分数越大，表示该样本越特异于目标客户端。该逻辑不筛除样本，不启用 ICLR 防御，
也不改变 FedMIA 论文训练配置中的 SGD、momentum、weight decay 或学习率日程。

```bash
# 只检查命令
bash scripts/run_resnet18_cifar100_fedmia_loss.sh --dry-run

# 正式运行（默认 GPU 0、seed 1）
bash scripts/run_resnet18_cifar100_fedmia_loss.sh

# 指定设备和数据目录
bash scripts/run_resnet18_cifar100_fedmia_loss.sh \
  --gpu 1 --data_root /path/to/data
```

论文附录 Table 3 写明学习率每轮乘 `0.99`，因此本配置以该表为准。作者公开
`run.sh` 使用 cosine schedule；若要对齐公开脚本而非论文表格，应另建配置，不能
把两种学习率日程的结果直接合并。

### 1. 先检查最终参数

修改配置或正式启动前，优先使用 dry-run。CLIP 入口会打印每个任务合并后的最终
参数，包括联邦方法、聚合权重、学习率、候选协议和攻击列表：

```bash
bash scripts/run_all_clip_fedmia_attacks.sh --dry-run --max-runs 1
```

当前输出应显示：CLIP-MLP 为 FedAvg；Visual Adapter 和 CLIP-LoRA 为 FedSGD；
三者均为 `aggregation_weighting: uniform`。

### 2. CLIP 三模型统一 sweep

无参数调用默认依次运行三种模型和五个图像数据集：

```bash
bash scripts/run_all_clip_fedmia_attacks.sh
```

常用筛选方式：

```bash
# 只运行 CLIP-MLP
bash scripts/run_all_clip_fedmia_attacks.sh --models clip_mlp

# 只运行 Visual Adapter 的两个数据集
bash scripts/run_all_clip_fedmia_attacks.sh \
  --models visual_adapter \
  --datasets caltech101,flowers

# 只运行两种 FedMIA 攻击
bash scripts/run_all_clip_fedmia_attacks.sh \
  --datasets cifar100 \
  --attacks fedmia_loss,fedmia_cosine

# 两张 GPU 并行调度不同任务
bash scripts/run_all_clip_fedmia_attacks.sh --gpus 0,1 --jobs 2

# Dirichlet 非 IID；仅传 alpha 时会自动切换为 dirichlet
bash scripts/run_all_clip_fedmia_attacks.sh --dirichlet-alpha 0.1
```

也可显式使用 `--partition-mode iid|dirichlet`。若同时传
`--partition-mode iid --dirichlet-alpha 0.1`，显式 partition mode 优先，alpha
只作为未使用配置保留。

仓库保留模型专用 wrapper，但当前协议以统一入口打印的最终参数为准。直接调用
wrapper 时必须自行核对其 spec 默认值和命令行覆盖，尤其不要跳过 CLIP-MLP
学习率 `0.1` 的最终参数检查。

### 3. 单个 CLIP 任务与短程调试

正式协议只运行一个任务时，仍建议通过统一入口展开：

```bash
bash scripts/run_all_clip_fedmia_attacks.sh \
  --models clip_mlp \
  --datasets caltech101 \
  --max-runs 1
```

以下单次 YAML 主要用于短程开发和功能调试，其轮数、学习率与数据划分不等同于
上文正式 sweep；运行前应直接检查文件或生成的 `run_config.yaml`：

```bash
python main.py --config configs/clip_mlp_privacy.yaml
python main.py --config configs/visual_adapter_privacy.yaml
```

纯训练配置可关闭审计，例如：

```bash
python main.py --config configs/clip_mlp_fedavg.yaml
```

### 4. BERT/GPT2 Adapter sweep

先检查 2 个模型 × 3 个数据集的任务展开：

```bash
bash scripts/run_all_fedllm_attacks.sh --dry-run
```

正式运行：

```bash
bash scripts/run_all_fedllm_attacks.sh
```

常用筛选方式：

```bash
# 只运行 BERT 的 CoLA 和 IMDB
bash scripts/run_all_fedllm_attacks.sh \
  --models bert \
  --datasets cola,imdb

# 只运行一种攻击，并关闭 ProjRes
bash scripts/run_all_fedllm_attacks.sh \
  --attacks blackbox_loss \
  --skip-projres

# 两张 GPU 并行执行两个任务
bash scripts/run_all_fedllm_attacks.sh --gpus 0,1 --jobs 2
```

单任务入口：

```bash
python scripts/run_fedllm_adapter.py \
  --config configs/bert_base_sst5_adapter.yaml \
  --dataset sst5

python scripts/run_fedllm_adapter.py \
  --config configs/gpt2_large_sst5_adapter.yaml \
  --dataset sst5
```

### 5. 独立 CLIP-MLP ProjRes

CLIP-MLP 的 ProjRes 不属于统一 MLP sweep。独立严格入口为：

```bash
bash scripts/run_projres_mlp_real.sh
```

默认配置见 [`configs/clip_mlp_projres.yaml`](configs/clip_mlp_projres.yaml)，威胁
模型和公式边界见 [`docs/projres_mlp_strict.md`](docs/projres_mlp_strict.md)。

## 审计调度

审计器按攻击所需信号调度计算，不会无条件生成全部梯度和前向结果：

- CLIP-MLP 的 `blackbox_loss`、`grad_cosine` 默认使用预先固定的最后审计轮。
- `loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine` 使用多轮轨迹。
- CLIP-MLP 的四种更新攻击每 10 个已完成通信轮统计一次。
- Visual Adapter 和 CLIP-LoRA 的六种 exact-batch 攻击每 10 轮使用该轮真实上传
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
池化的攻击为 10 种通用攻击；统一 exact-batch 协议要求单个目标客户端。

候选协议包括：

- `legacy`：客户端内类别匹配的固定候选集；
- `fedmia_mix`：目标客户端训练成员与独立测试/其他客户端训练来源混合；
- `low_fpr_full`：冻结 CLIP 特征模型的低 FPR 完整候选池；
- `balanced_holdout`：目标客户端训练集与其独立测试集按类别精确 1:1；
- `balanced_global_holdout`：目标训练成员与全局独立 evaluation 非成员按固定比例抽样；要求完整成员池时，类别不足会由其他类别确定性补足，并记录实际标签 TV 距离。

当前只有统一 CLIP-MLP sweep 使用 `balanced_holdout`；Visual Adapter、CLIP-LoRA、
BERT 和 GPT2 使用 `balanced_global_holdout`。

## 隐私防御与 ICLR 分析

仓库保留更新扰动、稀疏化、Mixup、采样、数据增强、CoFedMID、Prompt-DP、
MIST、SOFT、HAMP、Local-GGEUR/MIRAGE/VEIL 和 ICLR 等实现。当前 CLIP 三模型
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
# 只打印最终命令
bash scripts/run_record_dp_benchmarks.sh --model resnet --dry-run
bash scripts/run_record_dp_benchmarks.sh --model bert --dry-run

# 正式运行
bash scripts/run_record_dp_benchmarks.sh --model resnet
bash scripts/run_record_dp_benchmarks.sh --model bert
```

默认配置以 `target_epsilon: 3.0`、`delta: 1e-5` 为目标，根据每个客户端的
Poisson 采样率和计划 DP 步数自动校准共享 noise multiplier。实际逐客户端步数、
采样率、累计 epsilon 及其最大值写入 `defense_summary.json`。正式隐私运行必须保持
`reproducible_noise: false`；确定性噪声只用于测试，且会被标记为不具备正式 DP。
`privacy_audit/summary.json` 还会为每种攻击写入
`TPR <= min(1, exp(epsilon) * FPR + delta)` 在已报告 FPR 点上的理论上界；这些攻击
文件包含成员真值，只作为封闭实验审计数据，不属于 DP 发布物。

#### BERT Record-DP × ProjRes 隐私预算 sweep

专用入口固定 BERT-Base Adapter、SST-5、500 轮训练、目标客户端 0、
`delta=1e-5`、裁剪阈值 1.0，以及每 50 轮一次的 exact-batch ProjRes；同一 seed
下只有目标 epsilon 及由此自动校准的噪声强度发生变化。默认还会为每个 seed
运行一个无 DP 的 ProjRes 参考基线。

```bash
# 先核对最终展开的命令，不启动训练
bash scripts/run_bert_record_dp_projres_sweep.sh \
  --epsilons 1,3,5,8 --seeds 42 --dry-run

# 单 seed 正式 sweep：1 个无 DP 基线 + 4 个隐私预算
bash scripts/run_bert_record_dp_projres_sweep.sh \
  --epsilons 1,3,5,8 --seeds 42 --jobs 1

# 三个独立 seed、两张 GPU；每张 GPU 同时只安排一个任务
bash scripts/run_bert_record_dp_projres_sweep.sh \
  --epsilons 1,3,5,8 --seeds 42,43,44 --gpus 0,1 --jobs 2

# 快速检查 100 轮配置的命令展开，不重复无 DP 基线
bash scripts/run_bert_record_dp_projres_sweep.sh \
  --epsilons 1,3 --seeds 42 --no-nondp --rounds 100 --dry-run
```

正式实验默认要求 CUDA，额外参数会原样转发给 `run_fedllm_adapter.py`。
Record-DP 结果目录包含 `record_dp_eps<预算>`，实际 epsilon 和噪声强度记录在
`defense_summary.json`，ProjRes 指标记录在 `privacy_audit/summary.json` 和
`attack_round_metrics.csv`。对每个预算应同时比较下游准确率、AUC、
`TPR@0.1FPR` 和 `TPR@0.01FPR`；真实 batch 只有约 10 倍非成员，不适合把
`TPR@0.001FPR` 当作主要结论。

## 结果目录与文件

`results/` 下每个一级目录就是一次独立训练任务：

```text
results/
└── <时间>_<模型>_<数据集>_<方法>_seed<种子>_target<客户端>_<hash>/
    ├── run_config.yaml
    ├── run.log
    ├── training_metrics.csv
    ├── training_health.json
    ├── federated_method_summary.json
    ├── defense_summary.json
    ├── final_mlp.pt | final_visual_adapter.pt
    │   | final_clip_lora.pt | final_transformer_adapter.pt
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
- Visual Adapter、CLIP-LoRA、BERT 和 GPT2 的统一 ProjRes 与其他攻击共享上述
  输出，不生成独立 ProjRes 目录。

CLIP sweep 还在 `results/` 根目录生成实验组汇总：

- `<实验组>_summary_by_run.csv`
- `<实验组>_summary_by_client.csv`
- `<实验组>_summary_aggregate.csv`
- `<实验组>_summary_projres.csv`（仅独立 ProjRes）

仓库不会为数据分析生成 HTML。已有 `results/` 内容属于实验数据；重新运行会创建
新的时间戳目录，不会复用或覆盖旧任务。

## 分析历史结果时的注意事项

分析任何任务前先读取其 `run_config.yaml`。历史结果与当前协议可能存在以下差异：

- 较早实验可能使用 Dirichlet `alpha=0.1`，不能默认视为 IID。
- 历史 CLIP-MLP 可能按本地样本数加权聚合；当前为参与客户端等权 FedAvg。
- 历史 Visual Adapter 可能遍历完整 local epoch 并使用 FedAvg；当前为 one-batch
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
bash scripts/run_all_clip_fedmia_attacks.sh --dry-run --max-runs 1
git diff --check
```

小范围修改优先运行直接相关的测试文件。共享审计器、聚合核心、候选协议或发布前
高风险变更再运行完整套件。

## 延伸文档

- [`docs/attack_mapping.md`](docs/attack_mapping.md)：论文、公式与运行名映射
- [`docs/federated_methods.md`](docs/federated_methods.md)：FedAvg/FedSGD、聚合和协议消息
- [`docs/projres_mlp_strict.md`](docs/projres_mlp_strict.md)：独立 CLIP-MLP ProjRes
- [`docs/defenses.md`](docs/defenses.md)：防御与 ICLR 分析
