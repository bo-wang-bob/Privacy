# Federated PEFT Membership-Inference Benchmark

本仓库用于研究联邦参数高效微调（PEFT）中的成员隐私泄漏。当前维护的实验覆盖
CLIP-MLP、双侧 CLIP Adapter、CLIP-LoRA、BERT-Base Adapter 和
GPT2-Large Adapter，并在统一训练任务中完成模型训练、成员推理审计、低 FPR
指标计算、候选集留档和结果汇总。

项目只研究成员隐私，不包含后门触发器、恶意客户端投毒、攻击成功率（ASR）或
SEISMOGRAPH 防御。仓库中可能保留早期研究攻击的实现文件，但公共审计器目前只
注册本文档列出的 11 种攻击；未注册实现不能通过配置或命令行用于新实验。

## 当前实验协议

| 模型 | 可训练部分 | 联邦方法 | 每轮客户端训练 | 默认轮数 | 默认学习率 | 默认攻击 |
| --- | --- | --- | --- | ---: | ---: | --- |
| CLIP-MLP | 冻结 CLIP 图像编码器后的两层 MLP | FedAvg | 1 个完整 local epoch | 150 | 0.1 | 10 种通用攻击 |
| Visual Adapter | 图像、文本两侧瓶颈 Adapter | FedSGD | 1 个 mini-batch / 1 次 SGD step | 300 | 0.001 | 10 种通用攻击 + ProjRes |
| CLIP-LoRA | 图像、文本注意力 Q/K/V 的 LoRA 因子 | FedSGD | 1 个 mini-batch / 1 次 SGD step | 300 | 0.0002 | 6 种 FedMIA 系列攻击 + 独立 ProjRes |
| BERT-Base Adapter | 各 Transformer block 的 Adapter 和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.005 | 10 种通用攻击 + ProjRes |
| GPT2-Large Adapter | 各 Transformer block 的 Adapter 和分类头 | FedSGD | 1 个 batch，batch size 16 | 500 | 0.001 | 10 种通用攻击 + ProjRes |

当前正式 sweep 的共同约定：

- CLIP 三种模型使用 10 个客户端、batch size 32、IID 划分和参与客户端等权聚合。
- BERT/GPT2 使用 30 个客户端、IID 划分和 one-batch 等权 FedSGD。
- CLIP 普通任务每 5 个已完成通信轮评估一次；多轮攻击通常每 10 轮审计。
- BERT/GPT2 普通任务与攻击每 50 轮评估，并始终包含最后一轮。
- 所有聚合和保存操作只处理 `requires_grad=True` 的参数，冻结主干不上传。

`main.py` 仍保留通用 CLIP prompt 和 PromptFL 兼容入口，但当前重点维护和批量
复现的是上表中的五类模型。

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

### 各模型的实际支持范围

| 模型 | 六种 FedMIA 系列攻击 | `gradient_diff` / `score_diff` / `score_ratio` / `fta` | ProjRes |
| --- | --- | --- | --- |
| CLIP-MLP | 默认启用 | 默认启用 | 不进入统一 sweep；保留独立严格入口 |
| Visual Adapter | 默认启用 | 默认启用 | 统一 exact-batch 审计器，默认启用 |
| CLIP-LoRA | 默认启用 | 共享审计器可配置，但当前 sweep 不默认启用 | 在同一训练任务内独立运行 |
| BERT-Base Adapter | 默认启用 | 默认启用 | 统一 exact-batch 审计器，默认启用 |
| GPT2-Large Adapter | 默认启用 | 默认启用 | 统一 exact-batch 审计器，默认启用 |

这里的“六种 FedMIA 系列攻击”指四个 FedMIA 对照基线加
`fedmia_loss`、`fedmia_cosine`。CLIP-LoRA 的四种更新攻击虽然可以通过共享配置
路径启用，但当前缺少与 MLP/Adapter 同等级的默认 sweep 和专项回归覆盖，不应与
正式默认集合混为一谈。

## 候选集与成员定义

不同协议回答的成员问题不同，不能只按攻击名称合并比较。

| 模型/视图 | 成员 | 非成员 | 默认规模 |
| --- | --- | --- | --- |
| CLIP-MLP 通用攻击 | 目标客户端历史训练集样本 | 该客户端从未训练的独立测试样本 | 每客户端按类别精确 1:1，最多 5000/5000 |
| Visual Adapter 六种 exact-batch 攻击 | 当轮目标客户端真实 FedSGD batch | 全局独立 evaluation 池中按标签匹配样本 | 完整 batch 时 32/320 |
| Visual Adapter 其余五种攻击 | 固定历史训练成员 | 固定、从未训练的全局非成员 | 100/1000 |
| CLIP-LoRA 通用攻击 | 目标客户端历史训练集样本 | 该客户端独立测试样本 | 按类别精确 1:1，最多 5000/5000 |
| BERT/GPT2 六种 exact-batch 攻击 | 当轮真实 FedSGD batch | 按该 batch 标签直方图抽取的独立 evaluation 样本 | 16/160 |
| BERT/GPT2 其余五种攻击 | 固定历史训练成员 | 固定、从未训练的全局非成员 | 100/1000 |

六种 exact-batch 攻击为 `blackbox_loss`、`grad_cosine`、`gradient_diff`、
`score_diff`、`score_ratio` 和 `projres`。它们在每个审计轮独立使用真实上传
batch，不跨轮拼接不同成员集合。

固定 1000 个非成员的经验 FPR 分辨率为 0.001；32/320 或 16/160 的真实 batch
视图只正式报告可由样本量解析的低 FPR 指标。`summary.json` 会保存成员/非成员
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

文本任务支持 SST-5、CoLA 和 IMDB。准备 Hugging Face 模型和 SST-5 的辅助脚本：

```bash
python scripts/download_hf_sst5_models.py
```

CoLA 和 IMDB 分别从 `data/huggingface/cola`、`data/huggingface/imdb` 等本地路径
读取；evaluation split 从不参与任何客户端训练。

## 快速开始

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

- CLIP-MLP/LoRA 的 `blackbox_loss`、`grad_cosine` 默认使用预先固定的最后审计轮。
- `loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine` 使用多轮轨迹。
- CLIP-MLP 的四种更新攻击每 10 个已完成通信轮统计一次。
- Visual Adapter 的六种 exact-batch 攻击每 10 轮使用该轮真实上传 batch 独立评估。
- BERT/GPT2 的全部配置攻击每 50 轮统计一次；最终汇总使用最后一个审计轮，逐轮
  结果保存在 `attack_round_metrics.csv`。

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
- `balanced_global_holdout`：目标训练成员与全局独立 evaluation 非成员按固定比例匹配。

当前统一 CLIP-MLP 和 CLIP-LoRA sweep 使用 `balanced_holdout`；Visual Adapter、
BERT 和 GPT2 的 exact-batch 任务使用 `balanced_global_holdout`。

## 隐私防御与 ICLR 分析

仓库保留更新扰动、稀疏化、Mixup、采样、数据增强、CoFedMID、Prompt-DP、
MIST、SOFT、HAMP、Local-GGEUR/MIRAGE/VEIL 和 ICLR 等实现。当前 CLIP 三模型
sweep 只把 `none` 与观测型 `iclr` 作为正式可选项；不要把通用 prompt 防御配置
直接套到 MLP、Adapter 或 LoRA。

BERT 配置默认执行不修改训练的 ICLR 排名分析；GPT2 默认不执行。ICLR 输出写入
同一任务目录的 `defense_summary.json` 及对应逐轮 CSV，不会额外创建训练任务。
防御与威胁模型说明见 [`docs/defenses.md`](docs/defenses.md)。

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
        └── projres_strict.json       # 仅独立 ProjRes 路径
```

文件按实际启用功能生成，并非每次任务都有全部文件：

- `run_config.yaml` 是判断实验协议的唯一可靠入口，包含所有合并后的实际参数。
- `summary.json` 保存 AUC、低 FPR TPR、候选规模、FPR 分辨率和逐客户端指标。
- `predictions.csv` 保存逐候选分数；`signals.pt` 保存配置允许的跨轮信号。
- `candidate_selection.pt` 固定历史候选池；`exact_batch_candidate_selection.pt`
  保存每个真实 batch 审计轮的成员/非成员索引。
- `attack_round_metrics.csv` 保存周期攻击结果。
- Visual Adapter、BERT 和 GPT2 的统一 ProjRes 与其他攻击共享上述输出，不生成
  独立 ProjRes 目录。

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
