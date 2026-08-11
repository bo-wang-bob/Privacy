# Federated CLIP Soft-Prompt Membership Privacy Benchmark

本仓库用于研究联邦 CLIP soft-prompt tuning 中的成员隐私风险。系统冻结
CLIP 骨干，只训练可学习提示参数，并在统一训练流程中提供联邦方法、数据划分、
成员推理攻击、隐私防御和实验结果汇总。

仓库聚焦成员隐私审计，不包含后门触发器、恶意客户端投毒、ASR 指标或
SEISMOGRAPH 防御。

## 核心能力

| 模块 | 支持内容 |
| --- | --- |
| 联邦训练 | FedAvg、PromptFL |
| 数据划分 | IID、Dirichlet non-IID、pathological label split、全量和 few-shot 训练 |
| 隐私审计 | 单客户端攻击和多客户端池化攻击；统一输出 AUC 与低 FPR 下的 TPR |
| 隐私防御 | 更新扰动、稀疏化、Mixup、采样、数据增强、CoFedMID、Prompt-DP、MIST、SOFT、HAMP、VEIL |
| 实验管理 | YAML 配置、命令行覆盖、批量 sweep、断点识别和 CSV 汇总 |

所有训练和聚合操作只处理 `requires_grad=True` 的参数。客户端状态和跨轮次
审计信号使用 detached tensor 副本，避免共享可变状态。

## 仓库结构

```text
.
├── main.py                 # 训练与审计入口
├── aggregator/             # 联邦聚合与个性化方法
├── trainmodel/             # CLIP soft-prompt 模型
├── users/                  # 客户端本地训练逻辑
├── servers/                # 联邦训练循环
├── privacy_attacks/        # 成员推理攻击与统一审计器
├── privacy_defenses/       # 独立隐私防御
├── utils/                  # 数据划分和隐私会计
├── configs/                # 单次实验与 sweep 配置
├── scripts/                # 实验启动脚本
├── analysis_scripts/       # 结果汇总、重放和绘图工具
├── docs/                   # 方法与实现说明
└── tests/                  # 不依赖数据集或 CLIP 权重的轻量测试
```

## 环境准备

推荐使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 实验需要与当前 CUDA 环境匹配的 PyTorch。

### CLIP 权重

程序仅以 `local_files_only=True` 加载 `openai/clip-vit-base-patch32`，不会在
运行时下载权重。请提前准备本地 Transformers 缓存，并在配置中设置：

```yaml
cache_dir: ./checkpoints/clip-vit-base-patch32
```

### 数据集

数据加载器使用本地数据，`download=False`。请将数据放在 `data_root` 下，
并在 YAML 中设置数据集名称和路径：

```yaml
dataset_name: cifar100
data_root: ./data
```

当前加载器支持 MNIST、FashionMNIST、CIFAR-10、CIFAR-100、SVHN、
Tiny ImageNet、Caltech101、Oxford-IIIT Pets、Flowers102、Food101、
Caltech256、FGVC-Aircraft 和 DTD。不同数据集的目录约定见
[`utils/data_loader.py`](utils/data_loader.py)。

## 快速开始

### CLIP 图像编码器 + 两层 MLP 的 FedAvg 基线

该基线冻结完整 CLIP，只训练并聚合 `Linear -> ReLU -> Linear` 分类头，并使用
官方完整训练集而不做 few-shot 截断。默认还会在 Server 创建前一次性编码全部
训练/测试图片，之后的本地训练、周期测试和成员审计只读取 CPU 中的 CLIP 向量：

```bash
python main.py --config configs/clip_mlp_fedavg.yaml
```

当前性能配置使用训练 batch 32、缓存向量评估/审计 batch 512；普通任务会在
round 0 建立训练前基线，之后每 5 个已完成通信轮评估一次（最后一轮始终补做）。
原始图片的一次性 CLIP 编码仍单独使用
`precompute_batch_size: 64`，因此增大向量 batch 不会同步放大视觉主干的显存峰值。

训练器仍支持按通信轮进行阶梯学习率调度：

```text
decay_step = round_index // learning_rate_decay_interval
lr(round_index) = initial_learning_rate * learning_rate_decay ** decay_step
```

当前 MLP 与 Visual Adapter 的通用实验设置 `learning_rate_decay: 1.0`；MLP 的
150 轮和 Adapter 的 300 轮均保持初始客户端学习率不变。只有显式设置小于 1
的衰减系数时，
`learning_rate_decay_interval` 才会影响训练。
将 decay 设为 `1.0` 可关闭衰减，也可通过 `--learning-rate-decay` 和
`--learning-rate-decay-interval` 覆盖。

也可以在命令行切换模型；`--model_type clip_mlp` 会自动选择普通集中式
FedAvg、全量数据，并在未显式指定攻击时关闭隐私审计：

```bash
python main.py --model_type clip_mlp --mlp_hidden_dim 512 --attack none
```

该模型也支持仓库内全部隐私攻击。运行完整攻击集合：

```bash
python main.py --config configs/clip_mlp_privacy.yaml
```

运行单个攻击时可直接覆盖：

```bash
python main.py --model_type clip_mlp --attack fedmia_cosine
```

### 全数据双侧 CLIP Adapter

该模式冻结 CLIP 图像和文本编码器，在图像特征与类别文本特征两侧分别训练
`Linear -> ReLU -> Linear -> ReLU` 瓶颈。视觉侧按 `alpha` 做残差混合，文本侧
按 `text_alpha` 做残差混合，之后分别归一化并计算相似度。每个通信轮中，每个参与
客户端只使用一个 mini-batch 做一次 SGD 更新，服务器对两侧 Adapter 的更新直接
等权平均，不按 batch 或本地数据量加权。它与 CLIP-MLP 一样使用完整训练集，
不再执行按类别 few-shot 截断：

```bash
python main.py --config configs/visual_adapter_privacy.yaml
```

Visual Adapter 使用同一套预计算与缓存策略：CLIP 图像特征在启动时只计算一次，
训练、周期评估及隐私审计均直接复用特征张量；低 FPR 审计还会直接引用 CPU 缓存，
不再先复制到 GPU 后又复制回 CPU。

CIFAR-100 与 Caltech101 默认使用 `a photo of a {class}.`；OxfordPets 使用
`a photo of a {class}, a type of pet.`。可通过 `visual_adapter.template` 覆盖。
`text_reduction`、`text_alpha` 和 `text_output_relu` 控制文本侧 Adapter；省略时
分别继承视觉侧设置。历史配置若没有 `text_adapter_enabled`，仍按仅视觉侧运行。

### 全数据 CLIP-LoRA

`clip_lora` 冻结原始 CLIP，只在图像与文本 Transformer 每层自注意力的
Q/K/V 投影上训练 LoRA。基础配置使用 rank 2、`alpha=1`、dropout 0.25 和
`alpha/sqrt(rank)` 缩放；图像侧 LoRA 会改变编码结果，因此训练和审计直接使用
原始图像，不复用冻结特征缓存。数据协议使用完整训练集：

```bash
python main.py --config configs/clip_lora_privacy.yaml
```

基础联邦聚合直接分别平均各客户端的低秩因子：

```text
A^(t+1) = sum_k w_k A_k^t
B^(t+1) = sum_k w_k B_k^t
```

当前统一协议令参与客户端 `w_k=1/|S_t|`。冻结的 CLIP 主干不上传、不聚合；
这种逐因子线性聚合不做 SVD 重分解、稀疏化或异构 rank 对齐。普通 LoRA 配置
使用 FedAvg；带严格 ProjRes 的攻击 sweep 使用论文威胁模型要求的 one-batch
FedSGD。

实现上只有一个全局共享的冻结 CLIP 主干，每个客户端模型只持有自己的
`lora_A/lora_B`。轮到某客户端训练或推理时，其因子会临时绑定到共享主干的
LoRA 工作槽位，结束后恢复全局槽位；客户端上传也只导出自己的 A/B，不复制或
上传完整 CLIP。

### BERT-Base / GPT2-Large 的联邦 Adapter 微调

文本入口复现 Deng 等人的 Adapter-based PEFT 协议：SST-2 训练集以固定种子
IID 随机划分给 30 个客户端，每轮所有客户端各使用一个大小为 16 的本地 batch，
服务器执行同步等权 FedSGD。BERT 和 GPT2 的每个 Transformer block 后均插入
`down -> ReLU -> up + residual`，down-projection ratio 为 2；预训练主干全部冻结，
只训练各层 Adapter 与新增的二分类任务头。一个全局主干由所有客户端共享，每个
客户端只在 CPU 保存自己的 PEFT 参数，轮到该客户端时才绑定到主干所在设备。

先下载并验证模型与 SST-2：

```bash
python scripts/download_hf_sst2_models.py
```

分别运行两种模型：

```bash
python scripts/run_fedllm_adapter.py \
  --config configs/bert_base_sst2_adapter.yaml

python scripts/run_fedllm_adapter.py \
  --config configs/gpt2_large_sst2_adapter.yaml
```

默认训练 50 个通信轮、学习率 `1e-4`，BERT 配置允许 CPU 调试；GPT2-Large
默认要求 CUDA。SST-2 的官方 test 标签不可见，因此联邦训练使用 train split，
独立评估使用 validation split。两个配置默认同时运行六种通用攻击：
`blackbox_loss`、`loss_series`、`grad_cosine`、`avg_cosine`、`fedmia_loss` 和
`fedmia_cosine`，并在最后一轮运行严格 ProjRes。通用攻击对目标客户端构建
train/validation 严格 1:1、逐类别同数量的候选池；非成员只来自从未参与任何客户端
训练的 validation 数据。可用 `--attacks` 覆盖通用攻击列表，用 `--no-projres`
只关闭 ProjRes。

GPT2-Large 的逐样本梯度不会组成“候选数 × 全部 PEFT 参数”的常驻矩阵，而是逐样本
计算与真实上传的精确余弦后立即释放。ProjRes 观察目标客户端最后一轮真实 FedSGD
batch，攻击首个 Transformer block 后 Adapter 的 `down.weight`；成员定义严格等于
该 batch，非成员使用完整的独立 validation 分区。

`PromptFL` 是针对可训练 `prompt_learner` 定义的算法，不能直接应用到 MLP
分类头、Adapter 或 LoRA。三种参数高效微调模型使用共享全局状态：CLIP-MLP
使用 FedAvg，Visual Adapter 使用每客户端每轮一个 mini-batch 的 FedSGD；
CLIP-LoRA 普通训练使用 FedAvg，带论文严格 ProjRes 的统一攻击任务使用
one-batch FedSGD。训练协议不同，比较攻击结果时需要同时报告这一差异；服务器
聚合均对参与客户端直接等权平均，而不是按客户端样本数加权。

论文 ProjRes 在第一层 MLP 参数上的严格单轮 FedSGD 实现使用独立入口。它只取
`classifier.0.weight` 的一个真实 batch 更新，并以该层输入的 CLIP 表示计算
原始 L1 投影残差：

```bash
bash scripts/run_projres_mlp_real.sh
```

默认配置见 `configs/clip_mlp_projres.yaml`，严格威胁模型与公式映射见
`docs/projres_mlp_strict.md`。这里的严格 ProjRes 与通用审计器中的余弦
`promptres` 不是同一种算法。

在 MLP FedAvg 上运行损失、时序损失、单轮/时序梯度余弦和 FedMIA-I/II；统一
sweep 不再对 MLP 执行 ProjRes：

```bash
bash scripts/run_clip_mlp_fedmia_attacks.sh
```

该入口现在是多数据集 sweep，默认运行 Caltech101、OxfordPets、Flowers102、
Food101 和 CIFAR100。可以只选择其中一部分，也可以仅打印所有有效参数和命令：

```bash
bash scripts/run_clip_mlp_fedmia_attacks.sh \
  --datasets caltech101,flowers \
  --dry-run
```

每个训练任务都会创建一个新的时间戳目录，例如
`results/2026-08-05_14-30-52-123456_clip_mlp_caltech101_fedavg_seed42_targetall_<hash>/`。
目录名同时标明任务启动时间、模型实现、数据集、联邦方法、随机种子和审计范围；
重新运行不会复用或覆盖之前的任务目录。可复现配置保存在该目录的
`run_config.yaml`，默认攻击列表可通过 `--attacks` 覆盖。

Visual Adapter 的 ProjRes 直接复用主任务已缓存的 CLIP 向量，并观察实际通信轮的客户端上传更新，
不再启动第二个 Python 进程或重新加载数据；默认评估最后一轮，也可通过
`--projres-round 50` 对齐论文通常使用的第 50 轮。结果写入每个作业的
`privacy_audit/projres_strict.json`。可使用 `--projres-threshold` 调整阈值；仅运行
通用审计攻击时传入 `--skip-projres`。普通攻击汇总以实验组为文件名前缀写入
`results/<实验组>_summary_by_run.csv`、`<实验组>_summary_by_client.csv` 与
`<实验组>_summary_aggregate.csv`，
ProjRes 汇总写入 `<实验组>_summary_projres.csv`；不会生成 HTML 文件。

该脚本默认加载 `configs/clip_mlp_fedmia_attacks_sweep.yaml`，其基础配置是
`configs/clip_mlp_low_fpr_attacks.yaml`：默认只训练一次，并依次把每个客户端的
完整训练集作为该客户端的成员，使用所有客户端测试集及其他客户端训练集作为
相对于该客户端的互斥非成员；
全部联邦训练集和测试集在训练开始前只编码一次，后续 FedAvg、测试及攻击均复用
向量。每个客户端独立构建候选池和计算指标，再报告客户端宏平均；协议强制每个
客户端至少 1000 个非成员，因此
`TPR@0.1%FPR` 的经验分辨率达到要求；实际成员数、非成员数和
`fpr_resolution` 会写入 `privacy_audit/summary.json`。Adapter ProjRes 的成员
定义与被观察轮次的 FedSGD 协议一致，即当前实际 batch；它使用完整非成员池计算
低 FPR 指标。CLIP-MLP 的统一 sweep 不生成 ProjRes 结果。

Visual Adapter 对应的全数据五数据集 sweep 使用相同参数接口；严格 ProjRes
攻击 Adapter 的第一层下采样参数：

```bash
bash scripts/run_visual_adapter_fedmia_attacks.sh
```

推荐使用统一入口。无参数调用会依次完成 CLIP+MLP、CLIP+Adapter 和
CLIP-LoRA 的全部数据集 sweep：

```bash
bash scripts/run_all_clip_fedmia_attacks.sh
```

该入口默认运行三种模型和五个数据集。可通过 `--models clip_mlp`、
`--models visual_adapter` 或 `--models clip_lora` 只运行一个模型，并可用
`--learning-rate` 临时覆盖学习率。`--models all` 同样表示三种模型。ProjRes
在 Adapter 与 LoRA 主任务内执行；MLP 运行六种通用攻击。独立的 MLP ProjRes
验证入口仍保留用于单独复现实验。

常用筛选、覆盖和并行命令：

```bash
# 只打印三种模型的完整作业命令
bash scripts/run_all_clip_fedmia_attacks.sh --dry-run

# 只跑指定数据集与 FedMIA-I/II
bash scripts/run_all_clip_fedmia_attacks.sh \
  --datasets cifar100 \
  --attacks fedmia_loss,fedmia_cosine

# 两张 GPU 同时调度多个数据集作业
bash scripts/run_all_clip_fedmia_attacks.sh --gpus 0,1 --jobs 2

# 只运行 Adapter，并覆盖轮数与学习率
bash scripts/run_all_clip_fedmia_attacks.sh \
  --models visual_adapter \
  --rounds 50 \
  --learning-rate 0.0005

# 只打印 CLIP-LoRA 作业（等价入口：run_clip_lora_fedmia_attacks.sh）
bash scripts/run_all_clip_fedmia_attacks.sh --models clip_lora --dry-run
```

三种模型的攻击 sweep 默认运行 `blackbox_loss`、`loss_series`、`grad_cosine`、
`avg_cosine`、`fedmia_loss` 和 `fedmia_cosine`，并使用每客户端成员/非成员
1:1、类别精确配对的 `balanced_holdout` 协议。LoRA 因图像编码器可训练而不能
使用预计算特征的 `low_fpr_full` 候选池。

审计信号按攻击需求调度：`blackbox_loss` 和 `grad_cosine` 默认只使用最终轮，
`loss_series`、`avg_cosine`、`fedmia_loss` 和 `fedmia_cosine` 按论文定义逐轮
采集；其中两个时序基线只需要目标客户端，而 FedMIA 需要全部当轮客户端。
如果只选择单轮攻击或时序基线，调度器不会计算其他信号；默认六种攻击一起运行
时，由于 FedMIA 使用全部通信轮，仍会逐轮审计所有客户端。

## 多客户端成员隐私审计

设置 `audit.audit_client_ids: all` 或提供多个客户端 ID 时，审计器会先在每个
客户端上独立执行攻击，再汇总所有预测。候选集支持两种明确区分的协议：

- `candidate_sampling: legacy`：成员和非成员在客户端内按类别精确 1:1
  配对，用于控制类别分布混杂；
- `candidate_sampling: fedmia_mix`：沿用 FedMIA 的混合来源候选构造，以目标
  客户端训练样本为成员，非成员由完整独立测试池和其他客户端训练集按来源
  均衡抽取；本仓库默认成员与非成员数量为 1:1。

当前支持池化执行的攻击为：

- `loss_series`
- `grad_cosine`
- `avg_cosine`
- `promptres`
- `fedmia_loss`
- `fedmia_cosine`
- `nasr_passive`
- `transfer_representation`
- `rmia`
- `quantile_mia`

`promptres` 实现文档中的正方向余弦平方分数，可选使用其他客户端更新进行
leave-one-client-out 均值与截断 SVD 背景消除：

```yaml
audit:
  attacks: [promptres]
  promptres_background_rank: 0  # 0 为直接分数；正数启用背景残差化
  promptres_aggregation: mean   # mean、max 或 last
```

无需数据集或 CLIP checkpoint 的第一阶段可行性验证：

```bash
python analysis_scripts/verify_promptres_toy.py
```

非 FedMIA 分数使用客户端内、无成员标签的经验 CDF 秩校准后再合并；FedMIA
保留自身的零假设 CDF。汇总文件同时包含总体指标、客户端宏平均指标和逐客户
端指标。低 FPR 完整候选池与混合来源的 `fedmia_mix` 均支持一次训练审计所有
客户端；只审计单个客户端时可传 `--target-client 3`，恢复全部客户端时传
`--target-client all`。使用 `fedmia_mix` 测评时可配置为：

```yaml
audit:
  enabled: true
  audit_client_ids: all
  candidate_sampling: fedmia_mix
  nonmember_to_member_ratio: 1.0
  max_member_samples: 128
  max_nonmember_samples: 128
  match_candidate_labels: false
```

严格标签配对协议仍可通过 `candidate_sampling: legacy` 与
`match_candidate_labels: true` 使用。两种协议的结果不应直接混合比较；
`privacy_audit/summary.json` 会记录实际成员/非成员数量、比例和各非成员来源。

统一 CLIP 实验默认使用更严格的 `balanced_holdout`：对客户端 `k`，成员仅来自
该客户端训练集，非成员仅来自该客户端从未参与任何训练的独立测试集；两组按类别
无放回精确配对并保持 1:1。其他客户端训练样本不会被错误标记为全局模型非成员。

完整攻击定义和威胁模型见 [`docs/attack_mapping.md`](docs/attack_mapping.md)。

## 配置说明

实验主要通过 YAML 管理，常用字段包括：

- `model_type`：`prompt`、冻结 CLIP 图像编码器的 `clip_mlp`、
  `visual_adapter` 或 `clip_lora`；
- `clip_mlp`：两层 MLP 的隐藏维度、dropout 与特征归一化设置；
- `visual_adapter`：双侧特征维度、视觉/文本瓶颈 reduction、两侧残差混合系数、
  输出 ReLU、图像特征缓存和可选文本模板；该模式使用完整训练集；
- `clip_lora`：编码器侧、注意力目标投影、层范围、rank、alpha、dropout、缩放
  方式和可选文本模板；该模式使用完整训练集；
- `aggregator`：联邦训练方法；
- `total_users`、`sample_users`：客户端总数和每轮参与数；
- `partition_mode`：`iid`、`dirichlet`、`pathological` 或 `auto`；
- `fpl_shots`、`use_full_dataset`：few-shot 或全量训练；
- `audit`：攻击列表、审计客户端、候选数量和攻击参数；
- `defense`：独立防御名称和参数；
- `results_dir`：运行输出根目录。

命令行参数会覆盖 YAML 中对应字段。`configs/` 提供 CLIP-MLP、Visual Adapter
与 CLIP-LoRA 的单次运行和攻击 sweep，以及 MLP 严格 ProjRes 配置。

## 输出文件

每次运行会在 `results_dir` 下创建独立目录，主要包含：

- `run_config.yaml`：合并命令行覆盖后的实际配置；
- `run.log`：直接运行 `main.py` 时的训练、评估与审计日志；
- `training_metrics.csv`：各评估轮次的损失与准确率；
- `training_health.json`：训练状态健康检查；
- `final_prompt.pt`：最终可训练提示参数；
- `final_mlp.pt`：`clip_mlp` 基线最终可训练的两层 MLP 参数；
- `final_visual_adapter.pt`：图像侧和文本侧 CLIP Adapter 的最终可训练参数；
- `federated_method_summary.json`：联邦方法配置与诊断；
- `defense_summary.json`：防御配置、统计与隐私会计；
- `privacy_audit/summary.json`：攻击指标和逐客户端元数据；
- `privacy_audit/predictions.csv`：逐候选成员分数；
- `privacy_audit/signals.pt`：可选的跨轮次审计信号。

`signal_storage` 可设置为 `none`、`compact` 或 `full`，用于控制审计信号的
保存范围。

通过 sweep 脚本启动时，每个训练任务只使用一个
`results/<时间>_<模型>_<数据集>_<方法>_seed<种子>_target<客户端>_<hash>/`
目录。展开后的实际配置保存为其中的 `run_config.yaml`，标准输出和错误保存为
其中的 `run.log`，训练指标、健康检查和隐私审计结果也直接写入该任务目录；
ProjRes 日志并入 `run.log`，不再创建独立的 `projres_strict.log`、`configs/` 或
`launcher_logs/`。
进度日志按 `eval_interval` 输出；逐轮详情仅在 DEBUG 级别记录。

## 测试

日常修改默认运行快速核心回归，不需要数据集、GPU 或 CLIP 检查点：

```bash
python -m pytest -q
```

默认测试覆盖 few-shot 数据划分、实验入口与配置、轮次调度、信号压缩和
ProjRes 基本不变量。只有修改共享审计器、聚合核心或准备完整回归时，才显式运行
完整本地测试：

```bash
python -m pytest -q tests
```

小范围修改应优先直接指定相关测试文件，避免无关测试消耗时间和资源。
