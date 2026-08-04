# Federated CLIP Soft-Prompt Membership Privacy Benchmark

本仓库用于研究联邦 CLIP soft-prompt tuning 中的成员隐私风险。系统冻结
CLIP 骨干，只训练可学习提示参数，并在统一训练流程中提供联邦方法、数据划分、
成员推理攻击、隐私防御和实验结果汇总。

仓库聚焦成员隐私审计，不包含后门触发器、恶意客户端投毒、ASR 指标或
SEISMOGRAPH 防御。

## 核心能力

| 模块 | 支持内容 |
| --- | --- |
| 联邦训练 | FedAvg、PromptFL、FedOTP、FedPGP、DP-FPL、FedASK |
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

### 仅运行联邦提示学习

以下命令使用 PromptFL 训练，不启用攻击或防御：

```bash
python main.py \
  --config configs/federated_prompt_paper.yaml \
  --aggregator promptfl
```

将 `promptfl` 替换为 `fedavg`、`fedotp`、`fedpgp`、`dpfpl` 或 `fedask`
即可选择其他联邦方法。各方法专用参数位于配置文件的同名段中。

### CLIP 图像编码器 + 两层 MLP 的 FedAvg 基线

该基线冻结完整 CLIP，只训练并聚合 `Linear -> ReLU -> Linear` 分类头，并使用
官方完整训练集而不做 few-shot 截断。默认还会在 Server 创建前一次性编码全部
训练/测试图片，之后的本地训练、周期测试和成员审计只读取 CPU 中的 CLIP 向量：

```bash
python main.py --config configs/clip_mlp_fedavg.yaml
```

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

### 16-shot 视觉 CLIP Adapter

该模式冻结 CLIP 图像和文本编码器，只用 FedAvg 训练
`Linear -> ReLU -> Linear -> ReLU` 视觉瓶颈，并按 `alpha` 将适配特征与原始
图像特征残差混合。它固定复用 FPL 的每类 16-shot 划分：

```bash
python main.py --config configs/visual_adapter_privacy.yaml
```

CIFAR-100 与 Caltech101 默认使用 `a photo of a {class}.`；OxfordPets 使用
`a photo of a {class}, a type of pet.`。可通过 `visual_adapter.template` 覆盖。
命令行可使用 `--adapter_reduction` 和 `--adapter_alpha` 调整瓶颈与混合比例。

论文 ProjRes 在第一层 MLP 参数上的严格单轮 FedSGD 实现使用独立入口。它只取
`classifier.0.weight` 的一个真实 batch 更新，并以该层输入的 CLIP 表示计算
原始 L1 投影残差：

```bash
bash scripts/run_projres_mlp_real.sh
```

默认配置见 `configs/clip_mlp_projres.yaml`，严格威胁模型与公式映射见
`docs/projres_mlp_strict.md`。这里的严格 ProjRes 与通用审计器中的余弦
`promptres` 不是同一种算法。

在 MLP FedAvg 上运行损失、时序损失、单轮/时序梯度余弦、FedMIA-I/II，
并在随后使用相同配置执行严格单批次 ProjRes：

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

每个作业开始时都会输出完整参数块，并把可复现配置保存到
`results/clip_mlp_fedmia_attacks/runs/<run_id>/run_config.yaml`。默认攻击列表可通过
`--attacks` 覆盖。

严格 ProjRes 结果写入每个作业的
`privacy_audit/projres_strict.json`。可使用 `--projres-threshold` 调整阈值；仅运行
通用审计攻击时传入 `--skip-projres`。普通攻击汇总写入
`summary_by_run.csv` / `summary_aggregate.csv`，ProjRes 汇总写入
`summary_projres.csv`；不会生成 HTML 文件。

该脚本默认加载 `configs/clip_mlp_fedmia_attacks_sweep.yaml`，其基础配置是
`configs/clip_mlp_low_fpr_attacks.yaml`：通用攻击使用目标客户端
完整训练集作为成员，并使用所有客户端测试集及其他客户端训练集作为互斥非成员；
全部联邦训练集和测试集在训练开始前只编码一次，后续 FedAvg、测试及攻击均复用
向量。协议强制至少 1000 个非成员，因此
`TPR@0.1%FPR` 的经验分辨率达到要求；实际成员数、非成员数和
`fpr_resolution` 会写入 `privacy_audit/summary.json`。严格 ProjRes 的成员仍只能
来自产生上传梯度的实际 batch，但它也使用完整非成员池计算低 FPR 指标。

### 运行单个攻击与防御

```bash
python main.py \
  --config configs/fedprompt_privacy.yaml \
  --aggregator promptfl \
  --attack fedmia_loss \
  --defense none
```

`--attack none` 可关闭攻击，`--defense none` 可关闭防御。也可以通过
`--audit_attacks` 传入逗号分隔的多个攻击名称。

### 运行 Few-shot 多客户端审计

```bash
./scripts/run_fedmia_prompt_methods_fewshot.sh
```

该脚本默认运行 PromptFL。可覆盖方法、few-shot 数量、Dirichlet 系数和轮数：

```bash
./scripts/run_fedmia_prompt_methods_fewshot.sh \
  --methods promptfl,fedotp,fedpgp \
  --fpl-shots 8 \
  --dirichlet-alpha 0.5 \
  --rounds 75
```

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
端指标。使用混合来源的 `fedmia_mix` 测评时可配置为：

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

完整攻击定义和威胁模型见 [`docs/attack_mapping.md`](docs/attack_mapping.md)。

## 配置说明

实验主要通过 YAML 管理，常用字段包括：

- `model_type`：`prompt`、冻结 CLIP 图像编码器的 `clip_mlp`，或
  16-shot `visual_adapter`；
- `clip_mlp`：两层 MLP 的隐藏维度、dropout 与特征归一化设置；
- `visual_adapter`：视觉特征维度、瓶颈 reduction、残差混合系数 `alpha`、输出
  ReLU、特征缓存和可选文本模板；该模式固定使用 `fpl_shots: 16`；
- `aggregator`：联邦训练方法；
- `total_users`、`sample_users`：客户端总数和每轮参与数；
- `partition_mode`：`iid`、`dirichlet`、`pathological` 或 `auto`；
- `fpl_shots`、`use_full_dataset`：few-shot 或全量训练；
- `audit`：攻击列表、审计客户端、候选数量和攻击参数；
- `defense`：独立防御名称和参数；
- `results_dir`：运行输出根目录。

命令行参数会覆盖 YAML 中对应字段。可用配置位于 [`configs/`](configs/)，
联邦方法与防御的实现说明分别见
[`docs/federated_methods.md`](docs/federated_methods.md) 和
[`docs/defenses.md`](docs/defenses.md)。

## 输出文件

每次运行会在 `results_dir` 下创建独立目录，主要包含：

- `run_config.yaml`：合并命令行覆盖后的实际配置；
- `run.log`：直接运行 `main.py` 时的训练、评估与审计日志；
- `training_metrics.csv`：各评估轮次的损失与准确率；
- `training_health.json`：训练状态健康检查；
- `final_prompt.pt`：最终可训练提示参数；
- `final_mlp.pt`：`clip_mlp` 基线最终可训练的两层 MLP 参数；
- `final_visual_adapter.pt`：视觉 CLIP Adapter 最终可训练的瓶颈参数；
- `federated_method_summary.json`：联邦方法配置与诊断；
- `defense_summary.json`：防御配置、统计与隐私会计；
- `privacy_audit/summary.json`：攻击指标和逐客户端元数据；
- `privacy_audit/predictions.csv`：逐候选成员分数；
- `privacy_audit/signals.pt`：可选的跨轮次审计信号。

`signal_storage` 可设置为 `none`、`compact` 或 `full`，用于控制审计信号的
保存范围。

通过 sweep 脚本启动时，每个任务只使用 `runs/<run_id>/` 一个目录。展开后的
实际配置保存为其中的 `run_config.yaml`，标准输出和错误保存为其中的
`run.log`，训练指标与审计结果也直接写入同一目录，不再创建独立的
`configs/`、`launcher_logs/` 或时间戳子目录。完整超参数只保留在
`run_config.yaml`，不在日志中重复打印。进度日志按 `eval_interval` 输出；
逐轮详情仅在 DEBUG 级别记录。

## 测试

轻量测试不需要数据集、GPU 或 CLIP 检查点：

```bash
python -m pytest -q
```

测试覆盖联邦聚合、few-shot 数据划分、攻击与防御接口、审计池化、信号压缩
和 sweep 配置。
