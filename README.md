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
- `fedmia_text`
- `fedmia_text_gradient`
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

`fedmia_text`（本仓库扩展，简称 FedMIA-III）把攻击空间从提示参数扩展到 CLIP 文本特征
矩阵。每轮先计算各客户端训练前后的归一化文本特征矩阵变化，再沿候选样本
的提示梯度执行一个小幅虚拟下降步，并比较两者的 Frobenius 余弦相似度。
其他客户端的同类分数继续作为 FedMIA 的逐轮零假设：

```yaml
audit:
  attacks: [fedmia_loss, fedmia_cosine, fedmia_text, fedmia_text_gradient]
  fedmia_text_probe_norm: 0.001
  fedmia_text_candidate_batch_size: 8
  fedmia_text_aggregation: mean
  fedmia_text_tail: upper
  fedmia_text_gradient_aggregation: mean
  fedmia_text_gradient_tail: upper
  fedmia_text_gradient_project_tangent: false
```

FedMIA-III 只保存每个客户端与候选样本的最终相似度，不保存完整候选文本
特征矩阵。它需要协议能够重建逐客户端提示；`released_prompt` 视图不支持，
FedASK 在非 `full_whitebox` 视图下也不支持。

`fedmia_text_gradient`（FedMIA-IV）在假设分类 logit 为
`scale * image_features @ text_features.T` 时，直接计算候选样本交叉熵对文本
矩阵的负梯度 `-scale * (softmax(logits) - one_hot(label)) x.T`，并与客户端
实际文本矩阵变化计算 Frobenius 余弦。它不使用 JVP，也不重新编码候选 prompt；
可选的 `fedmia_text_gradient_project_tangent=true` 会将每一类梯度投影到归一化
文本特征的切空间。FedOTP 使用最优传输 logit，不满足这一假设，因此不支持
FedMIA-IV。

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
