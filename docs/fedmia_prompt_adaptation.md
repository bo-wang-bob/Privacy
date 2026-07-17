# FedMIA 方法在联邦 CLIP soft-prompt 中的适配

本文档说明 FedMIA 比较方法在本仓库中的定义、兼容性边界和复现实验入口。实现参考 FedMIA 论文及其官方代码，但训练和更新始终限制在 `requires_grad=True` 的 soft-prompt 参数；CLIP image/text encoder 保持冻结并只从本地缓存加载。

## 攻击定义

四个攻击都复用审计器在目标客户端上逐候选、逐轮采集的真实协议观察，不额外训练代理模型。

| 运行名 | 测量 | 时间信息 | 空间信息 | 本仓库评分 |
|---|---|---|---|---|
| `blackbox_loss` | 负交叉熵损失 | 单轮 | 仅目标客户端 | 固定轮次候选分数 |
| `loss_series` | 负交叉熵损失 | 多轮 | 仅目标客户端 | 各观测轮分数均值 |
| `grad_cosine` | 候选 prompt 梯度与客户端 prompt delta 的余弦相似度 | 单轮 | 仅目标客户端 | 固定轮次候选分数 |
| `avg_cosine` | 同上 | 多轮 | 仅目标客户端 | 各观测轮分数均值 |

单轮攻击由 `audit.fedmia_baseline_single_round` 选择 `first`、`last` 或一个已观测轮次索引，默认 `last`。FedMIA 官方绘图代码会使用真实成员标签在多个轮次中选择最高 TPR；这适合复现论文图表，却是一个评估时标签 oracle。本仓库固定轮次，避免把测试标签用于选择攻击分数。

`grad_cosine` 与 `avg_cosine` 依赖客户端更新，因此不能用于仅发布最终 prompt 的 `released_prompt` 审计视图。`blackbox_loss` 与 `loss_series` 可以在该视图中退化为最终发布状态上的查询。

## 防御定义与兼容性

| 运行名 | prompt 适配 | 主要参数 |
|---|---|---|
| `perturb` | 裁剪客户端 prompt delta，上传前加高斯噪声 | `perturb_clip_norm`、`perturb_noise_std` |
| `sparse` | 上传前将最小幅值的 prompt delta 置零 | `sparse_ratio` |
| `mixup` | 本地训练使用 Beta MixUp 与软组合损失 | `mixup_alpha` |
| `sampling` | 每批随机保留部分训练样本 | `sampling_ratio` |
| `data_aug` | 对已预处理 CLIP 张量做翻转、平移和颜色扰动 | `data_aug_*` |
| `data_aug_sampling` | 每批先抽样，再增强 | `sampling_ratio`、`data_aug_*` |

这些方法是 FedMIA 中的独立防御比较基线，而不是 DP-FPL 或 FedASK 的组成部分。为避免叠加机制后产生无法归因的结果，配置校验要求它们使用 `aggregator: fedavg` 和 `train_mode: centralized`。量化只出现在论文附录的额外讨论中，不属于用户指定的六个防御，因此本次未加入。

## 正式实验

基准配置同时启用四个比较攻击和 FedMIA-I/FedMIA-II：

```bash
python main.py --config configs/fedmia_prompt_benchmark.yaml --data_root /path/to/datasets --cache_dir /path/to/clip-cache --defense none
python main.py --config configs/fedmia_prompt_benchmark.yaml --defense perturb --perturb_noise_std 0.1
python main.py --config configs/fedmia_prompt_benchmark.yaml --defense sparse --sparse_ratio 0.9
python main.py --config configs/fedmia_prompt_benchmark.yaml --defense mixup --mixup_alpha 1.0
python main.py --config configs/fedmia_prompt_benchmark.yaml --defense sampling --sampling_ratio 0.5
python main.py --config configs/fedmia_prompt_benchmark.yaml --defense data_aug --data_aug_strength 0.1
python main.py --config configs/fedmia_prompt_benchmark.yaml --defense data_aug_sampling --sampling_ratio 0.5 --data_aug_strength 0.1
```

建议至少扫描以下论文范围或其中的对数子集，并为每个点固定数据划分、候选集合、目标客户端和随机种子：

- Perturb 噪声标准差：`0.01, 0.05, 0.1, 0.2, 0.5`；
- Sparse 比例：`0.1, 0.5, 0.9, 0.99`；
- MixUp alpha：`0.1, 1, 10`，需要复现论文全范围时再扩展到 `1e-5–1e5`；
- Sampling 比例：`0.1, 0.25, 0.5, 0.75, 1.0`；
- Data Aug 强度：`0.05, 0.1, 0.2`，组合防御同时扫描 Sampling 比例。

每次实验同时比较 `privacy_audit/summary.json` 的 AUC、TPR@10%FPR、TPR@1%FPR、TPR@0.1%FPR，以及 `training_metrics.csv` 的最终任务准确率。只有轻量 toy 测试通过不能作为防御效果结论；正式数值必须来自相同数据划分上的完整 CLIP 实验。

### 自动选卡的单卡串行复杂扫描

`analysis_scripts/run_fedmia_complex_sweep.py` 读取 `configs/fedmia_complex_sweep.yaml` 并展开防御参数。所有任务严格串行，每次只占用一张 GPU；每个任务启动前通过 `nvidia-smi` 比较候选 GPU 的空闲显存和利用率，自动选择当前最合适的卡。默认至少需要 7000 MiB 空闲显存，否则等待而不是抢占繁忙 GPU。

正式规格包含26个防御设置、3个随机种子，共78次20轮实验，本地 epoch 为2。每次同时审计 Blackbox-Loss、Loss-Series、Grad-Cosine、Avg-Cosine、FedMIA-Loss、FedMIA-Cosine、Nasr-Passive、RMIA 和 Quantile-MIA，不运行 Nasr-Active。候选最多为128个成员和2048个按成员标签比例分层抽取的非成员；非成员优先来自目标测试集及 few-shot 选择前被排除、从未分配给任何客户端的同域训练样本。后三个带校准集的攻击划分后仍约有1024个评估非成员，使经验 TPR@0.1%FPR 至少具有一个误报点的分辨率。

```bash
# 一键运行；自动在 GPU 0/1 中逐任务选择一张卡
./scripts/run_fedmia_complex.sh

# 只检查展开后的任务，不训练
./scripts/run_fedmia_complex.sh --dry-run
```

可用 `--gpus 1,0` 限制候选卡，或用 `--min-free-memory-mb 10000` 修改启动阈值。脚本为每个配置生成稳定的 run ID，并把进度写入 `results/fedmia_complex_tpr1/sweep_state.json`。中断后重新执行同一命令会跳过已有完整结果，只继续未完成或失败的任务。单次 smoke test 可加 `--max-runs 2`；只重建汇总可加 `--summarize-only`；确需重跑全部配置时加 `--force`。

主要输出为：

- `summary_by_run.csv`：逐 seed、目标客户端、防御和攻击的准确率、三档 TPR、AUC 与 Prompt-DP 隐私会计；
- `summary_aggregate.csv`：跨 seed 的均值和样本标准差，以 TPR@1%FPR 为主要隐私指标；
- `summary_tpr_matrix.csv`：TPR@10%FPR、TPR@1%FPR 和 TPR@0.1%FPR 的比较矩阵；
- `privacy_utility_pareto.csv`：逐攻击的隐私—准确率非支配配置；
- `launcher_logs/`：每个独立实验的完整控制台日志；
- `configs/`：实际传给 `main.py` 的展开后 YAML，便于精确复现。

## 当前验证状态

轻量测试不需要数据集或 CLIP checkpoint，覆盖四类攻击的单轮/多轮定义、六种防御的训练或上传钩子、确定性，以及所有攻击—防御组合的端到端执行。正式配置要求 CUDA、本地 `openai/clip-vit-base-patch32` 缓存和已有数据集；缺少任一资源时会明确失败，不会联网下载 CLIP 权重。

2026-07-17 已在 CIFAR100 和本地 CLIP ViT-B/32 上完成一组 GPU 快速对照：FedAvg、4 客户端全部参与、Dirichlet alpha 0.1、每类 4-shot、2 轮、128 个标签匹配候选、seed 42。下表为最终任务准确率与攻击 TPR@1%FPR；该指标也是审计器输出的 primary metric：

| 防御 | 准确率 | Blackbox-Loss | Loss-Series | Grad-Cosine | Avg-Cosine |
|---|---:|---:|---:|---:|---:|
| none | 0.6308 | 0.0469 | 0.0469 | 0.0000 | 0.0000 |
| Perturb (`std=0.1`) | 0.4776 | 0.0000 | 0.0312 | 0.0000 | 0.0156 |
| Sparse (`ratio=0.9`) | 0.6206 | 0.0156 | 0.0156 | 0.0000 | 0.0000 |
| MixUp (`alpha=1`) | 0.6374 | 0.0312 | 0.0469 | 0.0000 | 0.0312 |
| Sampling (`ratio=0.5`) | 0.6324 | 0.0000 | 0.0312 | 0.0000 | 0.0156 |
| Data Aug (`strength=0.1`) | 0.6276 | 0.0469 | 0.0312 | 0.0000 | 0.0000 |
| Data Aug + Sampling | 0.6314 | 0.0000 | 0.0312 | 0.0312 | 0.0156 |

这组快速实验用于验证真实数据、真实 CLIP 和完整协议下的运行效果，不足以形成最终统计结论：只有 2 个通信轮次和 1 个随机种子，64 个非成员候选也只能让经验 FPR 以约 1.56 个百分点为步长变化，因此标称 1% FPR 指标较粗。Perturb、Sparse 和 Sampling 均降低了 Blackbox-Loss 的 TPR@1%FPR，其中 Perturb 的准确率损失达到 15.32 个百分点；MixUp 没有降低 Loss-Series，并提高了 Avg-Cosine。完整结论应使用更多候选、20 轮、多随机种子和上述参数扫描。
