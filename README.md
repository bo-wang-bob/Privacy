# 联邦提示学习隐私攻击与防御基准

本仓库保留冻结 CLIP 骨干上的联邦 soft-prompt 学习，并提供统一的成员推理攻击、独立防御和联邦训练方法接口。后门触发器、恶意客户端逻辑与 SEISMOGRAPH 防御均不在本仓库中。

## 联邦训练方法

运行时用 `--aggregator` 选择：

- `fedavg`：按客户端样本数加权的标准 FedAvg。
- `dpfpl`：适配 ICLR 2025 DP-FPL。每个客户端持久保留本地提示，服务器同步全局提示；本地提示使用低秩梯度投影/重构和局部高斯机制，全局上传使用裁剪与服务器高斯机制。
- `fedask`：适配 NeurIPS 2025 FedASK。soft prompt 被参数化为冻结初始提示加 `B·A`；客户端固定 `A`，对完整 prompt 梯度做逐样本裁剪和扰动后更新 `B`，服务器执行两阶段随机草图、QR 与 SVD 重构 `A/B`。

方法细节和论文对应关系见 [docs/federated_methods.md](docs/federated_methods.md)。

## 攻击与防御

成员推理攻击包括 Nasr 被动/主动攻击、FedMIA、表示迁移、CodePoison、PIPRA、RMIA、IMIA、Quantile-MIA、YOQO、Canary 和 PromptMIA。映射说明见 [docs/attack_mapping.md](docs/attack_mapping.md)。

可独立选择 `cofedmid`、`prompt_dp`、`mist`、`soft`、`hamp` 五种防御，见 [docs/defenses.md](docs/defenses.md)。一次运行可指定：

```bash
# 一个方法 + 一个攻击 + 一个防御
python main.py --config configs/fedprompt_privacy.yaml \
  --aggregator dpfpl --attack fedmia_loss --defense prompt_dp

# FedASK + 仅攻击
python main.py --config configs/fedprompt_privacy.yaml \
  --aggregator fedask --attack pipra --defense none

# DP-FPL + 仅防御
python main.py --config configs/fedprompt_privacy.yaml \
  --aggregator dpfpl --attack none --defense hamp
```

方法参数在配置文件的 `dpfpl:` 与 `fedask:` 段中设置。DP-FPL 自动使用个性化客户端状态；FedAvg 和 FedASK 使用共享全局状态。

## 环境

推荐 Python 3.10+：

```bash
pip install -r requirements.txt
```

CLIP 始终通过 `local_files_only=True` 加载。请提前把 `openai/clip-vit-base-patch32` 放入 `cache_dir` 指定的本地缓存；程序不会下载模型权重。

## 输出

每次运行会生成：

- `run_config.yaml`：实际运行配置；
- `training_metrics.csv`：任务损失与准确率；
- `final_prompt.pt`：最终可训练提示状态；
- `federated_method_summary.json`：联邦方法、隐私参数及重构/裁剪诊断；
- `defense_summary.json`：所选防御及统计；
- `privacy_audit/`：攻击分数、指标和审计信号（启用攻击时）。

## 测试

轻量测试不需要数据集或 CLIP 权重：

```bash
python -m pytest -q
```

本仓库报告的是论文方法在联邦 CLIP soft-prompt 场景中的适配结果。模型、数据划分与威胁模型和原论文实验可能不同，不能把这里的数值直接视为论文复现数值。
