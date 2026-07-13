# Privacy attacks in Federated Prompt Tuning

本仓库从 SEISMOGRAPH 中提取了联邦提示微调（Federated Prompt Tuning, FPT）主流程，删除了全部后门攻击、触发器优化、恶意客户端逻辑和 SEISMOGRAPH 防御，仅保留：

- 冻结的 CLIP 图像/文本骨干；
- 可训练的 soft prompt；
- 客户端本地提示微调；
- 按样本数加权的 FedAvg；
- 面向联邦提示学习的成员推理隐私审计。

## 已实现的成员推理攻击

| 配置名 | 论文方法 | 在联邦提示学习中的适配 |
|---|---|---|
| `nasr_passive` | Nasr et al. 被动白盒 MIA | 组合提示梯度 token 范数、输出、损失与客户端提示更新关系，训练监督攻击头 |
| `nasr_active` | Nasr et al. 主动梯度上升 MIA | 对候选样本进行隔离的提示梯度上升探测，再观察目标客户端本地训练后的损失/梯度骤降；不污染正式全局训练 |
| `fedmia_loss` | FedMIA-I | 以所有非目标客户端的候选样本置信度构造单尾零假设，并跨通信轮平均 |
| `fedmia_cosine` | FedMIA-II | 以候选样本提示梯度与客户端 prompt update 的余弦相似度构造单尾零假设，并跨轮平均 |
| `transfer_representation` | Wu et al. 表示差异攻击 | 将目标客户端提示视为 student prompt，其他客户端更新形成 shadow prompts，学习其联合图文表示差异 |
| `codepoison` | Chen & Pattabiraman CodePoison MIA | 本地提示训练同时记忆由样本哈希确定的秘密合成伙伴，再以合成伙伴损失推断成员身份 |

详细对应关系、公开实现来源和适配边界见 [docs/attack_mapping.md](docs/attack_mapping.md)。

## 环境

推荐 Python 3.10+：

```bash
pip install -r requirements.txt
```

CLIP 始终以 `local_files_only=True` 加载。请提前将 `openai/clip-vit-base-patch32` 放入 `cache_dir` 指定的本地缓存。程序不会自动下载模型权重。

## 运行

完整配置：

```bash
python main.py --config configs/fedprompt_privacy.yaml
```

轻量配置：

```bash
python main.py --config configs/fedprompt_privacy_quick.yaml
```

可以从命令行覆盖常用参数：

```bash
python main.py --config configs/fedprompt_privacy.yaml \
  --dataset_name cifar100 \
  --num_global_iters 10 \
  --target_client_id 0 \
  --audit_attacks nasr_passive,fedmia_loss,fedmia_cosine
```

## 输出

每次运行会在 `results/<dataset>_fedprompt_privacy_<time>/` 生成：

- `run_config.yaml`：实际运行配置；
- `training_metrics.csv`：全局提示的干净任务指标；
- `final_prompt.pt`：最终可训练提示参数；
- `privacy_audit/summary.json`：各攻击 AUC 与低 FPR TPR；
- `privacy_audit/predictions.csv`：逐候选样本分数；
- `privacy_audit/signals.pt`：不含原始图像的跨客户端、跨轮审计信号。

## 测试

轻量测试不需要数据集或 CLIP 权重：

```bash
pytest -q
```

## 研究用途说明

这些实现用于获得统一、可复现的隐私风险基线。不同论文原本的模型、数据划分和威胁模型并不完全一致，因此本仓库报告的是“在联邦 soft-prompt 训练中的适配结果”，不能直接替代论文原始实验数字。
