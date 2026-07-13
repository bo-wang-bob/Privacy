# Privacy attacks in Federated Prompt Tuning

> 五种独立防御及“攻击/防御/攻击+防御”运行方式见 [docs/defenses.md](docs/defenses.md)。

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
| `pipra` | PIPRA, AAAI 2026 | 训练多个 shadow prompts，并仅从候选图像与标签条件 prompt 的跨模态几何关系推断成员，不使用目标预测输出 |
| `rmia` | RMIA, ICML 2024 | 使用非目标客户端提示作为 reference models，以已知 OUT 总体进行低成本相对似然比检验 |
| `imia` | IMIA, USENIX Security 2026 | 仅训练 prompt 的目标感知模仿模型，分别估计 OUT 行为与同类别 pivot 的 IN 行为 |
| `quantile_mia` | Quantile-MIA, NeurIPS 2023 | 在已知非成员样本上拟合条件置信度分位数，用目标置信度超出条件阈值的幅度打分 |
| `yoqo` | YOQO, ICLR 2024 | 由非目标 prompt 离线构造 improvement-area 查询，对目标 prompt 只保留一次 hard label 响应 |
| `canary` | Canary, ICLR 2023 | 用 prompt-only IN/OUT surrogate pairs 优化一组多样查询，再集成目标与 OUT prompts 的校准置信度 |
| `promptmia` | PromptMIA, 2026 预印本 | 按论文余弦区间算法构造多样化 adversarial keys；在当前共享 CoOp prompt 中以隔离 token-update probe 适配 |

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
  --audit_attacks pipra,fedmia_loss,rmia,imia,quantile_mia,yoqo,canary,promptmia
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
