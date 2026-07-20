# 联邦提示学习隐私攻击与防御基准

本仓库保留冻结 CLIP 骨干上的联邦 soft-prompt 学习，并提供统一的成员推理攻击、独立防御和联邦训练方法接口。后门触发器、恶意客户端逻辑与 SEISMOGRAPH 防御均不在本仓库中。

## 联邦训练方法

运行时用 `--aggregator` 选择：

- `fedavg`：保留历史提示词构造的标准 FedAvg；训练骨架与 PromptFL 相同。
- `promptfl`：论文 PromptFL 基线。客户端只优化共享 CoOp soft prompt，服务器按客户端样本数执行 FedAvg。
- `fedotp`：每个客户端同时优化全局/本地完整 prompt，以熵正则部分最优传输匹配图像 patch 与两类文本 prompt；只聚合全局 prompt。
- `fedpgp`：全局 prompt 加客户端低秩个性化项，使用论文的任务损失与 prompt-wise 对比损失；只聚合全局 prompt。
- `dpfpl`：适配 ICLR 2025 DP-FPL。每个客户端持久保留本地提示，服务器同步全局提示；本地提示使用低秩梯度投影/重构和局部高斯机制，全局上传使用裁剪与服务器高斯机制。
- `fedask`：适配 NeurIPS 2025 FedASK。soft prompt 被参数化为冻结初始提示加 `B·A`；客户端固定 `A`，对完整 prompt 梯度做逐样本裁剪和扰动后更新 `B`，服务器执行两阶段随机草图、QR 与 SVD 重构 `A/B`。

三个论文提示学习方法可直接使用统一配置运行：

```bash
python main.py --config configs/federated_prompt_paper.yaml --aggregator promptfl
python main.py --config configs/federated_prompt_paper.yaml --aggregator fedotp
python main.py --config configs/federated_prompt_paper.yaml --aggregator fedpgp
```

该配置关闭隐私攻击和额外防御，以单独验证论文训练目标。FedOTP/FedPGP 自动启用客户端个性化状态；PromptFL 使用共享全局状态。方法超参数分别位于配置文件的 `fedotp:`、`fedpgp:` 段中。

三种方法的 FedMIA-Loss/FedMIA-Cosine 完整审计可一键运行：

```bash
./scripts/run_fedmia_prompt_methods.sh
```

新增的 few-shot 设置先从整个联邦系统的训练集为每个类别抽取 16 张
图片，再以 Dirichlet α=0.1 将这个共享样本池划分给客户端；测试集不做
few-shot 截断。默认配置和一键命令分别为
`configs/fedmia_prompt_methods_fewshot_sweep.yaml` 与：

```bash
./scripts/run_fedmia_prompt_methods_fewshot.sh
```

可直接从脚本覆盖每类图片数和异构程度，例如：

```bash
./scripts/run_fedmia_prompt_methods_fewshot.sh \
  --fpl-shots 8 --dirichlet-alpha 0.5 --rounds 75
```

`--shots` 是 `--fpl-shots` 的简写。两个参数也可用于原有 sweep
启动器；命令行值优先于 YAML，并自动启用 Dirichlet 划分和关闭
full-data 模式。通信轮数默认是 50；可用 `--rounds N` 覆盖，
`--num-global-iters N` 是它的等价写法。

few-shot 隐私审计会继续执行逐客户端、逐类别的成员/非成员精确配对。
若极端 Dirichlet 划分使个别客户端不足两对候选，或该客户端没有参与
任何可用通信轮次，池化 FedMIA 会跳过该客户端并在 `summary.json` 中记录
原因；攻击指标只根据实际审计到的客户端和样本计算。样本量不足以解析
某档 FPR 时，该指标保持不可报告，而不会输出伪精确结果。

few-shot sweep 对所有数据集统一使用 10 个客户端、每轮 10 个客户端全部
参与、默认 50 个通信轮，并在每个客户端执行 2 个本地 epochs。

启动器默认使用 `--jobs 1` 顺序执行；可通过 `--jobs N` 设置最大并发任务数。每个任务只使用一张卡，但同一张候选 GPU 可以同时运行多个任务；每次启动任务时都会选择满足显存门槛且空闲显存最多的卡。因此并发数可以超过候选 GPU 数，但 `--jobs` 和显存门槛应按实际显存容量设置。实验支持断点续跑，汇总结果位于 `results/fedmia_prompt_methods/summary_privacy_metrics.csv`，其中 TPR 以百分数报告，并同时给出 FPR=0.1%、1%、10% 三档结果与 AUC。

该 sweep 在一次训练中对 10 个客户端执行 FedMIA 池化审计。每个客户端分别从本地训练集和同客户端测试集按类别一一配对成员与非成员，再合并攻击分数；因此 pathological 标签划分不会让攻击通过类别归属取巧。每个客户端最多贡献 128 对候选，合并后的 1280 个非成员可分辨 FPR=0.1% 档位。CIFAR100 有 50 个训练客户端，但同样固定审计前 10 个客户端以控制计算量。

方法细节和论文对应关系见 [docs/federated_methods.md](docs/federated_methods.md)。
Flowers102 同场景公平比较和 VEIL（原 Local-GGEUR）优化结果见
[docs/flowers_fair_method_comparison.md](docs/flowers_fair_method_comparison.md)。
AAAI 2027 论文源码位于 [paper/aaai2027/veil.tex](paper/aaai2027/veil.tex)。

## 攻击与防御

成员推理攻击包括 Nasr 被动/主动攻击、FedMIA、表示迁移、CodePoison、PIPRA、RMIA、IMIA、Quantile-MIA、YOQO、Canary 和 PromptMIA。映射说明见 [docs/attack_mapping.md](docs/attack_mapping.md)。

可独立选择 `cofedmid`、`prompt_dp`、`mist`、`soft`、`hamp`、`veil`
六种防御；`local_ggeur`、`mirage` 仅保留为 VEIL 的历史兼容名。详见
[docs/defenses.md](docs/defenses.md)。一次运行可指定：

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

DP-FPL/FedASK 参数在配置文件的 `dpfpl:` 与 `fedask:` 段中设置。DP-FPL 自动使用个性化客户端状态；FedAvg、PromptFL 和 FedASK 使用共享全局状态。

可以通过 `audit.audit_view` 明确攻击者可见性。默认的 `protocol_plus_released_prompts` 仅把真实上传消息用于更新攻击，并允许查询公开 prompt；也可选择 `released_prompt` 或用于强上界的 `full_whitebox`。方法摘要会报告保守的隐私预算和 FedASK 草图重构诊断。

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
