# CLIP-MLP、Visual Adapter 与 CLIP-LoRA 上的 ProjRes 实现

本实现对应论文 *Toward Efficient Membership Inference Attacks against
Federated Large Language Models: A Projection Residual Approach* 的 Algorithm
1。论文中的 adapter down-projection 层在 Visual Adapter 中直接对应
`adapter.net.0.weight`；在 CLIP-MLP 对照模型中对应第一层分类 MLP
`classifier.0.weight`。CLIP-LoRA 对应视觉编码器首个 Q 投影中的下投影因子
`clip_model.vision_model.encoder.layers.0.self_attn.q_proj.lora_A`。

## 数学映射

冻结 CLIP 产生批表示 `X ∈ R^(p×n)`，它就是 Visual Adapter 的输入隐藏
表示。被攻击的 down-projection 是 PyTorch
`Linear(n, m)`，其权重梯度为：

```text
G = dL/dW = Delta^T X ∈ R^(m×n)
```

因此 `G` 的每一行都是训练批表示行向量的线性组合。服务端对可见的
`W_before - W_after = learning_rate × G` 做 SVD，取它的行空间 `S`（与论文
Algorithm 1 / Appendix C 中通过 QR 得到正交投影等价）。对候选图像 `x`，
攻击者只执行：

```text
f(x) = frozen_CLIP_image_encoder(x)
f_rec(x) = projection_S(f(x))
r(x) = ||f(x) - f_rec(x)||_1
member iff r(x) < tau
```

实现使用论文的原始 L1 残差，默认 `tau=0.01`。为了使用统一的 ROC/AUC
接口，报告中的 `score=-r(x)`；这只是单调变换，不改变攻击排序。

### LoRA 映射

仓库的 LoRA 前向为 `W_eff = W_0 + scale * B A`，其中 `A` 的形状为
`(rank, hidden)`。对视觉注意力 Q 投影，客户端单步 SGD 上传满足：

```text
Delta_A = learning_rate * dL/dA
rowspan(dL/dA) is a subspace of the attacked-layer token inputs
```

实现从产生该上传之前的全局模型注册 forward hook，捕获进入 Q-LoRA 的
`[batch, tokens, hidden]` 表示。论文子空间由全部 token 共同形成；当前样本级
候选使用 CLIP class token 作为 `f(x)`，并将完整的
`batch_size * tokens_per_sample` 记入论文秩条件。服务端只读取真实上传的
`lora_A` 参数差，不构造代理梯度或合并后的稠密 `BA` 更新。

### 数值稳定性

实际 float32 SGD 上传通过 `W_before - W_after` 得到，参数相减可能产生理论梯度
中不存在的微小满秩方向。实现仍投影到同一个理论行空间，但使用以下等价的数值
处理避免 CUDA float32 SVD 失去正交性：

- 在 CPU float64 中执行 SVD，并沿用原始上传 dtype 的数值秩容差；
- 将使用秩限制为实际 batch size，因为 `rank(Delta^T X) <= batch_size`；
- 对选中的右奇异向量执行 QR，只更换同一子空间的正交基坐标。

这些步骤不修改原始 L1 残差、阈值、成员定义或分数方向。JSON 中的
`numerical_stabilization` 会记录秩上限、计算精度及 QR 前后的 Gram 误差。

## 严格威胁模型

- honest-but-curious 服务端；
- 只观察一个目标客户端的一次通信；
- 客户端只取一个真实 batch，执行一次无 momentum、无 weight decay 的
  vanilla SGD；
- 成员严格定义为“出现在产生该上传更新的 batch 中”，不是“曾经属于该
  客户端的数据集”；
- 不使用其他客户端更新、代理梯度、shadow model、学习型攻击头或成员标签
  来构造攻击子空间；
- CLIP 主干冻结；Visual Adapter 只读取首个 down-projection 权重更新，
  CLIP-MLP 对照模型只读取第一层分类 MLP 权重更新，CLIP-LoRA 只读取首个
  视觉 Q 投影的 `lora_A` 更新；
- 数据协议与对应正常训练保持一致：CLIP-MLP、Visual Adapter 和 CLIP-LoRA
  均使用完整训练集。

独立入口继续用于 CLIP-MLP 严格复现和 CLIP-LoRA/Visual Adapter 的单独诊断；
通用 `promptres` 是余弦代理攻击，不等同于本文的投影残差算法。正式 Visual
Adapter sweep 已将 ProjRes 接入共享多轮审计器，但不改变这里的投影残差公式。

## 统一训练入口中的真实轮次评估

论文第 2.1 节采用 FedSGD：客户端在通信轮 `t` 上传当前本地 batch 的梯度；
第 3.1 节要求的是目标客户端单个通信轮的上传梯度，而不是训练初始化时额外构造
一次更新。论文第 4.1 节的默认实验通常在第 50 个通信轮评估。

Visual Adapter 与 CLIP-LoRA 的统一 sweep 因此每 10 轮直接观察实际 one-batch
FedSGD 上传，ProjRes 与另外五种真实 Batch 攻击共享当轮 `N` 个成员和按标签匹配
的 `10N` 个非成员，并由共享审计器统一保存结果。独立诊断入口仍可显式选择观察
轮次。若独立入口观察的是包含
多个本地 batch 的 FedAvg 参数差，它是多步更新的累积，不再与论文的单-batch
FedSGD 梯度完全等价；结果元数据会明确记录这一差异。

文本 Adapter 的 rank cap 使用当轮 attention mask 中的有效 token 总数，而不是
`batch_size × padding_length`。因此 padding 不会人为扩大保留的梯度子空间；BERT
的 CLS、GPT2 的 last-token 候选表示仍保持不变，仅数值 SVD 的最大保留秩被收紧。

## 运行

```bash
bash scripts/run_projres_mlp_real.sh
```

单客户端示例：

```bash
python scripts/validate_projres_mlp_real.py \
  --config configs/clip_mlp_projres.yaml \
  --target-client 0 \
  --output results/projres_mlp_client0.json
```

Visual Adapter 单客户端示例：

```bash
python scripts/validate_projres_mlp_real.py \
  --config configs/visual_adapter_low_fpr_attacks.yaml \
  --target-client 0 \
  --output results/projres_visual_adapter_client0.json
```

输出是 JSON，不生成 HTML。ProjRes 采用 ranking-only 判断：保存原始 L1 残差及其
反向连续分数，通过 AUC 和可解析 FPR 下的 TPR 评价，不生成固定阈值二元预测。
结果还记录梯度秩、第一层维度，以及模拟上传梯度与参数更新的一致性误差。

独立入口默认使用当前上传梯度所对应 batch 中的全部成员候选，并从互斥非成员池
（所有客户端测试集和其他客户端训练集）中最多选取 20,000 个样本。非成员不得
少于 1000 个，因而 `TPR@0.1%FPR` 的经验 FPR 步长不大于 `1/1000`。可显式使用
`--max-nonmembers 0` 改为完整非成员池，或指定另一个不小于 1000 的上限。
Visual Adapter 与 CLIP-LoRA 正式统一 sweep 使用按真实 Batch 标签匹配的
`N/10N` 候选视图，不报告 `TPR@0.1%FPR`。无论哪条入口，目标客户端不在当前
batch 中的其他训练样本
都不能算作严格威胁模型下的成员。

## 适用边界

论文给出的有利秩条件在这里是 `p <= m` 且 `p < n`：batch 大小 `p` 不超过
第一层输出维度 `m`，并小于 CLIP 表示维度 `n`。Visual Adapter 的默认
down-projection 为 `n=512, m=128`，实际 `p` 是目标客户端首个 batch 的大小。
若改成多 batch、本地多步、动量、裁剪、噪声或安全聚合，上传更新不再是单个
batch 梯度的常数倍，就不再属于本入口声明的严格实验设置。

CLIP-LoRA 中 `p` 是 `batch_size * tokens_per_sample`，`m` 是 LoRA rank，
`n` 是被攻击注意力层的 hidden size。默认结果会明确写入
`paper_favorable_rank_condition`；当 rank 较小导致该条件不成立时，仍可计算
投影残差和 AUC，但不能宣称获得论文定理保证。
