# CLIP-MLP 与 Visual Adapter 上的严格 ProjRes 实现

本实现对应论文 *Toward Efficient Membership Inference Attacks against
Federated Large Language Models: A Projection Residual Approach* 的 Algorithm
1。论文中的 adapter down-projection 层在 Visual Adapter 中直接对应
`adapter.net.0.weight`；在 CLIP-MLP 对照模型中对应第一层分类 MLP
`classifier.0.weight`。

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
- CLIP 全冻结；Visual Adapter 只读取首个 down-projection 权重更新，
  CLIP-MLP 对照模型只读取第一层分类 MLP 权重更新；
- 数据协议与对应正常训练保持一致：CLIP-MLP 使用完整数据集，Visual
  Adapter 使用系统级 16-shot FPL 设置。

这也是为什么该入口独立于仓库的通用多轮审计器。通用 `promptres` 是余弦
代理攻击，不等同于本文的投影残差算法。

## 运行

```bash
bash scripts/run_projres_mlp_real.sh
```

单客户端示例：

```bash
python scripts/validate_projres_mlp_real.py \
  --config configs/clip_mlp_projres.yaml \
  --target-client 0 \
  --threshold 0.01 \
  --output results/projres_mlp_client0.json
```

Visual Adapter 单客户端示例：

```bash
python scripts/validate_projres_mlp_real.py \
  --config configs/visual_adapter_low_fpr_attacks.yaml \
  --target-client 0 \
  --threshold 0.01 \
  --output results/projres_visual_adapter_client0.json
```

输出是 JSON，不生成 HTML。除 AUC/低 FPR TPR 外，还记录原始残差、固定阈值
准确率、梯度秩、第一层维度，以及 `update / learning_rate` 与 autograd 梯度
的一致性误差。

默认评估使用当前上传梯度所对应 batch 中的全部成员候选，并流式编码完整的互斥
非成员池（所有客户端测试集和其他客户端训练集）。非成员不得少于 1000 个，因而
`TPR@0.1%FPR` 的经验 FPR 步长不大于 `1/1000`。可用
`--max-nonmembers 0` 保持完整非成员池，或指定一个不小于 1000 的上限。目标
客户端不在当前 batch 中的其他训练样本不能算作本文严格威胁模型下的成员。

## 适用边界

论文给出的有利秩条件在这里是 `p <= m` 且 `p < n`：batch 大小 `p` 不超过
第一层输出维度 `m`，并小于 CLIP 表示维度 `n`。Visual Adapter 的默认
down-projection 为 `n=512, m=128`，实际 `p` 是目标客户端首个 batch 的大小。
若改成多 batch、本地多步、动量、裁剪、噪声或安全聚合，上传更新不再是单个
batch 梯度的常数倍，就不再属于本入口声明的严格实验设置。
