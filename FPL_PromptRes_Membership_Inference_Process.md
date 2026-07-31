# 面向联邦提示学习的成员推理攻击：完整流程与公式推导

> **暂定方法名：PromptRes（Prompt-Update Residual Membership Inference）**
> 本文档给出一个面向 CLIP 文本侧联邦提示学习（Federated Prompt Learning, FPL）的被动成员推理攻击框架。核心目标是：服务器给定候选样本 \((x^\star,y^\star)\)，利用目标客户端上传的 soft-prompt 更新，判断该样本是否属于目标客户端的本地训练集。

---

## 1. 研究目标

传统联邦提示学习中，客户端不上传原始图像，也不更新完整视觉语言模型，只上传低维 soft prompt。尽管通信参数量较少，prompt 更新仍由本地样本的梯度驱动，因此可能隐式编码本地图像的成员信息。

本文研究如下成员推理问题：

\[
\mathcal{A}\!\left(
x^\star,y^\star,
p_G^{t-1},
\Delta p_i^t
\right)
\longrightarrow
\{0,1\},
\]

其中：

- \(x^\star\) 是候选图像；
- \(y^\star\) 是候选标签；
- \(p_G^{t-1}\) 是第 \(t\) 轮下发的全局 prompt；
- \(\Delta p_i^t=p_i^t-p_G^{t-1}\) 是客户端 \(i\) 上传的 prompt 更新；
- 输出 \(1\) 表示候选样本属于客户端 \(i\) 的训练集，输出 \(0\) 表示不属于。

本文默认攻击者为**诚实但好奇的服务器**：服务器正常执行联邦聚合协议，但会分析单个客户端上传的更新。

---

## 2. 联邦提示学习模型

### 2.1 模型组成

设系统中共有 \(N\) 个客户端。客户端 \(i\) 持有本地数据集

\[
D_i=\{(x_j^i,y_j^i)\}_{j=1}^{n_i}.
\]

每个客户端维护冻结的视觉语言模型，例如 CLIP：

- 图像编码器 \(g(\cdot)\)；
- 文本编码器 \(f(\cdot)\)；
- 可训练 soft prompt \(p\)。

设 prompt 包含 \(M\) 个上下文 token，每个 token 的维度为 \(d_p\)，则

\[
p\in\mathbb{R}^{M\times d_p}.
\]

将其向量化后记为

\[
\bar p=\operatorname{vec}(p)\in\mathbb{R}^{q},
\qquad
q=Md_p.
\]

对于类别 \(k\)，将 soft prompt 与类别名 \(c_k\) 组合后送入文本编码器。文本编码器首先产生未归一化特征

\[
u_k(p)=f_{\mathrm{raw}}(p,c_k)\in\mathbb{R}^{d},
\]

再执行 \(\ell_2\) 归一化：

\[
t_k(p)
=
\frac{u_k(p)}{\|u_k(p)\|_2}
\in\mathbb{R}^{d}.
\]

将 \(K\) 个类别的归一化文本特征按行排列：

\[
T(p)=
\begin{bmatrix}
t_1(p)^\top\\
t_2(p)^\top\\
\vdots\\
t_K(p)^\top
\end{bmatrix}
\in\mathbb{R}^{K\times d}.
\]

图像编码器同样输出归一化图像特征：

\[
z=g(x)\in\mathbb{R}^{d},
\qquad
\|z\|_2=1.
\]

---

### 2.2 分类概率

对于输入图像 \(x\)，类别 \(k\) 的 logit 为

\[
\ell_k(x;p)
=
\frac{t_k(p)^\top g(x)}{\tau_s},
\]

其中 \(\tau_s\) 是 CLIP 的温度参数。

预测概率为

\[
\pi_k(x;p)
=
\frac{
\exp\!\left(t_k(p)^\top g(x)/\tau_s\right)
}{
\sum_{j=1}^{K}
\exp\!\left(t_j(p)^\top g(x)/\tau_s\right)
}.
\]

对于标签 \(y\)，单样本交叉熵损失为

\[
\ell(p;x,y)
=
-\log \pi_y(x;p).
\]

客户端 \(i\) 的本地经验损失为

\[
\mathcal L_i(p)
=
\frac{1}{|D_i|}
\sum_{(x,y)\in D_i}
\ell(p;x,y).
\]

---

## 3. 联邦训练过程

在第 \(t\) 轮，服务器下发全局 prompt：

\[
p_G^{t-1}.
\]

客户端 \(i\) 初始化本地 prompt：

\[
p_{i,0}^{t}=p_G^{t-1}.
\]

假设客户端执行 \(S_i^t\) 个本地优化步骤。第 \(s\) 步使用 mini-batch \(\mathcal B_{i,s}^t\)，普通 SGD 更新为

\[
p_{i,s+1}^{t}
=
p_{i,s}^{t}
-
\eta_{i,s}^{t}
\nabla_p
\mathcal L_{i,s}^{t}
\left(p_{i,s}^{t}\right),
\]

其中

\[
\mathcal L_{i,s}^{t}(p)
=
\frac{1}{|\mathcal B_{i,s}^{t}|}
\sum_{(x,y)\in\mathcal B_{i,s}^{t}}
\ell(p;x,y).
\]

本地训练结束后，客户端上传

\[
p_i^t=p_{i,S_i^t}^{t}
\]

或者等价的 prompt 差值

\[
\Delta p_i^t
=
p_i^t-p_G^{t-1}.
\]

服务器执行 FedAvg：

\[
p_G^t
=
\sum_{i=1}^{N}
\frac{n_i}{\sum_{j=1}^{N}n_j}
p_i^t.
\]

---

## 4. 威胁模型与成员定义

### 4.1 攻击者能力

服务器知道：

1. 冻结的图像编码器 \(g(\cdot)\)；
2. 冻结的文本编码器 \(f(\cdot)\)；
3. 类别名称集合 \(\{c_k\}_{k=1}^{K}\)；
4. 全局 prompt \(p_G^{t-1}\)；
5. 每个客户端上传的 prompt 或 prompt 更新；
6. 优化器类型和训练超参数，或者至少知道其大致配置；
7. 候选样本 \((x^\star,y^\star)\)。

服务器不知道：

1. 客户端本地数据集；
2. 客户端当前 mini-batch；
3. 其他未上传的中间梯度；
4. 客户端本地样本顺序。

---

### 4.2 两种成员定义

#### 当前批次成员

\[
(x^\star,y^\star)\in\mathcal B_{i,s}^{t}.
\]

这种设置需要服务器观察单步或单 mini-batch 更新，最接近梯度级成员推理。

#### 本地数据集成员

\[
(x^\star,y^\star)\in D_i.
\]

这种设置允许客户端执行多个 batch 和多个 local epoch。候选样本可能只在部分本地步骤中出现，因此需要跨轮累积弱成员信号。实际 FPL 更适合这种定义。

---

## 5. 图像特征如何进入 prompt 更新

这是整个攻击的理论起点。

### 5.1 文本特征对 prompt 的 Jacobian

定义第 \(k\) 个归一化文本特征关于向量化 prompt 的 Jacobian：

\[
J_k(p)
=
\frac{\partial t_k(p)}
{\partial \bar p}
\in\mathbb{R}^{d\times q}.
\]

将全部类别 Jacobian 堆叠：

\[
J_T(p)
=
\frac{
\partial\operatorname{vec}(T(p))
}{
\partial \bar p
}
\in\mathbb{R}^{Kd\times q}.
\]

由于文本编码器冻结，服务器可以在已知 prompt 处通过自动微分计算 Jacobian-vector product 或 vector-Jacobian product，而不需要知道客户端数据。

---

### 5.2 单样本的文本特征梯度

定义分类误差系数

\[
e_k(x,y;p)
=
\pi_k(x;p)-\mathbb I(k=y).
\]

交叉熵关于第 \(k\) 个归一化文本特征的梯度为

\[
\frac{\partial\ell(p;x,y)}
{\partial t_k}
=
\frac{e_k(x,y;p)}{\tau_s}g(x).
\]

因此，单样本的文本特征梯度矩阵为

\[
H(x,y;p)
=
\nabla_T\ell(p;x,y)
=
\frac{1}{\tau_s}
e(x,y;p)g(x)^\top,
\]

其中

\[
e(x,y;p)
=
\begin{bmatrix}
e_1(x,y;p)\\
\vdots\\
e_K(x,y;p)
\end{bmatrix}
\in\mathbb R^K.
\]

注意 \(H(x,y;p)\) 是一个 rank-1 矩阵：

\[
\operatorname{rank}\big(H(x,y;p)\big)\leq 1.
\]

同时 softmax 误差满足

\[
\sum_{k=1}^{K}e_k(x,y;p)=0,
\]

所以

\[
\mathbf 1_K^\top H(x,y;p)=0.
\]

---

### 5.3 单样本 prompt 梯度

根据链式法则：

\[
\nabla_{\bar p}\ell(p;x,y)
=
J_T(p)^\top
\operatorname{vec}
\left(
H(x,y;p)
\right).
\]

展开后得到

\[
\boxed{
\nabla_{\bar p}\ell(p;x,y)
=
\frac{1}{\tau_s}
\sum_{k=1}^{K}
e_k(x,y;p)
J_k(p)^\top g(x)
}
\]

定义候选样本的 **prompt-gradient fingerprint**：

\[
a(x,y;p)
\triangleq
\nabla_{\bar p}\ell(p;x,y).
\]

于是

\[
\boxed{
a(x,y;p)
=
\frac{1}{\tau_s}
\sum_{k=1}^{K}
e_k(x,y;p)J_k(p)^\top g(x)
}
\]

该式表明：

> 图像特征先经过类别相关的文本 Jacobian 映射到 prompt 参数空间，再根据分类误差进行加权，最终形成单样本 prompt 梯度。

---

### 5.4 利用 softmax 的零和性质改写

由于

\[
\sum_{k=1}^{K}e_k=0,
\]

任选类别 \(1\) 作为参考，可得

\[
\sum_{k=1}^{K}e_kJ_k^\top g
=
\sum_{k=2}^{K}
e_k(J_k-J_1)^\top g.
\]

因此

\[
\boxed{
a(x,y;p)
=
\frac{1}{\tau_s}
\sum_{k=2}^{K}
e_k(x,y;p)
\left(J_k(p)-J_1(p)\right)^\top
g(x)
}
\]

这说明真正驱动 prompt 更新的是：

\[
\text{图像特征在不同类别文本响应之间的差异},
\]

而不是所有类别共享的公共方向。

---

### 5.5 文本特征归一化的影响

若

\[
t_k(p)
=
\frac{u_k(p)}{\|u_k(p)\|_2},
\]

则

\[
\frac{\partial t_k}{\partial u_k}
=
\frac{1}{\|u_k\|_2}
\left(
I-t_kt_k^\top
\right).
\]

令未归一化文本特征关于 prompt 的 Jacobian 为

\[
H_k^{\mathrm{raw}}(p)
=
\frac{\partial u_k(p)}
{\partial \bar p},
\]

则

\[
J_k(p)
=
\frac{1}{\|u_k(p)\|_2}
\left(
I-t_k(p)t_k(p)^\top
\right)
H_k^{\mathrm{raw}}(p).
\]

因此

\[
J_k(p)^\top g(x)
=
H_k^{\mathrm{raw}}(p)^\top
\frac{
I-t_k(p)t_k(p)^\top
}{
\|u_k(p)\|_2
}
g(x).
\]

这意味着 prompt 梯度主要编码图像特征相对于类别文本原型的**切向偏差**。

---

## 6. 一个 batch 下的梯度结构

设 mini-batch 大小为 \(B\)。将图像特征按行排列：

\[
G=
\begin{bmatrix}
g(x_1)^\top\\
\vdots\\
g(x_B)^\top
\end{bmatrix}
\in\mathbb R^{B\times d}.
\]

定义预测概率矩阵

\[
P\in\mathbb R^{B\times K},
\]

以及 one-hot 标签矩阵

\[
Y\in\mathbb R^{B\times K}.
\]

误差矩阵为

\[
E=P-Y.
\]

批次损失关于文本特征矩阵的梯度为

\[
\boxed{
\nabla_T\mathcal L_{\mathcal B}
=
\frac{1}{B\tau_s}
E^\top G
}
\]

记

\[
H_{\mathcal B}
=
\nabla_T\mathcal L_{\mathcal B}.
\]

则 prompt 梯度为

\[
\boxed{
\nabla_{\bar p}\mathcal L_{\mathcal B}
=
J_T(p)^\top
\operatorname{vec}(H_{\mathcal B})
}
\]

或写成逐样本形式：

\[
\boxed{
\nabla_{\bar p}\mathcal L_{\mathcal B}
=
\frac{1}{B}
\sum_{b=1}^{B}
a(x_b,y_b;p)
}
\]

进一步展开：

\[
\nabla_{\bar p}\mathcal L_{\mathcal B}
=
\frac{1}{B\tau_s}
\sum_{b=1}^{B}
\sum_{k=1}^{K}
e_{bk}
J_k(p)^\top g(x_b).
\]

---

## 7. 单步更新中的成员信息

若客户端只执行一步 SGD：

\[
\bar p_i
=
\bar p_G
-
\eta
\nabla_{\bar p}\mathcal L_{\mathcal B_i}
(\bar p_G),
\]

则上传更新为

\[
\Delta\bar p_i
=
-\eta
\nabla_{\bar p}\mathcal L_{\mathcal B_i}
(\bar p_G).
\]

定义服务器观测到的“正梯度方向”：

\[
u_i
\triangleq
-\Delta\bar p_i.
\]

则

\[
u_i
=
\frac{\eta}{B}
\sum_{(x_b,y_b)\in\mathcal B_i}
a(x_b,y_b;p_G).
\]

若候选样本 \((x^\star,y^\star)\) 是 batch 成员，则

\[
u_i
=
\frac{\eta}{B}
a^\star
+
\frac{\eta}{B}
\sum_{b\neq \star}
a_b,
\]

其中

\[
a^\star
=
a(x^\star,y^\star;p_G).
\]

若候选不是成员，则 \(u_i\) 中不存在确定的自相关项 \(a^\star\)。

---

## 8. 主攻击：候选梯度残差

### 8.1 候选指纹构造

服务器将候选样本输入冻结 CLIP，并在全局 prompt 处计算

\[
a_t^\star
=
\nabla_{\bar p}
\ell
\left(
p_G^{t-1};
x^\star,y^\star
\right).
\]

客户端更新转为

\[
u_i^t
=
-\operatorname{vec}
\left(
\Delta p_i^t
\right).
\]

如果学习率和单步更新已知，也可以使用

\[
\widehat g_i^t
=
-\frac{
\operatorname{vec}(\Delta p_i^t)
}{
\eta
}.
\]

由于后续分数对尺度不敏感，通常直接使用 \(u_i^t\) 即可。

---

### 8.2 一维非负投影

用候选指纹解释客户端更新：

\[
R_0
=
\|u_i^t\|_2^2.
\]

加入候选梯度方向后，最小残差为

\[
R_1
=
\min_{\alpha\geq 0}
\left\|
u_i^t-\alpha a_t^\star
\right\|_2^2.
\]

最优系数为

\[
\alpha^\star
=
\frac{
[\langle u_i^t,a_t^\star\rangle]_+
}{
\|a_t^\star\|_2^2+\epsilon
},
\]

其中

\[
[z]_+=\max(z,0).
\]

残差下降量为

\[
R_0-R_1
=
\frac{
[\langle u_i^t,a_t^\star\rangle]_+^2
}{
\|a_t^\star\|_2^2+\epsilon
}.
\]

定义单轮成员分数：

\[
\boxed{
s_{\mathrm{dir},i}^{t}
(x^\star,y^\star)
=
\frac{
[\langle u_i^t,a_t^\star\rangle]_+^2
}{
(\|u_i^t\|_2^2+\epsilon)
(\|a_t^\star\|_2^2+\epsilon)
}
}
\]

它可以理解为正方向余弦相似度的平方。

- 分数越大：候选梯度越能解释客户端上传更新；
- 分数越小：候选梯度与客户端更新无关或方向相反。

---

### 8.3 理论直觉

若候选是成员：

\[
\left\langle
u_i,a^\star
\right\rangle
=
\frac{\eta}{B}
\|a^\star\|_2^2
+
\frac{\eta}{B}
\sum_{b\neq\star}
\langle a_b,a^\star\rangle.
\]

若候选是非成员：

\[
\left\langle
u_i,a^\star
\right\rangle
=
\frac{\eta}{B}
\sum_{b}
\langle a_b,a^\star\rangle.
\]

两者的核心差异是成员样本多出自相关项：

\[
\boxed{
\frac{\eta}{B}
\|a^\star\|_2^2
}
\]

如果不同样本梯度之间的交叉内积在中心化后均值接近零，则成员分数在期望上更大。

---

## 9. 消除任务公共方向和数据异构性

普通余弦分数可能受到以下因素干扰：

- 全局任务公共优化方向；
- 高频类别方向；
- 非 IID 客户端分布；
- 不同客户端共有的 prompt 漂移；
- 优化器造成的公共尺度变化。

因此可以构造背景更新子空间。

### 9.1 背景更新中心

第 \(t\) 轮的平均更新为

\[
\mu^t
=
\frac{1}{N}
\sum_{j=1}^{N}
u_j^t.
\]

目标客户端的中心化更新：

\[
u_{i,c}^t
=
u_i^t-\mu^t.
\]

为了避免目标更新影响背景估计，也可以使用 leave-one-client-out 均值：

\[
\mu_{-i}^t
=
\frac{1}{N-1}
\sum_{j\neq i}
u_j^t.
\]

---

### 9.2 背景子空间

构造其他客户端的中心化更新矩阵：

\[
U_{-i}^t
=
\begin{bmatrix}
u_1^t-\mu_{-i}^t &
\cdots &
u_{i-1}^t-\mu_{-i}^t &
u_{i+1}^t-\mu_{-i}^t &
\cdots &
u_N^t-\mu_{-i}^t
\end{bmatrix}.
\]

执行截断 SVD：

\[
U_{-i}^t
=
Q^t\Sigma^t(V^t)^\top.
\]

取前 \(r\) 个左奇异向量：

\[
Q_r^t
\in\mathbb R^{q\times r}.
\]

背景投影矩阵为

\[
P_{\mathrm{bg}}^t
=
Q_r^t(Q_r^t)^\top.
\]

---

### 9.3 残差化

目标客户端更新的异常分量：

\[
\widetilde u_i^t
=
\left(
I-P_{\mathrm{bg}}^t
\right)
\left(
u_i^t-\mu_{-i}^t
\right).
\]

候选梯度的非公共分量：

\[
\widetilde a_t^\star
=
\left(
I-P_{\mathrm{bg}}^t
\right)
a_t^\star.
\]

定义异构性消除后的攻击分数：

\[
\boxed{
s_{\mathrm{res},i}^{t}
=
\frac{
[\langle
\widetilde u_i^t,
\widetilde a_t^\star
\rangle]_+^2
}{
(\|\widetilde u_i^t\|_2^2+\epsilon)
(\|\widetilde a_t^\star\|_2^2+\epsilon)
}
}
\]

该分数先去除其他客户端也普遍存在的方向，再检查目标客户端是否包含候选样本的独特梯度成分。

---

## 10. 多步本地训练下的更新

实际客户端通常执行多个本地步骤：

\[
\Delta \bar p_i^t
=
-
\sum_{s=0}^{S_i^t-1}
\eta_{i,s}^t
\nabla_{\bar p}
\mathcal L_{i,s}^t
\left(
p_{i,s}^t
\right).
\]

展开为

\[
\Delta \bar p_i^t
=
-
\sum_{s=0}^{S_i^t-1}
\frac{
\eta_{i,s}^t
}{
B_{i,s}^t\tau_s
}
\sum_{(x_b,y_b)\in\mathcal B_{i,s}^t}
\sum_{k=1}^{K}
e_{sbk}
J_k(p_{i,s}^t)^\top
g(x_b).
\]

因此最终上传 prompt 是以下因素的累积混合：

1. 多个本地 batch；
2. 多个 local epoch；
3. 不同步骤下的 prompt；
4. 不同步骤下的 Jacobian；
5. 不同样本对应的预测误差；
6. 可能存在的动量或 Adam 预条件。

这时单轮更新不能被视为某个 batch 的精确梯度，但本地数据集中的成员样本会在多个本地步骤和多个通信轮中重复影响 prompt。

---

## 11. 跨轮成员分数

在第 \(t\) 轮，服务器重新基于当前全局 prompt 计算候选指纹：

\[
a_t^\star
=
\nabla_{\bar p}
\ell
\left(
p_G^{t-1};
x^\star,y^\star
\right).
\]

得到单轮分数 \(s_i^t\) 后，进行跨轮聚合：

\[
\boxed{
S_i(x^\star,y^\star)
=
\sum_{t\in\mathcal R_i}
w_t
s_i^t(x^\star,y^\star)
}
\]

其中 \(\mathcal R_i\) 表示客户端 \(i\) 参与的通信轮集合，并满足

\[
w_t\geq 0,
\qquad
\sum_{t\in\mathcal R_i}w_t=1.
\]

一种简单选择是均匀权重：

\[
w_t
=
\frac{1}{|\mathcal R_i|}.
\]

也可以根据信号强度设置：

\[
w_t
=
\frac{
\|\widetilde a_t^\star\|_2
}{
\sum_{r\in\mathcal R_i}
\|\widetilde a_r^\star\|_2
}.
\]

或者根据客户端更新幅度设置：

\[
w_t
=
\frac{
\|\widetilde u_i^t\|_2
}{
\sum_{r\in\mathcal R_i}
\|\widetilde u_i^r\|_2
}.
\]

成员样本会跨轮稳定地产生正对齐，而非成员的偶然对齐通常不会长期持续。

---

## 12. 增强攻击：恢复文本特征梯度

直接 prompt-gradient residual 最容易实现，但不能直接显式展示本地图像语义子空间。可以进一步将 prompt 更新“提升”回文本特征梯度空间。

### 12.1 理想关系

单步 SGD 下：

\[
u_i
=
-\frac{\Delta \bar p_i}{\eta}
=
J_T(p_G)^\top
\operatorname{vec}(H_i),
\]

其中

\[
H_i
=
\nabla_T\mathcal L_i
=
\frac{1}{B\tau_s}
E_i^\top G_i.
\]

服务器知道 \(J_T(p_G)\)，但不知道 \(H_i\)。

---

### 12.2 欠定逆问题

服务器可以求解：

\[
\widehat H_i
=
\arg\min_H
\left\|
J_T(p_G)^\top
\operatorname{vec}(H)
-u_i
\right\|_2^2.
\]

由于通常 \(Kd>Md_p\)，该问题可能欠定。加入结构约束：

\[
\boxed{
\widehat H_i
=
\arg\min_H
\left\|
J_T^\top
\operatorname{vec}(H)
-u_i
\right\|_2^2
+
\lambda_F\|H\|_F^2
+
\lambda_\ast\|H\|_\ast
}
\]

约束：

\[
\mathbf 1_K^\top H=0.
\]

其中：

- \(\|H\|_F\) 是 Frobenius 范数；
- \(\|H\|_\ast\) 是核范数，用于鼓励低秩；
- \(\mathbf1_K^\top H=0\) 来自 softmax 误差和为零。

也可以只计算最小范数解：

\[
\operatorname{vec}(\widehat H_i)
=
J_T
\left(
J_T^\top J_T+\lambda I
\right)^{-1}
u_i.
\]

这实际上恢复的是文本特征梯度在 prompt 可控制子空间中的投影，而不是完整真实梯度。

---

## 13. 文本特征梯度与图像特征子空间

对于一个 batch：

\[
H_i
=
\frac{1}{B\tau_s}
E_i^\top G_i.
\]

因此必然有

\[
\operatorname{RowSpan}(H_i)
\subseteq
\operatorname{RowSpan}(G_i).
\]

如果满足

\[
\operatorname{rank}(E_i)=B
\]

且

\[
\operatorname{rank}(G_i)=B,
\]

则

\[
\boxed{
\operatorname{RowSpan}(H_i)
=
\operatorname{RowSpan}(G_i)
}
\]

这是因为 \(E_i^\top\in\mathbb R^{K\times B}\) 满列秩时存在左逆矩阵 \(L\)，满足

\[
LE_i^\top=I_B,
\]

从而

\[
G_i
=
B\tau_s\,L H_i.
\]

另一方面，

\[
H_i
=
\frac{1}{B\tau_s}
E_i^\top G_i
\]

说明 \(H_i\) 的行本来就是 \(G_i\) 各行的线性组合。因此两个行空间相等。

由于

\[
\operatorname{rank}(E_i)\leq K-1,
\]

完整保留 batch 图像子空间至少要求

\[
\boxed{
B\leq K-1
}
\]

并且

\[
B\leq d.
\]

---

## 14. 图像投影残差攻击

对恢复的文本梯度执行 SVD：

\[
\widehat H_i
=
U_i\Sigma_iV_i^\top.
\]

取前 \(r\) 个右奇异向量：

\[
V_{i,r}\in\mathbb R^{d\times r}.
\]

它们构成恢复的视觉语义子空间。候选图像特征为

\[
z^\star=g(x^\star).
\]

将其投影到该子空间：

\[
\widehat z^\star
=
V_{i,r}V_{i,r}^\top z^\star.
\]

投影残差为

\[
\boxed{
r_{\mathrm{proj},i}
(x^\star)
=
\left\|
z^\star-
V_{i,r}V_{i,r}^\top z^\star
\right\|_2
}
\]

由于 \(\|z^\star\|_2=1\)，也可以使用投影能量：

\[
\boxed{
s_{\mathrm{proj},i}
(x^\star)
=
\left\|
V_{i,r}^\top z^\star
\right\|_2^2
=
1-r_{\mathrm{proj},i}^2
}
\]

- 成员：图像特征更可能位于恢复子空间中，投影能量较高；
- 非成员：图像特征与恢复子空间匹配较弱，残差较大。

---

## 15. Rank-1 候选原子匹配

候选样本对应的理想文本特征梯度为

\[
H^\star
=
\frac{1}{\tau_s}
\left(
\pi(x^\star;p_G)-e_{y^\star}
\right)
g(x^\star)^\top,
\]

其中 \(e_{y^\star}\) 是标签 \(y^\star\) 的 one-hot 向量。

定义 Frobenius 对齐分数：

\[
\boxed{
s_{\mathrm{atom},i}
=
\frac{
[
\langle
\widehat H_i,
H^\star
\rangle_F
]_+^2
}{
(\|\widehat H_i\|_F^2+\epsilon)
(\|H^\star\|_F^2+\epsilon)
}
}
\]

其中

\[
\langle A,B\rangle_F
=
\operatorname{Tr}(A^\top B).
\]

该分数判断候选样本的 rank-1 文本梯度原子是否能够解释恢复出的客户端文本梯度。

---

## 16. 综合攻击分数

可以组合三类信号：

1. Prompt 参数空间直接对齐：

\[
s_{\mathrm{dir}}.
\]

2. 恢复文本梯度中的候选原子对齐：

\[
s_{\mathrm{atom}}.
\]

3. 图像嵌入对子空间的投影能量：

\[
s_{\mathrm{proj}}.
\]

综合分数：

\[
\boxed{
S_{\mathrm{final}}
=
\alpha
S_{\mathrm{dir}}
+
\beta
S_{\mathrm{atom}}
+
\gamma
S_{\mathrm{proj}}
}
\]

其中

\[
\alpha,\beta,\gamma\geq 0,
\qquad
\alpha+\beta+\gamma=1.
\]

在第一版实现中，建议仅使用

\[
S_{\mathrm{dir}}
\]

或

\[
S_{\mathrm{res}},
\]

因为它们不需要求解 Jacobian 逆问题。恢复文本梯度与图像投影残差可以作为增强模块和消融实验。

---

## 17. 标签未知场景

如果候选标签未知，服务器可以遍历全部类别：

\[
a_t^\star(y)
=
\nabla_{\bar p}
\ell
\left(
p_G^{t-1};
x^\star,y
\right),
\qquad
y\in\{1,\dots,K\}.
\]

最终使用最大分数：

\[
\boxed{
S_i(x^\star)
=
\max_{y\in\{1,\dots,K\}}
S_i(x^\star,y)
}
\]

或者按照模型预测概率进行加权：

\[
S_i(x^\star)
=
\sum_{y=1}^{K}
\pi_y(x^\star;p_G)
S_i(x^\star,y).
\]

需要注意，取最大值会增加假阳性，因此阈值必须在同样的 label-unknown 设置下校准。

---

## 18. 阈值校准

服务器准备一组确定为非成员的参考样本：

\[
D_{\mathrm{ref}}^{\mathrm{non}}.
\]

计算其攻击分数：

\[
\left\{
S_i(x,y):
(x,y)\in
D_{\mathrm{ref}}^{\mathrm{non}}
\right\}.
\]

给定目标假阳性率 \(\rho\)，阈值取为非成员分数的 \(1-\rho\) 分位数：

\[
\boxed{
\tau_{\mathrm{mia}}
=
Q_{1-\rho}
\left(
S_i(D_{\mathrm{ref}}^{\mathrm{non}})
\right)
}
\]

成员判断为

\[
\widehat m_i(x^\star,y^\star)
=
\mathbb I
\left[
S_i(x^\star,y^\star)
>
\tau_{\mathrm{mia}}
\right].
\]

也可以使用影子客户端估计成员和非成员分数分布，训练简单的 logistic regression，但第一版方法应尽量保留无影子模型特性。

---

## 19. 完整攻击流程

### 阶段 A：正常联邦训练

1. 服务器向客户端发送 \(p_G^{t-1}\)；
2. 客户端在本地数据上优化 prompt；
3. 客户端上传 \(p_i^t\) 或 \(\Delta p_i^t\)；
4. 服务器保存目标客户端逐轮更新。

### 阶段 B：构造候选 prompt-gradient fingerprint

对候选 \((x^\star,y^\star)\)：

1. 计算图像特征

   \[
   z^\star=g(x^\star);
   \]

2. 用全局 prompt 计算所有类别文本特征

   \[
   T(p_G^{t-1});
   \]

3. 计算预测误差

   \[
   e^\star
   =
   \pi(x^\star;p_G^{t-1})
   -
   e_{y^\star};
   \]

4. 对单样本交叉熵反向传播，得到

   \[
   a_t^\star
   =
   \nabla_{\bar p}
   \ell
   \left(
   p_G^{t-1};
   x^\star,y^\star
   \right).
   \]

### 阶段 C：直接残差打分

1. 令

   \[
   u_i^t=-\operatorname{vec}(\Delta p_i^t);
   \]

2. 可选地去除背景更新子空间；
3. 计算单轮残差下降分数；
4. 跨轮聚合得到最终分数。

### 阶段 D：增强的 Jacobian 提升

1. 计算 \(J_T(p_G^{t-1})\)；
2. 从 prompt 更新近似恢复 \(\widehat H_i^t\)；
3. 对 \(\widehat H_i^t\) 做 SVD；
4. 计算候选图像投影残差或 rank-1 原子对齐；
5. 与直接分数组合。

### 阶段 E：成员决策

将最终分数与阈值比较：

\[
S_i(x^\star,y^\star)
>
\tau_{\mathrm{mia}}
\Rightarrow
\text{member}.
\]

---

## 20. 算法伪代码

```text
Algorithm: PromptRes for Federated Prompt Learning

Input:
    Frozen image encoder g
    Frozen text encoder f
    Global prompts {p_G^{t-1}}
    Target-client prompt updates {Δp_i^t}
    Candidate sample (x*, y*)
    Attack rounds R
    Background rank r
    Decision threshold τ_mia

Output:
    Membership prediction m_hat

for each round t in R do
    1. Compute candidate image embedding:
           z* = normalize(g(x*))

    2. Reconstruct class text features using p_G^{t-1}

    3. Compute candidate prompt-gradient fingerprint:
           a_t* = ∇_p ℓ(p_G^{t-1}; x*, y*)

    4. Convert client update to gradient direction:
           u_i^t = -vec(Δp_i^t)

    5. Optional background removal:
           estimate Q_r^t from other clients' updates
           u_tilde = (I - Q_r^t Q_r^{tT})(u_i^t - μ_{-i}^t)
           a_tilde = (I - Q_r^t Q_r^{tT})a_t*
       Otherwise:
           u_tilde = u_i^t
           a_tilde = a_t*

    6. Compute direct residual score:
           s_t =
           [<u_tilde, a_tilde>_+]^2
           /
           ((||u_tilde||_2^2 + ε)(||a_tilde||_2^2 + ε))

end for

7. Aggregate across rounds:
       S = Σ_t w_t s_t

8. Membership decision:
       m_hat = 1 if S > τ_mia else 0

return m_hat
```

---

## 21. PyTorch 实现要点

### 21.1 候选梯度

```python
prompt = global_prompt.detach().clone().requires_grad_(True)

image_feat = image_encoder(x_star)
image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

text_feat = build_normalized_text_features(
    prompt=prompt,
    class_names=class_names,
    text_encoder=text_encoder,
)

logits = image_feat @ text_feat.T / temperature
loss = torch.nn.functional.cross_entropy(
    logits,
    y_star,
)

(candidate_grad,) = torch.autograd.grad(
    loss,
    prompt,
    retain_graph=False,
    create_graph=False,
)

a_star = candidate_grad.flatten()
```

### 21.2 客户端更新

```python
u_client = -(local_prompt - global_prompt).flatten()
```

### 21.3 直接分数

```python
eps = 1e-12
dot = torch.dot(u_client, a_star)
positive_dot = torch.clamp(dot, min=0.0)

score = positive_dot.pow(2) / (
    (u_client.pow(2).sum() + eps)
    * (a_star.pow(2).sum() + eps)
)
```

### 21.4 背景子空间残差化

```python
# updates: [num_clients, prompt_dim]
background_updates = updates[other_client_indices]
mu = background_updates.mean(dim=0, keepdim=True)
centered = background_updates - mu

# centered.T: [prompt_dim, num_other_clients]
U, S, Vh = torch.linalg.svd(
    centered.T,
    full_matrices=False,
)

Q = U[:, :rank_r]

u_res = u_client - Q @ (Q.T @ u_client)
a_res = a_star - Q @ (Q.T @ a_star)
```

---

## 22. 优化器影响

### 22.1 SGD

单步 SGD 下：

\[
-\Delta p
=
\eta\nabla_p\mathcal L,
\]

关系最清晰。

### 22.2 Momentum SGD

动量更新为

\[
m_s
=
\beta m_{s-1}
+
\nabla_p\mathcal L_s,
\]

\[
p_{s+1}
=
p_s-\eta m_s.
\]

最终上传更新包含历史梯度的指数加权和。服务器若知道 \(\beta\) 和初始动量状态，可以近似反演；否则应把 \(-\Delta p\) 视为平滑后的梯度轨迹。

### 22.3 Adam

Adam 使用

\[
m_s
=
\beta_1m_{s-1}
+
(1-\beta_1)g_s,
\]

\[
v_s
=
\beta_2v_{s-1}
+
(1-\beta_2)g_s^2,
\]

\[
p_{s+1}
=
p_s
-
\eta
\frac{\widehat m_s}
{\sqrt{\widehat v_s}+\epsilon}.
\]

此时更新方向是经过坐标级预条件的梯度，直接余弦对齐会受到影响。可以：

1. 在实验中先使用 SGD 验证基本泄漏；
2. 再测试 Adam 的鲁棒性；
3. 使用同一优化器计算候选的模拟一步更新；
4. 或使用更新白化降低坐标缩放影响。

---

## 23. 实验设计

### 23.1 基础配置

可采用：

- CLIP ViT-B/32；
- 20 个客户端；
- prompt 长度 \(M=32\)；
- batch size 32；
- local epochs 2；
- 40 个通信轮；
- Dirichlet \(\alpha=0.1\)；
- 16-shot per class；
- 3 次独立运行取平均。

### 23.2 数据集

建议优先使用类别数较多的数据集：

- Caltech101；
- CIFAR100；
- Flowers102；
- DTD；
- OxfordPets；
- Food101。

对于图像子空间恢复版本，类别数 \(K\) 应重点考虑，因为完整秩条件要求

\[
B\leq K-1.
\]

### 23.3 成员与非成员采样

为了避免攻击只识别类别或数据分布：

1. member 来自目标客户端训练集；
2. non-member 来自测试集或其他客户端；
3. member 与 non-member 尽量同类别；
4. 按样本损失或置信度进行难度匹配；
5. 保证候选样本不重复；
6. 对每个客户端分别评估。

### 23.4 评价指标

建议报告：

\[
\mathrm{AUC},
\]

\[
\mathrm{Attack\ Accuracy},
\]

\[
\mathrm{TPR@1\%FPR},
\]

\[
\mathrm{TPR@0.1\%FPR},
\]

\[
\mathrm{Precision},
\qquad
\mathrm{F1}.
\]

隐私研究中，低假阳性率下的 TPR 尤其重要。

---

## 24. 消融实验

至少包含：

1. 单轮与多轮聚合；
2. 是否去除背景子空间；
3. 背景秩 \(r\)；
4. batch size；
5. local epochs；
6. prompt 长度 \(M\)；
7. 客户端数量；
8. 类别数 \(K\)；
9. Dirichlet 参数 \(\alpha\)；
10. SGD、Momentum 和 Adam；
11. label-known 与 label-unknown；
12. 直接 prompt 对齐、文本梯度原子对齐和图像投影残差；
13. Jacobian 正则参数；
14. 候选难度匹配策略；
15. 梯度裁剪、差分隐私和安全聚合。

---

## 25. 方法边界与局限

### 25.1 多步更新稀释

随着 batch size 和 local epochs 增大，单个成员样本在最终更新中的权重下降：

\[
\text{member contribution}
\propto
\frac{1}{B\times S}.
\]

因此需要跨轮累积或更强的背景消除。

### 25.2 数据异构性混淆

同类别非成员可能与目标客户端更新高度对齐，因为客户端数据分布本身会形成稳定方向。必须使用同类 non-member、背景子空间消除或客户端历史基线。

### 25.3 Prompt Jacobian 压缩

从图像特征到 prompt 更新要经过

\[
J_T^\top,
\]

因此某些视觉方向可能落入 Jacobian 的零空间，无法从 prompt 更新观察。

### 25.4 归一化造成信息损失

文本特征归一化移除了径向分量，prompt 更新主要反映切向语义变化。

### 25.5 安全聚合

若服务器只能看到所有客户端更新的聚合结果，而不能看到单个客户端更新，则客户端级成员推理显著困难，本文威胁模型不再直接适用。

---

## 26. 推荐的实现顺序

### 第一阶段：可行性验证

实现单步 SGD、单 batch：

\[
s_{\mathrm{dir}}
=
\operatorname{PosCos}^2
\left(
-\Delta p_i,
\nabla_p\ell(x^\star,y^\star)
\right).
\]

目标：验证 prompt 更新是否包含明显成员信号。

### 第二阶段：真实 FPL

加入：

- 多 batch；
- 2 local epochs；
- 40 轮；
- 多轮聚合；
- 同类难度匹配；
- 背景子空间消除。

目标：验证本地数据集级成员推理。

### 第三阶段：语义提升

从 prompt 更新恢复

\[
\widehat H_i,
\]

并测试：

\[
s_{\mathrm{atom}},
\qquad
s_{\mathrm{proj}}.
\]

目标：建立图像特征、文本梯度和 prompt 更新之间更强的语义解释。

---

## 27. 可用于论文的方法概述

> We propose PromptRes, a passive membership inference attack tailored to text-side federated prompt learning. PromptRes characterizes each candidate sample by its prompt-gradient fingerprint, which is obtained by backpropagating the candidate loss through the frozen text encoder to the shared soft prompt. Since a client’s uploaded prompt update is an accumulated mixture of its local sample gradients, a member candidate contributes a deterministic self-correlation component to the observed update. PromptRes therefore measures how much the candidate fingerprint reduces the residual of the client update after removing task-common background directions. To further expose the underlying visual semantics, PromptRes optionally lifts the prompt update into the normalized text-feature-gradient space through the text encoder Jacobian and performs rank-one gradient-atom matching or image-feature projection-residual inference. Multi-round aggregation is used to accumulate weak but persistent membership evidence under multiple local steps and non-IID client data.

---

## 28. 核心公式总结

图像特征：

\[
z=g(x).
\]

归一化文本特征：

\[
t_k(p)
=
\frac{u_k(p)}{\|u_k(p)\|_2}.
\]

单样本文本特征梯度：

\[
H(x,y;p)
=
\frac{1}{\tau_s}
\left(
\pi(x;p)-e_y
\right)
g(x)^\top.
\]

单样本 prompt-gradient fingerprint：

\[
\boxed{
a(x,y;p)
=
J_T(p)^\top
\operatorname{vec}
\left(
H(x,y;p)
\right)
}
\]

等价展开：

\[
\boxed{
a(x,y;p)
=
\frac{1}{\tau_s}
\sum_{k=1}^{K}
\left(
\pi_k(x;p)-\mathbb I(k=y)
\right)
J_k(p)^\top g(x)
}
\]

单步客户端更新：

\[
\boxed{
-\Delta p_i
=
\frac{\eta}{B}
\sum_{(x_b,y_b)\in\mathcal B_i}
a(x_b,y_b;p_G)
}
\]

直接成员分数：

\[
\boxed{
s_{\mathrm{dir}}
=
\frac{
[\langle-\Delta p_i,a^\star\rangle]_+^2
}{
(\|\Delta p_i\|_2^2+\epsilon)
(\|a^\star\|_2^2+\epsilon)
}
}
\]

批次文本特征梯度：

\[
\boxed{
H_i
=
\frac{1}{B\tau_s}
E_i^\top G_i
}
\]

满秩条件下：

\[
\boxed{
\operatorname{RowSpan}(H_i)
=
\operatorname{RowSpan}(G_i)
}
\]

图像投影残差：

\[
\boxed{
r_{\mathrm{proj}}
=
\left\|
g(x^\star)
-
V_rV_r^\top g(x^\star)
\right\|_2
}
\]

跨轮聚合：

\[
\boxed{
S_i(x^\star,y^\star)
=
\sum_t
w_t
s_i^t(x^\star,y^\star)
}
\]

---

## 29. 参考工作

- Guilin Deng et al., *Toward Efficient Membership Inference Attacks against Federated Large Language Models: A Projection Residual Approach*, arXiv:2604.21197, 2026.
- PromptFL: traditional federated prompt learning based on CLIP prompt tuning.
- Standard passive membership inference and privacy auditing literature in federated learning.
