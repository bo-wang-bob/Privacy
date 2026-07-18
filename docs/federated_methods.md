# 联邦提示训练方法

## PromptFL (`aggregator: promptfl`)

对应论文 *PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models*。每个客户端在冻结 CLIP 上只训练一组共享 CoOp context token，并最小化本地交叉熵；服务器按客户端训练样本数执行 FedAvg。

历史 `fedavg` 入口已经具备相同的训练与聚合骨架，但它把学习 token 拼接在完整手工模板之前。严格的 `promptfl` 入口使用论文式 `[SOS] [learned context] [class] [EOS]` 构造，因此保留 `fedavg` 以兼容旧实验，论文对照应使用 `promptfl`。

论文：[arXiv](https://arxiv.org/abs/2208.11625)。

## FedOTP (`aggregator: fedotp`)

对应 CVPR 2024 论文 *Global and Local Prompts Cooperation via Optimal Transport for Federated Learning*。

- 每个客户端持久保留一个本地完整 prompt，并训练一个服务器同步的全局完整 prompt；
- 图像 patch token 与全局/本地文本特征构成代价矩阵，用熵正则部分最优传输求固定运输计划，再由 Wasserstein 相似度计算分类损失；
- 客户端只上传 `global_ctx`，服务器按样本数聚合；`local_ctx` 不通信且跨轮保留。

主要参数为 `epsilon`、`transported_mass`、`max_iterations` 和 `threshold`。论文：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Global_and_Local_Prompts_Cooperation_via_Optimal_Transport_for_Federated_CVPR_2024_paper.html)。官方代码：[FedOTP](https://github.com/HongxiaLee/FedOTP)。

## FedPGP (`aggregator: fedpgp`)

对应 ICML 2024 论文 *FedPGP: Federated Personalized Global Prompt for Vision-Language Models*。

- 每个客户端使用服务器同步的 `global_ctx`，并持久保留低秩个性化项 `U·V`；
- 分类使用 `global_ctx + U·V`；训练目标为交叉熵加 prompt-wise 对比损失，其中冻结的初始占位符 prompt 文本特征是正锚点，个性化 prompt 是负项；
- 只聚合 `global_ctx`，低秩因子不上传。

主要参数为 `rank`、`contrastive_weight` 和 `temperature`。论文：[PMLR](https://proceedings.mlr.press/v235/cui24c.html)。官方代码：[FedPGP](https://github.com/TianyuCuiOvO/FedPGP)。

## DP-FPL (`aggregator: dpfpl`)

对应 ICLR 2025 论文 *Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models*。

- 实际提示为 `global_ctx + local_ctx`。全局提示由服务器同步，本地提示按客户端跨轮持久保存。
- 每个私有步骤使用一次随机 power iteration 得到正交低秩因子 `U,V` 和残差 `R`，前向使用 `global + U·V + R`。
- 对每个样本分别计算并裁剪 `global/U/V` 梯度；只向 `U/V` 梯度加入局部高斯噪声，再按 RGP 公式重构完整本地梯度。
- 客户端协议消息只包含裁剪后的全局提示梯度。服务器按样本数加权时，GDP 噪声敏感度使用最大客户端权重，而不是错误地假设所有权重均为 `1/N`。

主要参数：

- `rank`、`local_steps`：低秩和每轮私有本地步数；
- `local_clip_norm`、`global_clip_norm`：逐样本本地裁剪和客户端全局更新裁剪；
- `local_noise_multiplier`、`global_noise_multiplier`：噪声标准差与对应敏感度之比；
- `local_target_epsilon`、`global_target_epsilon`：可选目标预算；设置后程序使用保守 Gaussian RDP accountant 反推噪声；
- `delta`；
- `reproducible_dp_noise`：仅测试时使用。默认 `false`，使用不写入结果文件的 OS 随机种子。

论文：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4431224d3762aa655f0aee4eaf04ff16-Abstract-Conference.html)。官方代码：[Privacy-Preserving-Paper1](https://github.com/coderanik/Privacy-Preserving-Paper1)。本仓库依据论文算法及该实现，将个性化全局/本地提示和 RGP 私有更新适配到冻结 CLIP soft prompt。

## FedASK (`aggregator: fedask`)

对应 NeurIPS 2025 论文 *Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix*。

- soft prompt 写成 `base_ctx + scaling·B·A`；`scaling=1` 等价于令原论文 LoRA 的 `α/r=1`。
- 客户端固定 `A`，计算完整 prompt 矩阵的逐样本梯度，经裁剪和高斯扰动后右乘 `A^T` 更新 `B`。
- 协议消息为 `Y_i=B_i(A_iΩ)` 和 `P_i=A_i^T(B_i^TQ)`，而不是完整客户端 `A/B`。
- 服务器对加权草图执行 QR、SVD 和 rank 截断，同时记录截断前误差、最终误差、客户端 `B` 子空间秩以及由秩宽度给出的最低 oversampling 诊断。该宽度条件不是重构精度保证，实际结果以两项重构误差为准。

主要参数：

- `rank`、`oversampling`、`scaling`、`local_steps`；
- `clip_norm`、`noise_multiplier`；
- `target_epsilon`：可选目标预算；
- `delta`、`reproducible_dp_noise`。

论文：[NeurIPS Proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/a686ddca183f72ee9f3f04896eb11bcb-Abstract-Conference.html)。官方代码：[PrivacyFedLLM](https://github.com/FLEECERmw/PrivacyFedLLM)。两阶段草图和 SVD 恢复遵循官方实现。

## 隐私会计

`federated_method_summary.json` 报告保守 Gaussian RDP 上界，不使用客户端抽样或数据抽样带来的隐私放大，因此不会虚报更小的 ε。DP-FPL 分别报告本地 LDP 和服务器 GDP；FedASK 报告本地 DP-SGD 的端到端保守上界。`nasr_active` 与 `promptmia` 对目标客户端发起的额外私有更新查询也会计入校准和最终 ε。

当噪声乘数为零，或显式开启 `reproducible_dp_noise` 并把确定性种子用于测试时，摘要中的 `formal_dp_enabled` 为 `false`。

SOFT 和 CoFedMID 含有未由该 accountant 覆盖的数据依赖样本选择；它们与 DP-FPL/FedASK 同时运行时仍报告便于比较的数值上界，但 `formal_dp_enabled` 会设为 `false`，避免把实验性组合误称为形式化 DP 保证。

## 攻击可见性

`audit.audit_view` 支持：

- `protocol_plus_released_prompts`（默认）：更新攻击只能使用真实协议消息，同时允许攻击公开发布的 prompt 检查点；
- `released_prompt`：不使用通信更新，只审计公开 prompt；
- `full_whitebox`：允许完整内部客户端状态，用作强攻击上界。

DP-FPL 的协议视图不会暴露 `local_ctx` 更新；FedASK 的协议视图只使用两阶段草图。审计摘要会保存所用视图和具体威胁模型。

## 与五种防御组合

DP-FPL/FedASK 不再先完成原生训练、再额外执行一遍普通防御训练：

- HAMP、SOFT、CoFedMID 的目标或样本变换进入同一次逐样本裁剪/加噪流水线；
- Prompt-DP 通过更严格的裁剪阈值和更大的噪声乘数组合，不增加普通 SGD 阶段；
- MIST 保留反事实第二阶段，但该阶段仍调用 DP-FPL/FedASK 自身的私有更新规则，并计入隐私 accountant。

因此“一个方法 + 一个防御”不会因无意中多跑完整本地 epoch 而获得不公平的训练预算，也不会在私有机制之后追加未保护的数据相关更新。
