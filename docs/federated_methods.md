# FedAvg 替代方法

## DP-FPL (`aggregator: dpfpl`)

对应论文：*Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models*，ICLR 2025。

- 模型提示为 `global_ctx + local_ctx`。全局提示由服务器同步，本地提示在每个客户端跨轮持久保存。
- 每个本地批次把完整本地提示分解为正交低秩因子 `U,V` 与残差 `R`，前向使用 `global + U·V + R`。
- 裁剪 `global/U/V` 梯度，只向低秩因子梯度加入局部高斯噪声，再按论文 RGP 公式重构完整本地提示梯度。
- 服务器裁剪客户端全局更新、按样本数加权，并加入全局高斯噪声。本地提示不会上传或聚合。

配置参数：`rank`、`local_clip_norm`、`local_noise_multiplier`、`global_clip_norm`、`global_noise_multiplier`、`delta`。

论文：[ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4431224d3762aa655f0aee4eaf04ff16-Abstract-Conference.html)。截至实现时未发现作者公开代码，因此本实现依据论文算法与公式完成。

## FedASK (`aggregator: fedask`)

对应论文：*Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix*，NeurIPS 2025。

- 将可训练 soft prompt 写成 `base_ctx + B·A`，把原论文 LoRA 的低秩更新映射到提示矩阵。
- 客户端固定 `A`，计算完整提示矩阵的逐样本梯度，裁剪、加噪后右乘 `A^T` 更新 `B`，避免直接在两个因子上注入更高维噪声。
- 聚合阶段使用共享随机矩阵 `Omega`：客户端先发送 `Y_i=B_i(A_i Omega)`；服务器 QR 得到 `Q`；客户端再发送 `P_i=A_i^T(B_i^T Q)`；服务器对加权 `P` 做 SVD 并重构新的 `B,A`。
- `federated_method_summary.json` 记录最后一轮草图重构相对误差，便于检查 `rank/oversampling` 是否足够。

配置参数：`rank`、`oversampling`、`clip_norm`、`noise_multiplier`、`delta`。

论文：[NeurIPS Proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/a686ddca183f72ee9f3f04896eb11bcb-Abstract-Conference.html)。官方实现：[PrivacyFedLLM](https://github.com/SII-FLEEECERmw/PrivacyFedLLM)。本仓库的两阶段草图与 SVD 因子恢复遵循该官方实现。

## 与攻击/防御的组合

攻击审计始终读取每个客户端实际训练后、聚合前的可训练状态。DP-FPL 下更新差值使用该客户端自己的个性化基线；FedASK 下审计 `A/B` 上传状态，同时黑盒攻击观察其实际 `base+B·A` 输出。

所选独立防御在论文方法的原生隐私训练之后执行为显式防御阶段；不会暗中组合两个防御。因而同一条命令可以比较无防御、仅防御、仅攻击或“一个攻击 + 一个防御”。
