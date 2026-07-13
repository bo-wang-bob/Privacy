# Paper-to-code mapping

## 1. Comprehensive Privacy Analysis (Nasr, Shokri, Houmansadr)

论文同时提出被动与主动白盒成员推理。原方法将梯度、隐藏层输出、模型输出、标签和损失交给分组件攻击网络。

本仓库的 `privacy_attacks/whitebox.py` 针对参数高效提示微调作了两点调整：

1. 冻结 CLIP 骨干不产生可观察训练梯度，因此白盒梯度组件仅分析 prompt 参数；完整高维梯度被转换为每个 prompt token 的梯度范数和全局统计量。
2. 主动攻击以隔离 probe 实现：服务器先对候选样本执行 prompt 梯度上升，再让目标客户端执行真实本地训练，使用候选损失与梯度范数的下降作为分数。probe 不进入 FedAvg。

参考工具：[Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)。

## 2. FedMIA

作者公开实现：[Liar-Mask/FedMIA](https://github.com/Liar-Mask/FedMIA)。

`privacy_attacks/fedmia.py` 重新实现论文的 all-for-one 单尾检验：

- FedMIA-I 使用负交叉熵（置信度）；
- FedMIA-II 使用候选样本 prompt gradient 与客户端 prompt update 的余弦相似度；
- 同一轮其他客户端构成 non-target/null 分布；
- 对多个通信轮的单尾 CDF 分数取平均。

参考仓库未附许可证，因此本仓库没有复制其源码。

## 3. Rethinking Membership Inference Attacks Against Transfer Learning

检索时未发现作者发布的官方实现，因此 `privacy_attacks/transfer.py` 按论文描述实现。

适配采用与 FPT 私有数据目标一致的 At.S & Ac.S 场景：冻结 CLIP 是共享 teacher，客户端 soft prompt 是 student。目标客户端的本地 prompt 是被攻击 student，其他当轮客户端 prompt 的均值是 shadow student。攻击特征包含二者联合图文表示的绝对差与 L2 距离，并在已知成员/非成员校准集上训练攻击头。

## 4. A Method to Facilitate Membership Inference

作者公开实现：[DependableSystemsLab/code_poison_MIA](https://github.com/DependableSystemsLab/code_poison_MIA)，MIT License。

`privacy_attacks/code_poison.py` 保留其关键机制：对每个训练样本进行稳定哈希，确定性生成秘密合成伙伴，并同时优化干净损失与合成伙伴损失。查询阶段以伙伴样本上的负损失作为成员分数。

原实现面向完整 CNN；这里仅更新 soft prompt，且不引入第二归一化层。该机制是训练代码投毒型隐私攻击，不包含触发器、目标标签或后门成功率逻辑。

## Shared evaluation

所有攻击统一输出 ROC AUC、TPR@10% FPR、TPR@1% FPR 和 TPR@0.1% FPR。需要训练攻击头的方法使用分层校准/评估拆分；FedMIA 与 CodePoison 是直接打分方法。
