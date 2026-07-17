# Paper-to-code mapping

## 1. Comprehensive Privacy Analysis (Nasr, Shokri, Houmansadr)

论文同时提出被动与主动白盒成员推理。原方法将梯度、隐藏层输出、模型输出、标签和损失交给分组件攻击网络。

本仓库的 `privacy_attacks/whitebox.py` 针对参数高效提示微调作了两点调整：

1. 冻结 CLIP 骨干不产生可观察训练梯度，因此白盒梯度组件仅分析 prompt 参数；完整高维梯度被转换为每个 prompt token 的梯度范数和全局统计量。
2. 主动攻击以隔离 probe 实现：服务器对候选样本反复执行 prompt 梯度上升和目标客户端真实本地训练，以最后的候选 prompt 梯度范数作为成员信号。该实现保留论文的主动 ascent/train 轨迹，但按候选隔离，且 probe 不进入 FedAvg。

参考工具：[Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter)。

## 2. FedMIA

作者公开实现：[Liar-Mask/FedMIA](https://github.com/Liar-Mask/FedMIA)。

`privacy_attacks/fedmia.py` 重新实现论文的 all-for-one 单尾检验：

- FedMIA-I 使用负交叉熵（置信度）；
- FedMIA-II 使用候选样本 prompt gradient 与客户端 prompt update 的余弦相似度；
- 同一轮其他客户端构成 non-target/null 分布；
- 对多个通信轮的单尾 CDF 分数支持 `mean`、`max`、`last` 与 `late3` 聚合。

早期实验曾自行组合两类 FedMIA 证据；该分数并非论文定义的攻击，现已从
执行入口、配置和测试中移除。结果分析器仅在读取旧目录时识别并丢弃对应字段，
它不会进入新生成的证据或实验表格。

参考仓库未附许可证，因此本仓库没有复制其源码。

## 3. Rethinking Membership Inference Attacks Against Transfer Learning

检索时未发现作者发布的官方实现，因此 `privacy_attacks/transfer.py` 按论文描述实现。

适配采用与 FPT 私有数据目标一致的 At.S & Ac.S 场景：冻结 CLIP 是共享 teacher，客户端 soft prompt 是 student。目标客户端的本地 prompt 是被攻击 student，其他当轮客户端 prompt 的均值是 shadow student。攻击特征包含二者联合图文表示的绝对差与 L2 距离，并在已知成员/非成员校准集上训练攻击头。

## 4. A Method to Facilitate Membership Inference

作者公开实现：[DependableSystemsLab/code_poison_MIA](https://github.com/DependableSystemsLab/code_poison_MIA)，MIT License。

`privacy_attacks/code_poison.py` 保留其关键机制：对每个训练样本进行稳定哈希，确定性生成秘密合成伙伴，并同时优化干净损失与合成伙伴损失。查询阶段以伙伴样本上的负损失作为成员分数。

原实现面向完整 CNN；这里仅更新 soft prompt，且不引入第二归一化层。该机制是训练代码投毒型隐私攻击，不包含触发器、目标标签或后门成功率逻辑。

## 5. PIPRA: Your Prompts Are Not Safe

论文：[AAAI 2026 官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/40839)。截至实现时未发现作者公开的官方代码，因此 `privacy_attacks/pipra.py` 按论文第 4--12 式实现：

- 将已知 OUT 数据划分并训练多个 prompt-only shadow prompts；
- 使用冻结 VLM 提取图像特征，以及由真实标签选择的 prompt 文本特征；
- 以 shadow prompt/成员为正对、shadow prompt/非成员为负对；
- 联合训练共享特征投影器、InfoNCE 对比目标和二元判别器；
- 推断阶段不使用目标模型 logits、损失或梯度。

## 6. RMIA

论文：[ICML 2024 / PMLR](https://proceedings.mlr.press/v235/zarifzadeh24a.html)。官方实现位于 [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/2024_rmia)。

`privacy_attacks/rmia.py` 使用同一通信轮的非目标客户端 prompts 作为 reference models，实现论文 offline marginal estimate 与 population likelihood-ratio dominance score，并采用论文低 FPR 实验常用的 `gamma=2`。已知 OUT 辅助集与最终查询集严格拆分。

## 7. IMIA

论文：[USENIX Security 2026 官方页面](https://www.usenix.org/conference/usenixsecurity26/presentation/du)。截至实现时未发现官方代码。

`privacy_attacks/imia.py` 实现论文的 weighted log-probability imitation loss、warm-up checkpoint、imitative OUT model、同类别低损失 pivot、imitative IN model，以及最终的非参数距离分数：

`(s_obs - mean(s_out))^2 - (s_obs - mean(s_in))^2`。

所有 imitative models 共享冻结骨干，只随机初始化和训练 soft prompt。

## 8. Scalable Membership Inference via Quantile Regression

论文：[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html)，[官方代码](https://github.com/amazon-science/quantile-mia)。

`privacy_attacks/quantile.py` 使用目标客户端已知 OUT 样本的 CLIP/prompt 表示和真实类别置信度训练 pinball-loss 分位数回归器。成员分数是观测置信度超过条件分位数阈值的幅度。

## 9. YOQO

论文：[ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/b3edfc1950c30c42e2ecf6637ab7fb09-Abstract-Conference.html)，[官方代码](https://github.com/WU-YU-TONG/YOQO)。

`privacy_attacks/query_attacks.py` 的 offline 适配使用非目标客户端 prompts 作为 OUT shadow models，为每个 OUT prompt 分别选择最高概率错误类，并按论文式 (5) 优化错误类交叉熵与 MSE 距离。实现采用官方代码的归一化 loss threshold 早停；由于输入是 CLIP 归一化张量，步长缩放为 `0.001`。对目标 prompt 只执行一次查询，并仅保留 argmax hard label；预测仍保持真实类别时判为更可能是成员。

## 10. Canary in a Coalmine

论文：[ICLR 2023](https://openreview.net/pdf?id=b7SBTEBFnC)，[官方代码](https://github.com/YuxinWenRick/canary-in-a-coalmine)。

本适配从非目标客户端 prompts 构造 OUT surrogates，并用单个候选样本对其 prompt-only 副本短暂微调得到 IN surrogates。查询优化同时让 IN surrogates 保持真实类、OUT surrogates 转向错误类，最终集成多条查询的 scaled-confidence difference。

## 11. PromptMIA: Leveraging Soft Prompts for Privacy Attacks in FPT

论文当前为 [arXiv:2601.06641](https://arxiv.org/abs/2601.06641)，尚不是正式顶会版本，也未发现公开代码。

`privacy_attacks/promptmia.py` 忠实实现论文 Algorithm 1/2：先从目标查询方向移除随机向量的平行分量，再合成具有指定余弦相似度和原查询范数的 key，并在 `s_max + delta_min` 到 `s_max + delta_min + span` 中生成多样 key 集合。

原论文使用带 key 的视觉 prompt pool，而本仓库使用所有样本共享的 CoOp 文本 prompt，没有 top-N prompt-selection event。因此这里明确实现为适配版：将 adversarial keys 写入隔离 prompt-token 副本，让目标客户端正常本地训练，再以这些 token 的更新强度和候选梯度对齐度评分。该 probe 不进入 FedAvg，不改变正式训练轨迹；结果元数据会记录此架构差异。

## Shared evaluation

所有攻击以 `TPR@1% FPR` 为主指标，并保留 ROC AUC、TPR@10% FPR 和 TPR@0.1% FPR 作为诊断信息。需要训练攻击头的方法使用分层校准/评估拆分；FedMIA 与 CodePoison 是直接打分方法。YOQO 只有二元硬标签分数，低 FPR 指标在有限样本下等价于观察零误报阈值，解释时需同时报告样本数。
