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

### FedMIA 实验中的四个基线

FedMIA 官方仓库将四个相关基线按“度量、时间信息、空间信息”区分：

| 运行名 | 度量 | 时间信息 | 空间信息 | 当前适配 |
|---|---|---|---|---|
| `blackbox_loss` | 候选负交叉熵 | 单轮 | 仅目标客户端 | 默认使用最后一个已审计轮次；可通过 `audit.fedmia_baseline_single_round` 固定为 `first` 或具体轮次 |
| `loss_series` | 候选负交叉熵 | 多轮 | 仅目标客户端 | 对目标客户端逐轮分数取均值，与官方代码的 temporal average 一致 |
| `grad_cosine` | 候选 prompt 梯度与目标客户端 prompt update 的余弦 | 单轮 | 仅目标客户端 | 默认使用最后一个已审计轮次 |
| `avg_cosine` | 同上 | 多轮 | 仅目标客户端 | 对逐轮余弦取均值 |

官方绘图代码还会查看每一轮的攻击效果并报告其中的最大 TPR。那种选择依赖
真实成员标签，不能作为部署时可实现的攻击规则，因此本仓库的单轮基线使用预先
固定的轮次，而不实现 oracle best-round。四个基线都只使用目标客户端，不使用
FedMIA 的跨客户端 null 分布；这正是它们与 FedMIA-I/II/III/IV 的对照意义。

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

## CLIP 图像编码器 + 两层 MLP 场景

`model_type: clip_mlp` 冻结 CLIP，只训练两层 MLP 分类头。普通 FedAvg 上传并
按样本数聚合 `classifier.0.{weight,bias}` 和
`classifier.3.{weight,bias}`。攻击适配遵循以下映射：

`clip_mlp.precompute_features: true`（默认）会在训练开始前把各客户端训练集和
测试集完整编码成 CPU 上的 CLIP 向量。模型对二维输入走向量快路径，所以本地
训练、Server 测试和审计不会在每轮重复运行冻结图像编码器。

- loss、confidence、概率序列、更新余弦、PromptRes 和 Nasr 白盒特征直接作用于
  全部可训练 MLP 参数；
- representation、Transfer 和 QMIA 使用 `logits`、隐藏表示以及隐藏表示与真实类
  决策向量的逐维乘积；
- PIPRA 使用隐藏表示和真实类最后一层权重作为对齐的样本/类别语义表示；
- PromptMIA 将最后一层类别决策向量视为 key-like vectors，测量隔离本地训练后
  沿候选梯度方向的有符号更新；
- YOQO 和 Canary 保留冻结 CLIP 参数，但允许梯度对输入传播，因此仍可优化查询；
- IMIA、RMIA、CodePoison 和主动 Nasr 复用仅训练 MLP 的影子模型或隔离 probe。

这些映射保留每种攻击的观测权限和 FedAvg 协议边界，但不把 MLP 类别向量称为
文本 prompt。完整入口见 `configs/clip_mlp_privacy.yaml`。

## 16-shot 视觉 CLIP Adapter 场景

`model_type: visual_adapter` 冻结 CLIP，只通过 FedAvg 训练视觉残差瓶颈，且配置
强制 `fpl_shots: 16` 与 `use_full_dataset: false`。损失、概率、表示、更新余弦、
Nasr、PromptRes、PIPRA、RMIA、IMIA、YOQO、Canary 和 CodePoison 均复用其原有
观测协议，但影子模型与 probe 只重置或更新 adapter 参数。

PromptMIA 使用 adapter 第一层的输入投影向量作为 key-like vectors。普通训练时
冻结 CLIP 不保留反向图；YOQO 与 Canary 优化输入时会临时保留输入到冻结 CLIP
的梯度。完整入口见 `configs/visual_adapter_privacy.yaml`。

### ProjRes 严格单轮入口

`privacy_attacks/projres_mlp.py` 另行实现 Deng 等人的 ProjRes Algorithm 1。
它攻击第一层 `classifier.0.weight`：从一次 vanilla FedSGD 更新的行空间恢复训练
batch 的 CLIP 表示子空间，再使用原始 L1 投影残差判定成员。成员集合严格等于
产生该更新的实际 batch，不使用客户端的整个历史训练集代替。真实数据入口是
`scripts/validate_projres_mlp_real.py`，配置是 `configs/clip_mlp_projres.yaml`；完整
边界说明见 `docs/projres_mlp_strict.md`。该入口没有复用名称相近但采用候选梯度
余弦相似度的 `promptres`。
