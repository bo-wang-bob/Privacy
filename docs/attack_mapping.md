# Paper-to-code mapping

当前公共审计器只注册 FedMIA 四个基线、FedMIA-I/II、Gradient-Diff、
Score-Diff、Score-Ratio、FTA 和 ProjRes。下文其他攻击的研究实现仍保留在仓库中，
但已从配置、命令行和公共审计器注册表移除，不能作为新实验攻击启用。

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
- FedMIA-II 使用候选样本梯度与客户端上传梯度的余弦相似度；
- 同一轮其他客户端构成 non-target/null 分布；
- 对多个通信轮的单尾 CDF 分数支持 `mean`、`max`、`last` 与 `late3` 聚合。

ResNet18/CIFAR100 论文复现配置使用 `fedmia_loss`、`mean` 和 `upper`：每轮先用
其余 9 个客户端对同一候选样本的负交叉熵估计 null，按公式 (8)--(10) 去掉高于
均值加三倍标准差的空间异常值，再计算目标客户端观测的高斯 CDF；最终对第
10、20、…、300 轮的 30 个 CDF 取均值。候选协议为 `fedmia_mix`：目标客户端
完整 5000 个训练成员，对 1000 个独立测试样本和其余 9 个客户端各 1000 个训练
样本组成的 10000 个目标客户端非成员。

该 ResNet18 复现入口还在完全相同的固定候选和审计轮上计算逐样本 ICLR 观测分数：

\[
s_{\mathrm{ICLR}}(x)=L(x;\theta_{-k})-L(x;\theta_k).
\]

其中 `theta_k` 是目标客户端本轮完成一个 local epoch 后上传的模型，`theta_-k`
通过 `theta_global = w_k theta_k + (1-w_k) theta_-k` 使用服务器实际聚合权重反解。
评分不参与客户端训练、候选过滤或聚合，因此 `defense.name` 仍为 `none`，论文的
FedAvg 优化过程保持不变。每轮 15000 个原始评分和按样本跨轮聚合的评分分别写入
`iclr_candidate_round_scores.csv` 与 `iclr_candidate_scores.csv`；后者使用稳定的
`sample_index` 与最终 FedMIA-Loss 分数一一连接。由于非成员同时包含独立测试样本
和其他客户端训练样本，结果还按候选来源报告均值及其与 FedMIA-Loss 的相关性。

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

默认 sweep 使用按需审计调度：两个单轮基线只在固定轮次使用信号；两个时序
基线每轮只需要目标客户端；FedMIA-I/II 根据公式 (5)、公式 (12) 和 Algorithm 1
对所有通信轮、所有当轮客户端构造跨客户端 null 分布，因此也显式配置为每轮
采集。若任务没有选择 FedMIA，调度器不会为它计算跨客户端信号。

官方绘图代码还会查看每一轮的攻击效果并报告其中的最大 TPR。那种选择依赖
真实成员标签，不能作为部署时可实现的攻击规则，因此本仓库的单轮基线使用预先
固定的轮次，而不实现 oracle best-round。四个基线都只使用目标客户端，不使用
FedMIA 的跨客户端 null 分布；这正是它们与 FedMIA-I/II 的对照意义。

### ProjRes 论文采用的四个传统基线

Deng 等人的 FedLLM ProjRes 实验还比较 Gradient-Diff、Score-Diff、Score-Ratio
和 FTA。本仓库以如下运行名实现，并统一将分数方向转换为“越大越像成员”：

| 运行名 | 原始信号 | 当前 FedLLM 实现 |
|---|---|---|
| `gradient_diff` | `||g_client||² - ||g_client - sum_y grad L(x,y)||²` | 直接使用目标客户端上传的 one-batch 梯度；逐候选流式计算对全部标签求和的梯度 |
| `score_diff` | `L_post(x)-L_pre(x)` | 同轮客户端更新前后各前向一次；FedSGD post-state 由 `base - learning_rate * uploaded_gradient` 构造；输出其相反数 `L_pre-L_post` |
| `score_ratio` | `(L_post(x)+c)/(L_pre(x)+c)` | FedSGD post-state 由 `base - learning_rate * uploaded_gradient` 构造；默认 `c=1e-6`；输出负比值，避免改变全仓库“高分为成员”的 ROC 约定 |
| `fta` | 多个 FL 模型快照上性能指标的变化斜率 | 默认对真实标签置信度使用实际通信轮编号做 OLS；首个检查点使用该轮更新前/后两个快照，也可设置 `audit.fta_measurement: loss` |

Gradient-Diff 来源于 [Li、Li 与 Ribeiro，ICLR 2023](https://openreview.net/forum?id=QsCSLPP55Ku)；
Score-Diff/Score-Ratio 来源于 [Jagielski 等，PoPETs 2023](https://petsymposium.org/popets/2023/popets-2023-0078.php)；
FTA（free training attack）来源于 [Chang 等，USENIX Security 2024](https://www.usenix.org/conference/usenixsecurity24/presentation/chang)。
在 `scripts/run_privacy_experiments.py` 生成的 BERT 默认任务中，FTA 每 10 个已完成通信轮
采集一次，Gradient-Diff、Score-Diff 和 Score-Ratio 仍每 50 轮采集一次；GPT2
默认任务中的四者仍统一每 50 轮采集。
BERT、GPT2-Large、CLIP-Adapter 与 CLIP-LoRA 的 Blackbox-Loss、Gradient-Diff、Score-Diff、Score-Ratio 与 Grad-Cosine 在每轮使用真实上传 Batch
作为成员，并按标签从从未训练的全局 evaluation 池抽取 10 倍非成员；每轮独立评估，
不改变攻击分数公式。四类模型的 FTA、Loss-Series、Avg-Cosine 和两个 FedMIA
变体均使用目标客户端完整训练集作为成员，并抽取类别尽力匹配的等量全局独立
evaluation 非成员。

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

所有攻击以 `TPR@1% FPR` 为主指标，并保留 ROC AUC 和 TPR@10% FPR。BERT、GPT2、CLIP-MLP、CLIP-Adapter 与 CLIP-LoRA 的五种固定攻击使用目标客户端完整训练集中的 `M` 个成员，以及类别尽力匹配的 `M` 个全局独立 evaluation 非成员；单类非成员不足时从其他仍有容量的类别确定性补足，并记录实际标签直方图与 TV 距离。仍可从中派生固定 100/100 论文对照视图，该视图也采用相同的尽力匹配规则。五类模型的六种真实 Batch 攻击使用当轮 `N` 个成员和 `10N` 个非成员；BERT/GPT2 完整 Batch 为 16/160，三种 CLIP 模型为 32/320。这些真实 Batch 攻击不使用固定历史成员或论文对照视图，也不计算 TPR@0.1% FPR。需要训练攻击头的方法使用分层校准/评估拆分；FedMIA 与 CodePoison 是直接打分方法。YOQO 只有二元硬标签分数，低 FPR 指标在有限样本下等价于观察零误报阈值，解释时需同时报告样本数。

## CLIP 图像编码器 + 两层 MLP 场景

`model_type: clip_mlp` 冻结 CLIP，只训练两层 MLP 分类头。正式协议每类抽取 16 张
训练图像；每个客户端每轮用一个真实 mini-batch 计算 FedSGD 梯度，服务器对
`classifier.0.{weight,bias}` 和 `classifier.3.{weight,bias}` 等权聚合。攻击适配遵循以下映射：

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

这些映射保留每种攻击的观测权限和 FedSGD 协议边界，但不把 MLP 类别向量称为
文本 prompt。模型基线见 `configs/models/clip_mlp.yaml`，任务由统一入口生成。

## Few-shot CLIP-Adapter 场景

`model_type: clip_adapter` 冻结 CLIP，只通过每客户端每轮一个 mini-batch 的
FedSGD 训练图像、文本两侧残差瓶颈，并与 CLIP-MLP 一样使用每类 16 张训练图像，
即 `fpl_shots: 16` 与 `use_full_dataset: false`；服务器对参与客户端梯度直接等权平均。损失、概率、表示、更新余弦、
Nasr、PromptRes、PIPRA、RMIA、IMIA、YOQO、Canary 和 CodePoison 均复用其原有
观测协议，但影子模型与 probe 只重置或更新 adapter 参数。

PromptMIA 使用 adapter 第一层的输入投影向量作为 key-like vectors。普通训练时
冻结 CLIP 不保留反向图；YOQO 与 Canary 优化输入时会临时保留输入到冻结 CLIP
的梯度。模型基线见 `configs/models/clip_adapter.yaml`，任务由统一入口生成。

### ProjRes 严格单轮入口

`privacy_attacks/projres_mlp.py` 另行实现 Deng 等人的 ProjRes Algorithm 1。
它攻击第一层 `classifier.0.weight`：从一次 vanilla FedSGD 更新的行空间恢复训练
batch 的 CLIP 表示子空间，再使用原始 L1 投影残差判定成员。成员集合严格等于
产生该更新的实际 batch，不使用客户端的整个历史训练集代替。真实数据入口是
`scripts/validate_projres_mlp_real.py`，配置是 `configs/models/clip_mlp.yaml`；完整
边界说明见 `docs/projres_mlp_strict.md`。该入口没有复用名称相近但采用候选梯度
余弦相似度的 `promptres`。

统一 sweep 对 CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA 提供共享 exact-batch ProjRes：
FedSGD 路径直接读取客户端上传梯度，不再从参数差反推。三者均每 10 轮运行；
CLIP-MLP 攻击 `classifier.0.weight`，CLIP-Adapter 攻击第一层 down-projection，
LoRA 攻击视觉 Q 投影的 `lora_A` 下投影因子，并使用各层对应的候选表示。三者真实
上传均只来自一个 batch，因此与论文 FedSGD 观测一致。ProjRes 以负 L1 残差做
ranking-only 评价，不再使用固定残差阈值。CLIP-MLP 的独立严格验证入口仍然保留，
用于单轮诊断而不是替代统一攻击任务。
