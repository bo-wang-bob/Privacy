# Repository guidance

## 用户偏好与基本规则

- 默认使用中文沟通。
- 数据分析不要生成 HTML 文件；优先输出 Markdown、CSV、JSON、PNG/PDF 等必要产物。
- 给出运行方式时尽量把稳定参数写进配置或脚本默认值，减少用户每次手动传参。
- `results/` 中的真实实验结果属于用户数据。除非用户明确要求，不要删除、覆盖或批量改写已有结果。
- 工作区可能包含用户自己的未提交修改；只处理当前任务涉及的文件。
- Git 提交信息和提交日志应在合理范围内尽量详细准确：标题概括核心协议或功能变化，正文列出训练协议、候选定义、兼容性影响和验证结果；避免只写笼统的 `update`、`fix` 或 `changes`。

## 当前统一实验入口

- 六个 PEFT 模型（CLIP-MLP/Adapter/LoRA、BERT Adapter/LoRA、GPT2 Adapter）均支持 FedSGD/FedAvg。使用 `--methods fedsgd,fedavg` 展开方法维度；不传保持各模型基线，ResNet18 仍只支持 FedAvg。不要增加尚未实现的 GPT2-LoRA 等模型。
- `--methods fedavg` 默认全部模型 100 轮、每轮 1 个完整本地 epoch、`aggregation_weighting: sample_count`；可用 `--local-epochs` 和 `--aggregation-weighting uniform` 覆盖。FedSGD 默认仍为 1 个 batch、等权，要求 `local_epochs=1`，原有模型轮数保留。方法默认值集中在 catalog 的 `method_overrides`；模型的数据、few-shot 和学习率保留，CLIP FedAvg 审计间隔使用下文的 BERT 同频设置。显式 `--rounds` 或 `--set num_global_iters=...` 覆盖所选全部方法的轮数。ResNet18 不传方法选择时保持 300 轮；显式选 FedAvg 则用 100 轮并关闭完整论文模式。相同轮数不代表相同训练预算。
- FedAvg 的 11 种攻击以目标客户端原始完整训练集 M 个成员与默认 M 个独立 evaluation 非成员为固定候选；类别尽力匹配并记录实际直方图/TV。六种单轮更新攻击自动从 `audit.exact_batch_membership_attacks` 转到 `audit.client_train_membership_attacks`，其余五种保持时序/跨客户端定义。不能以最后一个本地 batch 定义整个 FedAvg 上传的成员；重复 epoch 不重复计数。CoFedMID/Record-DP 下仍评价原始训练集身份，不宣称每个成员本轮都被访问。
- FedAvg 上传模型 delta，梯度类攻击只做一次 `-delta / round_learning_rate` 转换，明确记录为累计更新代理量；候选梯度取轮初模型。ProjRes 作为经验性多步适配，`paper_fedsgd_exact=false`、`batch_rank_bound=null`，候选上限自动为 0/0/0（完整固定池）。任意方法被攻击层上传为零时只跳过该轮 ProjRes，记录 `zero_observed_update`。
- FedAvg 使用 `client_train_update_candidate_selection.pt`、摘要 `client_train_membership` 和信号 `client_train_update_observations`；FedSGD 继续使用原 `exact_batch_*`。任务名标明方法，local_epochs 和权重进入汇总协议字段；所有低 FPR 可报告性仍依据独立非成员数，历史结果不改写。
- WWW、CoFedMID 与 BERT Adapter Record-DP 已支持多步 FedAvg，后者按多步 Poisson 机制校准预算；`local_client_dp` 仍只支持 FedSGD，混跑时跳过。旧 WWW post-round 私有 batch 分析仍只支持 FedSGD，可选的逐批 `www_record_diagnostics` 可用于 FedAvg，默认关闭。实现与限制见 `docs/federated_methods.md`。以下 one-batch 规则描述默认 FedSGD 协议。
- 本次 FedAvg 支持同时修复 `none` 分支忽略所配置 optimizer/momentum/weight_decay 的问题，ResNet18 也受此修正影响。旧结果即使 YAML 写有 momentum/weight_decay，也不能据此认定实际训练已使用这些值，须结合代码版本判断。现有默认 PEFT FedSGD 的 SGD/0/0 行为保持一致。

- 唯一批量入口是 `scripts/run_privacy_experiments.py`。它支持单个或多个模型、数据集、攻击、防御、seed 和目标客户端；模型 × 数据集 × 防御 × seed × 目标客户端展开为独立任务，同一组多攻击共享一次训练。
- 不传 `--models` 时保持原 CLIP 入口范围，依次运行 CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA；`--models all` 覆盖 7 个正式模型。
- 只运行单个模型时使用 `python scripts/run_privacy_experiments.py --models clip_mlp`；多模型用逗号分隔。
- `--attacks all` 按模型解析能力集：ResNet18 只有 `fedmia_loss`，PEFT 模型为全部 11 种；`--attacks none` 是纯训练。
- `--defenses all` 只展开模型正式支持的防御。显式攻击与模型不兼容时跳过该模型并打印原因；显式防御或数据集不兼容时跳过对应组合。
- `configs/experiment_catalog.yaml` 维护能力矩阵、防御深度覆盖和别名；`configs/models/` 下 7 个基线维护模型训练与默认候选协议。不要重新为数据集、攻击或防御复制整份模型 YAML。
- 统一入口在应用全部 `--set` 后自动推导启用的 exact-batch ProjRes 候选参数：普通训练和 WWW 使用最终 batch 大小及成员/非成员比例，`record_dp` 使用 `0/0/0` 动态候选。改变 `batch_size` 或混跑 `none,record_dp,www` 无需手动同步这三个值。显式传入的 ProjRes 候选参数保留并接受协议校验；干运行显示最终候选参数。
- CLIP 默认学习率由模型基线设置：CLIP-MLP 为 `0.1`，CLIP-Adapter 与 CLIP-LoRA 均为 `0.01`。应以统一脚本干运行打印的最终参数为准。
- CLIP-LoRA 自 2026-09-09 起的模型基线使用 `rank=32`（此前为 2），保留 `alpha=1`、`scaling=sqrt_rank`、dropout `0`、图像/文本编码器全部 Q/K/V 和原 batch/学习率；FedSGD/FedAvg 共用。LoRA 因子参数量为此前的 16 倍，不能据此保证 ProjRes 满秩或攻击有效。旧 rank 对照使用 `--set clip_lora.rank=2`，历史结果不改写。
- CLIP-Adapter 默认 `clip_adapter.variant: transformer`：视觉 ViT 的每个 block 后插入 BERT 式 `768→384→768` 残差 Adapter（reduction=2、ReLU、上投影零初始化），原始骨干与文本编码器冻结，保留图像/类别文本相似度分类。图像、文本特征均在线计算，只缓存提示词 token；`precompute_features: true` 被拒绝。客户端复用共享骨干、独立保存 Adapter 参数。旧 `variant: feature` 和缺少 variant 的旧配置保持末端特征 Adapter；历史结果不改写。新权重保存到 `final_clip_transformer_adapter.pt`，不是旧末端 Adapter checkpoint。见 `docs/clip_transformer_adapter.md`。
- 三个 CLIP 基线均显式使用 IID。不要在正常 IID 实验中传 `--dirichlet-alpha`；仅传该参数会自动切换为 `dirichlet`。如同时显式传 `--partition-mode iid --dirichlet-alpha 0.1`，显式 partition mode 优先，alpha 仅作为未使用的配置值保留。
- 自 2026-09-10 起，CLIP-Adapter/LoRA 默认在客户端划分前每类抽取 100 张训练图像，10 个 IID 客户端时每客户端每类 10 张；MLP 仍固定每类 16 张。CLIP-Adapter/LoRA 均可配置 `use_full_dataset: false` 与正整数 `fpl_shots` 选择每类样本上限，或 `use_full_dataset: true` 与 `fpl_shots: null` 使用完整训练分区；旧 16 张/类对照使用 `--set fpl_shots=16`。Adapter/LoRA 始终从完整源分区加载，每类限额只截取训练集，再进行客户端划分；独立 evaluation/test 分区不随 shots 截断。新 Adapter/LoRA 的 CIFAR100/Food101 不再先经过历史 200 张/类训练及 50 张/类测试子集；旧结果不改写，比较时核对实际分区。直接 CLI 的 shots 覆盖也不再隐式把 Adapter/LoRA 切为 Dirichlet。IID 仍要求每类训练样本数至少等于客户端数。
- 三个 CLIP 模型在 FedSGD/FedAvg 下默认只依次运行 CIFAR100、Food101。Caltech101、OxfordPets、Flowers102 仍支持通过 `--datasets` 显式选择，`--datasets all` 展开全部五个支持的数据集。单任务 CLIP 模型 YAML 的默认数据集为 CIFAR100。
- 三个模型在全部默认数据集上统一使用 10 个客户端和 batch size 32；CLIP-MLP 使用 150 个通信轮次，CLIP-Adapter/CLIP-LoRA 使用 300 个通信轮次。三者均使用 FedSGD，每个客户端每轮只执行 1 个 mini-batch/1 次 optimizer step；`local_epochs: 1` 是协议校验值，不表示遍历完整本地数据集。
- 三种微调方式的服务器端聚合都使用 `aggregation_weighting: uniform`，即对本轮参与客户端上传的梯度直接等权平均，不按客户端本地样本数或实际 batch 大小加权。三者任务目录方法名均为 `fedsgd`。
- 正常任务指标默认按 `eval_interval: 5` 在已完成的第 5、10、15、…轮评估；若总轮数不能被 5 整除，最后一轮仍会额外评估。`training_metrics.csv` 使用相同的一基轮次编号。

## 当前 WWW 防御（原 ICLR）

- 原观测型 `iclr` 已更名并更新为实际参与训练的 `www`；配置和新产物使用 `www_*`，历史 `results/` 与 `iclr_*` 产物不改写。
- 支持三个 CLIP 模型及 BERT Adapter/LoRA。每个真实训练 batch 按上一轮防御后模型的 `M_i = loss(theta_-k, z_i) - loss(theta_k, z_i)` 稳定升序排序。默认 `defense.www_tail_fraction=0.8`、`www_tail_basis=actual_batch`，高风险尾部为最后 `m=ceil(0.8*n)` 条；尾部升序第 j 条权重为 `r_i=(j-0.5)/m`，其余低风险样本权重为 0。batch=32 时为 6 条仅用 CE、26 条加正则。`expected_batch` 可显式沿用历史固定尾部宽度并右对齐短批次。首轮或缺少上一轮参考状态时所有 r 为 0，仅用 CE，诊断标记风险不可用。
- 当前损失为 `mean(CE_i + lambda*r_i*abs(p_i-q_i))`，默认 `defense.www_regularization_weight=1.0`，必须有限且非负，0 可做普通 CE 消融。`p_i=exp(-当前CE_i)` 参与求导；`q_i=exp(-上一轮theta_-k的CE_i)` 及风险权重停止梯度。这是受 MIST 启发的单步 FedSGD 适配：参考为其他客户端的参数聚合模型，不是平均客户端预测，也不是严格从未见过目标数据的 leave-one-out 模型。所有样本保留 CE，不裁剪、不使用 INO 权重、不加噪。
- 当前 WWW 使用 `sampling=shuffled_batches`，按普通 FedSGD 的种子规则打乱后分批，每轮取下一批，遍历结束再打乱；短 batch 保留并按实际样本数求均值。不提供 DP 保证。旧配置或混合 sweep 的 `target_epsilon`、`delta`、`adjacency`、`accountant`、`max_grad_norm`、`www_beta_alpha`、`www_beta_beta` 清空为 `null`，`noise_multiplier` 固定为 0；显式旧 `sampling=poisson` 会被拒绝。普通 DP 保留 Poisson 及原预算校准。历史 WWW 结果不改写，须区分带噪、Poisson 裁剪、尾部免裁剪与当前风险损失版本。
- WWW 训练直接对整批组合损失执行一次 backward/optimizer step 并上传实际梯度，逐样本求导只用于可选范数诊断。WWW 诊断与 BERT Adapter Record-DP 默认 `defense.grad_sample_backend=auto`、`defense.microbatch_size=4`，使用分块 batched VJP；WWW 诊断复用同一个完整 batch 前向图和 dropout，块大小仅控制逐样本梯度存储。共享 Transformer 启用 gradient checkpointing 时 `auto` 选择 `loop`；ResNet18 Record-DP 保留原 `vmap`。性能说明见 `docs/fedsgd_performance.md`。
- 默认 `www_record_diagnostics=false`，关闭额外逐样本梯度诊断；显式使用 `--set defense.www_record_diagnostics=true` 后，在任务目录的 `www_diagnostics/` 流式记录 `sample_gradients.csv`、`batch_summary.csv`、`summary.json`（schema_version=2）：每次训练访问的风险、样本身份、风险权重、真实类预测差异、CE/正则/总损失，以及对应的联合梯度范数；批次和风险分组保存分位数及 Pearson/Spearman 相关系数。记录 CE 梯度的有符号系数和方向反转数，避免把范数减小误判为方向不变。每批 flush，内存不随轮数增长；首轮风险、教师差异及不可定义的相关系数留空。关闭诊断不改变风险计算和训练，并省去额外逐样本求导，旧 `release_private_diagnostics` 只控制旧版额外诊断。
- WWW 始终记录 `formal_dp_enabled=false`、`client_upload_is_private=false` 和空的 epsilon/delta，不输出 DP 攻击理论上界。旧 `reproducible_dp_noise` 在 WWW 中不再生效。新记录只影响新启动任务，不能补录旧进程未保存的范数。
- WWW 下 ProjRes 与普通训练一样按 batch size 推导候选上限，完整 batch=32 时为 `32/320/320`，短 batch 仍按真实 n/10n 候选计算；当前真实类概率正则的逐样本梯度与 CE 共线，保留 batch 秩上限，`attacked_parameter_perturbed=false`，因修改训练损失仍标记 `paper_fedsgd_exact=false`。不要声称该损失必然防住 ProjRes。LoRA 初始化等导致被攻击层上传为零时，仅跳过 ProjRes 并记录 `zero_observed_update`，其余攻击继续执行。入口与完整公式见 `docs/defenses.md`。

## 风险指导的本地生成研究

- `risk_synthesis` 是 2026-09-12 新增的实验防御，当前只支持 CLIP transformer Adapter/LoRA + FedAvg。通过统一入口 `--methods fedavg --defenses risk_synthesis` 运行，默认参数集中在 catalog；FedSGD 组合跳过，直接配置会拒绝。
- 每客户端从原始完整本地训练集拟合每类输入 patch+position token 的低秩几何，向该客户端类内合并协方差收缩；固定原始 CLIP 教师检查语义，统计与虚拟编码不共享。
- 用户随后要求全部替换：catalog 默认 `replacement_policy: all`、`replacement_fraction: 1`、`warmup_rounds: 0`、范数下限 0.1。首轮缺少参考时 r=0 但仍加几何噪声；后续复用 WWW 上一轮 own/other 损失差和尾部秩风险，所有位置生成 `(1-r)*h+r*mu_without_self+0.1*L*epsilon`，风险只控制原始编码系数，类别中心均匀。不存在概率请求或 batch 替换上限。
- 按用户确认，最多两次语义检查后仍不达标则保留语义最好的有效虚拟候选，记录 `quality_passed=0`，不能回退原图。无法产生有限、范数合格且实际改变的编码则在该 batch 优化前报错。新版本 `local_token_geometry_v4_all_replacement` 记录 `selected_attempt`、`retained_original_fraction`、`original_distance`，全部访问均计入 synthetic_steps。普通 CE、batch 大小、本地 epoch、聚合权重及原始完整客户端训练集的成员身份不变，无 DP 保证。见 `docs/risk_synthesis_all_replacement.md`。
- 旧非空 synthesis 配置缺少 replacement_policy 时保持历史 risk_probability 协议；新实验用该策略复现时还需显式恢复 replacement_fraction=0.25、warmup_rounds=1。旧部分替换的 Adapter/CIFAR100 三种子确认通过事前标准，但风险排序的额外低 FPR 收益不稳定；这些收益不能继承给新的全部替换版本。历史结果不改写。
- 2026-09-13 已核验新版 Adapter/CIFAR100、seed43/客户端0、100轮全替换对照：无防御 Accuracy/最大AUC/最大TPR@1%FPR 为 81.58%/0.674120/37.10%，真实风险为 80.90%/0.609326/18.50%，打乱风险为 81.53%/0.612777/20.70%。本次通过预设三个判据，两组各100万次访问全部替换；10/11种正式AUC下降但5种TPR@1%FPR上升。真实风险相对打乱风险的最大AUC/TPR候选区间均跨零，不能声称稳定排序收益。仅一个种子、一个目标、已研究来源，无新版跨模型完整效果结论；见 `docs/risk_synthesis_all_replacement_results.md`。混合历史/新版机制导出使用列并集，旧版缺失 quality_failed 留空，不能填成零。
- 独立确认可显式设置 `confirmation_split_manifest`；当前仅支持 CLIP transformer Adapter/LoRA、CIFAR100、FedAvg、10 个 IID 客户端、全局每类 100 张及 none/www/risk_synthesis。清单预留原始训练源中的 30,000 张训练抽样池和 10,000 张独立 evaluation，并排除探索用过的 10,000 张；先按清单隔离，再截取每类 100 张训练。默认数据路径不变，文本入口拒绝该参数。配置解析保存清单 SHA256，加载时核对图像/标签指纹，任务内保存 `confirmation_split.json` 与 `data_partition.json`，审计分析将候选位置恢复为原始图片身份；不能复用旧分区基线作为确认对照。
- 历史 risk_probability 模式仍支持 `center_weighting: previous_risk`：使用上一参与轮各原始记录平均 assigned/used-risk，按 a=1-r 加权排除自身的类别中心，本轮冻结；打乱风险必须同时打乱建中心的风险，重复 epoch 对原始记录求平均。全部替换模式只支持 uniform，避免风险同时改变锚点。历史加权运行保留 anchor_* 字段和分析器重放核验。

## 当前 CoFedMID 防御

- `--defenses cofedmid` 已支持三个 CLIP 模型、BERT Adapter/LoRA 和 GPT2 Adapter。默认 `cofedmid_clients: all`，所有客户端协作，且每轮要求全员参与；显式列表可设置至少两个联盟成员。
- 默认三个模块全部开启：动态类别分配、EXP3 回收与软目标正则、聚合中性上传扰动。保留模型基线的 IID/few-shot、学习率、轮数和 one-batch 等权 FedSGD。第 11 轮首次回收，回收池上限为完整本地训练集的 5%。
- 默认从独立 evaluation 分区按类预留约 10% 作为防御验证集，剩余部分用于任务评估和审计非成员；训练集保持不变。统一入口 `--defenses none,cofedmid` 会让两组共享预留规则。独立 `none` 对照须显式设置 `--set defense.cofedmid_validation_fraction=0.1`；以 `defense_validation_split.json` 的 hash 核对实际划分。
- 默认噪声标准差 `0.01` 使用参数空间单位，FedSGD 上传梯度按 `-δ/lr` 转换，作用于统一可训练参数向量尾部 20%；按真实聚合权重抵消。攻击只读取防御后的上传。原训练成员身份不变，实际训练/回收/评分暴露另存 `cofedmid_sample_exposure.pt`。
- CoFedMID 下 ProjRes 仍审计真实 batch；若攻击层被噪声覆盖则取消无噪声 batch 秩上限。不要把 `paper_fedsgd_exact: false` 误解为候选不是实际 batch。
- 这属于六个 PEFT 模型的适配，已验证小模型端到端执行；真实数据上的防御效果、ResNet18/FedAvg 原论文复现、IFL/SeqMIA 和全局模型威胁视图尚未验证。参数和差异说明见 `docs/defenses.md`、`docs/cofedmid_implementation_plan.md`。

## 当前成员推理审计协议

- 注册攻击共 11 种：`blackbox_loss`、`loss_series`、`grad_cosine`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine`、`gradient_diff`、`score_diff`、`score_ratio`、`fta` 和 `projres`。
- CLIP-MLP、CLIP-Adapter 与 CLIP-LoRA 均运行全部 11 种攻击，并使用和 BERT/GPT2 相同的分组：
  - `loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine`、`fta` 使用目标客户端完整训练集 `M` 个成员和类别尽力匹配的 `M` 个全局独立 evaluation 非成员；类别不足时从其他类别确定性补足。
  - `blackbox_loss`、`grad_cosine`、`gradient_diff`、`score_diff`、`score_ratio`、`projres` 使用当轮真实上传 batch 的 `N` 个成员和严格按标签直方图抽取的 `10N` 个从未训练非成员；完整 batch 为 32/320，每轮候选独立构造。
- 固定候选保存到 `privacy_audit/candidate_selection.pt`，真实 Batch 候选保存到 `privacy_audit/exact_batch_candidate_selection.pt`；`summary.json` 记录候选规模、来源、标签直方图和 FPR 分辨率。

## 当前 BERT PEFT 协议

- BERT-Base Adapter 与 BERT-Base LoRA 均使用 30 个 IID 客户端、batch size 16、one-batch 等权 FedSGD；两者均使用 500 轮，Adapter 默认学习率为 `0.005`，LoRA 默认学习率为 `0.015`。
- BERT-LoRA 默认在全部 12 层自注意力 Query/Value 投影中训练 `rank=16`、`alpha=32`、`scaling=rank`、LoRA dropout `0` 的因子，并同时训练分类头，分类头 dropout 保持 `0.1`；冻结 BERT 主干不上传。自 2026-09-09 起仅将 LoRA dropout 从 `0.1` 改为 `0`，隔离输入随机掩码的影响；FedSGD/FedAvg 共用，历史结果不改写，旧对照使用 `--set lora.dropout=0.1`。服务器与 CLIP-LoRA 一样分别聚合同名 `lora_A`、`lora_B`，不先合成稠密 `BA` 更新。
- BERT Adapter/LoRA 均支持全部 11 种注册攻击，并使用上述 5 种固定候选攻击与 6 种真实 Batch 攻击划分；完整真实 Batch 候选为 16/160。
- BERT-LoRA 默认配置为 `configs/models/bert_lora.yaml`，默认只展开 CoLA，SST-5/IMDb 仍可通过 `--datasets` 显式选择；可由 `scripts/run_privacy_experiments.py --models bert_lora` 启动。ProjRes 默认观察最后一个已训练 Query 的 `lora_A` 上传，以该层输入 CLS 作为样本表示；显式首层 Query/Key/Value + CLS 会被拒绝，仍可使用 mean 作对照。

## 按需审计频次

- 当前三个 CLIP 配置不设置共享的 `audit_interval`；逐轮攻击全部使用各自的显式间隔。
- FedSGD 下 CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA 的全部 11 种攻击每 10 轮测量。MLP 测量到第 150 轮，Adapter/LoRA 测量到第 300 轮。
- FedAvg 下三个 CLIP 模型与 BERT 一致：`blackbox_loss`、`grad_cosine`、`gradient_diff`、`score_diff`、`score_ratio`、`projres` 每 50 轮，`loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine`、`fta` 每 10 轮；默认 100 轮分别测量 2 次和 10 次。覆盖保存在 catalog 的 `method_overrides.fedavg.models`，ProjRes 声明间隔同步为 50，不修改 FedSGD 模型基线。
- 审计间隔按已完成的通信轮数计数；例如 `attack_audit_intervals: 10` 对应零基内部索引 9、19、…，而不是索引 0、10、…。三者每轮均只训练 1 个真实 batch。
- 三种 CLIP 模型的 `blackbox_loss` 和 `grad_cosine` 都属于真实 Batch 协议，必须按配置轮次分别审计。
- 同一任务运行多个攻击时，调度器取各攻击所需轮次的并集，但仍只计算该轮实际需要的信号族。只要包含任一逐轮攻击，对应信号仍会每轮计算，这是协议需求而不是调度失效。

## ProjRes 特例

- 所有 ProjRes 默认使用目标客户端本轮训练后、服务器可重建的模型表示（`representation_state: client_post_update_model`）：FedAvg 为 `base + uploaded_delta`，FedSGD 为 `base - lr * uploaded_gradient`。FedSGD 若配置非普通 SGD 的本地优化器，该表示仍是公开梯度对应的 SGD 端点，不读取隐藏优化器状态。成员/非成员使用同一个目标状态，独立入口逐客户端重算动态非成员表示；审计完成后恢复共享模型。MLP/旧 feature Adapter 使用冻结输入缓存，前后表示相同。结果记录 `representation_state_source` 与 `representation_training_invariant`；新默认只影响新运行，旧结果不改写。其他梯度攻击仍取轮初候选梯度。
- 当前统一入口对 CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA 执行共享 exact-batch ProjRes；三者每 10 轮读取真实 one-batch FedSGD 上传，分别攻击 `classifier.0.weight`、最后视觉 Adapter 的 down 权重（ViT-B/32 为 `clip_model.vision_model.encoder.layers.11.adapter.down.weight`）与最后一个已训练视觉 Q 投影的 `lora_A`。旧 feature Adapter 仍攻击 `adapter.net.0.weight`。
- 逐层视觉 Adapter 的 ProjRes 默认 `attacked_parameter: null` 自动选择实际最后层，使用进入该 Adapter 的 CLS；`token_reduction: auto` 同样解析为 CLS。最后 Adapter 的 patch 输出不影响分类损失。显式层名与 `mean` 仍可复现首层/均值对照，其他审计的 key parameter 保持首层。成员/非成员比例不变：FedAvg 默认 M:M，FedSGD N:10N。为隔离换层变更，FedSGD 仍沿用输入 token 总数的保守秩上限和布局计数，FedAvg 无 batch 秩上限；保留 `paper_fedsgd_exact=false`、目标客户端训练后表示和原始负 L1 残差。零初始化导致首轮 down 上传为零时仅跳过 ProjRes。新结构只支持统一审计入口，不使用独立的缓存特征 ProjRes 或 `low_fpr_full`。2026-09-09 之前的逐层 Adapter 历史结果使用首层 down + mean，不改写。
- CLIP-LoRA ProjRes 默认攻击最后一个已训练视觉 Query 的 `lora_A`，`token_reduction: cls`，`auto` 同样使用 CLS；`mean` 可显式作对照。首层 Q/K/V 输入的 CLS 在图像间恒定，显式选择该组合会被拒绝；不得把历史恒定 CLS 分数当成隐私保护效果。统一审计器的 token 总数按实际候选样本数 × 每图 token 数计算，FedAvg 仍不施加 batch 秩上限。修复只影响新任务，历史结果不改写。
- FedSGD 成员严格等于该轮实际 batch，非成员与其他真实 Batch 攻击共享同一 1:10 标签匹配视图。普通无防御 MLP/旧 feature Adapter 的输入始终冻结，保留其原 `paper_fedsgd_exact` 标记；其余模型采用训练后表示，均为经验性适配，`paper_fedsgd_exact=false`。FedAvg 仍为完整客户端训练集协议。
- 三者统一 ProjRes 使用 `max_candidates: 32`、`min_nonmembers: 320`、`max_nonmembers: 320`，并与其他攻击共享 `predictions.csv`。CLIP-MLP 的独立严格入口仍保留用于单独诊断并生成自己的严格 JSON 输出，但不是统一 sweep 的替代品。

## 结果目录与日志规范

- 严格遵循“`results/` 下每个一级文件夹就是一次训练任务”。新任务目录名包含精确时间、模型、数据集、方法、seed 和目标客户端。
- 一次任务所需文件放在同一目录内，主要包括 `run_config.yaml`、`run.log`、模型/训练指标，以及 `privacy_audit/` 下的 `summary.json`、`predictions.csv`、`signals.pt`、`candidate_selection.pt` 和 `exact_batch_candidate_selection.pt`（按实际启用项生成）。三种 CLIP 模型的统一 ProjRes 复用这些输出；只有独立 ProjRes 路径生成 `projres_strict.json`。进程内日志写入同一 `run.log`，不要重新创建独立 `projres_strict.log`。
- `runs` 在汇总表中表示同一数据集/攻击分组包含的独立训练任务数量，不是额外的目录层级。代码仍可读取历史 `.../runs/...` 布局，但新实验不得继续生成该布局。
- 每个 `run.log` 开头只记录一次时间、训练任务、模型、数据集、防御、阶段和 GPU；后续子进程日志不再为每行重复这些固定字段。不要输出逐通信轮次的训练状态，只在配置的正式评估轮次和最终轮输出 `Progress`，记录轮次、loss、accuracy/MCC、学习率、参与客户端和审计累计数。不要重新引入只有脚本名而没有任务身份的日志命名。
- 每个任务结束后由共享服务器输出一次 `RUN RESULTS`：最终任务指标、精简隐私会计和对齐的攻击指标表；批量入口末尾输出 `EXPERIMENT OVERVIEW`，区分 OK/FAILED/PARTIAL 并显示最终主指标和耗时。不要把完整防御字典或逐客户端状态打印到控制台，它们保存在 `defense_summary.json`。显示值统一精度，Accuracy/TPR 显示百分比、MCC/AUC 显示小数；攻击优先读取 `reportable_metrics`，不可报告的值显示 `N/A`，不能回退到原始低 FPR TPR。
- 批量 CSV 保存 `primary_metric`、`primary_score`、`primary_metric_reportable` 和三个 FPR 下的 TPR，附成员/非成员数量及 FPR 分辨率。CSV 和控制台共用可报告性判断；不可报告的值留空，真实零值保留。固定候选池 Gradient-Diff 与真实 Batch 路径一样，直接使用 FedSGD 梯度；只有模型 delta 才按符号及学习率转换一次。

## 分析现有结果时的注意事项

- 先读取每个任务的 `run_config.yaml` 再判断实验协议。提交 `a789143` 之前产生的许多历史结果使用 Dirichlet `alpha=0.1`，不能默认视为 IID，也不应与新 IID 结果直接合并比较。
- 2026-08-09 修改前的已有结果均不使用当时切换后的协议：历史 MLP 是按本地样本数加权的 FedAvg；历史 Adapter 是遍历完整 local epoch 的 FedAvg，也按样本数加权。2026-08-09 切换后的实验为 MLP/Adapter 每类 16-shot、one-batch、客户端等权 FedSGD；Adapter/LoRA 又于 2026-09-10 将默认每类样本数改为 100。必须依据 `run_config.yaml` 区分，不能直接把历史曲线当作新配置基线。
- 旧非 IID 实验的 `TPR@0.001FPR` 曾明显受到成员/非成员标签分布不匹配影响。分析攻击有效性时至少同时检查类别直方图、按类别 TPR/FPR、标签匹配或类别加权 ROC，避免把类别识别能力误判为成员识别能力。
- `match_candidate_labels: false` 在 `low_fpr_full` 下不会执行精确标签配对；当前通过 IID 划分与分别分层抽样缓解标签偏移，但仍应在结果分析中实测标签分布，而不是假定完全一致。
- “让非成员来自目标客户端相同潜在分布”目前只完成了可行性分析，尚未加入正式候选采样代码。现有 α=0.1 Caltech101 候选池若保持完整成员类别比例，每客户端只能保留约 31–259 个比例匹配非成员，无法解析 `TPR@0.001FPR`；不要声称当前已经实现了低 FPR 精确分布匹配。
- 历史结果目录中可能残留早期生成的 `candidate_geometry.csv` 等文件；相关原型几何分析源代码已按用户要求撤销，不要把这些残留文件当成当前受支持的正式分析管线。

## 当前异构实验进展与已验证结论

- 当前 Dirichlet `alpha=0.1` 批次只有历史协议下的 CLIP-MLP/Caltech101 完整结束：`results/2026-08-08_17-09-20-748154_clip_mlp_caltech101_fedavg_seed42_targetall_c91a2b9bdd`。OxfordPets 同批任务停在第 235/300 轮且没有 `privacy_audit/summary.json`；其余三个 MLP 数据集和全部 Adapter 异构任务没有完成。不要把这批结果描述为跨模型、跨数据集结论。
- Caltech101 α=0.1 的原始客户端宏平均 AUC 显著高于 IID，例如 `avg_cosine` 为 0.8991、`fedmia_cosine` 为 0.8987；但按“客户端 × 类别”消除分数基线后的 AUC 分别只有 0.5640 和 0.5478。每客户端成员/非成员类别 TV 从 IID 的约 0.3066 升到 α=0.1 的约 0.7834，说明原始提升主要受客户端类别分布泄漏驱动。
- 这类异构结果仍代表真实的客户端分布隐私泄漏，但不能直接等同于“同类别内具体样本的 record-level membership inference”。报告时应同时给出原始 AUC/低 FPR TPR、类别分布统计和类内/类别条件指标。
- 已生成但被 Git 忽略的分析产物包括：`analysis_scripts/alpha01_vs_iid_progress_20260808.md`、`analysis_scripts/heterogeneous_auc_curves_caltech101.png`、对应 CSV 和可复现绘图脚本。异构 AUC 曲线使用第 10、20、…、300 轮信号；单轮攻击的逐轮线是诊断轨迹，正式 Blackbox-Loss/Grad-Cosine 仍只取最后观测轮。

## 性能与验证

- 主要耗时来自余弦类攻击的逐样本梯度计算；候选样本数和逐轮攻击数量决定大部分审计时间。若继续优化，优先考虑 CLIP-MLP/Adapter 可解析或向量化的梯度余弦、批量客户端前向和在线聚合，且必须保持攻击定义不变。
- 文本审计默认 `audit.grad_sample_backend=auto`、`audit.grad_sample_chunk_size=4`，分块计算逐记录梯度，在每次信号计算内缓存上传向量的 float64 表示并批量点积。缓存不超过 `audit.gradient_update_cache_mb=2048` 且不超过当前空闲显存四分之一时放在 GPU，否则使用 CPU；设为 0 强制 CPU。不得跨客户端状态/轮次复用缓存；余弦继续求真实标签 CE 梯度，Gradient-Diff 继续求所有标签损失之和的梯度。`audit.grad_sample_backend=loop` 可回到逐记录求导。
- 三个 CLIP 模型的常规余弦/Gradient-Diff 审计也使用上述分块与 float64 归约，MLP/旧 feature Adapter 可复用缓存图像特征，逐层视觉 Adapter/LoRA 直接计算原始图像梯度；不再保留整个候选池的梯度或计算未使用的梯度特征。`signal_storage=full` 需要额外信号时保留原完整信号路径。
- BERT Adapter/LoRA 与 GPT2 Adapter 默认 `performance.evaluation_backend=shared`，全局评估只加载一次共享模型，保留各客户端测试分区、batch 边界及本地状态；`clients` 可复核原逐客户端加载路径。默认 `performance.enabled=true`、`performance.cuda_events=true`，在 `performance_summary.json` 记录累计阶段耗时。CUDA event 延迟读取，避免每块求导强制同步；父子阶段为包含关系，不能直接相加。计时覆盖服务器训练过程及文件输出，不包含模型/数据加载。
- 修改实验配置后先干运行核对最终参数：
  - `python scripts/run_privacy_experiments.py --dry-run --max-runs 1`
- 当前干运行应看到：MLP/Adapter/LoRA 均为 `federated.aggregator: fedsgd` 和 `federated.aggregation_weighting: uniform`；MLP 默认为每类 16 张，Adapter/LoRA 默认为每类 100 张且支持配置全量；三者均启用统一 ProjRes。
- 测试环境使用 `/root/.local/share/mamba/envs/pfedba/bin/python`。小范围修改优先只运行直接相关的测试文件和 `git diff --check`；不要习惯性执行完整套件。
- `/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q` 是日常快速核心回归；完整本地套件必须显式使用 `python -m pytest -q tests`，仅在修改共享审计器、聚合核心、候选池协议或准备高风险发布时运行。
