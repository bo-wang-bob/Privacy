# Repository guidance

## 用户偏好与基本规则

- 默认使用中文沟通。
- 数据分析不要生成 HTML 文件；优先输出 Markdown、CSV、JSON、PNG/PDF 等必要产物。
- 给出运行方式时尽量把稳定参数写进配置或脚本默认值，减少用户每次手动传参。
- `results/` 中的真实实验结果属于用户数据。除非用户明确要求，不要删除、覆盖或批量改写已有结果。
- 工作区可能包含用户自己的未提交修改；只处理当前任务涉及的文件。
- Git 提交信息和提交日志应在合理范围内尽量详细准确：标题概括核心协议或功能变化，正文列出训练协议、候选定义、兼容性影响和验证结果；避免只写笼统的 `update`、`fix` 或 `changes`。

## 当前统一实验入口

- 唯一批量入口是 `scripts/run_privacy_experiments.py`。它支持单个或多个模型、数据集、攻击、防御、seed 和目标客户端；模型 × 数据集 × 防御 × seed × 目标客户端展开为独立任务，同一组多攻击共享一次训练。
- 不传 `--models` 时保持原 CLIP 入口范围，依次运行 CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA；`--models all` 覆盖 7 个正式模型。
- 只运行单个模型时使用 `python scripts/run_privacy_experiments.py --models clip_mlp`；多模型用逗号分隔。
- `--attacks all` 按模型解析能力集：ResNet18 只有 `fedmia_loss`，PEFT 模型为全部 11 种；`--attacks none` 是纯训练。
- `--defenses all` 只展开模型正式支持的防御。显式攻击与模型不兼容时跳过该模型并打印原因；显式防御或数据集不兼容时跳过对应组合。
- `configs/experiment_catalog.yaml` 维护能力矩阵、防御深度覆盖和别名；`configs/models/` 下 7 个基线维护模型训练与默认候选协议。不要重新为数据集、攻击或防御复制整份模型 YAML。
- 统一入口在应用全部 `--set` 后自动推导启用的 exact-batch ProjRes 候选参数：普通训练使用最终 batch 大小及成员/非成员比例，`record_dp`/`www` 使用 `0/0/0` 动态候选。改变 `batch_size` 或混跑 `none,record_dp,www` 无需手动同步这三个值。显式传入的 ProjRes 候选参数保留并接受协议校验；干运行显示最终候选参数。
- CLIP 默认学习率由模型基线设置：CLIP-MLP 为 `0.1`，CLIP-Adapter 为 `0.001`，CLIP-LoRA 为 `0.0002`。应以统一脚本干运行打印的最终参数为准。
- 三个 CLIP 基线均显式使用 IID。不要在正常 IID 实验中传 `--dirichlet-alpha`；仅传该参数会自动切换为 `dirichlet`。如同时显式传 `--partition-mode iid --dirichlet-alpha 0.1`，显式 partition mode 优先，alpha 仅作为未使用的配置值保留。
- CLIP-MLP 和 CLIP-Adapter 使用每类 16 张训练图像的 Few-shot 训练集；CLIP-LoRA 使用完整训练集。Few-shot 先在全局训练分区内按类确定性抽取，再进行客户端划分；独立 evaluation/test 分区不截断。
- 默认数据集为 Caltech101、OxfordPets、Flowers102、Food101 和 CIFAR100。
- 三个模型在全部默认数据集上统一使用 10 个客户端和 batch size 32；CLIP-MLP 使用 150 个通信轮次，CLIP-Adapter/CLIP-LoRA 使用 300 个通信轮次。三者均使用 FedSGD，每个客户端每轮只执行 1 个 mini-batch/1 次 optimizer step；`local_epochs: 1` 是协议校验值，不表示遍历完整本地数据集。
- 三种微调方式的服务器端聚合都使用 `aggregation_weighting: uniform`，即对本轮参与客户端上传的梯度直接等权平均，不按客户端本地样本数或实际 batch 大小加权。三者任务目录方法名均为 `fedsgd`。
- 正常任务指标默认按 `eval_interval: 5` 在已完成的第 5、10、15、…轮评估；若总轮数不能被 5 整除，最后一轮仍会额外评估。`training_metrics.csv` 使用相同的一基轮次编号。

## 当前 WWW 防御（原 ICLR）

- 原观测型 `iclr` 已更名并更新为实际参与训练的 `www`；配置和新产物使用 `www_*`，历史 `results/` 与 `iclr_*` 产物不改写。
- 支持三个 CLIP 模型及 BERT Adapter/LoRA。每个真实训练 batch 按上一轮防御后模型的 `M_i = loss(theta_-k, z_i) - loss(theta_k, z_i)` 升序排序，默认 `defense.www_tail_fraction=0.8`，按固定期望 batch 大小 b 预设尾部宽度 `ceil(0.8 * b)`，实际 batch 有 n 条样本时，尾部包含最后 `min(n, ceil(0.8*b))` 条；短于尾部的 batch 使用重要性函数的最右侧区间。首轮或缺少上一轮参考状态时统一裁剪并加噪。
- 所有样本联合 L2 裁剪到默认 `defense.max_grad_norm=8`，再乘 INO-SGD 的 Beta 尾部函数积分权重；默认 Beta(1,1)。先裁剪再缩放，不能替换成直接裁剪到缩放后的阈值。
- 默认全程预算 `defense.target_epsilon=3`、`delta=1e-5`。与普通 `record_dp` 共用 Poisson 采样，按 `add_remove`、敏感度 C 和 Poisson sampled-Gaussian RDP 校准全程噪声；梯度和加噪后除以固定期望 batch 大小。空 batch 不重抽，仍上传纯噪声并计入预算。防御每轮运行，诊断间隔不影响训练保护。
- WWW 与 BERT Adapter Record-DP 默认 `defense.grad_sample_backend=auto`、`defense.microbatch_size=4`，使用分块 batched VJP；分块只影响计算，整个真实 batch 汇总后仍只加一次噪声和执行一次 FedSGD step。共享 Transformer 启用 gradient checkpointing 时 `auto` 选择 `loop`；ResNet18 保留原 `vmap`。性能说明和独立基准见 `docs/fedsgd_performance.md`。
- 默认不导出未私有化分数诊断；固定噪声或 `release_private_diagnostics=true` 将使 `formal_dp_enabled=false`。受保护发布范围为防御后的上传与模型，审计私有信号与标签属于本地研究资料。
- WWW 下 ProjRes 按真实 Poisson batch 动态构造候选，三个候选规模参数均为 0；取消无噪声 batch 秩上限，`paper_fedsgd_exact=false`。空 batch 跳过真实 Batch 攻击并记录原因，固定候选攻击继续执行。入口与完整公式见 `docs/defenses.md`。

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
- BERT-LoRA 默认在全部 12 层自注意力 Query/Value 投影中训练 `rank=16`、`alpha=32`、dropout `0.1` 的 LoRA 因子，并同时训练分类头；冻结 BERT 主干不上传。服务器与 CLIP-LoRA 一样分别聚合同名 `lora_A`、`lora_B`，不先合成稠密 `BA` 更新。
- BERT Adapter/LoRA 均支持全部 11 种注册攻击，并使用上述 5 种固定候选攻击与 6 种真实 Batch 攻击划分；完整真实 Batch 候选为 16/160。
- BERT-LoRA 默认配置为 `configs/models/bert_lora.yaml`，默认只展开 CoLA，SST-5/IMDb 仍可通过 `--datasets` 显式选择；可由 `scripts/run_privacy_experiments.py --models bert_lora` 启动。ProjRes 观察首层 Query 的 `lora_A` 上传，并以 attention-mask 加权的有效 token 输入均值作为每个样本的表示，避免首层 CLS 在不同文本间恒定导致分数退化。

## 按需审计频次

- 当前三个 CLIP 配置不设置共享的 `audit_interval`；逐轮攻击全部使用各自的显式间隔。
- CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA 的全部 11 种攻击每 10 轮测量。MLP 测量到第 150 轮，Adapter/LoRA 测量到第 300 轮。
- 审计间隔按已完成的通信轮数计数；例如 `attack_audit_intervals: 10` 对应零基内部索引 9、19、…，而不是索引 0、10、…。三者每轮均只训练 1 个真实 batch。
- 三种 CLIP 模型的 `blackbox_loss` 和 `grad_cosine` 都属于真实 Batch 协议，必须按配置轮次分别审计。
- 同一任务运行多个攻击时，调度器取各攻击所需轮次的并集，但仍只计算该轮实际需要的信号族。只要包含任一逐轮攻击，对应信号仍会每轮计算，这是协议需求而不是调度失效。

## ProjRes 特例

- 当前统一入口对 CLIP-MLP、CLIP-Adapter 和 CLIP-LoRA 执行共享 exact-batch ProjRes；三者每 10 轮读取真实 one-batch FedSGD 上传，分别攻击 `classifier.0.weight`、`adapter.net.0.weight` 与首个视觉 Q 投影的 `lora_A`。
- 成员严格等于该轮实际 batch，非成员与其他真实 Batch 攻击共享同一 1:10 标签匹配视图，元数据中的 `paper_fedsgd_exact` 应为 `true`。
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
- 2026-08-09 本轮修改前的已有结果均不使用当前协议：历史 MLP 是按本地样本数加权的 FedAvg；历史 Adapter 是遍历完整 local epoch 的 FedAvg，也按样本数加权。新实验则是 MLP/Adapter 每类 16-shot、one-batch、客户端等权 FedSGD。必须依据 `run_config.yaml` 区分，不能直接把历史曲线当作新配置基线。
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
- 三个 CLIP 模型的常规余弦/Gradient-Diff 审计也使用上述分块与 float64 归约，MLP/Adapter 复用缓存图像特征，LoRA 直接计算原始图像梯度；不再保留整个候选池的梯度或计算未使用的梯度特征。`signal_storage=full` 需要额外信号时保留原完整信号路径。
- BERT Adapter/LoRA 与 GPT2 Adapter 默认 `performance.evaluation_backend=shared`，全局评估只加载一次共享模型，保留各客户端测试分区、batch 边界及本地状态；`clients` 可复核原逐客户端加载路径。默认 `performance.enabled=true`、`performance.cuda_events=true`，在 `performance_summary.json` 记录累计阶段耗时。CUDA event 延迟读取，避免每块求导强制同步；父子阶段为包含关系，不能直接相加。计时覆盖服务器训练过程及文件输出，不包含模型/数据加载。
- 修改实验配置后先干运行核对最终参数：
  - `python scripts/run_privacy_experiments.py --dry-run --max-runs 1`
- 当前干运行应看到：MLP/Adapter/LoRA 均为 `federated.aggregator: fedsgd` 和 `federated.aggregation_weighting: uniform`；MLP/Adapter 为每类 16-shot，LoRA 使用完整训练集；三者均启用统一 ProjRes。
- 测试环境使用 `/root/.local/share/mamba/envs/pfedba/bin/python`。小范围修改优先只运行直接相关的测试文件和 `git diff --check`；不要习惯性执行完整套件。
- `/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q` 是日常快速核心回归；完整本地套件必须显式使用 `python -m pytest -q tests`，仅在修改共享审计器、聚合核心、候选池协议或准备高风险发布时运行。
