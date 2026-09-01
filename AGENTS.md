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
- CLIP 默认学习率由模型基线设置：CLIP-MLP 为 `0.1`，CLIP-Adapter 为 `0.001`，CLIP-LoRA 为 `0.0002`。应以统一脚本干运行打印的最终参数为准。
- 三个 CLIP 基线均显式使用 IID。不要在正常 IID 实验中传 `--dirichlet-alpha`；仅传该参数会自动切换为 `dirichlet`。如同时显式传 `--partition-mode iid --dirichlet-alpha 0.1`，显式 partition mode 优先，alpha 仅作为未使用的配置值保留。
- CLIP-MLP 和 CLIP-Adapter 使用每类 16 张训练图像的 Few-shot 训练集；CLIP-LoRA 使用完整训练集。Few-shot 先在全局训练分区内按类确定性抽取，再进行客户端划分；独立 evaluation/test 分区不截断。
- 默认数据集为 Caltech101、OxfordPets、Flowers102、Food101 和 CIFAR100。
- 三个模型在全部默认数据集上统一使用 10 个客户端和 batch size 32；CLIP-MLP 使用 150 个通信轮次，CLIP-Adapter/CLIP-LoRA 使用 300 个通信轮次。三者均使用 FedSGD，每个客户端每轮只执行 1 个 mini-batch/1 次 optimizer step；`local_epochs: 1` 是协议校验值，不表示遍历完整本地数据集。
- 三种微调方式的服务器端聚合都使用 `aggregation_weighting: uniform`，即对本轮参与客户端上传的梯度直接等权平均，不按客户端本地样本数或实际 batch 大小加权。三者任务目录方法名均为 `fedsgd`。
- 正常任务指标默认按 `eval_interval: 5` 在已完成的第 5、10、15、…轮评估；若总轮数不能被 5 整除，最后一轮仍会额外评估。`training_metrics.csv` 使用相同的一基轮次编号。

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
- 修改实验配置后先干运行核对最终参数：
  - `python scripts/run_privacy_experiments.py --dry-run --max-runs 1`
- 当前干运行应看到：MLP/Adapter/LoRA 均为 `federated.aggregator: fedsgd` 和 `federated.aggregation_weighting: uniform`；MLP/Adapter 为每类 16-shot，LoRA 使用完整训练集；三者均启用统一 ProjRes。
- 测试环境使用 `/root/.local/share/mamba/envs/pfedba/bin/python`。小范围修改优先只运行直接相关的测试文件和 `git diff --check`；不要习惯性执行完整套件。
- `/root/.local/share/mamba/envs/pfedba/bin/python -m pytest -q` 是日常快速核心回归；完整本地套件必须显式使用 `python -m pytest -q tests`，仅在修改共享审计器、聚合核心、候选池协议或准备高风险发布时运行。
