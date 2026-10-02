# 联邦训练方法与攻击口径

## 统一切换 FedSGD / FedAvg

生成视图防御 `risk_synthesis` 的 direct 模式也支持两种方法；FedSGD 使用原始 batch 成员和生成视图真实梯度，详见 [生成视图的 FedSGD 协议与命令](risk_synthesis_fedsgd.md)。

六个 PEFT 模型（CLIP-MLP/Adapter/LoRA、BERT Adapter/LoRA、GPT2 Adapter）
均支持两种方法。模型 YAML 保持默认 FedSGD；统一入口按
模型 × 数据集 × 防御 × seed × 目标客户端 × 方法生成独立任务。
三个 CLIP 模型在两种方法下均默认只运行 CIFAR100、Food101；其余图像数据集
通过 `--datasets` 显式选择，`--datasets all` 展开全部五个支持的数据集。
ResNet18 不传方法选择时保留原有 300 轮 FedAvg 基线；显式 `--methods fedavg`
也使用 100 轮，并退出要求完整 300 轮的 `paper_protocol` 模式。

```bash
python scripts/run_privacy_experiments.py --models clip_mlp,clip_adapter,clip_lora,bert_adapter,bert_lora,gpt2_adapter --methods fedsgd,fedavg --defenses none
```

`--methods fedavg` 只运行 FedAvg；`--methods fedsgd` 切回 FedSGD；`all` 展开
各模型支持的方法。不传时保持模型基线。运行前可追加 `--dry-run` 核对配置。
同一模型/数据集/防御/seed/客户端下，两种方法各训练一次，各自共享全部所选攻击。

| 项目 | FedSGD | FedAvg |
| --- | --- | --- |
| 默认通信轮数 | CLIP-MLP 150；CLIP-Adapter/LoRA 1000；BERT/GPT2 500 | 全部 100 |
| 本地训练 | 每轮 1 个打乱后的 mini-batch | 每轮遍历 `local_epochs` 个完整本地 epoch，默认 1 |
| 短批次 | 保留，按实际大小求均值 | 保留，作为本地 optimizer step |
| 协议消息 | 可训练参数的真实梯度 | 本地训练后参数相对轮初全局参数的 delta |
| 默认聚合权重 | 等权 | 本地训练集样本数占比 |
| 优化器状态 | 无动量 SGD | 每客户端每轮重新创建，epoch 之间保留状态 |
| LoRA 聚合 | 分别聚合同名 A/B 梯度 | 分别聚合同名 A/B 参数，不合成 BA |

`--local-epochs 3` 调整 FedAvg 本地训练次数；FedSGD 仍要求该值为 1。
`--aggregation-weighting uniform` 可令 FedAvg 等权，便于只对比本地训练协议。
方法默认值集中在 `configs/experiment_catalog.yaml` 的 `method_overrides`。
数据划分、few-shot、batch size 和学习率均沿用模型基线。
FedAvg 的通信轮数统一为 100；FedSGD 继续使用模型基线轮数。`--rounds` 和
`--set num_global_iters=...` 可显式覆盖轮数，后者优先；混跑时覆盖作用于两种方法。
只缩短 FedAvg 不需要手动传轮数，也不需要复制模型 YAML。
FedAvg 下三个 CLIP 模型的攻击间隔与 BERT 一致：`blackbox_loss`、`grad_cosine`、
`gradient_diff`、`score_diff`、`score_ratio`、`projres` 每 50 轮一次，
`loss_series`、`avg_cosine`、`fedmia_loss`、`fedmia_cosine`、`fta` 每 10 轮一次。
100 轮中分别测量第 50/100 轮和第 10/20/…/100 轮。ProjRes 的声明间隔同步为 50。
此覆盖保存在 `method_overrides.fedavg.models`，FedSGD 的 CLIP 基线仍全部每 10 轮。
相同轮数下 FedAvg 通常执行更多 optimizer steps，比较效果时须同时考虑训练预算。

配置解析顺序为模型基线 → 方法默认值 → 防御覆盖 → CLI 参数 → `--set`。
最后根据实际 `aggregator` 推导攻击候选协议。`--set aggregator=fedavg` 也可切换，
但会保留其他基线参数（例如等权）；需要标准方法默认值时用 `--methods`。
视觉模型优化器配置使用 `fedavg.*` / `fedsgd.*`，文本模型使用 `optimization.*`。
默认均为无动量、无 weight decay 的 SGD。FedSGD 拒绝会改变真实梯度语义的
AdamW、动量、weight decay 或优化器侧裁剪；FedAvg 可配置 SGD/AdamW。

兼容性修正：普通 `none` 防御分支现在实际使用所配置的优化器、momentum 和
weight decay；旧实现曾在该分支固定创建无动量 SGD。这也修正了 ResNet18 的
配置执行行为。因此旧结果中仅有 `run_config.yaml` 的优化器字段，不足以证明
这些参数曾实际生效；比较此修改前后的相关实验还应核对代码版本。

### FedAvg 的成员推理评价

全部 11 种攻击使用固定候选：目标客户端原始训练集的完整 `M` 个成员，及全局
独立 evaluation 池中默认等量 `M` 个从未训练非成员。非成员按类别尽力匹配，
类别不足时确定性补足；实际直方图、匹配标志及 TV 距离保存在候选文件中。
不再将本轮最后一个 batch 视作 FedAvg 上传的成员集合，也不按重复 epoch
重复计数成员。CoFedMID/Poisson-DP 可能只访问部分训练记录，仍评价原始训练集
成员身份，不能把该身份解释为本轮实际访问；曝光和 DP 步数另按各防御记录。

- `blackbox_loss`、`grad_cosine`、`gradient_diff`、`score_diff`、`score_ratio`、
  `projres` 使用 `audit.client_train_membership_attacks`，分别计算审计轮的分数，
  最终报告最后一个实际观测轮。迁移原 `exact_batch_membership_attacks` 配置时自动路由。
- 另外五种攻击沿用固定候选的时序/跨客户端定义和原审计调度；不从真实标签挑选最佳轮。
- 损失类攻击的 post-state 是目标客户端本地训练完成的可观测模型；Score-Diff/Ratio
  的 pre-state 是该轮训练开始的全局模型。
- 梯度类攻击在轮初模型求候选梯度，客户端向量用 `-delta / round_learning_rate`，
  只转换一次。普通固定学习率 SGD 下这是各本地步骤梯度之和；多个模型位置上的梯度
  并不等于单 batch 梯度。使用动量、AdamW 或扰动时只能解释为累计更新代理量。
  Gradient-Diff 保留 `2<u,g_sum_labels> - ||g_sum_labels||²`；余弦不受正标量缩放影响。
- ProjRes 使用目标客户端训练后表示：由 `轮初参数 + 目标客户端上传 delta` 重建模型，
  同一模型提取成员和非成员表示，再投影到该 delta 的行空间，按原始负 L1 残差排名；
  `paper_fedsgd_exact=false`、`batch_rank_bound=null`。多个本地步骤及中间表示变化
  会破坏单步精确条件，因此这是经验性适配。零更新只跳过该轮 ProjRes，记录
  `zero_observed_update`，其余攻击继续。
- `projres.max_candidates/min_nonmembers/max_nonmembers` 自动清零，表示使用完整
  客户端候选，不按 batch 截断。FedAvg 的候选数量与训练 batch size 解耦。

固定候选保存在 `privacy_audit/candidate_selection.pt`，单轮更新攻击的轮次与身份
保存在 `client_train_update_candidate_selection.pt`；摘要和信号使用
`client_train_membership` / `client_train_update_observations`。
FedSGD 继续使用原 `exact_batch_*` 文件及字段；历史结果不改写。
FedAvg 的全部攻击统一报告 AUC 和 TPR@10%/1%/0.1%FPR；FedSGD 的六种
真实 batch 攻击继续只报告 10%/1%FPR。所有低 FPR 指标按实际独立非成员数量判定是否可报告，
重复轮次、重复 epoch 均不提高 FPR 分辨率。
默认 FedSGD batch 成员与 FedAvg 训练集成员的问题不同，跨方法比较时应同时列出
候选定义、数量和类别分布；五种固定候选攻击的训练集成员口径相同。

支持原有 WWW、CoFedMID 及 BERT Adapter Record-DP 的 FedAvg 路径。
WWW 每个本地 batch 使用同一上一通信轮的 own/other 参数参考，逐批更新当前模型；
诊断与训练步数随 epoch 累计。Record-DP 按实际计划的多步 Poisson 机制重新校准预算。
CoFedMID 保留实际聚合权重下的参数空间扰动抵消，训练成员标签保持原始分区。
`local_client_dp` 仅支持 one-batch FedSGD，方法混跑时明确跳过不兼容组合。
旧 WWW `release_private_diagnostics=true` 的 post-round batch 分析仍只支持 FedSGD；
可选的 `www_record_diagnostics` 逐批诊断支持两种方法，默认关闭。

任务名、控制台概览、批量 CSV/manifest 均标明方法；CSV 同时记录 `local_epochs`、
聚合权重和成员定义。`federated_method_summary.json` 保存上传类型、训练协议、
优化器配置和更新量解释。当前验证覆盖 CPU 小模型端到端，不代表真实数据的攻击效果。

以下模型细节描述默认 FedSGD 协议；选择 FedAvg 时应用上述训练与评价规则。

## FedAvg (`aggregator: fedavg`)

正式支持上述六个 PEFT 模型的多 batch 本地训练和参数聚合。另保留冻结 CLIP
上的 soft-prompt 兼容入口，学习 token 拼接在手工模板之前。

## PromptFL (`aggregator: promptfl`)

对应论文 *PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models*。每个客户端在冻结 CLIP 上只训练一组共享 CoOp context token，并最小化本地交叉熵；服务器按客户端训练样本数执行 FedAvg。

严格的 `promptfl` 入口使用论文式 `[SOS] [learned context] [class] [EOS]` 构造。论文：[arXiv](https://arxiv.org/abs/2208.11625)。

## CLIP-LoRA 的因子级聚合

`model_type: clip_lora` 冻结 CLIP 主干，在图像与文本编码器注意力投影中插入
`W_eff = W_0 + (alpha/sqrt(r)) B A`。客户端只优化并上传 `lora_A`、`lora_B`，
服务器分别线性聚合同名因子。当前正式 sweep 使用 one-batch FedSGD，并对本轮
参与客户端的各同名因子梯度等权平均：

```text
A_global = (1 / |S|) sum_k A_k
B_global = (1 / |S|) sum_k B_k
```

2026-09-09 起，模型基线将 `clip_lora.rank` 从 2 提高为 32，保留 `alpha=1`、
`scaling=sqrt_rank`、dropout `0`、batch size 32 和学习率 `0.01`。
该配置同时用于图像与文本编码器的全部 Q/K/V LoRA，并由 FedSGD/FedAvg 共用；
LoRA 因子参数量为原 rank=2 的 16 倍，冻结主干大小不变。这是新的训练配置，
不保证 ProjRes 的实际上传满秩或攻击效果提升。旧配置对照可通过统一入口传入
`--set clip_lora.rank=2`；已有结果和 checkpoint 不改写，不同 rank 的权重不能直接互载。

该基础方案不聚合冻结主干，也不先把 `B_k A_k` 合成稠密更新；后者与分别平均
因子并不数学等价，属于需要另行实现和对照的聚合变体。

内存模型采用“一个共享 CLIP + 每客户端独立 LoRA”的结构。共享模型保存冻结
CLIP 主干和服务器全局 LoRA 工作槽位；每个客户端模型类只注册该客户端的 A/B
参数。客户端执行时临时把自己的 Parameter 绑定到工作槽位，使优化器直接更新
该客户端参数；执行结束后恢复全局槽位。服务器收到和聚合的状态字典因此只含
同名的 `lora_A/lora_B`，不会包含冻结 CLIP 权重。

## FedLLM Adapter 的同步 FedSGD

`bert_adapter` 和 `gpt2_adapter` 按 Deng 等人的实验协议，在每个 Transformer
block 输出后插入残差瓶颈：

```text
h(x) = x + W_up ReLU(W_down x + b_down) + b_up
```

down-projection ratio 为 2，即瓶颈宽度等于主干隐藏宽度的一半。预训练主干冻结，
新增 Adapter 和任务分类头可训练，其中 SST-5 为五分类，CoLA 和 IMDB 为二分类。
数据入口支持 SST-5、CoLA 和 IMDB；前两者
使用 `train/validation`，IMDB 使用 `train/test`，评估 split 从不参与训练。
默认使用 30 个 IID 客户端，BERT 与 GPT2-Large 均使用 batch size 16。
BERT Adapter 三个数据集均训练 500 轮并保持学习率 `0.005`；GPT2-Large 训练 500 轮并保持
`0.001`。Adapter 的 down-projection 使用主干 initializer range 做小随机初始化，
up-projection 与 bias 置零，使残差分支在初始化时严格保持主干隐藏表示；全部可训练参数
使用同一个标量学习率，客户端梯度不执行 norm clipping。
两者均由所有客户端同步参与；每个客户端每轮在一个 batch 上计算一次无 momentum、
无 weight decay 的梯度并上传，服务器等权聚合真实梯度后执行
`theta_(t+1) = theta_t - learning_rate * mean(g_k)`，再下发新的全局模型。
这些普通任务优化不关闭或绕过审计：上传仍是攻击器观察到的真实 one-batch 梯度，
全部 11 种注册攻击均保持启用，ProjRes 每 50 轮运行一次。
普通任务评估中，SST-5 与 IMDB 以 accuracy 为主；CoLA 以 MCC 为主，并继续记录
accuracy 以兼容已有结果分析。

内存中只保留一个共享预训练主干。客户端持有独立的 Adapter/分类头状态并常驻 CPU，
训练或评估时临时移动到主干设备并绑定，结束后立即卸载回 CPU；客户端上传和最终
checkpoint 均不包含冻结的 BERT/GPT2 参数。

### BERT-LoRA

`bert_lora` 使用相同的 30 客户端、one-batch、等权 FedSGD 协议。BERT 主干冻结，
默认在全部 12 层自注意力的 Query 和 Value 投影中加入
`W_eff = W_0 + (alpha/r) B A`，使用 `rank=16`、`alpha=32`、LoRA dropout `0`，
并同时训练任务分类头，分类头 dropout 保持 `0.1`。每个客户端只在 CPU 保存自己的 LoRA 因子和分类头；执行时
临时绑定到共享 BERT 主干。

2026-09-09 起，模型基线仅将 `lora.dropout` 从 `0.1` 改为 `0`，用于隔离 LoRA
分支随机掩码对训练输入与 ProjRes 审计表示一致性的影响；rank、alpha、缩放规则、
batch size 和学习率保持不变。FedSGD/FedAvg 共用此默认值；历史结果不改写，
原 dropout 对照可通过统一入口传入 `--set lora.dropout=0.1`。

聚合与 CLIP-LoRA 一致：服务器分别线性平均每个同名 `lora_A`、`lora_B`，不会先
合成为稠密的 `BA` 更新；分类头参数也按相同客户端权重线性聚合。FedSGD 下等价于
分别平均各可训练张量的真实 batch-mean 梯度，再执行一次全局 SGD step。默认配置为
`configs/models/bert_lora.yaml`，默认数据集为 CoLA，训练 500 轮并使用恒定学习率 `0.015`。
ProjRes 默认观察最后一个已训练 Query 投影的 `lora_A` 上传，并在目标客户端训练后
模型提取该层输入 CLS；显式首层 Query/Key/Value + CLS 会被拒绝，mean 仍可作对照。

`scripts/run_privacy_experiments.py` 是全仓库统一入口，可按模型、数据集、防御、
seed 和目标客户端展开独立进程；同一任务所选的多种攻击共享训练。
`scripts/run_fedllm_adapter.py` 只执行统一入口传入的单个文本任务。统一模式支持
`--models`、`--datasets`、`--attacks`、`--defenses`、`--gpus`、`--jobs`、
`--dry-run` 和 `--max-runs`，默认单 GPU 串行调度。dry-run 只打印命令；
完整解析配置仅在对应任务实际启动后写入终端和该任务的 `run.log`。

这些文本模型复用通用审计器的 Blackbox/Fed-loss、Loss-Series、Grad-Cosine、
Avg-Cosine、两种 FedMIA 信号，以及 Gradient-Diff、Score-Diff、Score-Ratio 和
FTA。默认只审计客户端 0。BERT Adapter、BERT-LoRA 与 GPT2-Large 的 Blackbox-Loss、Grad-Cosine、Gradient-Diff、
Score-Diff、Score-Ratio 和 ProjRes 将每轮真实上传 Batch 定义为成员，并从全部客户端的独立 evaluation 分区
逐类别抽取 10 倍从未训练的非成员；每轮候选集独立构造和评估。其余攻击从目标客户端
训练分区使用完整的 `M` 个历史成员，再从相同 evaluation 池抽取类别尽力匹配的
等量 `M` 个非成员，并跨轮复用固定候选池。余弦攻击比较候选样本对全部可训练
PEFT/分类头参数
的精确梯度与服务器收到的真实 FedSGD 上传；GPT2-Large 使用逐样本流式点积控制内存。

固定候选攻击的主视图为 `M/M`；另从中派生最多 100/100 的论文对照视图。后者复用
同一分数，不重新执行模型前向或梯度计算。主视图的最低可解析 FPR 由 `M` 决定；
论文视图只有 100 个非成员，不能解析 TPR@0.1%FPR。六种真实 Batch 攻击固定为
16 成员/160 非成员，只
统计 AUC、TPR@10%FPR 与 TPR@1%FPR，不生成 TPR@0.1%FPR。

BERT Adapter 的统一入口默认启用 WWW 风险损失防御，BERT-LoRA 可通过
`--defenses www` 显式启用。每个真实 batch 按上一轮防御模型间的逐样本损失差升序排序，
使用普通打乱后分批，每轮取下一批；最高风险的 `ceil(0.8*n)` 条额外加入
`lambda*r_i*abs(p_i-q_i)`，其中 `r_i` 按尾部风险名次递增，`q_i` 为上一轮其他客户端
参数聚合参考模型的冻结真实类概率。默认 lambda=1；所有样本保留交叉熵，
整批损失按实际样本数求均值并执行一次 FedSGD 更新，不裁剪、不加噪、不提供 DP 保证。
预算、裁剪阈值和旧 Beta 参数在 WWW 中清空，不影响普通 Record-DP。
首轮或缺少上一轮客户端参考状态时仅用交叉熵。算法及诊断字段详见
[WWW 文档](defenses.md#www)。GPT2-Large 当前保持 `defense.name: none`。

ProjRes 每 50 个已完成通信轮观察目标客户端真实 one-batch 上传：BERT/GPT2 Adapter
默认攻击最后层 down-projection，BERT-LoRA 默认攻击最后一个已训练 Query 的 `lora_A`。
BERT 使用该层输入 CLS，GPT2 使用最后一个有效 token。候选表示在目标客户端的
训练后模型下提取；FedSGD 由 `轮初参数 - 学习率 × 上传梯度` 重建公开 SGD 端点。
BERT Adapter、BERT-LoRA 与 GPT2-Large 的成员和非成员直接复用共享真实 Batch 候选视图，即当轮
`N` 个成员及按标签匹配的 `10N` 个从未训练 evaluation 样本；完整 Batch 时为
16/160。结果与其他攻击统一写入审计器输出，`projres.max_candidates: 16` 不会截断
实际上传 Batch。训练后表示可能不同于产生上传梯度时的表示，因此文本模型与
CLIP Transformer Adapter/LoRA 均标记 `paper_fedsgd_exact=false`。MLP/旧 feature Adapter
的冻结输入前后相同，保留原论文条件标记。WWW 修改训练损失，也标记 `false`。
当前概率差异正则的逐样本梯度与 CE 共线，仍保留 batch
秩上限，且 `attacked_parameter_perturbed=false`，成员身份仍由真实 batch 确定。

所有 ProjRes 结果记录 `representation_state: client_post_update_model`、
`representation_state_source` 和 `representation_training_invariant`。两个候选组共享同一个
目标客户端端点；独立入口按客户端重新计算动态非成员表示，提取结束恢复共享模型。
该端点由目标上传重建，不是聚合后的全局模型。FedSGD 若本地使用 momentum/Adam 等
优化器，公开梯度只能给出上述 SGD 端点，不能据此恢复隐藏的优化器状态。
FedAvg 最终状态仍不能还原所有中间本地步骤的表示，不保证残差更小或攻击更强。
其他梯度攻击继续使用轮初模型。历史结果不改写。

## 攻击可见性

`audit.audit_view` 支持：

- `protocol_plus_released_prompts`（默认）：更新攻击使用真实协议消息，同时允许攻击公开发布的 prompt 检查点；
- `released_prompt`：不使用通信更新，只审计公开 prompt；
- `full_whitebox`：允许完整内部客户端状态，用作强攻击上界。

在 FedSGD 的默认 `protocol_plus_released_prompts` 视图中，服务器直接观察每个客户端
上传的完整可训练参数梯度，并用 `base - learning_rate * gradient` 构造与该上传对应的
虚拟 client post-step state。Blackbox-Loss、
Score-Diff、Score-Ratio 以及基于 loss/confidence 的时序攻击使用这个可观测客户端状态，
而不是各客户端聚合后的 global post-state；审计器不会读取模拟器内部未上传的状态。

审计摘要会保存实际使用的视图和威胁模型。
