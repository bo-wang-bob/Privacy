# 联邦提示训练方法

## FedAvg (`aggregator: fedavg`)

历史兼容入口。客户端在冻结 CLIP 上训练一组 soft-prompt 参数，服务器按客户端训练样本数聚合。学习 token 会拼接在配置的手工模板之前。

## PromptFL (`aggregator: promptfl`)

对应论文 *PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models*。每个客户端在冻结 CLIP 上只训练一组共享 CoOp context token，并最小化本地交叉熵；服务器按客户端训练样本数执行 FedAvg。

严格的 `promptfl` 入口使用论文式 `[SOS] [learned context] [class] [EOS]` 构造。论文：[arXiv](https://arxiv.org/abs/2208.11625)。

## CLIP-LoRA 的基础 FedAvg

`model_type: clip_lora` 冻结 CLIP 主干，在图像与文本编码器注意力投影中插入
`W_eff = W_0 + (alpha/sqrt(r)) B A`。客户端只优化并上传 `lora_A`、`lora_B`，
服务器分别线性聚合同名因子。普通配置使用 FedAvg；严格 ProjRes 配置使用
one-batch FedSGD。两者默认均对本轮参与客户端等权：

```text
A_global = (1 / |S|) sum_k A_k
B_global = (1 / |S|) sum_k B_k
```

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
BERT 三个数据集均训练 500 轮并保持学习率 `0.005`；GPT2-Large 训练 500 轮并保持
`0.001`。Adapter 的 down-projection 使用主干 initializer range 做小随机初始化，
up-projection 与 bias 置零，使残差分支在初始化时严格保持主干隐藏表示；全部可训练参数
使用同一个标量学习率，客户端梯度不执行 norm clipping。
两者均由所有客户端同步参与；每个客户端每轮在一个 batch 上计算一次无 momentum、
无 weight decay 的梯度并上传，服务器等权聚合真实梯度后执行
`theta_(t+1) = theta_t - learning_rate * mean(g_k)`，再下发新的全局模型。
这些普通任务优化不关闭或绕过审计：上传仍是攻击器观察到的真实 one-batch 梯度，
十种通用攻击与每 50 轮一次的 ProjRes 均保持启用。
普通任务评估中，SST-5 与 IMDB 以 accuracy 为主；CoLA 以 MCC 为主，并继续记录
accuracy 以兼容已有结果分析。

内存中只保留一个共享预训练主干。客户端持有独立的 Adapter/分类头状态并常驻 CPU，
训练或评估时临时移动到主干设备并绑定，结束后立即卸载回 CPU；客户端上传和最终
checkpoint 均不包含冻结的 BERT/GPT2 参数。

`scripts/run_all_fedllm_attacks.sh` 是统一 sweep 入口，不带参数时按
`BERT/GPT2 × SST-5/CoLA/IMDB` 展开 6 个独立进程；
`scripts/run_fedllm_adapter.py` 只执行 Shell 入口传入的单个任务。统一 Shell 模式
支持 `--models`、`--datasets`、`--gpus`、`--jobs`、`--dry-run`、
`--max-runs` 和 `--skip-projres`，默认单 GPU 串行调度。dry-run 只打印命令；
完整解析配置仅在对应任务实际启动后写入终端和该任务的 `run.log`。

这两个文本模型复用通用审计器的 Blackbox/Fed-loss、Loss-Series、Grad-Cosine、
Avg-Cosine、两种 FedMIA 信号，以及 Gradient-Diff、Score-Diff、Score-Ratio 和
FTA。默认只审计客户端 0。BERT 与 GPT2-Large 的 Blackbox-Loss、Grad-Cosine、Gradient-Diff、
Score-Diff、Score-Ratio 和 ProjRes 将每轮真实上传 Batch 定义为成员，并从全部客户端的独立 evaluation 分区
逐类别抽取 10 倍从未训练的非成员；每轮候选集独立构造和评估。其余攻击从目标客户端
训练分区使用完整的 `M` 个历史成员，再从相同 evaluation 池抽取类别尽力匹配的
等量 `M` 个非成员，并跨轮复用固定候选池。余弦攻击比较候选样本对全部可训练
Adapter/分类头参数
的精确梯度与服务器收到的真实 FedSGD 上传；GPT2-Large 使用逐样本流式点积控制内存。

固定候选攻击的主视图为 `M/M`；另从中派生最多 100/100 的论文对照视图。后者复用
同一分数，不重新执行模型前向或梯度计算。主视图的最低可解析 FPR 由 `M` 决定；
论文视图只有 100 个非成员，不能解析 TPR@0.1%FPR。六种真实 Batch 攻击固定为
16 成员/160 非成员，只
统计 AUC、TPR@10%FPR 与 TPR@1%FPR，不生成 TPR@0.1%FPR。

BERT 同时启用周期性 ICLR 排名分析。服务器在第 50、100、…、500 个通信轮完成聚合后，
根据该轮真实客户端上传和等权聚合系数，为每个客户端精确重建其他客户端的聚合模型，
并在该轮真实 one-batch 训练样本上计算 `L(x; theta_-k) - L(x; theta_k)`。该分析只记录
排名与统计，不筛选样本、不改变优化步骤；逐轮结果写入 `iclr_round_metrics.csv`、
`iclr_round_samples.csv` 和 `iclr_series.json`，最终
与具有稳定本地索引的固定攻击成员候选对齐。六种真实 Batch 攻击另存逐轮成员本地索引，
不会错误套用固定候选池的位置。GPT2-Large 当前保持 `defense.name: none`。
由于 ICLR 和 ProjRes 使用同轮同客户端的真实 FedSGD batch，两者还会按通信轮、客户端、
batch 位置和本地样本索引进行严格逐样本连接，并报告分数相关性与 Top-K 富集倍数。
ProjRes 命中率使用同轮共享的 160 个非成员连续分数分别在 10% 和 1% FPR 下复算，再比较高低
ICLR 组；ProjRes 采用 ranking-only 协议，不再产生固定残差阈值预测。

严格 ProjRes 每 50 个已完成通信轮观察目标客户端真实 one-batch 上传，使用首层 Adapter
down-projection 权重更新构造子空间，并在同一全局模型下提取进入该层的样本级隐藏
表示。BERT 与 GPT2-Large 的成员和非成员直接复用共享真实 Batch 候选视图，即当轮
`N` 个成员及按标签匹配的 `10N` 个从未训练 evaluation 样本；完整 Batch 时为
16/160。结果与其他攻击统一写入审计器输出，`projres.max_candidates: 16` 不会截断
实际上传 Batch；完整 16 条 Batch 的结果元数据会将 `paper_fedsgd_exact` 记为 `true`。

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
