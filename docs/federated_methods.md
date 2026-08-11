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
新增 Adapter 和 SST-2 二分类头可训练。默认使用 30 个 IID 客户端、batch size 16、
所有客户端同步参与；每个客户端每轮执行一次无 momentum、无 weight decay 的 SGD，
服务器对客户端上传等权平均，构成 one-batch FedSGD。

内存中只保留一个共享预训练主干。客户端持有独立的 Adapter/分类头状态并常驻 CPU，
训练或评估时临时移动到主干设备并绑定，结束后立即卸载回 CPU；客户端上传和最终
checkpoint 均不包含冻结的 BERT/GPT2 参数。

这两个文本模型复用通用审计器的 Blackbox/Fed-loss、Loss-Series、Grad-Cosine、
Avg-Cosine 和两种 FedMIA 信号。默认只审计客户端 0，成员来自该客户端训练分区，
非成员来自该客户端独立 validation 分区，并按标签精确配成 1:1。余弦攻击比较候选
样本对全部可训练 Adapter/分类头参数的精确梯度与服务器收到的真实 FedSGD 上传；
GPT2-Large 使用逐样本流式点积控制内存。

严格 ProjRes 在最终通信轮观察目标客户端真实 one-batch 上传，使用首层 Adapter
down-projection 权重更新构造子空间，并在同一全局模型下提取进入该层的样本级隐藏
表示。成员只取该轮保留的真实 batch，非成员只取从未训练的 validation 样本。

## 攻击可见性

`audit.audit_view` 支持：

- `protocol_plus_released_prompts`（默认）：更新攻击使用真实协议消息，同时允许攻击公开发布的 prompt 检查点；
- `released_prompt`：不使用通信更新，只审计公开 prompt；
- `full_whitebox`：允许完整内部客户端状态，用作强攻击上界。

审计摘要会保存实际使用的视图和威胁模型。
