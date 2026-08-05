# 联邦提示训练方法

## FedAvg (`aggregator: fedavg`)

历史兼容入口。客户端在冻结 CLIP 上训练一组 soft-prompt 参数，服务器按客户端训练样本数聚合。学习 token 会拼接在配置的手工模板之前。

## PromptFL (`aggregator: promptfl`)

对应论文 *PromptFL: Let Federated Participants Cooperatively Learn Prompts Instead of Models*。每个客户端在冻结 CLIP 上只训练一组共享 CoOp context token，并最小化本地交叉熵；服务器按客户端训练样本数执行 FedAvg。

严格的 `promptfl` 入口使用论文式 `[SOS] [learned context] [class] [EOS]` 构造。论文：[arXiv](https://arxiv.org/abs/2208.11625)。

## 攻击可见性

`audit.audit_view` 支持：

- `protocol_plus_released_prompts`（默认）：更新攻击使用真实协议消息，同时允许攻击公开发布的 prompt 检查点；
- `released_prompt`：不使用通信更新，只审计公开 prompt；
- `full_whitebox`：允许完整内部客户端状态，用作强攻击上界。

审计摘要会保存实际使用的视图和威胁模型。
