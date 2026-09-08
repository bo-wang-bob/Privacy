# CLIP 逐层视觉 Adapter

`configs/models/clip_adapter.yaml` 默认使用 BERT 式逐层残差 Adapter。统一入口和模型
名称仍为 `clip_adapter`，结构由 `clip_adapter.variant` 记录。

## 结构与训练

- 在 CLIP ViT-B/32 的全部 12 个视觉 Transformer block 输出后分别添加
  `h + W_up ReLU(W_down h + b_down) + b_up`。
- reduction=2，视觉隐藏维度 `768 → 384 → 768`，共 7,091,712 个可训练参数。
  down 权重按标准差 0.02 正态初始化，up 权重和所有 bias 为零，初始网络保持原始
  CLIP 的预测；首次 backward 的 down 梯度为零是这一初始化的正常行为。
- 原始视觉/文本编码器、图像/文本投影、logit_scale 全部冻结，不训练线性分类头或
  末端 Adapter。使用归一化图像向量与类别提示词文本向量计算 CLIP 相似度 logits。
- 图像、文本每次前向都重新编码，不缓存 feature。只保留固定类别提示词的 token。
  `precompute_features: true` 和二维特征输入会被拒绝；冻结骨干不意味着切断视觉
  反向图。冻结骨干保持 eval 模式，新增 Adapter 参与训练，无 dropout 随机性。
- 客户端通过 BERT 已有的参数绑定机制复用一个骨干；每个客户端独立保存 Adapter
  参数，离开本地训练会话后恢复服务器参数并卸载客户端梯度。
- 学习率 0.01、batch=32、10 个 IID 客户端、全局每类 16-shot 保持当前 CLIP
  基线。评估 batch=32，审计前向 batch=16，以适应原图计算；全局独立评估集不截断。

配置的结构部分为：

```yaml
clip_adapter:
  variant: transformer
  reduction: 2
  activation: relu
  zero_init_up: true
  text_adapter_enabled: false
  precompute_features: false
  template: null
```

当前支持 FedSGD/FedAvg 和 none/WWW/CoFedMID。默认 FedSGD 是 300 轮、每轮一个
batch、uniform 聚合；选择 FedAvg 后默认 100 轮、每轮一个完整本地 epoch、sample_count
聚合。入口保留配置中的实际聚合权重，不再在 CLIP 运行时强制改成 uniform。

沿用此前 FedAvg、uniform 的对照方式：

```bash
python scripts/run_privacy_experiments.py --models clip_adapter \
  --methods fedavg --defenses none --aggregation-weighting uniform
```

默认依次运行 CIFAR100、Food101，全部 11 种攻击共享训练。追加 `--dry-run` 可查看
variant、reduction、是否预计算特征和最终协议。纯训练使用 `--attacks none`。

## 攻击协议

损失、时序、余弦、Gradient-Diff 和其他注册攻击使用真实在线前向。普通梯度审计按
小块计算全部可训练 Adapter 参数上的候选梯度，保留原有 float64 归约与上传单位转换。
共享骨干不影响每个客户端上传状态的独立性。

ProjRes 默认攻击最后一个视觉 Adapter 的 down 权重；ViT-B/32 对应
`clip_model.vision_model.encoder.layers.11.adapter.down.weight`，提取进入该 Adapter
前的 CLS。`projres.attacked_parameter: null` 根据实际视觉层数选择最后层，
`token_reduction: auto` 也解析为 `cls`。显式指定其他 down 权重及 `mean` 仍可做对照。
该 CLS 已包含图像信息，且最后 Adapter 后只取 CLS 分类，patch 输出不再贡献损失梯度。
其他审计使用的 key parameter 保持首层，旧 feature Adapter 与 CLIP-LoRA 的默认攻击面不变。

```yaml
projres:
  attacked_parameter: null  # 自动选择最后视觉 Adapter 的 down.weight
  token_reduction: cls
```

- FedSGD 候选仍是当轮真实 batch 的 N 个成员与标签匹配的 10N 个独立非成员。
  为隔离本次攻击面变更，继续沿用输入 token 总数作为保守秩上限；默认每图 50 个
  输入 token，元数据中的 `candidate_hidden_vector_count` 仍为 N×50，并不表示最后层
  有 50 个损失活跃 token。最后层只有 CLS 贡献梯度，本次不增加新的数值截断规则。
- 保留 `paper_fedsgd_exact=false` 及现有 FedSGD 解释标记
  `empirical_token_aggregate_gradient_projection`；实际层名与 CLS 表示写入攻击元数据。
- 零初始化导致 down 上传为零时只跳过该轮 ProjRes，记录 `zero_observed_update`；
  其他攻击继续。FedAvg 仍使用完整原始客户端训练集及默认 M:M 的独立非成员，
  `batch_rank_bound=null`，使用累计更新。最后层 CLS 在本地多步更新中可能发生表示漂移，
  因此继续作为经验性多步适配，表示取轮初模型。
- CoFedMID 扰动覆盖被攻击参数时取消无噪声秩上限。WWW 保留真实 batch 身份和真实上传。
- 使用统一审计路径；独立缓存特征 ProjRes 和 `low_fpr_full` 不支持这个结构。
- 分数继续为原始负 L1 投影残差；相对残差、端点插值与中心化没有启用。

50 轮 FedAvg/uniform 对照可直接运行，不必再传攻击层或 token 参数：

```bash
python scripts/run_privacy_experiments.py --models clip_adapter \
  --methods fedavg --defenses none --aggregation-weighting uniform \
  --rounds 50 --attacks projres
```

2026-09-09 此次修改之前的逐层 Adapter 结果使用首层 down + mean；复现旧攻击面需同时
指定 `--set projres.attacked_parameter=clip_model.vision_model.encoder.layers.0.adapter.down.weight`
和 `--set projres.token_reduction=mean`。仅指定首层不再自动使用 mean。

## 与历史结果区分

缺少 variant 的旧配置解释为 `feature`，保留原末端图像/文本 Adapter 的实现与字段。
新配置明确记录 `transformer`，攻击元数据记录
`trainable_scope: clip_visual_transformer_adapters` 和 `adapter_variant: transformer`。
可训练权重保存为 `final_clip_transformer_adapter.pt`；旧末端权重是
`final_clip_adapter.pt`，两者不能混用；即使使用部分状态加载，旧末端 checkpoint
也会明确报错，避免未载入 Adapter 却继续训练。已有 `results/` 文件不修改。

使用旧末端结构时应提供完整的 feature 配置块，包括原 reduction、alpha、文本分支等；
仅把新配置中的 variant 改为 feature 不能恢复原双侧超参数。

新结构需要保存内部视觉层的反向图，计算和显存开销高于原末端 Adapter。新增容量
不保证提高成员推理效果，需在数据、候选、方法、聚合权重和训练预算一致时比较。

## 验证

`tests/test_clip_transformer_adapter.py` 覆盖初始化预测一致性、冻结参数不变、两步梯度
传播、在线文本编码、拒绝缓存特征、客户端状态隔离、首末层真实 token 表示、最后层
CLS 独立重建 down 梯度、默认攻击面与候选比例、FedSGD/FedAvg
结合三种防御的全部 11 种攻击，以及 main 入口保持聚合权重。测试使用真实 Transformers
模块构成的小型随机 CLIP，不需要下载权重或数据。真实数据的攻击效果需另行实验。

另用本地预训练 ViT-B/32 在 CPU 对两个合成 224×224 输入执行了两步 SGD：确认
12 层、7,091,712 个可训练参数，初始化 logits 与原始 CLIP 完全相等；第一步 down
梯度为零，第二步首层 down 梯度范数约 0.005859，冻结参数没有梯度，客户端会话后
服务器参数正确恢复。ProjRes 表示为 2×768、成员 token 总数为 100。这是接口与梯度
冒烟验证，不是 CIFAR100/Food101 的训练或攻击效果评估。
