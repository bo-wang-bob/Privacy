# FedSGD 逐记录计算优化

优化以 WWW Poisson DP 实现 `d63b5a6` 为基准，覆盖 WWW、BERT Adapter 普通样本级 DP 的梯度计算，以及 BERT/GPT2 的文本梯度审计。

## 默认行为

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `defense.grad_sample_backend` | `auto` | WWW 与 BERT Adapter Record-DP 使用 batched VJP |
| `defense.microbatch_size` | `4` | 一次求导保留的记录数 |
| `audit.grad_sample_backend` | `auto` | 文本余弦/Gradient-Diff 审计使用 batched VJP |
| `audit.grad_sample_chunk_size` | `4` | 一次审计求导保留的候选数 |
| `audit.gradient_update_cache_mb` | `2048` | 上传向量 GPU 缓存上限（MiB），0 表示使用 CPU |

原来的循环逐个调用 backward。新的实现先对最多 4 条完整记录前向，得到各自的 loss，再用 batched VJP 一次求出这些 loss 对客户端可训练参数的独立梯度。文本中的一条记录仍是完整序列。它直接使用当前计算图和客户端参数对象，兼容 BERT 共享主干的参数绑定；不通过 functional_call 替换共享模型参数。

每条梯度仍在全部可训练参数上计算联合 L2 范数。普通 DP 裁剪到 C；WWW 在此基础上乘该记录的 INO 积分权重。所有块累加后才添加一次噪声，除以固定期望 batch 大小，执行一次 optimizer step 并捕获一次上传。服务器继续对参与客户端等权聚合。实际 Poisson batch 可能跨越任意多个计算块；尾部排序仍基于整个实际 batch，尾部宽度仍由期望 batch 决定。空 batch 保留纯噪声更新和一次会计步数。

文本审计在一次信号计算内缓存各客户端上传向量的 float64 表示，并对一块样本一起计算点积。缓存不超过配置上限且不超过当前空闲显存四分之一时，放在 GPU 上，避免每块梯度搬回 CPU；其余情况使用 CPU。缓存会在该次调用结束后释放，避免复用过期模型/上传。余弦使用真实标签交叉熵梯度；Gradient-Diff 使用所有标签损失之和的梯度，目标客户端顺序、上传符号、学习率缩放与 float64 归约保持原定义。30 个客户端、约 709 万个可训练参数时，缓存约占 **1.59 GiB**，不随候选池大小增长；逐记录梯度仅保留当前块。

`auto` 在启用 Transformer gradient checkpointing 时选择 `loop`。ResNet18 保留原 functional `vmap` 后端。自定义额外逐记录损失走已有循环实现。显式 `loop` 可用于复核或处理新模型不支持的批量求导算子。没有静默捕获 OOM 后跳过记录的行为。

FP32 下改变批矩阵计算与求和顺序会产生小数值差异；存在 dropout 时，相同公开 seed 也不保证分块前后抽到完全相同的随机掩码。实现保留 dropout 配置和私有采样/噪声生成器，不能据此承诺逐位一致的训练轨迹。

## 运行

继续使用统一入口即可启用优化，无需新增参数。例如沿用预算 8、阈值 8：

```bash
python scripts/run_privacy_experiments.py \
  --models bert_adapter --datasets cola,imdb \
  --defenses www,record_dp --attacks all \
  --set defense.target_epsilon=8 --set defense.max_grad_norm=8
```

将期望 batch 调整为 32 时加 `--set batch_size=32`；预算改为 16 时把 epsilon 参数改为 `--set defense.target_epsilon=16`。计算块默认仍为 4。实际采样率和对应噪声会按最终配置重新校准。

统一入口会在应用全部覆盖参数后自动匹配 ProjRes 候选配置。普通训练按最终 batch 大小和成员/非成员比例推导，WWW/Record-DP 使用 `0/0/0` 表示真实 Poisson batch 的动态候选；无需分别填写这些参数。显式指定的候选值仍保留，不符合协议时由现有校验报错。以下一条命令比较无 DP、普通 DP、WWW，共六个训练任务：

```bash
python scripts/run_privacy_experiments.py \
  --models bert_adapter --datasets cola,imdb \
  --defenses none,record_dp,www --attacks all --gpus 1 \
  --set batch_size=32 \
  --set defense.target_epsilon=16 --set defense.max_grad_norm=8
```

如需逐记录求导对照，增加：

```bash
--set defense.grad_sample_backend=loop --set audit.grad_sample_backend=loop
```

普通 DP 的旧配置使用 `defense.microbatch_size=16`；复核旧配置时同时指定这一值。循环实现中的 GPU 标量重复同步已经消除，文本审计仍会复用本次上传缓存；增加 `--set audit.gradient_update_cache_mb=0` 可指定 CPU float64 归约。

## 独立速度基准

```bash
/root/.local/share/mamba/envs/pfedba/bin/python \
  scripts/benchmark_fedsgd_gradients.py --device cuda:1 \
  --output /tmp/fedsgd_gradient_benchmark.json
```

默认读取仓库本地 BERT-Base 权重，使用 batch 16/32、序列长度 128、计算块 4、30 个合成客户端上传向量。每个阶段先在 eval 模式检查梯度/信号一致性，再预热并重复测量；梯度计时使用 train 模式，审计使用 eval 模式。旧普通 DP 与旧审计算法保留在基准脚本中作为对照。脚本不读取真实训练记录、不训练 500 轮，也不修改 `results/`；JSON 已存在时拒绝覆盖。

报告包括每次耗时、中位数、GPU 峰值已分配显存、eval 最大绝对误差与相对 L2 误差。测量范围是逐记录梯度/裁剪及文本梯度审计，**不包括 WWW 排序、加噪、optimizer、上传、服务器聚合或其余攻击**，因此阶段加速倍数不能直接当作完整实验加速倍数。

## 2026-09-06 实测

pfedba 环境、PyTorch 2.7.1+cu128、RTX 4090（GPU 1），使用真实 BERT-Base 主干、7,093,250 个可训练参数和合成输入。此次测量时 WWW 尾部比例为 20%，当前默认已调整为 80%；重新运行基准会使用并记录当前比例。预热 2 次、重复 5 次，以下为中位数；同机其他工作可能影响耗时，完整原始数据见 [基准 JSON](benchmarks/fedsgd_gradients_20260906.json)。

| 阶段 | 样本数 | 原实现 | 优化后 | 加速倍数 |
| --- | ---: | ---: | ---: | ---: |
| WWW 梯度与裁剪 | 16 | 159.6 ms | 89.3 ms | 1.79× |
| WWW 梯度与裁剪 | 32 | 291.9 ms | 190.5 ms | 1.53× |
| 普通 DP 梯度与裁剪 | 16 | 260.9 ms | 95.3 ms | 2.74× |
| 普通 DP 梯度与裁剪 | 32 | 519.5 ms | 184.9 ms | 2.81× |
| 文本余弦及 Gradient-Diff 审计，30 个上传向量 | 16 | 4.016 s | 1.116 s | 3.60× |

梯度阶段优化后峰值已分配显存约 845–848 MiB，原 WWW 约 620 MiB、原普通 DP 约 1,439 MiB；审计因缓存上传向量，从约 617 MiB 上升到约 2,518 MiB。这些是基准进程的已分配显存，未包含完整真实训练任务的全部驻留数据，也不是 `nvidia-smi` 的进程占用值。

eval 数值核对中，裁剪梯度和最大绝对误差不超过 `3.86e-5`，相对 L2 误差不超过 `3.04e-5`；审计信号最大绝对误差为 `6.11e-5`，相对 L2 误差为 `4.16e-7`。提交前完整本地回归 272 项通过、11 项 GPU 用例因沙箱无法访问设备而跳过；这 11 项在 pfedba 的 GPU 可用环境下随 28 项专项测试全部通过。测试涵盖六种 PEFT 的逐记录梯度、CPU/GPU float64 审计、WWW/普通 DP 的采样及加噪上传和服务器 FedSGD 聚合，以及入口参数和结果日志。没有运行完整 500 轮来测量总耗时或防御效果。

## 2026-09-07：CLIP 审计与共享评估

三个 CLIP 模型的常规余弦和 Gradient-Diff 审计接入相同的分块求导路径，默认块大小为 4。
常规路径使用 `audit.grad_sample_chunk_size`；`low_fpr_gradient_batch_size` 仅供保留的旧缓存路径使用。
MLP/Adapter 继续使用冻结编码器的缓存特征，LoRA 使用原始图像。每个计算块结束后立即
计算与各上传向量的点积和范数，只保留最终分数；逐记录梯度内存从原始 CLIP-LoRA 路径的
`O(候选数 × 参数数)` 降为 `O(块大小 × 参数数)`。上传缓存仍按 2048 MiB 与空闲显存四分之一
限制放置。普通攻击不再计算未使用的梯度特征；显式 `signal_storage=full` 所需的完整信号路径保留。

统一使用真实标签 CE 的余弦以及所有标签损失之和的 Gradient-Diff。FedSGD 上传已经是梯度，
不再除以学习率；FedAvg 模型 delta 则取负并除以学习率一次。修正覆盖固定候选池下的
CLIP Gradient-Diff，原先的额外缩放可能改变分数和排序。默认真实 Batch 的候选定义和采样不变。
CLIP 的归约改为 float64，因此与旧 float32 路径可能有小数值差异，不保证逐位一致。

BERT Adapter/LoRA 和 GPT2 Adapter 在评估时只加载一次全局参数，随后依次遍历原有
客户端测试 loader，保留分区、样本顺序、batch 边界和 padding；按全部记录累加 loss、
正确数量和混淆矩阵。客户端本地状态不参与覆盖，异常时也恢复共享模型状态与模式。
训练仍是一轮一批、客户端等权 FedSGD，WWW 的上一轮参考状态保持完整。

以下默认设置会保存到新任务的 `run_config.yaml`，原运行命令即可使用：

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `audit.grad_sample_backend` | `auto` | 六个 PEFT 模型的常规梯度审计分块求导 |
| `audit.grad_sample_chunk_size` | `4` | 控制当前块的梯度存储 |
| `performance.evaluation_backend` | `shared` | BERT/GPT2 共享全局评估；其余模型保留客户端路径 |
| `performance.enabled` | `true` | 输出累计阶段耗时 |
| `performance.cuda_events` | `true` | 同时记录当前 CUDA stream 上的事件耗时 |

对照原评估方式使用 `--set performance.evaluation_backend=clients`。
关闭计时使用 `--set performance.enabled=false`；只记录主机耗时则设置
`--set performance.cuda_events=false`。

### 阶段耗时

每个任务在正常结束或训练异常退出后写入 `performance_summary.json`（进程被强杀时无法保证写出）。
文件中的 `status` 表示训练调用是否成功，`stages` 按阶段保存 `calls`、`wall_seconds` 和
`cuda_stream_seconds`。CPU 任务的 CUDA 值为 `null`。

| 阶段 | 范围 |
| --- | --- |
| `run` | `ServerBase.train`，包括训练前评估、训练、审计和最终输出；不含模型/数据加载 |
| `train.client_update` | 客户端完整训练调用，包括参数绑定、采样和防御 |
| `train.www_ranking` | WWW 参考状态切换与两次 loss 计算、排序 |
| `train.record_gradients` | WWW/Record-DP 逐记录前向、求导、裁剪及加权归约 |
| `train.noise_and_step` | 添加一次噪声、归一化及一次 optimizer step |
| `train.aggregation` | 服务器聚合 |
| `audit.observe` | 一轮审计信号采集 |
| `audit.gradient_measurements` | 上传缓存、梯度前向、求导、归约的总耗时 |
| `audit.forward` / `audit.backward` / `audit.gradient_reduce` | 常规审计中的前向、求导、点积及范数归约 |
| `audit.projres` / `audit.finalize` | ProjRes 评分 / 最终攻击指标与产物生成 |
| `evaluation` | 正式任务评估及其指标写入 |
| `outputs.write` | 服务器、防御和审计的 CSV/JSON/模型写入，包含初始任务元数据 |

主机时间使用 `perf_counter`；CUDA 操作是异步的，不能将主机时间直接当作 GPU 计算时间。
CUDA 事件在记录阶段不会逐块同步，通常在轮末读取已完成事件，结束时统一等待；待处理事件
到达 4096 组时回收，必要时同步以限制内存。事件时间是同一 stream 的经过时间，也会包含
CPU 发射间隙或文件输出时的空闲，**不等于 CUDA kernel 时间之和**。父子计时为包含关系，
例如 `audit.observe` 已包含 `audit.forward`，不能把这些行相加当作总耗时。

### 独立复测命令

```bash
/root/.local/share/mamba/envs/pfedba/bin/python scripts/benchmark_runtime_optimization.py \
  --device cuda:1 --audit-candidates 64 --warmups 2 --repeats 5 \
  --output /tmp/runtime_optimization_benchmark.json
```

脚本读取本地 CLIP/BERT 权重，输入与上传均为合成数据。CLIP 覆盖 64 个候选、30 个上传、
10 个类别；LoRA 在视觉与文本编码器全部层的 Q/K/V 使用 rank 2。BERT Adapter 使用 30 个
客户端，每客户端 32 条长度 128 的合成文本，eval batch 为 32。优化后的审计包含默认计时开销。
测量前验证数值一致性，输出每次计时、中位数、GPU 已分配显存峰值和梯度张量的理论大小；
后者不包括前向图、上传缓存或 allocator 缓存。输出文件已存在时拒绝覆盖。

### 本次 GPU 基准

pfedba、PyTorch 2.7.1+cu128、RTX 4090 GPU 1；参数如上述命令。每个阶段预热 2 次，
重复 5 次，以下为中位数。全部原始计时、误差和显存数据见
[runtime_optimization_20260907.json](benchmarks/runtime_optimization_20260907.json)。

| 阶段 | 原实现 | 优化后 | 加速倍数 |
| --- | ---: | ---: | ---: |
| CLIP-MLP 余弦审计 | 305.0 ms | 135.6 ms | 2.25× |
| CLIP-Adapter 余弦审计 | 494.7 ms | 109.0 ms | 4.54× |
| CLIP-LoRA 余弦审计 | 2.897 s | 1.567 s | 1.85× |
| BERT Adapter 全局评估，30 × 32 条文本 | 3.486 s | 0.963 s | 3.62× |

原实现的计时波动较大，例如 BERT 评估为 2.60–5.31 s，优化后为 0.859–0.988 s；
加速倍数是本次合成阶段基准，受机器负载、候选数量和模型配置影响，不能直接换算成
完整 500 轮、全部 11 种攻击的总时间收益。本次没有执行完整训练来重新测量防御效果。

三个 CLIP 阶段的最大分数绝对误差分别约 `1.91e-8`、`6.10e-8`、`2.67e-7`，
相对 L2 误差最大 `2.94e-5`。BERT 评估的 loss 总和、正确数、样本数和混淆矩阵完全一致。
LoRA 在 64 个候选下的完整梯度张量从 45 MiB 降为最多 2.8125 MiB 的当前块；
由于 GPU 上传缓存和批量前向图，GPU 峰值已分配显存仍会增加：MLP 约 596→774 MiB、
Adapter 598→692 MiB、LoRA 672→775 MiB。BERT 评估两条路径均约 638 MiB。

完整本地测试 `python -m pytest -q tests` 为 301 项通过、20 项 GPU 用例因沙箱设备访问受限跳过；
65 项 GPU 可用环境下的专项测试全部通过，覆盖上述 20 项。新增验证包含三种 CLIP 的独立
Gradient-Diff 公式、FedSGD/FedAvg 上传单位、六种 PEFT 的分块梯度信号、BERT/GPT2 的
Accuracy/MCC 与状态恢复、阶段计时，以及批量 CSV 的主指标/零值/不可报告值。77 个受支持
实验组合的配置校验以及 BERT CoLA/IMDb 六组 batch 32、预算 16、裁剪 8 的干运行通过。
