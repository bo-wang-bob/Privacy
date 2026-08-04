# ProjRes 在 PromptFL 中的验证代码

本验证套件把 Deng 等人的投影残差理论分成四个可比较层次：

1. `oracle_projres_t`：直接观察中间文本特征梯度
   `G_T = scale * E.T @ X / B`，是论文理论在 `logits = scale * T x`
   场景中的机制上界。
2. `delta_text_projres`：攻击者使用训练前后 prompt 在本地重算归一化文本
   特征，取 `T_after - T_before` 后先减去类别行均值，使其满足
   交叉熵文本梯度的零行和结构，再按
   `min(local_examples, classes - 1, feature_dimension - 1)` 截断行空间。
   这是可观测的 secant proxy，不假定文本矩阵变化等于文本梯度。
3. `lifted_projres_p`：只使用 PromptFL 上传方向，通过文本编码器
   Jacobian 的矩阵无关岭逆恢复文本梯度，再执行图像子空间投影。
4. `direct_prompt_atom`：不反演 Jacobian，直接比较客户端 prompt 更新与
   候选样本 prompt-gradient fingerprint。

核心实现位于 `privacy_attacks/projres_promptfl.py`。轻量测试不依赖数据集或
CLIP checkpoint。

## 1. 运行轻量测试

```bash
python -m pytest -q tests/test_projres_promptfl.py
```

也可以直接运行预设好的理论验证：

```bash
bash scripts/run_projres_promptfl_synthetic.sh
```

测试覆盖：

- 解析文本梯度与 autograd 一致；
- softmax 误差的零和性质；
- 满秩条件下梯度行空间与成员图像特征行空间一致；
- 子空间饱和后非成员也获得满投影能量；
- 稠密 Jacobian 提升与矩阵无关提升一致；
- 候选 prompt-gradient fingerprint 与逐样本 autograd 一致；
- 稠密 Jacobian 的内存保护。

## 2. 合成 rank/Jacobian sweep

```bash
python scripts/validate_projres_promptfl_synthetic.py \
  --classes 8 \
  --dimension 16 \
  --batch-sizes 1,2,4,7,8,12,16 \
  --prompt-widths 4,8,16,32 \
  --trials 10 \
  --output results/projres_synthetic.json
```

重点检查：

- `error_rank` 是否在 `classes - 1` 截断；
- 边界内 `oracle_member_mean_residual` 是否接近零；
- `maximum_principal_angle_degrees` 是否接近零；
- `lifted_auc` 随 Jacobian 压缩宽度如何变化；
- `candidate_fingerprint_mean_relative_error` 是否接近机器精度。

理论上的有效低秩边界为：

```text
batch_size <= min(classes - 1, feature_dimension - 1)
```

其中还需要实际 `rank(E) = rank(X) = batch_size`。

## 3. 真实 PromptFL 单次验证

真实入口只从本地缓存加载 CLIP，内部调用始终保持
`local_files_only=True`。数据集也必须已经位于本地。

```bash
python scripts/validate_projres_promptfl_real.py \
  --config configs/federated_prompt_paper.yaml \
  --batch-size 8 \
  --local-steps 1 \
  --max-candidates 32 \
  --ridge 1e-4 \
  --lift-iterations 20 \
  --output results/projres_real_b8_s1.json
```

等价的预设启动脚本为：

```bash
bash scripts/run_projres_promptfl_real.sh
```

该脚本默认与 `scripts/run_fedmia_prompt_methods.sh` 的基础 PromptFL 配置对齐：
`caltech101`、`./data`、本地 `clip-vit-base-patch32`、全量 IID 数据、batch 16、
学习率 `1e-4`、prompt 长度 16、seed 42 和一个本地 batch step，并默认审计
全部 10 个客户端。输出同时包含 pooled 与 client-macro 指标。通常不需要传入
任何参数。

仅在消融时覆盖改变的选项，例如：

```bash
# 只把本地 batch step 改为 2
bash scripts/run_projres_promptfl_real.sh \
  --local-steps 2 \
  --output results/projres_validation/caltech101_all_b16_steps2.json
```

只调试客户端 0 时才显式设置：

```bash
bash scripts/run_projres_promptfl_real.sh --target-client 0
```

可以通过环境变量替换 Python、配置和结果目录：

```bash
PROJRES_CONFIG=configs/federated_prompt_paper.yaml \
PROJRES_RESULTS_DIR=results/projres_validation \
bash scripts/run_projres_promptfl_real.sh
```

`PROJRES_PYTHON` 应为单个可执行文件路径。若使用 micromamba，推荐直接运行：

```bash
micromamba run -n pfedba bash scripts/run_projres_promptfl_real.sh
```

矩阵无关提升使用 JVP/VJP 和共轭梯度，不会构造大小为
`(classes * feature_dimension) x prompt_parameters` 的稠密 Jacobian。
该入口会让 CLIP 文本编码器使用 Transformers 的 `eager` attention 实现，
因为 PyTorch efficient SDPA 尚不支持 JVP 所需的 forward-mode AD；这只改变
注意力算子的实现路径，不改变模型权重或注意力公式。

若只验证 Oracle 和 Direct 两条路径，可使用：

```bash
python scripts/validate_projres_promptfl_real.py \
  --config configs/federated_prompt_paper.yaml \
  --local-steps 5 \
  --skip-lift
```

真实验证从目标客户端抽取实际训练 batch，并优先从独立测试集合中为每个成员
选择同标签 non-member；测试池容量不足时，从其他客户端训练集补充同标签样本。
两类样本对目标客户端都是真实 non-member，具体来源数量记录在
`candidate_controls.nonmember_source_counts`。输出包含：

- 四种攻击的 AUC、TPR@10%/1%/0.1%FPR 和均值分数差；
- 每步 softmax 误差秩、Oracle/Lifted 梯度秩和奇异值；
- 文本特征变化的原始秩、零行和投影后的秩、理论截断秩、变化范数、
  相对变化量和被移除的公共漂移比例；
- Oracle 子空间与实际成员子空间的 principal angles；
- 文本特征变化子空间与 Oracle/成员子空间的 principal angles；
- Lifted 子空间与 Oracle 子空间的 principal angles；
- 固定初始 Jacobian 对多步本地更新的相对拟合误差；
- 岭提升的共轭梯度收敛信息和 measurement residual；
- 成员与同标签 non-member 的基准损失。

少量候选不足以解析 1% 或 0.1% FPR；正式报告这些指标时，non-member 数量
分别至少需要 100 和 1000。AUC 和分数差可用于小规模机制检查。

## 4. 生成或执行完整 sweep

先只生成 manifest，检查实验规模：

```bash
python scripts/run_projres_promptfl_validation_sweep.py \
  --config configs/federated_prompt_paper.yaml \
  --output-dir results/projres_sweep \
  --batch-sizes 1,4,8,16,32 \
  --local-steps 1,2,5 \
  --n-ctx 4,8,16,32 \
  --dirichlet-alphas 0.1,0.5,1.0 \
  --seeds 42,43,44
```

确认后添加 `--execute` 顺序运行。已有 JSON 会被跳过，可以断点续跑。

也可以使用分阶段 shell 入口：

```bash
# 15 个任务：优先验证 batch size 与 local steps
bash scripts/run_projres_promptfl_sweep.sh mechanism

# 12 个任务：验证 prompt 长度/Jacobian 可观测性
bash scripts/run_projres_promptfl_sweep.sh jacobian

# 9 个任务：验证 Dirichlet non-IID 与随机种子
bash scripts/run_projres_promptfl_sweep.sh heterogeneity

# 540 个任务：完整笛卡尔积，成本很高
bash scripts/run_projres_promptfl_sweep.sh full
```

只生成 manifest 而不执行：

```bash
PROJRES_DRY_RUN=1 bash scripts/run_projres_promptfl_sweep.sh mechanism
```

完整笛卡尔积成本较高。推荐先固定 `n_ctx=16, alpha=0.1, seed=42` 扫描
batch/local steps，再分别展开 prompt 长度、异构性和随机种子。

## 5. 输出判读

- Oracle 成功而 Lifted 失败：主要瓶颈是 prompt Jacobian 压缩或反演病态。
- Oracle 成功而 `delta_text_projres` 失败：训练前后文本矩阵变化即使去除
  类别公共漂移，也没有保留足够准确的成员图像子空间；端点变化不能直接当成
  文本特征梯度。
- Oracle 与 Lifted 成功而 Direct 失败：候选自相关被 batch 中其他梯度方向抵消。
- 三者都在同类 non-member 上失败：信号更可能是语义/类别相似性，而不是严格
  样本成员关系。
- `fixed_base_jacobian_relative_error` 随 local steps 快速上升：固定初始
  Jacobian 不适合解释多步本地更新，应优先使用逐轮 Direct 分数或更精确的轨迹
  Jacobian。
- Oracle 梯度秩达到特征维度：子空间已经饱和，投影残差不能再区分成员与
  非成员。

该套件只进行被动成员隐私审计，不改变联邦训练目标，也不引入主动样本操作。
