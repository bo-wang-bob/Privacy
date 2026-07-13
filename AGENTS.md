# AGENTS.md

本文件适用于整个 `SEISMOGRAPH` 仓库。该分支只支持：

- 联邦提示学习（FPL）；
- `cerberus`、`a3fl`、`sabre` 三种攻击；
- `aggregator/seismograph_aggregator.py` 这一种聚合器。

运行 Python 时优先使用 `conda activate pfedba`，或等价地使用
`micromamba run -n pfedba ...`。Git 提交信息应详细说明精简范围和验证结果。

## 运行入口

```bash
python main.py --config configs/fpl/seismograph/cerberus_seismograph_fpl.yaml --gpu 0
python main.py --config configs/fpl/seismograph/a3fl_seismograph_fpl.yaml --gpu 0
python main.py --config configs/fpl/seismograph/sabre_seismograph_fpl.yaml --gpu 0
```

FPL 使用 `CLIPProcessor.from_pretrained(..., local_files_only=True)` 和
`CLIPModel.from_pretrained(..., local_files_only=True)`，默认依赖本地缓存
`checkpoints/clip-vit-base-patch32`。缺少缓存时不要改成联网下载逻辑，除非用户
明确要求。

## 关键执行路径

1. `main.py::parse_args()` 合并默认值、YAML 和 CLI，并验证最小分支范围。
2. `main.py::main()` 构建 FPL 数据划分、CustomCLIP、触发器和 seismograph 聚合器。
3. `servers/serverbase.py::ServerBase.train()` 执行客户端采样、三种攻击训练、
   指标记录和聚合。
4. `utils/seismograph_text_feature_analysis.py` 计算 raw top-1 奇异值并维护历史异常状态。
5. `aggregator/seismograph_aggregator.py` 过滤客户端后按样本数加权聚合 prompt 参数。
6. `context/context.py::Context.continue_to_next_round()` 提升跨轮状态。

## 约定

- 恶意客户端 ID 固定为整数 `[0, malnum)`。
- `sample_users` 传入服务器后对应 `user_per_round`。
- `train_mode` 只接受 `centralized` 或 `local`。
- `dirichlet_alpha >= 10` 走 IID，否则走 Dirichlet non-IID。
- FPL 的 `poisonratio` 是每个 batch 的投毒样本数，不是比例。
- `poison_label` 必须落在当前数据集类别范围内。
- 只聚合 `Context.trainable_param_names` 中的 prompt 参数。
- state dict 跨用户或跨轮保存时使用 `detach().clone()`，避免共享 Tensor 引用。
- 不要重新引入其他聚合器、攻击或非 FPL 模型，除非用户明确要求扩展范围。

## 目录职责

- `main.py`：配置、范围校验、CLIP/数据/服务器组装。
- `aggregator/`：seismograph 聚合器、基类和唯一构建入口。
- `servers/serverbase.py`：FPL 联邦训练循环与三种触发器优化。
- `users/user.py`：正常训练、三种攻击本地训练和后门评估。
- `context/context.py`：跨轮模型与文本特征状态。
- `trainmodel/`：CustomCLIP 和模型基类。
- `utils/`：数据加载、触发器、投毒、seismograph 文本特征与更新分析。
- `configs/fpl/seismograph/`：三种受支持攻击的配置。

## 修改与验证

- 保持现有 Python 风格，日志使用 `logging.getLogger(__name__)`。
- 设备相关 Tensor 显式放到目标 device。
- 修改随机性时同步考虑 PyTorch、NumPy 和 `random` 的 seed。
- 不做无关格式化或大范围行为改写。
- 优先运行不需要完整数据或 CLIP 权重的轻量单元测试；完整训练缺少本地数据或
  CLIP 缓存时，只说明缺失原因，不自动下载。

## Git 与产物

不要提交或删除用户的运行产物，包括：

- `data/`
- `logs*/`
- `results*/`
- `checkpoints/`
- `saved_models/`
- `__pycache__/`
- `.pytest_cache/`

工作区可能已有用户改动。不要执行 `git reset --hard`、`git checkout --` 或删除
无关实验产物，除非用户明确要求。
