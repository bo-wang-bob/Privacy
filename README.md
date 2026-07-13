# SEISMOGRAPH

SEISMOGRAPH 只保留联邦提示学习（FPL）下的三种后门攻击，以及
`aggregator/seismograph_aggregator.py` 聚合器：

- `cerberus`
- `a3fl`
- `sabre`
- 唯一聚合方式：`seismograph`

程序会在配置解析和运行入口同时检查这个范围。`fpl: false`、其他攻击或其他
`defense` 会直接报错，不会静默退回其他实现。

## 环境与本地模型

```bash
conda activate pfedba
```

FPL 使用本地 CLIP 缓存，默认目录为
`checkpoints/clip-vit-base-patch32`。加载始终设置
`local_files_only=True`，不会自动联网下载。

## 运行

```bash
python main.py --config configs/fpl/seismograph/cerberus_seismograph_fpl.yaml --gpu 0
python main.py --config configs/fpl/seismograph/a3fl_seismograph_fpl.yaml --gpu 0
python main.py --config configs/fpl/seismograph/sabre_seismograph_fpl.yaml --gpu 0
```

CLI 可以覆盖 YAML 中仍受支持的参数，例如：

```bash
python main.py \
  --config configs/fpl/seismograph/a3fl_seismograph_fpl.yaml \
  --dataset_name cifar100 \
  --poisonratio 2 \
  --gpu 1 \
  --seed 123
```

FPL 中 `poisonratio` 表示每个 batch 的投毒样本数，必须是
`0 <= poisonratio <= batch_size` 的整数。

SEISMOGRAPH 历史异常过滤参数通过 `defense_params` 配置：

```yaml
defense: seismograph
defense_params:
  seismograph_k: 1.0
  seismograph_h: 5.0
```

## 核心流程

1. `main.py` 读取 YAML/CLI，验证最小分支能力范围并加载本地 CLIP。
2. `utils/trigger.py` 为 Cerberus、A3FL 或 SABRE 创建 FPL 触发器。
3. `servers/serverbase.py` 完成客户端采样、本地正常/攻击训练和指标记录。
4. `utils/seismograph_text_feature_analysis.py` 计算客户端文本特征的 raw top-1
   奇异值，并维护历史异常分数。
5. `aggregator/seismograph_aggregator.py` 排除持续异常的客户端，再按样本数
   加权聚合其可训练 prompt 参数。

主要运行输出位于 `results/`，包括：

- `run_config.yaml`
- `detailed_metrics.csv`
- `summary_metrics.csv`
- `text_feature_raw_top1_history/`
- `update_metric_analysis/`
- `saved_models/final_analysis/`

## 保留的源码结构

```text
aggregator/   seismograph 聚合器、基类和唯一构建入口
configs/      三种攻击各一个 FPL + seismograph 配置
context/      跨轮联邦状态
servers/      FPL 训练循环与三种攻击的触发器优化
trainmodel/   CustomCLIP 与模型基类
users/        FPL 客户端正常/攻击训练和评估
utils/        数据加载、触发器、投毒与 seismograph 分析
```
