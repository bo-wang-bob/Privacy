# 首轮全局类别分布

2026-09-13。用户要求加入 Ma 等《Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning》的首轮统计共享，随后进一步要求中心也使用全局同类统计。用户随后明确要求不排除自身。当前默认是**全局同类别几何＋包含自身的全局类别均值**，实现版本为 `local_token_geometry_v12_direct`。此前 v8 的全局噪声＋本地中心保留为对照。

## 聚合协议

在第1轮任何客户端优化之前，所有已配置客户端从各自原始完整训练集计算每类样本数、均值和协方差，上传类别统计。只使用训练集，不使用 evaluation，也不以首轮参与抽样或实际 mini-batch 代替整个客户端训练分区。统计只执行一次，后续轮次不刷新。

按论文 §3.1、式(4)，类别c的全局统计为：

\[
N_c=\sum_k n_{k,c},\qquad
\mu_c=\sum_k\frac{n_{k,c}}{N_c}\mu_{k,c},
\]
\[
\Sigma_c=\sum_k\frac{n_{k,c}}{N_c}
\left[\Sigma_{k,c}+(\mu_{k,c}-\mu_c)(\mu_{k,c}-\mu_c)^T\right].
\]

协方差采用总体除数 n，与论文公式及现有本地几何一致。必须包含客户端均值差异项；仅平均本地协方差会漏掉跨客户端变化。权重来自当前类别样本数，与模型上传的聚合权重设置无关。客户端缺少某一类别时不贡献该类统计，但仍接收全局所有存在类别的分布。

这里聚合的是**不同客户端的同一类别**，没有恢复先前删除的本地跨类别合并协方差，也没有恢复 `pooled_rank` 或 `shrinkage`。

## 表示和生成

继续使用当前 CLIP 输入 patch+position 编码（不含CLS），保持 Adapter/LoRA 在线训练。ViT-B/32 的维度为49×768=37,632，单个float32稠密协方差约5.27GiB。因此上传与下发采用协方差的谱因子：`covariance_factor @ covariance_factor.T`。计算用float64，因子保存float32；只移除数值零方向。

局部上传包含完整数值秩因子，**不会先按 class_rank 截断本地统计**。服务器拼接加权因子和均值偏移列后做特征分解，保存完整数值秩的全局协方差因子及特征值；全局分布不是仅五维的统计。生成阶段才按已有 `class_rank=5` 取前几个方向，没有新增秩或混合权重超参数。资源开销随类内样本数和数值秩增加，完整数据集设置可能明显增加初始化成本。

\[
\tilde h_{i,v}=(1-r_i)h_i+r_i\mu^{\mathrm{global}}_c
+0.1L_c^{\mathrm{global}}\epsilon_{i,v},\qquad
L_c^{\mathrm{global}}=U_{c,1:q}\sqrt{\Lambda_{c,1:q}}.
\]

生成中心直接使用首轮下发的 `mu_global_c`，包含当前原始记录，所有客户端同一类别的全部样本与视图使用相同的参考中心。均值按同类原始样本数加权，不是客户端均值的简单平均。不再逐样本扣除自身，不需要额外通信或重新统计。每条原始记录对全局同类均值的贡献为1/N_c；当前全局每类100张时为1%。风险控制的加噪前生成位置仍是 `(1-r_i)*h_i+r_i*mu_global_c`，所以风险小于1时该位置仍因原样本而异。全局协方差包含原始训练集中的该样本，不能称为全局 leave-one-out 统计。两个替身独立加噪并共同训练，仍对每条原始记录的K个视图平均CE，每个原始batch只做一次optimizer step。风险与原始成员候选池保持原定义；v12生成后直接训练，没有语义重试或候选有效性检查。

本次只接入论文的统计聚合/下发步骤。论文在最终CLIP embedding上生成，式(5)的噪声使用特征值λ；这里仍保留输入token、低秩及sqrt(λ)噪声。因此是现有方法的扩展，不是整篇论文逐项复现。

## 产物和通信范围

- `client_*_moment_upload.pt`：模拟上传的逐类count、mean、covariance_factor，不含原始图片、逐记录编码、标签向量或记录ID。
- `global_distribution.pt`：全类别均值、样本数、完整数值秩协方差因子、特征值、用于生成的截断因子和贡献来源。
- `client_*_global_receipt.json`：每个客户端接收同一分布的SHA256、可用类别、接收时间点和是否用于生成。
- `synthesis_summary.json`：共享字段、只执行一次、数值秩策略、统计上传的审计范围。旧本地分布及source_codes继续作为本地统计/诊断产物保存。

仓库在单进程中模拟客户端和服务器；“下发”实现为每个客户端持有同一只读约定的映射对象，并保存各自回执，避免在磁盘复制多份巨大分布文件。没有新增网络通信框架。

统计共享没有DP保护，既有11种攻击未专门攻击这些共享统计。全局生成实验中的原攻击指标仍可衡量原上传/模型视图，但不能据此声称完整新协议获得同等隐私改善。

## 运行与验证

新版已从 `research/risk-synthesis-global` 快进整合到 `/root/Privacy` 的 `main`，当前运行统一使用正式目录。整合前主分支保存在 `archive/pre-global-integration-20260914`，研究分支和已有实验结果保留；历史结果仍按产生它们的版本解释。

```bash
cd /root/Privacy
PY=/root/.local/share/mamba/envs/pfedba/bin/python
$PY scripts/run_synthesis_multiview.py --dry-run
# GPU空闲时运行；使用统一批量入口，默认K=2、FedAvg100轮、每类100张。
$PY scripts/run_synthesis_multiview.py --gpus 0
```

catalog 默认 `defense.synthesis.global_distribution=generate` 和 `defense.synthesis.center_source=global_class`。

- `--set defense.synthesis.center_source=local_class`：保留全局噪声，恢复 v8 本地中心对照。
- 同时加 `--set defense.synthesis.global_distribution=disabled --set defense.synthesis.center_source=local_class`：关闭统计交换，作 v7 本地中心＋本地噪声对照。
- `share_only` 要求 `center_source=local_class`，仅共享、不使用全局统计生成。

直接加载缺少新字段的旧配置，global_distribution缺省disabled、center_source缺省local_class，避免悄悄修改历史训练协议。全局中心要求global_distribution=generate及uniform中心权重；不会把本地历史风险权重冒充全局加权中心。center_source是协议对照选项，没有新增连续混合系数。

验证覆盖：不等样本数权重、缺失类别、零类内方差时的客户端均值差异项、完整局部数值秩上传、全局采样与本地/全局中心的组合、包含自身的全局均值中央计算对照、同类别跨客户端中心一致性、全部客户端接收全部类别，以及Adapter/LoRA的多轮多视图与11种攻击集成。独立核验脚本 `scripts/verify_synthesis_global_geometry.py` 校验文件哈希、接收回执、权重/均值，并用每类8个固定探针方向检查聚合协方差；探针检查不等于对真实高维矩阵的逐元素完整重算，中央协方差的完整逐元素对照由小规模数值测试完成。

当前没有 v8/v9/v10 全局几何或全局中心的真实数据隐私或准确率效果结论。v9排除自身协议保留在提交4eab520中，复现它须使用该代码版本；旧结果核验仍识别原协议，新版本不改写历史结果。

v9验证：162项相关测试通过（17.20秒）；统一入口干运行明确显示 `global_distribution:generate`、`center_source:global_class`、`views_per_record:2`、FedAvg100轮、每类100张，未启动真实训练或创建结果目录。`git diff --check` 通过。

历史v10验证：162项相关测试通过（13.23秒），覆盖全局均值包含自身、同类别跨客户端参考中心一致、源记录变化按1/N_c进入均值，以及Adapter/LoRA多视图与全部11种攻击的集成。`git diff --check`通过；未启动真实数据训练。

正式验证命令见 [Adapter/LoRA单种子验证](risk_synthesis_validation_commands.md)。入口默认只运行seed43的无防御与当前方案对照，实际超参数以解析配置为准。

## 2026-09-14 范数门槛修复（v11）

全替换候选不再受 `norm_ratio_min/max` 限制；这两个配置项已删除，范数比仅记录诊断。全局均值的范数可明显小于个体编码，旧下限会拒绝有限且已改变的高风险候选，耗尽重试后触发全部替换中止。当前仍保留有限值、实际改变、视图不重复和固定教师语义检查；语义全失败仍选最佳有效候选，不恢复原始编码。没有调大噪声或重试次数，也没有强行把替身范数归一到原始样本。

新摘要标记 `norm_ratio_filter_enabled=false`、`norm_ratio_role=diagnostic_only`。核验工具按记录版本解释，历史 v10 等仍核对旧范数边界。不能直接用旧失败任务中含上下限的 `run_config.yaml` 启动新版；应使用当前统一入口重新生成配置，旧配置复现用原代码版本。只补跑防御的命令见 [验证命令](risk_synthesis_validation_commands.md)。相关168项测试通过，包括上下界外候选直接采用、多视图及Adapter/LoRA全部攻击接口；该验证不代表100轮真实数据效果。

实样本生成检查使用失败Adapter任务的client3/原始ID440保存编码（类别44），重新设定风险25.5/26、抽样seed20260914，调用当前生成流程及真实冻结CLIP教师。原始编码范数166.1693，全局同类均值范数16.8986；两个不同替身的范数比分别为0.088886和0.108100，教师语义间隔变化分别为+0.042902和+0.039170，均首个候选通过。前者会被旧0.1下限拒绝。未恢复原始编码，未执行optimizer；失败批次的实际风险及随机状态未保存，因此这不是精确重放，也不代表防御效果。诊断保存于 `analysis_scripts/synthesis_norm_fix_20260914/probe.json`，脚本为 `scripts/check_synthesis_norm_fix.py`。防御组的双模型干运行核对通过，未启动完整训练。

## 当前v12：候选直接共同训练

按用户要求删除有效性和固定教师语义检查。每条原始记录的V个视图各抽样一次、全部训练，不检查有限值/范数/距离/实际改变/视图重复，不拒绝、不重试、不择优。关闭教师初始化和语义统计；批量采样按类别执行，原始输入编码在同批所有视图间复用。语义、范数、距离字段留空，不能作为零失败率解释。训练损失有限性检查仍保留，不参与候选选择。原始成员身份、风险混合均值、几何因子和视图平均损失不变；旧结果与已运行进程保持原版本。完整说明见 [当前方法](risk_synthesis_method.md)。

验证：185项相关测试通过；最后采样接口调整后直接生成与全局几何27项复测通过。双模型默认2视图及Adapter4视图命令干运行通过；未启动完整真实数据训练，不据此声明加速倍数或隐私效果。
