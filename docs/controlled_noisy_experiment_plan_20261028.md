# 周会要求对应的受控含噪实验与代码改造计划

日期：2026-10-28；对应 `../DPCO_dev.md` 的 DPCO-0004。本文件仅为分析和待授权实施计划，不是代码变更或训练启动授权。

依据：[用户提供的周会总结](/Users/jason/Documents/Codex/2026-09-02/new-chat/outputs/教授周会_10.27_新项目实验重点总结.md)。本次阅读的是书面总结，没有独立核听原始录音；总结内已经区分教授要求与整理者建议，本文继续保留该区别。

## 1. 教授这轮实际要交付什么

| 层次 | 内容 | 证据地位 |
|---|---|---|
| 首要实验 | 可区分 clean/corrupted 的 noisy dataset，展示两组学习行为，重点看固定 clean subset 的普通 loss | 总结转述的教授明确要求 |
| 核心对照 | self-influence + discard；self-influence + flip；历史信号加当前信号 + flip | 总结转述的教授明确要求 |
| 推进顺序 | 优先 Adam，随后补 SGD；先完成受控验证，再扩大规模/应用 | 总结转述的教授要求与建议 |
| 历史机制 | 每批平均 signal 与衰减历史相结合，借鉴 Adam 的 β 衰减 | 有方向性要求，精确公式未给定 |
| 额外控制 | 普通 Adam、干净验证/测试指标、多随机种子、固定污染与统计口径 | 总结整理建议及本次实验设计建议 |
| 尚未确定 | 数据集、模型、污染率、β、r 的时间索引、LO、轮的单位、学习率、预算 | 不能当成教授已确定的参数 |

本轮不以“不同方法各自的加权训练目标更低”作为主结论。负权重下的训练 surrogate 可能为负，且各方法的 w 不同；应使用同一固定样本集合、同一普通交叉熵评价所有模型。

corrupted subset 的高损失只有在明确采用污染标签时，才可能表示抵抗错误标签记忆。高损失单独也可能来自模型失效，必须联合 clean 表现、正确标签表现与数值稳定性解释，不能以拉高污染损失为训练目标。

## 2. 对上一轮建议的修订

上一轮在缺少周会详情时，推荐 past-only 的 lagged EMA，便于条件理论分析；它仍是合理的消融候选，但不是教授已经指定的时序。

本次新增材料强调“当前 signal + 历史衰减 signal”。因此建议以 **包含当前 batch 的 EMA reference** 作为主原型，past-only 版本作为时间索引消融。这是本次选择的可执行解释，不是恢复出了录音未提供的精确公式。

上一轮讨论的“只反射 reference 平行分量”属于另一种编辑规则，本轮主实验继续采用教授要求的整向量 flip，不把两个变化同时混入比较。原理论的目标迁移条件与反例继续有效，不撤回，也不将它们解释为 Adam 情形的实际保证。

## 3. 第一轮建议固定的实验设置

建议先做 CIFAR-10 的可控标签污染，而非直接扩大到 ImageNet/iNaturalist。它是一个比手写数字更复杂的彩色物体分类任务；会议并未指定该数据集，也不能据此声称已满足某个未给定的样本数量规模要求。

[CIFAR-10 官方说明](https://cave.cs.toronto.edu/kriz/cifar.html)：32×32 RGB、10 类，50,000 个训练样本和 10,000 个测试样本。

| 设置项 | 本次建议的初始方案，不是教授指定 |
|---|---|
| 划分 | 从官方训练集固定分层留出 5,000 个干净验证样本，剩余 45,000 个用于训练；官方测试集保留干净 |
| 主污染 | 训练集中精确选定 40% 样本，类别均衡抽取；标签均匀改为其余 9 类之一，保证被选样本实际发生变化 |
| 清洁对照 | 后续补 0% 污染 sanity；corrupted 组为空时指标明确标为不适用，不能除零 |
| 模型 | 小图版 ResNet-18：basic block、3×3 stride-1 stem，不使用现有 ImageNet 的 7×7 stride-2 + max-pool stem；随机初始化 |
| 数值与输入 | 首轮 FP32；标准 CIFAR 小图 crop/flip 与固定归一化，评估关闭随机增强；不混入 MixUp/CutMix/label smoothing |
| 优化器 | 标准 Adam，β₁=.9、β₂=.999、ε=1e−8；首版不加入 weight decay，避免混淆 Adam 与 AdamW |
| reference | 当前+历史 EMA，β_ref=.9 作为待验证起点；β_ref 与 Adam β₁ 是独立参数/状态 |
| batch 与预算 | 建议 B=128；正式预算初定 100 个完整数据加载 epoch，所有方法相同；尾批规则、对应更新数和 epoch 定义在运行前固定 |
| 干预频率 | Adam：每个 optimizer step；“轮”按 batch 更新步实现是本次建议，会议没有确认该单位 |
| 初始化控制 | 同一 run seed 下，四组共用相同初始权重、数据顺序、增强序列、污染 manifest 和训练预算 |
| 主结果 | 预定 final checkpoint；可另报按干净验证 CE 选择的 best checkpoint，不能按测试集选 checkpoint |

污染发生在验证集划分之后，仅改变训练所用标签；固定保存 sample_id、原始标签、实际训练标签、是否被污染、split 与 corruption_seed。原数据保留不覆盖，生成独立 manifest，记录哈希和实际比例。不能每个 epoch 重新生成污染。

建议固定 split/noise seeds，主结果用 3 个配对的训练 seed。第一轮误差条表示初始化/抽样/增强的训练随机性，不代表跨污染实现的稳定性；后续如扩展 corruption seeds，要单独报告。

训练器只读取 images、observed_label 和有效 mask；original_label 与 corrupted 标记只给离线评估/观测。不得用真实污染标记决定 discard、flip、构建 reference 或提前过滤训练样本。干净验证集用于各组同等预算的调参和选模型，其监督信息使用需明确披露；它不参与 reference。

## 4. 主实验四组，先控制信号与干预两条轴

设当前观察标签下的样本梯度为 g_i，原始 batch 平均为 G=B⁻¹Σg_i。本文“self-influence”明确指 q_i=Gᵀg_i，不能误写成 g_iᵀg_i，否则永不出现负分数。

| 编号 | 对外显示名 | reference | 样本权重 | 参数更新器 |
|---|---|---|---|---|
| A | Plain Adam | 不使用 | 全部 +1 | 同一个标准 Adam |
| B | Self + Discard + Adam | G | q_i<0 为 0，其余 +1 | 同一个标准 Adam |
| C | Self + Flip + Adam | G | q_i<0 为 −1，其余 +1 | 同一个标准 Adam |
| D | EMA-current + Flip + Adam | 当前与历史的 EMA | q_i<0 为 −1，其余 +1 | 同一个标准 Adam |

共同方向定义为 d=B⁻¹Σw_i g_i；B 为原有效 batch 大小，不能在 discard 组改除以保留样本数。不更改下一步数据采样，也不因丢弃而减少计划训练步数。

现有 DTV 是 **gradient discard**：置零梯度贡献，不等于在 forward 前把样本移除。带 BatchNorm 时被丢弃样本仍影响批次统计；首轮明确采用现有语义，汇报中应写清。若需要字面上的数据删除，那是另一协议，不能当作同一 baseline。

C 对 D 回答历史信息是否有额外收益；B 对 C 回答 discard 与 flip 的区别；A 用于检查全部干预是否真正胜过普通优化器。主要比较都发生在同一个 Adam 内，不先把优化器差异当作方法收益。

会议中的 LO 没有完整定义，不能直接认定等于当前代码的 `_loo`，也不能宣称教授同意取消它。首轮采用明确的 full common-reference baseline，不启用当前 `_loo` 或局部层评分；如需其他定义，另列消融，不悄悄混用。

## 5. 当前+历史 reference 的建议精确定义

第 t 步（t=1,2,…）更新前，使用 observed labels 计算原始全 batch 平均 G_t。历史状态为 H_0=0，先形成

\[
H_t=\beta_{\rm ref}H_{t-1}+(1-\beta_{\rm ref})G_t,
\qquad
\boxed{r_t=H_t/(1-\beta_{\rm ref}^{\,t})}.
\]

随后 q_{t,i}=r_tᵀg_{t,i}，权重按严格负号翻转，零分数不翻。构建 r、评分与求编辑方向时权重/reference 均 stop-gradient；本步成功后才提交新的历史状态。历史更新使用原始 G_t，不能使用编辑后的 d_t，更不能直接复用 Adam 的一阶矩。

该主原型有两个可验证的边界：

- β_ref=0：r_t=G_t，D 在数值上退化为 C；无需为该消融另跑一套重复正式实验。
- 第一步：r_1=G_1，因此与 self + flip 的评分一致，而不是无历史时随机取方向。

正的 bias correction 不改变 sign，但使状态幅值定义完整。EMA 的 β_ref=.9 与 Adam 的 β₁=.9 即使数值相同，也有不同用途：H_t 跟踪未经编辑的 G_t；Adam 一阶/二阶矩跟踪送入优化器的 d_t。两套状态必须独立保存。

不在第一版给 reference 除以平方梯度均方根，不添加置信阈值、不引入 error/correct 子集 reference，也不将历史状态当作当前 clean objective 的精确梯度。当前项进入 reference 后，上一轮 past-only 的条件独立性论证不能原样套用。

更重要的是，G_t 使用 observed labels，历史 EMA 首先追踪的是含噪训练目标的梯度，而不是干净目标。平均可以降低抽样波动，却不能一般性地消除标签污染导致的系统偏差；固定错误标签也可能被稳定记住。因此“历史更稳定 -> 更容易识别污染贡献 -> clean 学习更好”是本次要检验的机制假设，不是 Adam 原论文或旧 SAGE 定理已证明的结论。

## 6. 评估必须回答教授的问题

每个固定评估节点用同一 checkpoint、确定性输入和 inference-mode BN，在固定分组上计算未加权、未翻转的普通 CE：

\[
L_{\rm clean}=|C|^{-1}\sum_{i\in C}\operatorname{CE}(f(x_i),y_i^{true}),
\]
\[
L_{\rm corrupt,obs}=|N|^{-1}\sum_{i\in N}\operatorname{CE}(f(x_i),\widetilde y_i),\qquad
L_{\rm corrupt,true}=|N|^{-1}\sum_{i\in N}\operatorname{CE}(f(x_i),y_i^{true}).
\]

其中 C、N 固定于污染 manifest，绝不能按当前预测是否错误、是否被 flip 动态改变成员。正式版本优先评估完整训练分组；若为了成本采用固定审计子集，必须运行前冻结 sample IDs、分组数量与采样规则，不按结果挑样本。

| 记录内容 | 用途 |
|---|---|
| clean train 普通 CE 与 accuracy | 教授关心的干净子集学习情况 |
| corrupted observed-label CE | 是否记住错误标签；高值要与其他指标共同解释 |
| corrupted true-label CE 与 accuracy | 是否仍学到了被污染样本的正确类别 |
| clean validation CE/accuracy | 调参和预定义模型选择，不进入 reference |
| clean test CE/accuracy | 最终独立评估，不用于挑参数/β/epoch |
| edited/discarded fraction，分 clean/corrupted 的干预率 | 解释干预行为；污染标记仅用于观测 |
| reference 范数、与当前 G 的 cosine、编辑前后方向范数 | 识别尺度改变和历史滞后；cosine 稳定不等于正确 |
| 学习率、步数、样本曝光、数值 finite、时间/峰值内存 | 排除预算或数值失效导致的假结果 |

普通 train_ce 保留其原始定义，不能改成 signed surrogate 后继续沿用同名指标。不要只展示不同方法各自的 Σw_iℓ_i。报告污染组高 loss 时，必须说明 observed/true 标签口径；若所有组 loss 都高、accuracy 低或参数非有限，这是失败，不是所需的选择性学习。

建议主图：clean CE、corrupted observed CE、corrupted true CE、干净验证 accuracy 四个面板；正式测试结果放最终表。额外提供干预率与耗时表。step/epoch、原始曲线、平滑规则和样本数统一。

## 7. 当前代码有哪些可复用、哪些确实缺少

以下是基于现有文件的只读检查，不是未检查就建议重写整个系统。

| 模块与文件（相对 DPCO/） | 已有能力 | 待授权后所需改动 |
|---|---|---|
| `src/methods/dtv.py`、`dpco.py`、`base.py` | 原 B 归一化的 discard/flip，完整参考 VJP/JVP | 直接复用 B/C；为 D 加纯函数历史状态和 reference 构建路径，复用 raw G 的 VJP、以 JVP 得分，不在正式训练物化 [B,P] 雅可比 |
| `src/training/updates.py` | 只有 SGDUpdate 及可选 momentum/Nesterov，make_updater 始终返回 SGD | 增加独立 Adam updater/adapter；建议复用标准 Optax Adam 并显式声明依赖版本；保留旧 SGD 默认行为 |
| `src/config.py`、`src/training/launch.py`、`scripts/launch_tpu.py` | 三个现有数据集、ResNet50/101 和既有方法别名 | 明确区分 optimizer、contribution method、reference；增加污染、模型、EMA、调度字段及父子 CLI 转发/校验；suite 补上已有 dtv 的入口 |
| `src/data/classification.py`、`images.py`、`batch.py` | ImageNet/iNaturalist 分发；ImageNet resize/crop/mean/std；batch 仅三字段 | 增加 CIFAR 数据适配、专用小图变换和独立污染 manifest/label view；训练 batch 继续只暴露 observed labels，评估侧按 ID 读取真值 |
| `src/models/resnet.py`、`src/models/__init__.py` | 仅 ImageNet bottleneck 模型和大图 stem | 增加明确命名的小图模型/适配，不把现有 resnet50 静默改成其他模型 |
| `src/training/trainer.py`、`local_mesh.py` | TrainState 已分 method_state 与 optimizer_state，支持编译和数据并行 | 将 reference H/count 放 method_state，Adam moments 放 optimizer_state；全局 batch 统一历史，不允许每设备各自生成 reference |
| `src/training/experiment.py`、`checkpoints.py` | 整个 train_state、data/RNG 恢复；编译候选丢弃 | 目前有状态方法初始化/验证写死 policy/warmup；需支持新历史状态，校验 count/step、污染 fingerprint 和 optimizer 类型；评估/编译候选不推进正式 EMA 或 Adam 状态 |
| `scripts/train_classification.py` | TPU-only 入口，明确拒绝 data.mode 非 clean；元信息写死 ImageNet/SGD 描述 | 显式接入新数据和优化器，正确记录模型/变换/污染/历史定义，不能仅把 mode 改名字后数据仍是干净的 |
| `src/training/evaluation.py`、`recording.py` | 单验证流，标准 CE/accuracy，支持私有 schema | 增加固定 clean/corrupted 分组、多标签视图和新 CSV 流；保存 counts/manifest ID，保留旧 CSV 字段含义 |
| `scripts/plot_training_results.py` | 只认识 SGD、DPCO-E/L 及既有比较组 | 新增独立 controlled-report 绘图/汇总入口，按 optimizer+signal+action 命名；不把 method=sgd、updater=adam 的普通 Adam 标为 SGD |

[Optax Adam 官方文档](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html) 可作为参数更新语义依据；这不是现在安装依赖的授权。现有项目没有显式声明 optax，不能仅因某环境碰巧存在就认为依赖完整。

不要把已有旧 theorem/diagnostic/mechanism observers 简单解除校验后挂到 Adam 上。它们不少假设 plain SGD、固定参考或指定轨迹；对 Adam，编辑会同时影响当步一阶与二阶矩，旧 q/Δ 恒等式不是实际一步下降保证。首轮关闭不兼容的旧 observers，仅记录新任务需要的数值指标。以后如做同状态 Adam 反事实，必须复制完整 optimizer state 和计数，并丢弃候选状态，不能用新建 Adam 代替。

训练入口当前规定 TPU-only；本轮计划不自动改变执行设备策略。未来授权后的本地 CPU 检查用于微型夹具，正式入口和 TPU smoke 应依项目既有约束运行；本次没有启动任何作业。

## 8. 按依赖顺序改造和验收

### 第一阶段：数据、评价与普通 Adam 闭环

先实现 Adam、CIFAR/小图模型、固定污染与分组评价，跑微型 fixture 和显式的短烟测。先验证原始/训练标签对照、污染数、固定 IDs、分组普通 CE，以及普通 Adam 能稳定训练。没有可靠分组评价，不进入历史机制比较。

### 第二阶段：接通 B/C 与有状态 D

复用 DTV/DPCO 权重逻辑，新增 EMA reference 状态、更新计数、父子配置和 checkpoint 校验。四组走同一数据、模型和 Adam 实现；不另外改 flip 几何、LO 或评分层。

授权后需要的有意义检查包括：

- 原数据/划分可复现，精确污染率且 true≠observed；验证与测试无污染；各方法污染 manifest 一致。
- discard/flip 的权重、严格负号、padding、原 B 归一化正确；污染 mask 不进入权重决策。
- 小型模型上显式 per-example 结果与 VJP/JVP 一致；β_ref=0 与 C 数值一致，第一步评分一致，H 用 raw G 更新而非 d。
- 所有方法全局 BN 状态每步只提交一次；编译 warmup、评估、诊断不推进 H/count 或 Adam moments。
- Adam 数步结果符合标准实现；checkpoint 恢复后 Adam/EMA/data-RNG/调度与连续运行一致，改变污染或时序则拒绝恢复。
- 固定分组评估覆盖每个 sample_id 一次、标签视图正确、counts 与手算 CE 一致；空分组明确不适用。
- 旧干净 SGD/DPCO 的适当回归通过，默认接口、训练语义和旧 CSV 未被意外改变。

### 第三阶段：四组短试跑，然后主实验

建议先用普通 Adam 在固定干净验证集上对 3 个相邻学习率（如 1e−4、3e−4、1e−3）做等预算短校准；它们只是待验证起点，不是教授指定范围。把选定的合理 LR/调度冻结用于四组主比较，以隔离 signal/action；对外结论先限定在该设置。若进一步主张普遍优于充分调优的 Adam，则要给各组相同的小范围调参预算，不能只调新方法。

先跑一个 seed 的四组短闭环（建议 10–20 epoch）检查工程稳定性，负结果也保存；随后按预先固定的预算跑 4 组 × 3 seeds = 12 个正式 run。正式 run 建议 100 epoch，不因某组曲线难看临时改变预算或选择汇报 seed。所有值是建议，启动前仍需落实硬件与实际步时。

### 第四阶段：少量消融，再扩展优化器

先复用 C 作为 β_ref=0 消融，再增加 β_ref=.5 与 past-only β_ref=.9；0% 污染作为无噪声控制。分阶段开展，不一开始展开所有 noise×β×LR×optimizer×model 的组合。主配置和消融均应记录全部结果。

随后补 SGD：把“每两轮一次”明确定义为每两个 optimizer batch steps 干预一次，例如第 2、4、6…步；三种干预方法使用相同开关，非干预步执行普通贡献，历史每个训练步仍用 raw G 更新。该相位和单位是提议，会议未明确；不额外把交替执行解释成必须观测 error reference。不要把 SGD/Adam 的不同调度结果当作只改变 optimizer 的严格对照。

## 9. 汇报判据与交付物

第一轮交付：固定设置/数据 manifest 表、四组 baseline 表、三种标签口径的固定分组 loss 曲线、干净验证/最终测试表、干预率与成本、配对 seed 的均值和波动，以及有无历史的消融。

有意义的支持是：D 相对 C/A 在同一 clean subset 上有更好的普通 CE/学习过程，同时正确标签视图与独立干净评价不显示整体崩坏；若 corrupted observed loss 更高而 true-label loss 更低，则更能支持减少错误标签记忆的解释。并不要求所有步 loss 单调下降，也不能把这些经验观察称为无条件理论证明。

如果 D 只让所有 loss 变大、只让 signed surrogate 更低、只在单个 seed 获益，或只胜过未合理校准的基线，都不足以支持教授的目标。如果负结果持续，按预定设置汇报和诊断，而不是持续叠加新机制直到得到期望曲线。

总结中的“下个礼拜”为原会议时间记录，不能当成本次自动启动、设定提醒或压缩验证步骤的授权。大模型、ICL、RL/post-training 和投影反射等后续范围暂不进入这个最小闭环。

## 10. 当前状态

本次仅新增本计划并追加开发记录；没有修改 Python、配置、脚本或依赖，没有下载数据、执行测试训练、连接服务器、提交或推送。上述设置与文件改动都等待用户后续明确授权实施。
