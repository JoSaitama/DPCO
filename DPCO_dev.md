# DPCO 开发协议与变更记录

## 一、开发协议（2026-10-28 生效）

### 1. 修改授权

- 只有用户明确要求修改代码，才可新增、修改、重命名或删除代码文件，以及改变程序行为的配置、依赖、启动脚本。
- “读论文”“解释”“分析”“讨论”“给方案”只授权相应的研究与说明工作，不授权实现代码。代码修改授权仅对用户明确指定的任务和必要关联文件有效。
- 本次用户明确要求把 SAGE 文件夹及 Python 代码中的 SAGE 名称改为 DPCO，因此本次可以进行一致性更名；不能据此实现 Adam、调整梯度算法或改动训练默认行为。

### 2. 记录要求

- 所有代码和理论变动统一记入本文件，包括新增、更正、撤回，以及对理论前提、公式、结论或适用范围的修改。
- 每条记录写清：日期、授权依据、目的、涉及文件、变更内容、验证与结果、假设与限制、后续待办。
- 研究讲解也应记录核心结论和来源；较长推导可另存 `docs/` 并在本文件链接，关键更正必须在本文件说明。
- 追加记录，不删除或覆盖旧记录；历史错误通过追加更正保留可追溯性。

### 3. 目录和命名

- 后续代码开发与修改统一在 `DPCO/` 内进行。
- 复用原 SAGE 代码，目录使用 `DPCO`，Python 模块、方法字符串及标识符使用对应的 `DPCO` / `dpco`，同步更新调用所依赖的配置、脚本和说明。
- 名称更换不代表新算法已经定义或取得改进；更名前后的数学更新规则应相同。
- 历史实验数据、检查点、旧开发日志及 Git 历史不改写。旧名称的历史产物与新名称接口之间的迁移，需要单独分析与授权。

### 4. 理论和验证

- 按面向博一学生的细讲风格说明目标函数、符号、推导、动机、例子、假设和反例。
- 分清原文主张、已证结论、教学近似和未验证设想；分清优化速度、最终训练效果、测试泛化和实际成本。
- 保留用户现有未提交修改；本地验证应与授权范围对应，不擅自启动正式训练或执行 Git 提交、推送。

### 5. 执行入口

项目根目录 `../AGENTS.md` 与本目录 `AGENTS.md` 供后续开发读取；本文件是代码和理论变动的统一记录入口。原 `docs/development.md` 保留为复用代码的历史档案，不再作为新变更的唯一记录。

## 二、变更记录

### 2026-10-28 / DPCO-0001：建立协议并执行项目一致性更名

- **授权依据：** 用户原话：“所有代码和理论的变动都要写入dpco文件夹下的DPCO_dev.md文档”；“只有我明确说明修改代码后，才修改文件”；“包括文件夹名字SAGE在内的py文件，都把SAGE的名字改为DPCO”。
- **目的：** 建立后续开发约束，并将原 SAGE 代码作为 DPCO 项目的起点。
- **现有状态：** 原 SAGE 仓库已有修改和未跟踪文件；更名基于当前文件内容进行，不回滚这些工作。原仓库 Git 历史和原始结果保留。
- **涉及文件：** 根目录 `AGENTS.md`；本文件；本目录 `AGENTS.md`；原 `SAGE/` 下含 SAGE 名称的 Python 文件、相关配置、启动脚本、项目元信息及当前说明文档。逐文件清单见本次工作完成后的验证记录。
- **变更内容：** `SAGE/` → `DPCO/`；Python 中名称及导入 `SAGE` / `sage` → `DPCO` / `dpco`，保留大小写风格；同步相关文件名、配置键、CLI 参数、环境变量、结果字段和脚本引用。普通英文单词中的子串（如 `message`、`usage`）不替换。`docs/development.md` 保留原文。
- **行为边界：** 仅改名称，不改变原更新公式、超参数数值、训练策略和参考梯度语义。新接口不自动兼容旧名称的命令、配置和检查点；历史产物保持原样，不能宣称已完成旧检查点迁移。
- **验证进展：** 已核对 70 个既有文件均仅作名称替换；97 个 Python 文件 AST、13 个 TOML、4 个 shell 脚本语法通过；153 个历史日志/结果文件哈希保持一致。首轮既有 CPU 回归运行 209 项，206 项通过，1 项暴露测试字面量中转义换行后的旧配置名漏替换，2 项因测试环境缺少 matplotlib 报错。已补齐该测试字面量更名，复用本机已有绘图环境路径重跑，不更改项目依赖。最终结果见末尾。未启动正式训练。

### 2026-10-28 / DPCO-0002：阅读 Adam 并建立与 SGD 的对照讲解

- **授权依据：** 用户要求“按照给博一学生的细讲风格，把adam这篇论文对比着sgd给我讲一下”。这项授权用于阅读和理论说明，不用于实现 Adam。
- **论文来源：** 项目根目录 `1412.6980v9.pdf`，Kingma 与 Ba，*Adam: A Method for Stochastic Optimization*，ICLR 2015，所给文件为 arXiv v9（2017-01-30），15 页。
- **记录位置：** `docs/adam_vs_sgd.md` 保存完整推导、示例及文献边界。
- **核心结论：** Adam 使用梯度的一阶指数移动平均估计方向，使用逐坐标的二阶原始矩指数移动平均估计尺度，并修正零初始化造成的权重缺失。它可能改善噪声梯度、坐标尺度差异和稀疏特征下的优化；不存在普遍优于 SGD 的保证。
- **关键边界：** 二阶原始矩不是方差或 Hessian；平均梯度的平方不等于样本梯度平方的平均；一阶矩的平滑收益必须与带动量 SGD 公平比较。初始化偏差修正不消除非平稳滞后、梯度本身的偏差或更新比值的偏差。
- **理论证据：** 原文在线凸优化遗憾分析与深网的非凸优化、最终泛化不同；后续 Reddi 等（ICLR 2018）指出原分析的问题并给出 Adam 不收敛的凸例子。Wilson 等（2017/2018）提供自适应方法泛化劣于 SGD 的例子；这些结果也不意味着 Adam 总是较差。
- **项目理论状态：** 本次未定义新的 DPCO 算法，也未把 Adam 的机制或保证赋予更名后的旧代码。待用户指定 DPCO 的新目标和定义后，再讨论或实现相应理论变动。

### DPCO-0001 附：本次更名文件清单

- 目录：`SAGE/` → `DPCO/`。下面路径均相对于 `DPCO/`；共 70 个既有文本文件发生名称替换，其中 15 个文件同时更名。
- `AGENTS.md`、`DPCO_dev.md` 和 Adam 讲解为新增文档，另计；旧历史日志及结果文件按字节哈希保留。

- `run_sage_theory_suite.sh` → `run_dpco_theory_suite.sh`
- `run_inaturalist_suite.sh`（内容名称替换）
- `run_sage_mechanism_suite.sh` → `run_dpco_mechanism_suite.sh`
- `pyproject.toml`（内容名称替换）
- `README.md`（内容名称替换）
- `run_sage_suite.sh` → `run_dpco_suite.sh`
- `tests/test_tpu_mechanisms.py`（内容名称替换）
- `tests/test_sage_warmup.py` → `tests/test_dpco_warmup.py`
- `tests/test_alternating_first.py`（内容名称替换）
- `tests/test_plot_training_results.py`（内容名称替换）
- `tests/test_observation.py`（内容名称替换）
- `tests/test_training_diagnostics.py`（内容名称替换）
- `tests/local_mesh_check.py`（内容名称替换）
- `tests/test_updates.py`（内容名称替换）
- `tests/test_inaturalist_training.py`（内容名称替换）
- `tests/test_directions.py`（内容名称替换）
- `tests/test_inverse_sqrt.py`（内容名称替换）
- `tests/test_checkpoint_controls.py`（内容名称替换）
- `tests/test_suite.py`（内容名称替换）
- `tests/test_config.py`（内容名称替换）
- `tests/test_trainer.py`（内容名称替换）
- `tests/test_sage_policy.py` → `tests/test_dpco_policy.py`
- `tests/test_references.py`（内容名称替换）
- `tests/test_sage_r.py` → `tests/test_dpco_r.py`
- `tests/test_theory_validation.py`（内容名称替换）
- `tests/test_classification.py`（内容名称替换）
- `tests/mechanism_mesh_check.py`（内容名称替换）
- `tests/test_tpu_pipeline.py`（内容名称替换）
- `docs/inaturalist_subset.md`（内容名称替换）
- `docs/training_diagnostics.md`（内容名称替换）
- `docs/tpu_mechanisms.md`（内容名称替换）
- `docs/recording.md`（内容名称替换）
- `docs/result_csv_guide.md`（内容名称替换）
- `docs/tpu_training.md`（内容名称替换）
- `docs/theory_validation.md`（内容名称替换）
- `configs/imagenet1k_resnet101_sage_e.toml` → `configs/imagenet1k_resnet101_dpco_e.toml`
- `configs/imagenet1k_resnet50_sage_e.toml` → `configs/imagenet1k_resnet50_dpco_e.toml`
- `configs/imagenet100_resnet50_sage_e.toml` → `configs/imagenet100_resnet50_dpco_e.toml`
- `configs/imagenet100_resnet101_sage_e.toml` → `configs/imagenet100_resnet101_dpco_e.toml`
- `scripts/validate_sage_theory.py` → `scripts/validate_dpco_theory.py`
- `scripts/sanity_recording.py`（内容名称替换）
- `scripts/plot_training_results.py`（内容名称替换）
- `scripts/train_classification.py`（内容名称替换）
- `scripts/launch_tpu.py`（内容名称替换）
- `scripts/run_suite.py`（内容名称替换）
- `scripts/sanity_core.py`（内容名称替换）
- `src/config.py`（内容名称替换）
- `src/__init__.py`（内容名称替换）
- `src/cli.py`（内容名称替换）
- `configs/tpu/inaturalist2021_resnet50.toml`（内容名称替换）
- `configs/tpu/imagenet100_resnet50_sage_e.toml` → `configs/tpu/imagenet100_resnet50_dpco_e.toml`
- `src/probes/classification.py`（内容名称替换）
- `src/probes/mechanism_config.py`（内容名称替换）
- `src/probes/theory_validation.py`（内容名称替换）
- `src/probes/tpu_mechanisms.py`（内容名称替换）
- `src/probes/gradient_statistics.py`（内容名称替换）
- `src/probes/training_diagnostics.py`（内容名称替换）
- `src/methods/sage.py` → `src/methods/dpco.py`
- `src/methods/sage_warmup.py` → `src/methods/dpco_warmup.py`
- `src/methods/__init__.py`（内容名称替换）
- `src/methods/sage_policy.py` → `src/methods/dpco_policy.py`
- `src/methods/base.py`（内容名称替换）
- `src/references/correct_subset.py`（内容名称替换）
- `src/training/results.py`（内容名称替换）
- `src/training/recording.py`（内容名称替换）
- `src/training/runtime.py`（内容名称替换）
- `src/training/experiment.py`（内容名称替换）
- `src/training/launch.py`（内容名称替换）
- `src/training/schedules.py`（内容名称替换）
- `src/training/status.py`（内容名称替换）

### 2026-10-28 / DPCO-0001、0002 完成验证

- **更名复核：** 将更名前的当前工作内容按确定的名称映射变换后，与全部 70 个修改后的既有文件逐字比较一致；包含修正转义换行后测试配置键的替换。未引入算法计算式或超参数数值变动。97 个 Python 文件解析通过，名称扫描无旧项目标识；`usage`、`message` 等普通英文保持原样。13 个 TOML 解析与 4 个 shell 脚本语法检查通过，Git 差异空白检查通过。
- **完整回归：** 第二轮 `python -m unittest discover -s tests -t .`，209 项全部通过，用时 84.405 秒。使用本机既有临时 JAX/Flax 测试环境，并通过 PYTHONPATH 复用已有 matplotlib 包；未安装或修改依赖。测试日志在 `/private/tmp/dpco-regression-final.log`（临时诊断材料，不是长期档案）。
- **历史保护：** 原 `docs/development.md` 及 `results/` 中共 153 个文件 SHA-256 与更名前一致。原有未提交/未跟踪工作基于其当前内容更名并保留；未进行提交、推送或远程操作。测试仅运行本地既有测试夹具，不是正式训练或 TPU 验证。
- **理论复核：** 阅读给定 PDF 共 15 页的提取内容，渲染核对 Algorithm 1、初始化修正、理论假设及图 1-4 所在完整页面；手算示例与默认系数的首步比值另用标量计算复核。长篇讲解保存在 `docs/adam_vs_sgd.md`，涵盖公式、噪声与尺度动机、偏差修正、对照实验、原文符号问题、后续收敛反例及与现有代码的区别。
- **新增文档：** 根目录 `../AGENTS.md`、本目录 `AGENTS.md`、本文件及 `docs/adam_vs_sgd.md`。两个执行规则文件供后续会话读取；后续所有代码及理论变更仍以本文件为统一记录入口。
- **限制及后续：** 未实现 Adam、未定义新的 DPCO 理论、未验证真实训练性能。旧 SAGE 命令/配置/结果字段和检查点中的旧名称未提供兼容层；原始结果及历史日志继续保留旧名，未来读取或迁移时须单独检查，不能直接假定新代码兼容。下一次代码实现或行为改变仍需用户明确授权。

### 2026-10-28 / DPCO-0003：历史 batch 梯度 reference 的可行性分析（理论候选）

- **授权依据：** 用户新加入 `SAGE_Proof_byProf/`，要求从理论分析历史 gradient mean reference 是否可行及如何定义。仅授权阅读、理论分析与按协议记录，未授权实现或修改代码。用户补充教授没有给出具体算法，只表示考虑历史 batch 的 gradient mean。
- **用户提供的经验状态：** 用户报告现有实验框架中 SAGE-E、SAGE-L 均低于普通 SGD；教授认为应考虑历史梯度，并报告 MNIST 有效。本次未复核原始实验输出，也未获得教授的 MNIST 方法细节，不能将本次 EMA 定义称作已复现或已验证的教授算法。
- **阅读来源：** `../SAGE_Proof_byProf/` 中三定理 PDF、`SAGE_theoretical_framework_cn.pdf` 及 DIVE 背景论文 `2608.22522v1.pdf`。原文件名称和内容全部保留。结合 Adam 一阶矩结构及动量/梯度投影相关原始文献核对术语。
- **涉及文件：** 新增 `docs/historical_gradient_reference_analysis.md`；本文件追加记录。未更改既有 Adam 阅读笔记、算法源文件、配置、依赖或启动脚本。
- **原理论适用范围澄清：** 定理 1 是任意 reference 下的代数最优；定理 2、3 需要 reference 等于真实目标梯度或另行证明目标迁移。总框架已提出 mismatch 条件，定理 3 也明确上界不直接排列不同完整轨迹。本次实验负结果不自动否定这些条件命题，但不支持无条件的实际优越性解释。不能确认 reference 是唯一失效原因。
- **候选 A（只更换 reference）：** 设 G_t 是未编辑的原始全 batch 平均梯度，m_0=0，r_1=0，t≥2 时 r_t=m_{t−1}/(1−β^{t−1})；本步评分后用 m_t=βm_{t−1}+(1−β)G_t 更新记忆。第一步/零 reference 回退普通 SGD。使用过去历史、全样本、未裁剪/未预条件/未翻转的梯度；不使用 S_t 写回，避免自我强化。β=0.9 仅为可讨论起点，尚未确定实验参数。
- **偏差修正边界：** 正的偏差修正因子不改变 hard sign 的判定，因此不应声称它改善翻转选择；保留它用于定量比较 reference 与真实梯度。缓存旧参数处梯度与保存样本后在当前参数重算梯度不同，前者不能直接充当历史目标在当前点的精确梯度。
- **降噪与滞后：** 固定参数/独立 batch 模型下，EMA 稳态噪声二阶矩因子为 (1−β)/(1+β)；β=.9 相当于约 19 个等权观测的噪声因子，平均梯度年龄约 10 步。实际轨迹引入参数漂移。笔记给出有限历史权重平方和、平均滞后以及在 L-smooth、鞅差噪声和有界位移条件下的 L² 误差上界；没有把这些假设当成实际网络已验证事实。
- **新分析的核心条件：** C=S−G，Δ=rᵀC≥0，真正目标的一阶增益 Λ=∇FᵀC≥Δ−||r−∇F||·||C||。故 reference 误差小于 Δ/||C||（C≠0）是充分条件。笔记补充条件期望版本、严格的有限步长差值界和在已知确定误差界时可用的负投影裕量；EMA 本身没有提供可直接部署的误差证书。
- **反例：** 构造 Hessian 为单位矩阵的二维强凸目标，在 h=(1,.01)、r=(1,0)、g₁=(−.001,1)、g₂=(2.001,−.98) 时，整向量翻转产生 C=(.001,−1)，Δ=.001>0，但 Λ=−.009<0，且每个正步长的目标都劣于同状态 SGD。reference cosine 约 .99995 仍不足以保证好转。另给出独立且无偏的单批 reference 也可能使编辑方向反向的一维例子。
- **候选 B（单独的编辑规则备选，未决定采用）：** 对负投影只反射沿 r 的分量：g̃=g−2min(rᵀg,0)r/||r||²，r=0 时不改。聚合修正为 Δr/||r||²，保留相同 reference 投影增益及样本范数；只要 hᵀr>0 且 Δ>0，就有严格一阶增益。它不再属于原定理 1 的标量权重可行集，不能当成只换 reference。此为标准几何构造分析，不宣称新颖性，不宣称实验证实有效，也不等于 PCGrad 的冲突分量去除。
- **与 Adam 的区别：** 候选 A 借鉴一阶 EMA；在欧氏 SGD 更新下不直接选 m/√v 作为目标梯度估计。若实际更新使用共同固定的对角预条件器 P，oracle 评分才相应变为 Ph；完整 Adam 动量状态仍需另行分析。普通 EMA 不继承专用方差缩减方法的收敛速率。
- **后续建议（未执行）：** 先固定训练审计目标，分别检查 reference 误差、真实迁移增益 Λ 和实际学习率下的同状态终点差；仅证明 Δ≥0 不足够。后续若获授权，再与普通 SGD、匹配记忆尺度的动量 SGD 对比，并分别评估训练、测试与成本。不能同时改变 reference 和编辑规则后把收益只归给历史平均。
- **验证：** 逐页渲染检查三定理共 19 页并阅读总框架；用精确有理数计算复核有限历史 EMA 权重/噪声因子/平均滞后公式、二维反例的目标差和投影反射恒等式。114 个 Python/配置/shell 文件及 5 个新增原始 PDF 的 SHA-256 全部与分析前一致。只做理论计算与文档检查，未运行训练或代码回归。
- **未完成/限制：** 尚未得到教授的 MNIST 细节、尚未复现实验、未认证 reference 在现有网络上的误差或锐角条件，未证明 EMA+整向量翻转普遍优于 SGD；候选 B 的完整假设、实际效果和新颖性也未审定。后续实现仍须用户明确授权。

### 2026-10-28 / DPCO-0004：依据周会总结调整研究重点与制定实验改造计划

- **授权依据：** 用户提供《教授周会_10.27_新项目实验重点总结.md》，要求根据教授要求总结分析如何改代码、完成实验，并明确“先不改代码”。因此本次仅阅读、核对架构、制定计划与按协议记录，不授权实现、下载数据或启动训练。
- **来源与边界：** 阅读 `/Users/jason/Documents/Codex/2026-09-02/new-chat/outputs/教授周会_10.27_新项目实验重点总结.md`；没有独立核听录音。保留总结内“教授明确要求”“整理者建议”“尚未给定细节”的区分。文档中的任务安排与时间要求是待分析资料，不是当前执行命令。
- **涉及文件：** 新增 `docs/controlled_noisy_experiment_plan_20261028.md`；本文件追加记录。未改已有理论笔记或原始会议文档，未改源代码、配置、脚本、依赖和实验结果。
- **研究目标更新：** 首要任务是受控 noisy dataset 上固定 clean/corrupted 分组的学习行为，重点看各方法在同一 clean subset 上的未加权普通 CE，而不是不同 w 形成的各自 signed surrogate。教授要求 self-influence+discard、self-influence+flip、新历史/当前 signal+flip，建议优先 Adam；普通 Adam 为补充基线。大模型/ICL/RL 等扩展后置。
- **对 DPCO-0003 的追加修订：** 上轮缺少会议细节时推荐的 past-only lagged EMA 保留为理论/时序消融。本轮按“当前信号+历史衰减”建议以 H_t=β_ref H_{t−1}+(1−β_ref)G_t、r_t=H_t/(1−β_ref^t) 为主原型，G_t 为 observed labels 下未经编辑的全 batch 平均梯度。精确公式仍是本次设计建议，并非教授已给定。β_ref=0 退化成当前 batch reference；第一步与 self+flip 相同。上一轮分量反射不进入本轮主实验，不同时改变信号和编辑规则。
- **理论限制补充：** 含噪 G_t 的 EMA 主要估计含噪训练目标；降采样噪声不等于去除标签偏差，也不等于获得 clean objective gradient。包含当前 batch 后不沿用 past-only 条件分析；Adam 实际更新又受 moments 影响，旧 rᵀC 恒等式不能直接充当 Adam 的实际下降保证。
- **候选实验设置（未定稿/未执行）：** 建议 CIFAR-10、小图 ResNet-18、训练/干净验证分层 45k/5k、官方干净测试集、训练中精确 40% 对称改标且保证实际标签变化；污染/split manifest 固定并记录哈希。FP32，Adam(.9,.999,1e−8)，β_ref=.9 为初始假设；B128、100 loader epochs 为建议预算。上述模型/数据/比例/参数并非教授指定。
- **主对照：** Plain Adam、Self+Discard+Adam、Self+Flip+Adam、EMA-current+Flip+Adam。共享原 B 归一化、数据顺序、初始化/增强、训练预算及同一 Adam；用 C vs D 隔离历史收益，用 B vs C 比较 discard/flip。现有 DTV 已实现 gradient discard；有 BN 时并不等于 forward 前删除数据。第一版不混入 LO 或局部层评分；总结中 LO 的精确定义仍未确认。
- **评价口径：** 固定 clean train 的 true-label CE；固定 corrupted train 的 observed-label CE 与 true-label CE；干净验证/最终测试、按分组的干预率、普通 CE/accuracy、数值有限性与时间/内存。污染标记与原始标签仅用于评价，不进入评分/权重。高 corrupted observed CE 必须联合 clean/true-label 表现解释，不以提高污染损失为训练目标，不按结果动态挑选子集。
- **现有代码核对：** `updates.py` 只有 SGD updater；`method_state` 和 `optimizer_state` 已分离；DTV/DPCO 及 VJP/JVP 可复用；runner 的状态初始化/验证当前绑定 policy/warmup；正式入口 TPU-only 且拒绝非 clean mode；数据为 ImageNet/iNaturalist，模型为 ImageNet ResNet50/101；旧绘图将 method=sgd 直接标成 SGD，新增 Adam 后必须按 optimizer+method 区分。
- **待授权改造范围：** 配置/父子 CLI/元信息、标准 Adam adapter、CIFAR 小图数据/模型与固定污染、历史 reference 的纯函数状态、checkpoint 恢复校验、分组评价/独立 CSV 与报告。reference 的 H/count 独立于 Adam moments；两者只在真实训练更新时提交。全局 batch 统一历史；编译 warmup、评价、反事实不推进状态。旧不兼容 theorem/diagnostic/mechanism 不简单解禁复用到 Adam。
- **实施与实验顺序建议：** 先普通 Adam+污染数据+分组评价闭环，再接现有 discard/flip 与 EMA；通过微型解析/VJP-JVP一致性、β=0退化、原B归一化、BN单次提交、Adam语义及精确断点恢复后进行短试跑；主实验 4组×3配对seed=12正式run。LR先做小范围普通Adam合理校准后四组冻结比较，广泛优越性另需等预算调参。少量β/时序/零噪声消融后再补SGD交替。
- **调度与未决项：** 提议把“轮”定义为 optimizer batch step，Adam 每步编辑，SGD 后续每2步编辑且三种干预统一相位，历史每步按 raw G 更新。会议未确认单位/相位，不可宣称教授指定。具体数据/模型/标签口径/β/LO/预算/硬件仍需落实，但不因此阻碍本次给出明确候选计划。
- **验证：** 只读检查上述代码接口与限制，查阅 CIFAR-10 官方说明和 Optax Adam 官方文档；未执行训练或代码测试。通过 SHA-256 保护性检查，现有 114 个 Python/TOML/shell 文件及会议原文保持不变；本次只新增方案并追加本记录。
- **未完成：** 所有实现、数据准备、烟测、正式运行、作图与实证结论均未开始；未把计划当成完成结果。原会议“下个礼拜”未转换为自动调度或任何外部操作。下一步代码实现仍须用户明确授权。

### 2026-10-28 / DPCO-0005：澄清历史 reference、梯度编辑与基础优化器的组合关系

- **授权依据：** 用户询问借鉴 Adam 构造 reference 后，对负向 gradient samples 做 flip/filter 是否已不属于原始 SGD/Adam、而是一种结合。本次仅解释算法结构与理论边界，未授权代码修改。
- **涉及文件：** 仅本文件追加理论澄清；现有代码、配置、实验计划和历史原文均不修改。
- **核心区分：** 将训练过程分为 reference 构造、样本贡献编辑、参数更新器三层。若 reference 仅被计算而不影响更新，训练算法未因此改变；当它通过权重 w 改变聚合方向 d=B⁻¹Σw_i g_i，整体训练方法一般已不同于原始损失上使用原始 batch 平均 G 的普通 SGD/Adam，但底层 updater 的公式可以保持标准形式。
- **组合定义：** 保留普通 SGD/Adam 的更新器，分别得到“历史 reference + flip/filter + SGD”和“历史 reference + flip/filter + Adam”。可作 DPCO-SGD / DPCO-Adam 的描述性候选命名，但未据此确定正式算法名或宣称新颖性。这不是必须将 SGD、Adam 两个更新器混合；借鉴 Adam 的 EMA 来评分，也不等于已经使用 Adam 做参数更新。
- **状态与方向：** reference 的历史 H 由原始 G 更新；样本权重对 filter 为0/1、对 flip 为−1/+1，零内积保持+1，负向仅表示与给定 r 冲突。送入 SGD/Adam 的是编辑后 d；Adam moments 应由 d 和逐坐标 d²更新，独立于 reference 的 H。单步没有编辑也不能自动推出与普通 Adam 的完整轨迹相同，因为已有 moments/参数可能不同。
- **目标边界：** 编辑后的 d 一般不再是原始目标梯度的无偏估计。冻结当步权重时可视为当步加权 surrogate 的梯度；由于权重依赖当前梯度和历史，不自动存在一个全程固定的加权目标。不能直接照搬原始目标上的性质，也不能将负内积样本自动称为污染或有害样本。
- **实验含义：** 同一 updater 内比较普通训练、self+discard、self+flip、历史+flip，隔离贡献处理和信号的作用；SGD 与 Adam 的组合需分别标注。reference 与 updater 的两层历史作用和额外状态应在方法说明中写清。
- **验证与未完成：** 本次是对已讨论公式和模块分工的数学澄清，无需运行训练或代码回归；只追加文档。尚无实现或新增实证结果，仍未得到普遍优越性或收敛保证。

### 2026-10-28 / DPCO-0006：展开解释当前+历史 EMA reference 的工作机制

- **授权依据：** 用户选中 DPCO-0004 对应的 G_t、H_t、r_t 原型，要求进一步阐述思路。仅授权概念解释和按协议记录，不修改实现。
- **涉及文件：** 仅本文件追加记录；既有代码、配置、其他笔记和实验结果不变。
- **定义与直觉：** G_t 为当前观察标签下的原始 batch 平均梯度，H_t=β_ref H_{t−1}+(1−β_ref)G_t，r_t=H_t/(1−β_ref^t)。展开后 r_t 是近期权重大、远期权重小的全部已观察 batch 梯度的归一化加权平均，不是只保留上一批，也不是存储历史样本重算。
- **初始化与衰减：** H_0=0；第一步 r_1=G_1。第二步 r_2=(β_ref G_1+G_2)/(1+β_ref)，体现早期权重归一化。β_ref=0 退化当前 batch 信号，β_ref 越大通常记忆越长也更滞后；正的 bias correction 不改变评分符号。旧梯度来自旧参数，不能将历史平均直接称为当前 clean gradient。
- **评分解释：** q_{t,i}=r_tᵀg_{t,i}，其符号由 β_ref H_{t−1}ᵀg_{t,i}+(1−β_ref)G_tᵀg_{t,i} 决定，综合历史趋势和当前 batch 方向；判断的是相对 reference 的冲突，不是真实污染标签。评分与权重固定后形成 d=B⁻¹Σw_i g_i，再交基础优化器。原始 G_t 写入历史，不写入编辑后的 d。
- **教学例子：** 取 β_ref=.9、H_{t−1}=(1,0)、当前两样本梯度 g₁=(−1,1)、g₂=(1.2,1)，则 G=(.1,1)、H_t=(.91,.1)。由于 bias correction 为正，可用 H_t 判断符号：当前均值给 g₁ 的分数为 .9，历史融合给 −.81。原规则保留 g₁，融合规则将其识别为冲突；flip 后 d=(1.1,0)，filter 后 d=(.6,.5)。该例仅说明历史如何改变选择，不证明该选择对真实目标更好；若正确目标主要沿第二坐标下降，flip 反而可能有害。
- **数学与经验边界：** flip 保证 rᵀd=B⁻¹Σ|rᵀg_i|，filter 保证 rᵀd=B⁻¹Σmax(rᵀg_i,0)，因此相对原 batch 均值不降低 reference 投影。但这不等于改善 clean 目标或实际 Adam 更新，尤其不能把稳定的历史含噪梯度当作无偏 clean 信号。降噪/有用信号时间一致性的解释是待实验检验的机制假设。
- **状态区分与验证：** reference 的 H 来源于 raw G；Adam moments 来源于编辑后 d，二者独立。通过精确有理数计算复核示例的均值、EMA、评分、flip/filter 聚合和 reference 投影恒等式；未运行训练、修改代码或声称实验有效。

### 2026-10-28 / DPCO-0007：reference 与 Adam 衰减超参数的取值说明

- **授权依据：** 用户询问“两个 beta”是否为超参数、应如何取值以及 Adam 是否需要指定 beta。本次仅解释与提出候选设置，不授权修改代码或实验配置。
- **涉及文件：** 仅本文件追加研究记录；实现、配置和既有实验计划不变。
- **参数区分：** reference 公式中的 β_ref 和 (1−β_ref) 互为补数，只有一个独立参数；若底层采用 Adam，另有一阶矩 β₁ 和二阶矩 β₂，合计三个独立衰减超参数。reference 由 raw G 更新，Adam moments 由 edited d 更新；同取 .9 不表示同一状态或必须绑定。
- **默认值与来源：** 核对 Optax Adam 官方文档 `https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html`，默认 b1=.9、b2=.999。构建 Adam 时这些值必然确定，但可由库默认提供，不要求每次手工传入；可复现实验应显式记录实际值。
- **候选设置建议：** 首轮所有 Adam 组统一固定 β₁=.9、β₂=.999，仅考察 β_ref∈{0,.5,.9}；.9 为主候选，0 复用 self+flip 对照，.5 为较短记忆消融。取值建议而非教授指定或已证明最优，不同时搜索三类 beta。后续若有依据再研究更长记忆或优化器 beta。
- **直觉与限制：** EMA β∈[0,1)，记忆时间尺度粗略为1/(1−β)，.5/.9/.99约对应2/10/100步量级，不是严格有限窗口或平均年龄的等式。较大β更平滑也更滞后，较小β更敏捷也更依赖当前噪声；本原型将其作为固定配置，不通过普通参数反向传播自动学得。
- **选择与验证边界：** 在固定污染manifest、相同optimizer/学习率预算下，用干净验证表现及固定分组曲线比较候选，测试集不参与选beta；报告相邻beta与配对seed的敏感性，不把默认值视为普遍最优。本次仅核对文档与数学含义，未运行训练、代码测试或改变实际参数。

### 2026-10-28 / DPCO-0008：结合现有 ImageNet/iNaturalist 框架调整含噪数据集建议

- **授权依据：** 用户认为 CIFAR-100 相对简单，询问近期常用、难度较高的含噪数据集，并说明近期已使用 ImageNet 和 iNaturalist。本次仅进行文献核查、数据选择建议及记录，未授权数据下载、实现或训练。
- **涉及文件：** 仅本文件追加研究建议；既有 CIFAR 计划作为先前候选保留，不改代码、配置、原数据和旧结果。
- **近期使用证据：** 核查 ICCV 2025《Joint Asymmetric Loss for Learning with Noisy Labels》原论文/作者官方代码，仍采用 CIFAR/CIFAR-N、WebVision、Clothing1M。AAAI 2025《Revisiting Interpolation for Noisy Label Correction》也列 CIFAR、TinyImageNet、WebVision、Clothing1M。可说明这些基准仍在近期相关研究使用，不据有限论文给出全领域流行度排名，也不把数据集创建年份等同于是否仍有研究价值。
- **数据类型区分：** 大规模/细粒度分类难度、真实标签噪声、人工可控标签污染是不同轴。ImageNet/iNaturalist 不能仅因分类困难就当作已知 clean/corrupted 划分的真实噪声基准；合成污染可明确“是否被本实验改标”，不保证未改标样本原注释全部正确。图像损坏/分布外测试也不同于本轮训练标签噪声。
- **推荐优先级修订：** 鉴于用户已有大图训练和数据框架，首选复用固定 ImageNet-100 类别 manifest、ResNet-50/既有输入设置，增加可复现训练标签污染；建议0%控制及20%/40%候选，先完成一个预定污染率下四组Adam闭环，再扩展。此为实验设置建议而非已经选定/运行。自定义 Noisy ImageNet-100 应报告100类清单、split、污染机制/比例/seed，不宣称它是唯一统一的公开标准切分。数据/模型阶段只调整计划，不同时添加其他机制。
- **后续困难任务：** 保留当前 iNaturalist 2021 子集，作为细粒度/多类别扩展，可另加固定合成标签污染。当前项目文档/配置为 full-train 比例抽取140k训练、20k验证、10k类（训练平均14张/类，验证2张/类），并非官方500k train_mini；本次只核对配置与协议，未重新核验实际数据或训练结果。高污染下少样本类别支持更少，建议分组报告污染前后类频率及按训练频率分层的效果，避免把少数/困难类梯度与噪声混淆。历史参考偏向频繁类是待验证风险，不是本次证明或实测结果。
- **iNat版本边界澄清：** iNat2021官方说明10k物种、约2.7M全训练、500k mini，明确其不一定聚焦long-tail；不能笼统把所有年份/子集称为同一种极端长尾基准。当前子集保留原类比例，实际不平衡程度仍应以manifest统计，不作未经核查的断言。
- **真实噪声候选：** WebVision1.0（原论文约2.4M网页图像、1k ImageNet语义类）或常用前50类Google Mini-WebVision，适合后续与ImageNet类别对接的真实网页噪声验证；Clothing1M为大规模服饰网页标签噪声基准，类别数不能当作整体难度排名；Food-101N约310k图像/101类，提供部分人工正确性核验标签，适合固定核验子集上的clean/noisy行为分析。Food-101N正确性标记并不自动提供所有错误样本的正确类别，也不覆盖全部样本。
- **对教授目标的适配：** 第一阶段需要精确控制分组和原标签视图，因此更适合现有ImageNet-100加固定合成污染；真实噪声数据一般不能直接得到全体训练样本的可靠clean/corrupt真值。第二阶段按有核验标签的范围作分组分析，并用独立人工标注验证/测试集评价。若选Food-101N，仅能在其核验范围内声称识别真实噪声，不能把二值核验当成纠正后的类别标签。
- **来源：** ICCV2025作者代码 `https://github.com/cswjl/joint-asymmetric-loss` 与论文 `https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Joint_Asymmetric_Loss_for_Learning_with_Noisy_Labels_ICCV_2025_paper.html`；AAAI2025 `https://ojs.aaai.org/index.php/AAAI/article/view/35489`；WebVision原论文 `https://arxiv.org/abs/1708.02862`；Food-101N作者项目 `https://kuanghuei.github.io/Food-101N/`；iNat2021官方协议 `https://github.com/visipedia/inat_comp/blob/master/2021/README.md`；本地 `docs/inaturalist_subset.md` 与现有数据配置。核查日期2026-10-28。
- **验证与限制：** 仅查阅原始论文、作者/数据集官方材料和现有配置；未测量硬件成本、未进行下载可用性全流程验证、未复现实验、未统计数据集使用频次。没有修改代码、实际污染标签、种子或实验预算；下一步实现仍需用户明确授权。

### 2026-10-28 / DPCO-0009：澄清 batch-self 对照与历史信号主方法的定位

- **授权依据：** 用户标注原四组表，确认 A 是否普通训练，询问 B/C 的 self 是否指 batch 内 self，并强调已有实验中 batch 内 self 效果差，应考虑借鉴 Adam 的历史+当前信号。本次只澄清定义和研究定位，未授权代码或配置变更。
- **涉及文件：** 仅本文件追加记录；原四组计划与已有结果保留，不擅自重新命名、改动实现或启动新实验。
- **定义确认：** A 是未经样本贡献编辑的标准 Adam 训练；B/C 是用当前 batch 共同平均 G_t 评分 q_i=G_tᵀg_i，再分别 discard/flip。这里 self 不是 g_iᵀg_i，也不是已经包含历史。原 D 才是当前+历史 EMA reference 下的 flip。
- **实验定位澄清：** 原表是对照设计，不代表四个同等优先的新算法。B/C 用作原有 batch-self 方法和“去掉历史”的对照；本轮待开发和验证的主方向是历史+当前信号。对当前 EMA 原型，β_ref=0 恰好退化为当前 batch reference，故这些对照有助于隔离历史的额外作用，不意味着继续以优化 batch-self 为研究重点。
- **可选扩展及数量：** 若同时评价历史 reference 的 filter 与 flip，两种新方法应分别定义为 EMA+filter、EMA+flip。它们与 plain Adam 构成三组核心方法；若同时保留原 batch-self 的 filter/flip，则完整设计是五组，不能把 B/C 改成历史版后仍声称旧 D 是不同的第四组。该五组/分阶段安排仅为澄清后的候选，不代表用户已确定新增运行预算。
- **已有负结果边界：** 接受用户报告的既有 batch-self 负结果；本次未复核实验数据。若已有数据/污染、优化器、模型、预算及评价口径与新实验一致，可复用作为对照；若从旧 SGD/无污染设置切换到新 Adam/含噪任务，不能直接用旧结果代替匹配对照，也不能因此断言在所有设定下一定失败。
- **验证与限制：** 核对本轮表格及 EMA 的 β_ref=0 代数退化关系；无新经验结论，无代码测试、实现修改或正式训练。

### 2026-10-28 / DPCO-0010：用户确认先聚焦三组 Adam，SGD 后续考虑

- **授权依据：** 用户标注普通 Adam、EMA+Filter+Adam、EMA+Flip+Adam 三组并表示“可以先关注这三个，再额外考虑sgd”。这是实验范围和优先级的确认，不是修改代码或启动训练的授权。
- **当前主线：** 首轮聚焦普通 Adam、历史+当前 EMA reference 下的 Filter+Adam、相同 EMA reference 下的 Flip+Adam。后两组仅在负分数样本的处理方式上不同，reference 构造/衰减、原B归一化和其他设置保持一致，以便解释 filter/flip 的差异。
- **阶段安排更新：** 原四组/五组表保留为历史讨论；当前三组主实验优先，batch-self(β_ref=0)作为后续消融，不自动列入首轮。SGD 后续额外考虑，届时再确定仅普通SGD对照还是对应历史编辑组合，不据此自动扩展为六组或增加正式运行预算。
- **结论范围：** 三组主实验回答历史reference与编辑的整体组合是否改善普通Adam，以及filter/flip哪种更合适；若要单独归因于历史而非样本编辑本身，仍需后续β_ref=0等消融。此前数据集、β、学习率和seed数量建议未因本次范围确认变成已批准执行参数。
- **涉及文件与验证：** 仅本文件追加范围决策；未修改代码、配置、其他方案文档或实验结果，未下载数据、运行测试或启动训练。核对三组定义和实验归因关系，待用户后续明确实现授权。


### 2026-10-28 / DPCO-0011：实现三组 Adam 对照及可配置负 alignment 权重

- **授权依据：** 用户明确要求“按照这仨组先完成代码修改，方法名叫做dpco，然后对dpco找出来的负align再有参数控制filter和flip……参数可以用-1或0这种，以后还可以直接类似权重的形式修改”。本次授权覆盖 DPCO-0010 确认的普通 Adam、EMA+Filter+Adam、EMA+Flip+Adam 的方法/更新器、必要配置与启动/恢复/记录链路及本地验证。它是后续新增的明确实现授权，取代此前这些功能处于“待授权”的状态；不扩展为正式训练、数据下载或所有旧实验的重构授权。
- **目的与命名：** 主方法保存为 `method.name="dpco"`、`reference.name="ema"`；filter/flip 由同一个 `method.negative_weight` 控制。普通基线保存为 `method.name="adam"`、`training.optimizer="adam"`。启动别名 `dpco_filter` / `dpco_flip` 只方便排列三组，并不生成两个独立实现。旧 `dpco_e` / `dpco_l` 等入口仍表达历史算法，不能把旧结果重新解释为当前历史 reference 算法。
- **核心实现：** 新增 `src/methods/dpco_history.py`，使用 observed-label 普通损失的原始有效全 batch 平均 G_t，维护 H_t=β_ref H_(t−1)+(1−β_ref)G_t、H_0=0，r_t=H_t/(1−β_ref^t)，包含当前 batch。仅当 q_i=r_tᵀg_i<0 时赋予 `negative_weight`，否则权重为1；有效样本数 B 不随 filter 改变。无效 padding 权重0，零 alignment 保留。评分/权重按 stop-gradient 处理。JVP 后端用 VJP/JVP 避免训练时构造 B×P Jacobian；audit 后端保留用于独立校验。模型状态只提交一次正常 forward 的结果。
- **权重语义：** 0 为 filter/discard；−1 为整向量 flip；−.5 为半幅反向；.5 为半幅保留；1 为普通贡献对照。接受任意有限实数，不自动裁剪、不按剩余样本重新平均。非有限数和布尔值拒绝；固定 filter/flip 别名与显式冲突权重同时给出时拒绝。权重1的多步轨迹已验证与普通 Adam 一致。filter 指 forward 后的贡献置零，仍参与该 batch 的 BN 统计。
- **Adam 与状态：** `src/training/updates.py` 增加基于 Optax `scale_by_adam(eps_root=0)` 的标准 Adam adapter，一/二阶矩分别接收编辑后 d_t 和逐坐标 d_t²，偏差修正、epsilon 在平方根外；不是每样本梯度平方的均值，也不是 AdamW。reference 的 H 独立保存且只接收 raw G。保留已有 LR schedule 与默认关闭的 kernel-only coupled L2；L2 在编辑后、Adam moments 前加入，不写入 H。新增 Optax train/test 依赖，使用本机已安装 0.2.8 验证，未安装或升级本机环境。
- **启动与配置：** `src/config.py`、`src/cli.py`、`src/training/launch.py`、`scripts/launch_tpu.py`、`scripts/train_classification.py` 接通 `--method adam/dpco`、`--negative-weight`、`--reference-beta`、Adam 两个 beta/epsilon 和 optimizer。主三组共用 Adam 配置；父 launcher 和子 worker 都接收全部参数。新增独立 `configs/tpu/dpco/imagenet100_resnet50.toml`，不改旧 preset 的学习率/算法；β_ref=.9、Adam(.9,.999,1e−8)、LR=.001、100步仅为未调优 pilot，可被覆盖。现有 ImageNet/iNaturalist 模型与数据入口复用；本次不增加 SGD 正式实验网格。
- **恢复与记录：** `src/training/experiment.py` 在编译前初始化 H/count 和 Adam 状态，恢复时先检查类型、累计步数、树结构、形状、有限性与二阶矩非负，再 rewind CSV。参数权重/beta 改变不能覆盖恢复同一实验。旧元信息缺少新增字段时按旧默认值比较，保留原 metadata 字节；缺失新算法必要状态时明确失败，不静默从零继续。warmup candidate 丢弃，评价不推进历史。`src/training/trainer.py` 为新状态增加有限性检查；仅 EMA run 添加 `dpco_steps.csv`，保留旧 stream/cursor 语义；记录 negative/edited count、权重、历史步数、norm、cosine 和 reference gain。目录/status 标签区分 beta、权重和 Adam，metadata 保存全部参数。
- **理论边界：** 延续 DPCO-0004/0006 的当前+历史原型，而非复现教授尚未提供细节的 MNIST 算法。β_ref=0 退化当前 batch reference；第一步 r_1=G_1。正偏差修正不改变 sign。对于负样本权重≤1，rᵀ(d−G)≥0 是代数投影性质；权重>1甚至不必满足它。该量不代表 clean-gradient 增益或 Adam 实际下降保证，也不把负 alignment 认定为错误标签。EMA 不能自动消除标签偏差。靠近1的 beta 使用 Python 标量 log 与 expm1 避免 FP32 的 1−β^t 消减问题；实验仍需检查数值有限性。
- **涉及文件：** 新增 `src/methods/dpco_history.py`、`configs/tpu/dpco/imagenet100_resnet50.toml`、`tests/test_dpco_history.py`、`docs/dpco_adam.md`。修改 `src/methods/__init__.py`、`src/config.py`、`src/cli.py`、`src/training/{updates,trainer,experiment,launch,recording,results,status}.py`、`scripts/{launch_tpu,train_classification,plot_training_results}.py`、`pyproject.toml`、`README.md`、`tests/{local_mesh_check,test_config,test_updates,test_plot_training_results}.py`，并追加本文件。旧绘图入口增加 Adam 拒绝检查，避免将 uniform+Adam 标成 SGD；未重写其历史图表与结果。
- **验证与修正过程：** 首轮旧209项回归仅有 method 枚举测试因新增 Adam 必须搭配 Adam updater 而失败；补齐测试的正确配置。新增独立零 reference 检查曾使用1/6相消，被 FP32 残差影响；改用可精确表示的整数梯度测试严格零，不引入 alignment 容差或改变算法。新增19项测试全部通过；之后全量 **228项 CPU 回归通过**（86.931秒），包含多步 NumPy EMA/Adam 独立递推、0/−1/任意权重、β=0退化、零/正分数、padding、audit/JVP、真实微型 ResNet BN 单次提交、三组真实入口 CPU fixture、编译不消耗状态、完整恢复/坏状态拒绝、父子参数一致、dry-run 不导入 JAX/Optax/Flax、双 CPU 设备全局历史与 Adam moments。全量后扩充已有两项检查，再次通过旧 metadata 缺失所有新字段的精确恢复与旧绘图对 Adam 的拒绝检查。98个 Python 文件 AST、所有配置 TOML 解析、`git diff --check` 通过；三组 dry-run 和自定义−.5配置检查通过。测试仅使用临时目录和合成 fixture，未把测试 loss 视为实证改进。
- **未完成项与证据限制：** 当前完成的是三组方法/优化器代码与现有真实数据入口的接通。尚未实现讨论方案中的固定噪声标签注入、污染 manifest、clean/corrupted 分组 CE 或三组新报告绘图；现有入口仍只接受 `data.mode=clean`（表示没有额外注入，不认证原始标签绝对无误）。没有下载数据、启动正式训练、进行 TPU 硬件/吞吐/内存实测、调参、得到准确率优势、提交或推送。所有既有未提交研究工作、原始实验结果与历史记录保持；后续实现按用户指定范围继续授权。


### 2026-10-29 / DPCO-0012：面向 noisy dataset 的受控验证选型建议

- **授权依据：** 用户报告已在服务器 `/home/jason_chia925_gmail_com/Project_DPCO/DPCO` 整理代码，当前要挑选 noisy dataset 并按教授要求做 control verification。本轮是选型与研究设计讨论，不构成新增代码、下载数据或启动实验的明确授权；仅按协议追加本记录。
- **来源核对：** 重读用户提供的《教授周会_10.27_新项目实验重点总结.md》及当前本地训练入口。会议明确关注固定 clean/corrupted 子集上的行为，未指定数据集、污染比例或精确 EMA 公式。当前三组仍按后续用户确认的普通 Adam、历史 DPCO filter、历史 DPCO flip；不把原会谈 batch-self 两组当作当前主线。未连接或核验服务器文件版本，终端截图只提供工作路径。
- **选型建议：** 第一阶段复用现有 ImageNet-100 及 ResNet-50，先把 noise 明确限定为训练标签噪声，固定样本 ID、原始标签、观察标签和注入标记；只改训练标签，不改图片与验证标签。对选定的约/精确40%样本从其他99类均匀选择新标签，保证实际发生改变并记录实际改标数量/比例、映射版本和哈希。40%与0%是本轮建议，非教授指定、非已执行参数。未改标子集/原标签仅作为相对原始标注的控制，不能认证 ImageNet 原标签绝对正确。
- **真实噪声备选：** 官方 Controlled Noisy Web Labels 提供 Blue Mini-ImageNet（合成）、Red Mini-ImageNet（真实web噪声）及 Stanford Cars 两种版本；官网描述每图3–5名标注员、可控0–80%噪声级别、共享clean验证集与 is_clean 字段。因此 Red Mini-ImageNet 可作为需要现成真实噪声及clean/corrupted标记的补充，不能默认每个错误样本都带有可计算原正确类别CE的标签。官网 https://google.github.io/controlled-noisy-web-labels/ 。本次仅查阅官方描述，未下载、检查数据文件或确认所有图像URL仍可取回。也核对 Clothing1M 作者仓库 https://github.com/Cysu/noisy_label ，不因其训练集有噪声就推断每个样本有经核实的真标签/污染标记。
- **规模与归因：** 第一阶段在可控 ImageNet-100 上建立机制闭环；更大 ImageNet-1K 和已有 iNaturalist 全类别比例子集可作为后续扩展。iNaturalist 的类别不均衡/稀有类与标签噪声是不同因素，类别多、样本难或梯度冲突不自动代表标注错误；先避免把它们同时混入首个归因实验。此选择依据现有代码适配与控制目标，并非“最流行”或“最难”的排行榜断言。
- **评价与公平性：** 三组共享污染manifest、初始化/训练seed、batch顺序、增强、模型和合理校准后冻结的Adam设置/预算；污染seed与训练seed分离。固定未改标子集看原标签普通CE；固定改标子集同时看观察错标签CE和原标签CE；另看未额外加噪验证集CE/accuracy。评价不使用各方法signed/weighted surrogate，不动态按模型评分重新定义clean组。注入标记与原标签不能用于训练评分、reference或权重，避免oracle泄漏。评价使用固定无随机增强的eval方式及相同BN口径。
- **成功判据与额外观察：** 改标子集的错标签CE升高仅说明更少拟合该标签，不能单独判定DPCO有效；需同时检查未改标子集及未加噪验证集是否改善、改标子集原标签预测是否改善。分组记录负alignment/干预比例，检查是否大量误伤难而未改标样本。负alignment不等同噪声检测器。只有三组主对照仍不能把收益单独归因于历史；β_ref=0为后续消融，不自动扩首轮网格。
- **候选运行顺序：** 3方法×2噪声级别(0%,40%)×1训练seed=6组作为初步闭环；稳定后3训练seed共18组，固定污染实例以隔离训练随机性。更换污染seed、20%中间强度、非对称/实例依赖噪声及真实噪声为后续扩展。先验证普通Adam训练预算足以表现学习/记忆行为；当前100步pilot不自动构成完整实验预算。所有数量均是建议，未创建运行任务。
- **实现边界与未完成项：** 本地 `scripts/train_classification.py` 仍拒绝非clean data.mode，尚无固定注入manifest、原标签保留/评价视图、分组普通CE或分组干预率。不能只把参数改成noisy就声称已开展该受控验证。后续实施需要用户明确代码授权。此次仅追加本文件，无源代码/配置/依赖变动，无下载、训练、提交、推送或新经验结论。


### 2026-10-29 / DPCO-0013：核对公开 ImageNet 衍生噪声基准，明确 CNWL 与现有 ImageNet-100 的区别

- **授权依据：** 用户追问“imagenet-100是不是已经有带噪声的标准数据集”。本轮仅查证公开基准、修订选型解释并追加研究记录；不授权代码改动、下载数据或训练。
- **查证结论：** 存在公开的100类 ImageNet 衍生可控标签噪声基准，最直接相关的是 Controlled Noisy Web Labels (CNWL) 的 Red/Blue Mini-ImageNet，出自 Jiang 等 ICML 2020《Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels》。不能断言用户具体记忆必然是它，也不能把它直接等同当前项目自定义类别清单的 ImageNet-100。检索还发现论文各自构造的 noisy ImageNet-100 设置，名称相似并不能保证数据/类别/噪声协议一致；不将未检出唯一命名版本当作证明不存在其他版本。
- **一手来源：** 数据集 https://google.github.io/controlled-noisy-web-labels/ ；论文 https://arxiv.org/abs/1911.09781 ；TFDS官方目录 https://www.tensorflow.org/datasets/catalog/controlled_noisy_web_labels ；作者介绍 https://research.google/blog/understanding-deep-learning-on-controlled-noisy-labels/ 。只读公开描述，未下载数据或执行TFDS。
- **可核对的协议：** TFDS明确提供 `mini_imagenet_red`（真实web噪声）、`mini_imagenet_blue`（合成标签噪声）；100类、84×84图像、十个0%–80%训练split及共享5000张clean validation。Red每级约5万训练图（train_40列为50010），Blue每级60000。输出含id/image/label/is_clean，可按is_clean固定clean/corrupted组。按官方说明需手工准备原始ImageNet档案与web图片；元数据压缩包大小不等于完整图像数据已经可用，未确认所有旧web URL当前可取回。
- **选型解释修订：** DPCO-0012已提及Red Mini-ImageNet备选，但上一轮把自行给现有ImageNet-100加噪作为首选，未充分展开现成公开基准的意义。如果优先条件是现成公开噪声级别与clean/corrupted标记，应优先核验CNWL Red Mini-ImageNet的获取和适配；自行加噪仍是方便保留完整原标签、精确追踪改标的控制方案，并非唯一方案。没有宣称CNWL是当前项目同一类别清单的即插即用换标签文件。
- **控制目标的重要区别：** 作者说明Red通过将部分原图替换为带错误web标签的图片形成污染，可能含目标100类之外的物体；它不是始终保持同一张图只改变标签。TFDS仅列label与is_clean，不能推断每个corrupted样本都有100类内的修正标签。因此原标签CE对所有Red噪声样本未必有定义/可计算；可先比较clean CE、corrupted observed-label CE与共享clean validation，另外单独审定是否存在修正标签。三方法必须共用同一个train_40的固定样本；不同噪声split中clean成员未必相同，不能直接当成同一固定子集排序。
- **验证与限制：** 核对上述官网、官方数据目录、原论文页面与作者说明；未独立复核原始注释、训练/验证成员交集、真实可下载图像数量或类别与当前manifest的对应关系。无源代码/配置/依赖变化、无测试训练、无下载/提交/推送；本轮仅本文件追加记录。


### 2026-10-29 / DPCO-0014：讲解 CNWL/MentorMix 论文并筛选其他含噪数据集

- **授权依据：** 用户标注 Red/Blue Mini-ImageNet 的介绍，表示可以使用，要求先讲论文在做什么、为何加噪，再调查其他数据集。该表态接受候选数据集，不等于授权下载、实现适配或启动训练。本轮仅文献分析、选型和本记录追加。
- **论文阅读：** 阅读 Jiang等 ICML2020《Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels》PMLR正文（12页，含参考文献）和补充材料（10页），并核对作者Google Research说明、数据集官网。来源 https://proceedings.mlr.press/v119/jiang20c/jiang20c.pdf 、https://proceedings.mlr.press/v119/jiang20c/jiang20c-supp.pdf 、https://research.google/blog/understanding-deep-learning-on-controlled-noisy-labels/ 、https://google.github.io/controlled-noisy-web-labels/ 。使用网页/PDF文本读取，未向项目下载论文或数据。
- **三项贡献及加噪目的：** 提供可控制强度的真实web标签噪声基准，提出MentorMix，并研究不同噪声分布/模型/初始化下的行为。数据构造的标签错误是实验中的受控因素，目的是检验鲁棒性及人工随机噪声结论的可迁移性；不应解释为“故意给数据错误监督以提升模型”。这与方法中的Mixup正则化是不同层次。
- **Blue与Red：** Blue保留图像并以受控比例将标签换成其他类；正文明确排除换回原正确标签。Red通过搜索收集web候选、去除验证近重复、3–5人核验标签正误、用经确认的错标web图替换部分原图而保留类别标签。标签正误人工信息用于构造/评价，不输入算法训练；Red可能包含目标类之外的内容，也引入图像分布变化，不是严格只对同一x改y的干预。补充材料对纯web图与去除image-to-image检索的构造变体另作检查。Blue/Red训练样本量也不同，不宜把两者任何数值差异完全归因于标签机制。
- **教学分解（本次说明，非论文新定理）：** 对固定clean集合C、corrupted集合N及rho=|N|/(|C|+|N|)，普通未加权训练目标精确分解为 L_obs=(1−rho)L_C+rho L_N，其中L_N使用观察到的错误标签；其梯度作同样分解。普通训练会试图同时降低两部分的观察标签CE，但错误标签CE下降不等于真实任务改善。用易懂的动物错标例子区分均匀随机错标与视觉相近/实例依赖错标。对于Red的open-set样本，不假设存在已提供的任务内正确类别。
- **MentorMix机制：** 正文Algorithm1先用普通样本loss和batch loss分位数的EMA阈值计算简化MentorNet二值可信权重，再以softmax权重偏向选择小loss伙伴、执行图像和one-hot标签的Mixup；lambda依据原样本可信度分配较大/较小混合比例，混合后可选择再次加权。不是零权重伙伴采样概率必然为零，也不是必须额外训练一个大型教师。Mixup公式为x_mix=lambda*x_i+(1−lambda)*x_j、y_mix=lambda*onehot(y_i)+(1−lambda)*onehot(y_j)。这里的EMA平滑标量损失阈值gamma；不等同DPCO原始梯度向量EMA，方法不使用负权重整梯度flip。
- **结论与证据边界：** 文中实验发现真实web噪声在其设置下通常比随机噪声造成更小退化，真实噪声上peak到final差值可能很小、早停收益不如随机噪声稳定，预训练架构的微调表现也有相关性。作者提出视觉/语义相近等解释，但不是普适定理，不能写成噪声有益、模型总先学clean后记noise、或一定支持DPCO。其主要评价包括clean validation的peak和final accuracy，并不证明clean CE必降/错误标签CE必升；教授要求的固定两组CE需单独实现。当前三组Adam试验也不能单独识别历史的净贡献，仍需后续beta_ref=0消融。
- **备选与控制信息核对：** CIFAR-100N（ICLR2022，100类、5万训练、32x32）提供原标签与真实人工噪声标签，能相对原标签明确划分一致/不一致，适合机制核验；需严格匹配PyTorch/TFDS图像顺序，原标签仍可能有歧义。Food-101N（CVPR2018，101类约310009图、估计20%噪声）有52868训练图和4741验证图的verification标签，仅核验部分可用于可信clean/corrupted观察，未知标签正误不能当clean。CNWL Red Stanford Cars（196类、8144训练）提供另一细粒度可控web噪声任务。Clothing1M（14类、约百万noisy图）有额外clean标注集合，不能把它等同全量noisy训练样本都有正误真值。WebVision1.0（1000类、超过240万图）适合扩大真实web噪声训练规模，官网的clean验证集不代表训练集有全量污染掩码。Animal-10N（10类、5万训练/5000测试、64x64）的约8%是估计，官网明确全量训练真值未知，不列为首轮分组机制验证优先项。
- **备选一手来源：** https://github.com/UCSC-REAL/cifar-10-100n 及 https://www.tensorflow.org/datasets/catalog/cifar100_n ；https://kuanghuei.github.io/Food-101N/ ；https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf ；https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html ；https://dm.kaist.ac.kr/datasets/animal-10n/ 。未把搜索到的非官方汇总页或过往SOTA名次当作最终推荐依据；未验证所有实际下载渠道。
- **本轮建议而非新执行决定：** 用户已接受的CNWL Red Mini-ImageNet继续作为主候选；若要补最直接的样本级机制核验，CIFAR-100N可作为低成本辅助，不强迫将其替换为主数据集；更大且有部分正误核验的Food-101N适合后续真实数据扩展。全量WebVision/Clothing1M更多回答规模下的泛化，不能替代完整clean/corrupted机制观察。不得因“noisy”名称就以模型loss/负alignment反过来定义评价真值；核验信息仅供评价，不进入reference/weight。
- **涉及文件、验证与未完成：** 仅本文件追加。本次是纸面公式与官方数据接口核对，不运行代码测试；未修改源代码/配置/依赖，未下载数据、连接服务器、训练、提交或推送。数据成员、图像URL实际可取回率、预训练与评价划分是否有重叠仍需落地前核验；未宣称DPCO已在这些基准有效。


### 2026-10-29 / DPCO-0015：建议先用 CIFAR-100N 建立固定样本的机制验证

- **授权依据：** 用户询问是否应先使用“cifar-100N这种官方的”数据。本轮仅研究选型澄清，不授权代码、下载或训练；仅本文件追加。
- **优先级解释：** 建议针对教授当前固定clean/corrupted机制观察先做CIFAR-100N，再用CNWL Red Mini-ImageNet验证真实web场景。理由是前者同一图片保留原标签和人工噪声标签，可定义完整、固定的对照并计算改标子集对原类别的CE，不是因为“官方”必然比其他公开基准更有效。CNWL同样是论文作者公开发布的正规基准；不要求所有研究必须从CIFAR开始。此为当前目标下的建议修订，非用户已明确选择/启动。
- **一手来源核对：** https://github.com/UCSC-REAL/cifar-10-100n 为ICLR2022作者正式发布；https://www.tensorflow.org/datasets/catalog/cifar100_n 明确同一原CIFAR-100训练图另有人工标注、字段label/noise_label、5万训练、32x32。作者同时提供PyTorch与TFDS顺序映射，不能混用数组顺序。所谓官方指噪声基准作者发布，不是原始CIFAR标注绝对正确的保证。
- **分组与训练：** 以原标注y和发布人工标签y_tilde定义C={i:y_tilde_i=y_i}、N={i:y_tilde_i!=y_i}，全程固定。三方法训练都使用同一人工噪声标签，C/N和原标签仅供评价，不能输入reference、过滤或翻转权重。分组相对官方原标注定义，不排除原标注歧义。记录C组原标签普通CE、N组人工标签CE、同一N组原标签CE，以及不额外改标测试/验证表现，联合解释而不是只奖励N组观察标签CE升高。
- **噪声口径更正：** 第一轮使用发布的人类噪声标签，噪声率由该文件与原标签逐样本比较得到；不重新随机制造40%或把固定发布版本误称可任意设置噪声强度。若补全原标签训练对照，两种训练使用同一图片和原CIFAR-N文件定义的固定C/N观察组，有助于控制图像因素；不得据此自动增加正式运行预算或用测试集选超参数。
- **实现边界与未完成：** 当前项目仍需CIFAR-100N标签/图像顺序校验、32x32数据/模型适配和分组评价，不能只把ImageNet配置dataset名称改成CIFAR就启动。未更改源代码/配置/依赖，未下载、运行训练、连接服务器、提交或推送。无新增实验结论，未承诺DPCO在CIFAR-100N有效。


### 2026-10-29 / DPCO-0016：实现官方 CIFAR-100N 下载、标签模式与 DPCO+Adam 完整接入

- **授权依据：** 用户明确要求“按照cifar100n的官方接口模式，参照imagenet的代码文件架构，完成cifar100n的代码，包括下载加载数据，按照官方接口的释义来定义noise 百分比参数等，然后接入现有dpco的adam训练流”。本次授权覆盖数据准备/下载/加载、必要的小图模型、配置/启动、固定分组评价与本地验证；不授权正式训练、修改既有DPCO/Adam数学规则、提交或推送。
- **起始状态与保护：** 工作开始时仓库HEAD为 `2d6ec52`，仅本文件存在用户研究讨论追加记录；保留这些记录和原有结果。所有实现位于DPCO目录；真实源数据与测试产物放在 `/private/tmp/`，未纳入Git、未覆盖服务器数据、未连接服务器或操作远端。
- **官方依据与版本：** 查阅并获取作者ICLR2022发布仓库README、main.py、data/cifar.py、data/utils.py、data/datasets.py，固定commit `49df7d8a69e355470c77c1c2f2424916325a394b`。官方来源 https://github.com/UCSC-REAL/cifar-10-100n 、https://www.tensorflow.org/datasets/catalog/cifar100_n 、https://www.cs.toronto.edu/~kriz/cifar.html 。Python原图包使用发布MD5 `eb9058c3a382ffc7106e4002c42a8d85`；官方ordered注释NPY的SHA256为 `603b16e5f88f77a0db27ffe41b62cf59ba438da2b8bd4a085d502ad72e2b3dbf`，image_order_c100.npy为 `423c3118865b4d912ff9082423b992e43fa94ef0f503c4180bd415f915eec018`。
- **下载与准备架构：** 新增 `src/data/cifar100n_prepare.py`、`scripts/prepare_cifar100n.py`。独立plan/download/verify入口不导入JAX/Torch/TensorFlow；默认目录 `../data/cifar100n` 在代码仓库外。下载完整校验后原子发布文件，坏的既有源/缓存明确失败，不静默替换；保留失败partial（重试目前重头下载）。目录文件锁、命名tar成员读取而非extractall、非pickle的prepared NPY/NPZ、manifest/hash/complete标记保证缓存与恢复可追踪。生产准备严格检查50000训练/10000测试、原始类别均衡、像素和标签形状/范围。
- **官方接口与顺序：** 复用作者官方TFDS ordered NPY，不要求安装PyTorch。NPY键为clean_label/noise_label，与PT的clean_label/noisy_label区别明确。作者mapping满足TFDS的i对应Python的order[i]，实现转换到原CIFAR Python顺序，并逐项断言所有原标签等于原图文件fine_labels；不使用易相消的标签差总和代替逐项检查。source哈希在读取官方object NPY之前验证；运行缓存不再允许pickle。`example()`暴露id/image/label/noise_label/observed_label供审计，训练批仍只有images/labels/valid_mask。
- **noise参数释义：** 新增DataConfig及父子CLI参数noise_type(clean100/noisy100)、is_human、noise_seed、noise_percent、download。人工默认使用作者发布版本：实测20100/50000不一致，actual_noise_rate=.402，actual_noise_percent=40.2。clean100训练使用原标签，实际比例0。`--noise-percent`（别名`--expected-noise-percent`）单位0–100，是加载后实际比例的校验断言，不生成指定强度的人类噪声；不一致即报错，不忽略参数。作者旧CIFAR100构造器中的noise_rate默认.2并未参与这两条加载/采样路径，因此不能把它误当人工噪声控制。切换label模式/合成设置时清除未显式给定的人工40.2断言；clean100默认校验0。完整标签先处理并计量，再限制smoke可见样本，避免把smoke比例错作官方完整比例。
- **合成对照兼容：** `--no-is-human` 以完整原标签→人工标签的经验类转移矩阵，按作者RandomState(seed).multinomial顺序采样；默认seed0，与训练seed分离。矩阵显式行归一化，在官方每类500张上等价于原仓库广播表达式；空类fixture行设identity，不影响实际抽样。原图、类条件概率与采样顺序保持，既不独立均匀重造40%也不强制最终比例等于40.2。官方完整数据seed0实测20108/50000=40.216%。human模式非零noise_seed拒绝。实际noise_or_not/actual_noise_rate反映训练标签，分组参考mask另行保存，不混为一谈。
- **数据加载与模型：** 新增 `src/data/cifar100n.py` 的只读mmap图像、标签视图、指纹/比例和源元信息；缓存逐文件SHA256验证，关闭后读取明确失败。共享ImageBatchSource/evaluation_batches通过dataset transform分派，ImageNet/iNaturalist原处理不变。CIFAR遵照作者data/datasets.py：pad4、randomcrop32、水平flip、均值(.5071,.4867,.4408)/标准差(.2675,.2565,.2761)，评价原生32无随机增强。保持现有seed/epoch/sample顺序和augmentation恢复协议，不宣称逐次随机增强与PyTorch完全相同。新增独立 `cifar_resnet18/34` basic-block模型，3×3 stride1 stem、无初始maxpool；旧ResNet50/101不改。接口/数据语义兼容不等于复现原文整套模型/优化器训练配方。
- **DPCO+Adam接入：** 统一分类入口按数据名选择模型和reader，沿用现有任务、global batch VJP/JVP、原始梯度EMA reference、负权重0/−1及任意有限权重、标准Adam与完整checkpoint；未修改方法/Adam更新源文件。新增 `configs/tpu/cifar100n/dpco_adam.toml` 与 `run_cifar100n_suite.sh`，默认三组adam/dpco_filter/dpco_flip顺序worker0运行。新preset为未调优100步pilot、B128、LR.001、FP32、beta_ref.9、Adam(.9,.999,1e−8)，不是正式基准预算。增加新数据参数的父子命令透传、dry-run元信息、配置检查、数据/模型切换校验和run-name中的clean100/noisy100与human/synthetic区分。data extras增加必要NumPy/Pillow/certifi，不升级本机训练环境。
- **固定分组评价：** 新增GroupedEvaluation及CIFAR构造器，在固定参考噪声标签相对原标签定义的C/N上，记录clean原标签CE、corrupted_noisy参考噪声标签CE、同一corrupted_original原标签CE（含top1/top5）。human参考为发布人类标签，synthetic参考为同一固定seed生成标签；clean100训练对照也保留同一参考分组和评价目标，便于比较相同图片。状态不提交、BN使用eval统计、padding按有效数量加权、空组计数0且指标留空。group_evaluation.csv为新私有stream，断点恢复同步rewind。评价时间包括分组开销。原标签与mask仅在审计/评价读取，不进入训练评分和reference。
- **官方test口径：** CIFAR原始接口只有train/test，没有另一个官方validation；metadata明确common evaluation/summary的val_*字段在此对应官方test监测，为兼容保留列名。CIFAR GroupedEvaluation关闭自动按该test挑best，故不生成best slot；fixed-budget/final保留。并未藉此声称用户看过曲线调参后仍是盲测，正式超参数选择需要另立验证协议。ImageNet原有best逻辑不变。
- **恢复兼容：** 指纹包含prepared manifest、训练/参考标签哈希、噪声模式/seed和smoke view；mode变化或数据不一致不能继续同一旧轨迹。旧ImageNet元信息缺少新增DataConfig字段时按原默认值比较，不修改旧metadata。原恢复测试扩展为移除全部新data字段后验证，确保新增接口不破坏历史off路径。
- **验证结果：** 首轮既有228项回归通过；新增18项CIFAR检查通过；全量 **246项CPU回归通过**（91.508秒）。随后针对actual_noise_rate/noise_or_not、关闭后读取保护、分组计时和原生CIFAR模型跨设备补查，18项新测试+双CPU全局BN/完整Jacobian oracle+旧元信息恢复共20项再次通过（11.605秒）。涵盖checksum发布/缓存复用/坏源拒绝、tar+ordered NPY完整准备路径fixture、官方顺序反转换与逐项原标签校验、实际百分比断言、human/clean/synthetic语义、官方多项式RNG一致、native transform、audit真值不进入训练batch、分组CE独立NumPy校验/空组、三组共享真实入口tiny CPU模型训练与p50精确恢复、group CSV无重复、test不选best、父子参数及无运行时dry-run、原模型兼容。103个Python AST、TOML、shell语法和git diff --check通过。
- **完整真实源验证：** 在 `/private/tmp/dpco-cifar-real` 实际取得169001437字节官方原图包并通过发布MD5，验证固定注释/映射SHA256，执行正式prepare生成50000训练/10000测试cache并完成全部50000项原标签匹配。网络慢的最初顺序下载由临时辅助Range读取补齐、拼接后整包MD5确认，停止重复慢下载后调用正式离线准备；该临时加速工具不是生产下载器支持并发/断点续传的声明。加载实测human 40.2%(C29900/N20100)、clean100实际0%但保留同组、synthetic seed0 40.216%(C29892/N20108)。真实标签与读入计量不等于方法性能实验；未用完整数据执行优化训练。
- **涉及文件：** 新增 `src/data/{cifar100n,cifar100n_prepare}.py`、`src/models/cifar_resnet.py`、`scripts/prepare_cifar100n.py`、`configs/tpu/cifar100n/dpco_adam.toml`、`run_cifar100n_suite.sh`、`tests/test_cifar100n.py`、`docs/cifar100n.md`；修改 `src/{config,cli}.py`、`src/data/{classification,images}.py`、`src/training/{launch,evaluation,experiment,results}.py`、`scripts/{launch_tpu,train_classification}.py`、`tests/{local_mesh_check,test_updates}.py`、`pyproject.toml`、`README.md`、`docs/dpco_adam.md`，追加本文件。旧“尚未实现CIFAR/分组”记录保留为历史状态，当前说明以本条和新文档为准。
- **理论与未完成边界：** C/N正确性相对原始benchmark标签定义，不认证语义绝对无歧义；合成噪声匹配类转移不能替代实例依赖的人类噪声。没有证明DPCO对clean目标或泛化必然改善，也不以错误标签CE升高单独判成功。尚未正式训练/调参、TPU硬件吞吐/内存测试、绘制新实验结果或统计每组实际训练flip率；没有增加通用ImageNet噪声注入、均匀可调人类噪声或修改历史数据。未提交/推送；服务器需同步代码并自行按文档准备数据后运行。
