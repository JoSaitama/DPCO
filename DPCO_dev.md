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


### 2026-10-29 / DPCO-0017：服务器长任务统一 nohup 与中断下载恢复说明

- **用户授权依据：** 用户明确要求“写入咱俩交流的协议，以后都是nohup启动”，并报告终端断开、询问下载一半能否续传。本次授权更新协议及记录；未将询问续传解释为修改下载器源代码的授权。
- **目的与具体变动：** 在根目录与DPCO目录的AGENTS.md中约定，服务器下载、数据准备、依赖安装、训练及批量实验等长任务统一nohup后台启动，显式设置标准输入、独立日志、PID及查看方法，使用明确的项目Python路径；短时检查可前台运行。此约定不自动授权训练或扩大其他开发范围。
- **下载行为核对：** 现有src/data/cifar100n_prepare.py的download_asset保留失败.partial，但再次请求使用partial.open("wb")，会重头写该文件，并非原生续传；完整且校验正确的源文件会复用。不能让用户直接重跑并称其为断点续传。终端断开后原任务是否退出、文件已下载多少，尚无服务器证据。
- **恢复建议与边界：** 提供服务器端nohup包装的curl -C -手动续传方案，读取已有原图包.partial，持有同一.prepare.lock防止与旧准备进程并发写入；完整包先核对项目固定MD5再发布到正式文件名，随后交还现有prepare/verify入口完成其他源与缓存。续传依赖远端Range支持；若不支持则保留partial并报错，不声称必能续传。不改变生产下载器，也不绕过原有完整性检查。nohup仅针对终端挂断，不是机器重启或训练checkpoint恢复机制。
- **涉及文件与验证：** 仅修改根AGENTS.md、DPCO/AGENTS.md并追加本文件；阅读现有下载器、准备入口与近期开发记录，核对curl官方-C选项说明 https://curl.se/docs/manpage.html#-C 。文档差异检查通过，无需运行算法测试。
- **未完成项与证据限制：** 未连接服务器，未确认用户旧进程/partial实际状态、远端Range响应或完成续传；未改源代码、配置或依赖，未启动下载/训练、提交或推送。服务器上的协议文件需用户后续同步；本地规则现已生效。


### 2026-10-29 / DPCO-0018：核对训练准备、results三目录与自动保存/probe边界

- **用户要求与授权范围：** 用户询问是否可以训练，要求所有输出保存在results并按files/runs/logs分别存放检查文件、checkpoint和日志，提醒原SAGE有自动保存与probe。本次核对现有实现、记录输出约定并提供nohup启动命令；未把询问训练准备解释为远程启动正式训练或新增probe算法的授权。
- **依据与当前状态：** 用户此前粘贴的服务器日志包含prepared及随后独立verified，50000训练/10000官方测试、human mismatch为20100即40.2%，并打印三组dry-run；这证明数据准备已完成，不证明TPU优化训练已通过。现有CIFAR preset仍为未调优100步pilot。
- **目录核对与具体变动：** ResultsPaths已将相对results根目录锚定到DPCO根目录，RunRecorder将CSV写files/<run_name>，CheckpointStore写runs/<run_name>，launcher与recorder将stdout/train.log、完整配置/环境/启动信息写logs/<run_name>，跨运行状态写logs/run_status.tsv。无需修改源代码、配置或依赖。仅根及DPCO/AGENTS.md追加输出约定，且本文件追加；后续套件外层nohup日志和PID也放results/logs，不延续此前独立顶层logs写法。原始数据位置不作迁移。
- **保存与恢复：** 默认checkpoint开启，固定initial/p25/p50/p75/final和滚动latest；checkpoint包含参数、BN、EMA reference、Adam矩、训练/data/RNG状态及CSV游标。latest会被后续保存替换，未被逻辑slot引用的旧快照按既有策略清理，并非保存每一步；恢复必须显式指定单个原run及一致配置，非自动重启。CIFAR不依据官方test选best。
- **probe事实与边界：** 原theorem/diagnostics/mechanism代码仍保留，但配置验证明确禁止CIFAR100N或EMA/Adam路径启用旧observer。新路径已有每步train_steps、DPCO两组的dpco_steps（负alignment/干预数量、范数等）、evaluation与固定clean/corrupted_noisy/corrupted_original的group_evaluation，以及timing/summary。固定分组CE并不等于旧SAGE定理probe，更不等于按clean/corrupted分别统计负alignment的新增识别诊断；本轮未新增这些算法诊断，也无新理论结论。
- **验证结果：** 本地以三组100步、checkpoint每25步、evaluation每25步的CLI dry-run通过，确认三组配置/子进程透传及results路径。定向运行tests.test_results、tests.test_checkpoint_controls、tests.test_cifar100n共23项CPU测试全部通过（6.603秒），涵盖结果命名/路径、自动保存控制和CIFAR分组/恢复等既有检查；文档diff检查通过。
- **未完成项：** 没有连接服务器、启动真实数据优化或TPU实机测试；三组100步仅建议先验证流程，不能作为训练充分或算法有效的证据。正式预算与调参/统计重复尚未制定；未修改程序行为、提交或推送，既有未提交协议记录保留。


### 2026-10-29 / DPCO-0019：核验三组TPU pilot日志并提出长程参数与敏感性方案

- **授权依据与目的：** 用户提供三份100步训练日志，询问按近期CIFAR-100N的steps、lr、beta如何选择。本次为研究分析、参数建议与可复查命令，仅追加研究记录；没有修改源代码、配置、脚本或依赖，也没有连接服务器启动任务。
- **用户实验事实：** 阅读附件0f55ef65-8186-4bbf-9ff9-c0e5b9019af1、abfdbcd3-af30-4de5-b3bf-b8ba7e950383、93cd41ec-9fe4-4137-a02f-ba03a3bbd976中的Pasted text.txt，分别为Flip、Filter、Adam。三者均4个本地TPU、同一数据manifest、seed42、初始化test CE=6.364174609375、batch128、lr=.001、100步12800样本，均以completed结束。最终官方test top1分别2.04%、3.68%、5.66%；test CE分别5.1727828125、5.617509765625、4.1575671875。此前“未实机验证”描述以此追加用户日志证据更新，但未独立检查服务器checkpoint文件/CSV。100步仅0.256样本曝光轮，不能推断充分训练后的优劣。
- **稳定性观察：** 第25步test CE为Flip219.1931、Filter11.191984375、Adam10.71313671875，后续回落；Flip同节点clean原标签CE245.72566889632108。训练CE与eval CE使用不同BN口径，早期BN统计与快速变化参数不匹配是待排查可能，不能据日志确诊实现错误、宣称已发散，或保证降低lr一定修复。
- **论文与原实现依据：** 数据作者固定版本main.py使用ResNet34、batch128、SGD(momentum.9、weight_decay.0005)、100epochs，alpha_plan为前60轮.1后40轮.01，实际覆盖传入lr：https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/49df7d8a69e355470c77c1c2f2424916325a394b/main.py 。TMLR2025的Coreset-Driven Re-Labeling论文Table3使用CIFAR100N 300epochs、SGD .02/momentum.9/weight_decay.0005/B128，针对DivideMix/SOP+等重标注流程，不是DPCO+Adam配方：https://openreview.net/pdf?id=Tk78vb2Qd7 （正文页面触发验证，检索索引可读到Table3；作者及TMLR页面亦确认出版信息）。Adam原文默认lr.001、betas(.9,.999)、epsilon1e-8：https://arxiv.org/pdf/1412.6980 。因此不存在数据集统一规定的收敛steps或DPCO reference beta，不能把SGD lr.1/.02套给Adam。
- **近期相关证据及边界：** 查阅2024TMLR的Exponential Moving Average of Weights in Deep Learning https://arxiv.org/html/2411.18704v1 ，其CIFAR100N/ResNet34实验为权重EMA，不能将论文alpha=.998或SGD大lr视为本项目梯度reference的验证。另核对2026-10-26论文Detecting noisy labels using checkpoint ensembles https://link.springer.com/article/10.1007/s00521-026-12575-1 ，说明其噪声识别与分类最优时间及scheduler关系依赖具体方法；不把其checkpoint检测/过滤的SGD设置宣称为在线DPCO共通定理。
- **本轮建议（尚未执行、非论文最优值）：** 保持native ResNet34/官方human noisy100/同一增强与FP32/B128。第一轮三组统一200个drop-last数据遍历，即floor(50000/128)=390步/遍历、总78000步；100/300遍历对应39000/117000步，原样本曝光epoch_progress在78000步为199.68。建议共享lr峰值3e-4、cosine降至1e-6、Adam beta1=.9/beta2=.999/eps1e-8，沿用L2=0以保持当前机制实验口径；这不是最终调优或论文复现基线。3e-4是考虑早期CE冲高的保守试验起点，.001保留为预先列出的敏感性值，不能保证3e-4更好。reference beta主设.99，Filter0/Flip-1，两者与Adam共享其他设置；每390步完整固定分组/test评价，latest每1950步，其余固定milestone保存不变。
- **beta解释与设计假设：** beta1/beta2控制编辑后方向的Adam moments，beta_ref控制原始batch梯度reference，独立设置。.9/.99/.999的名义衰减时间尺度1/(1-beta)约10/100/1000步，不是等权独立batch数；.99可减少当前batch在reference中的相对影响但引入历史滞后，尚无本项目性能证据。第一步无历史，偏差修正后r1=G1，增大beta无法改变这一点。后续先固定Adam betas，考察reference .9/.99/.999；lr候选1e-4/3e-4/1e-3，三组公平校准及报告，不能仅挑有利DPCO的设置。若增加正则对照，需三组一致，当前耦合kernel L2不等同AdamW。梯度编辑不是原始带噪CE的标准梯度下降，不能用该训练CE单调降低或固定步数单独定义DPCO收敛/成功。
- **评价与后续选择：** 当前代码官方test被兼容命名为val，尚无独立留出validation；本轮固定配置只能作探索，不能反复按test曲线挑lr/beta后声称盲测优势。正式调参应先制定独立固定validation方案；原标签/污染真值不进入reference/权重，不用corrupted_original组当训练oracle。同设置先seed42，后报告42/43/44三种子配对结果及固定预算final，不能以错误标签CE单独升高视为成功。长程改总预算/调度需从共同初始化新建run，不能按当前严格恢复协议从100步pilot改配置续训。
- **本地验证与未完成：** 用现有CLI对78000步、三组、B128、lr3e-4/cosine/min1e-6、beta_ref.99、Adam(.9,.999,1e-8)、L2=0、eval390/latest1950/seed42做dry-run，解析JSON逐组断言通过；不创建run、不导入训练运行时、不消耗TPU。文档diff检查通过。仅追加本文件；未改动其他未提交记录、启动正式训练、提交或推送。参数有效性、长程数值稳定性、独立validation与新reference识别诊断仍待验证。


### 2026-10-30 / DPCO-0020：三组训练曲线与时间效率绘图方案讨论（未实现）

- **用户请求与边界：** 用户拟在DPCO下新增plots目录，按明确指定的Adam/Flip/Filter三个results/files子目录读取train_steps.csv及timing.csv，画高宽比0.8:1的CE/accuracy折线，并询问timing是否结合CE；明确“先思考，先不改代码”。本轮只读取与讨论，按开发协议追加研究度量建议；没有创建plots目录或改动代码、配置、脚本、原始CSV。
- **字段及现有代码核对：** train_steps含step/train_ce/train_top1/valid_batch_size/elapsed_wall_s；accuracy为0–1比例，图中转为百分数。timing含event/duration_s/window_steps/window_samples/steps_per_s/samples_per_s/elapsed_wall_s等。experiment.py在train_window窗口排空时device_get，窗口时长包含该窗口输入处理、训练派发和等待设备完成；每步train_steps.elapsed_wall_s是host派发时间戳，不等同每步设备完成时间。现有scripts/plot_training_results.py针对旧ImageNet/SGD族，不能直接声称兼容当前CIFAR/Adam三组。未检查服务器完整长程CSV，也未据本轮读取推断长程结果。
- **主图建议（待用户确认实现）：** 两张step图：原始带噪标签普通train CE与100*train_top1；另建议对应的两张time图，横轴为训练循环累计wall-clock分钟，纵轴同前。每图独立三条曲线，纵横比按高/宽=.8解释；后续可固定axes box aspect=.8并另外固定画布，不能仅依靠数据坐标比例。按用户原文Adam蓝实线、Flip红实线、Filter蓝虚线；重复蓝色使用线型区分，不擅自将Filter改成绿色。可选统一后向平滑、保留淡色原始线并记录窗口，不能各方法单独平滑、隐藏前期异常或改原CSV。
- **时间对齐和口径：** 时间主图优先使用timing中train_window的已完成窗口端点elapsed_wall_s，将其覆盖的(step-window_steps,step]内训练CE/accuracy按有效样本数聚合，避免将异步派发时间戳当逐步精确GPU/TPU耗时。窗口指标是区间均值，横轴为端点，需在图注说明。循环wall-clock包括循环内初始评价、JIT、周期评价/保存及训练开销，但不包括循环启动前的数据准备和模型初始化；不冒称全程序端到端时间。可选active训练时间为仅train_window.duration_s累计，排除独立记录的JIT、评价和checkpoint阶段，但仍包含输入与同步，并非纯kernel时间。不能将所有event时长混加当总耗时。
- **辅助效率图：** 可选以窗口末step为横轴、1000*duration_s/window_steps(ms/step)为纵轴，或给出加权总平均1000*sum(duration_s)/sum(window_steps)；不对不同长度窗口的单步均值简单平均。不把duration_s本身误标为每步时长。保持三组相同设备、batch、数据和评价/保存设置，并如实报告是否计入编译和监测开销。
- **方法依据与研究边界：** MLPerf Training以到达指定质量所需时间评价效率，来源 https://mlcommons.org/benchmarks/training/ 。优化器比较亦同时考察墙钟与性能/计算代价，参考 https://proceedings.mlr.press/v199/metz22a/metz22a.pdf 。本项目只读取这两个CSV时，时间-CE/acc图衡量的是带噪标签拟合效率；更低训练CE/更高带噪train accuracy不自动说明DPCO更好。正式泛化效率图应另读evaluation.csv的官方test指标，教授机制图应另读group_evaluation.csv；不在本轮私自扩展实现输入或按test阈值调参。只在预先指定的统一质量目标有意义时报告time-to-target，未达阈值记未达到，不能编造到达时间。
- **拟定文件组织与验证：** plots作为后续绘图代码目录，图像及绘图参数/来源清单建议输出results/files/plots/<comparison_name>/，默认PNG/PDF，遵守既有results三目录规则；三个输入目录由参数显式指定，不自动选“最新”run。此为拟议接口而非已完成能力。检查本轮文档diff通过；无代码修改，不运行算法或绘图测试，未生成示意研究曲线、启动训练、提交或推送。


### 2026-10-30 / DPCO-0021：实现显式三组CSV绘图、可调平滑与ms/step图

- **用户授权依据：** 用户明确要求“可以，按照你的建议来修改代码”，平滑窗口由参数控制，并加入ms/step vs Step图。授权覆盖DPCO/plots离线绘图入口、必要绘图依赖、文档和本地验证；不改训练算法、CSV生产模式或既有实验数据。
- **起始状态与保护：** 开始时只有DPCO_dev.md存在未提交的历史研究/协议追加记录，全部保留。没有创建新worktree、修改训练源文件或既有历史绘图器；没有连接服务器、启动训练、提交或推送。
- **代码与接口：** 新增plots/plot_dpco_training.py及plots/README.md。用--adam-dir/--flip-dir/--filter-dir显式指定三个不同run目录，支持绝对路径、相对DPCO根路径或results/files下单个run名；不自动选择最新/最佳实验。只读train_steps.csv、timing.csv，存在对应logs元信息时额外校验方法/优化器/EMA负权重，报告配置与数据指纹差异；无元信息的CSV导出仍可画图但明确标签来源限制。训练runtime完全不导入，--help在无site-packages解释器下亦可用。
- **图形与输出：** 默认生成train_ce_vs_step、train_accuracy_vs_step、train_ce_vs_time、train_accuracy_vs_time、ms_per_step_vs_step五张独立图，各PNG+PDF。Adam蓝实线、Flip红实线、Filter蓝虚线。画布5×4英寸、绘图区高/宽=.8，PNG220DPI（1100×880），不使用tight裁边改变导出高宽比；accuracy按100倍转为百分数，线性CE不丢弃早期异常值。输出到results/files/plots/<自动唯一时间戳或--output-name>/，已有目录拒绝覆盖；另存step_curves.csv、timing_curves.csv和最后写入的plot_manifest.json，保留输入/脚本SHA256、平滑参数、计时说明、设置差异、实际曲线和加权ms/step。失败目录无完成manifest，旧结果不自动清除。
- **平滑定义：** --smooth-window默认100，正整数，1关闭额外平滑。Step图按最近W次更新的有效样本数加权，开头按已有前缀平均，保持因果性。时间及ms/step图以完整train_window为最小单位，从当前窗口往前累计到至少W步（开头不足则已有窗口）；不拆窗口、不对host派发时间作伪精确设备完成插值。W=1保留原始窗口均值。耗时用sum(duration)/sum(window_steps)，而非对不同长度窗口的ms/step简单平均。原始淡色线默认开启，--no-raw可关闭；输出CSV仍保留原始窗口/逐步值，实际平滑步数写入timing_curves.csv。
- **计时与输入完整性：** --time-axis wall默认取已完成train_window端点elapsed_wall_s/60，包含此前循环内监测/编译/保存，排除循环前数据准备和模型初始化；窗口指标为区间平均，末点不是最终评价/保存之后的进程结束时间。active选项仅累计train_window duration，仍含输入与设备同步。ms/step图取1000*duration/window_steps。校验连续完整训练step、窗口无重叠无缺口且覆盖所有训练行、正窗口时长、时间一致/单调、数值有限和accuracy比例。只读输入前后哈希复核，文件变化时拒绝产生完成manifest；不把训练中两个文件异步刷盘的不一致快照自动修补。
- **涉及文件：** 新增plots/plot_dpco_training.py、plots/README.md、tests/test_dpco_training_plots.py；pyproject.toml新增可选plot extra（matplotlib>=3.8,<4，不改变train/tpu依赖）；docs/cifar100n.md追加新绘图入口说明；追加本文件。提供服务器nohup命令将绘图依赖安装与绘图串行执行，外层日志和PID同样位于results/logs。
- **验证结果：** 新增8项测试与旧绘图10项合计18项检查全部通过（0.633秒）。验证独立数值预期的样本加权平滑、因果前缀、不等长计时窗口加权及分钟/毫秒单位、编译事件不计active、忽略host派发戳、gap/overlap/缺失/非有限/时间倒退拒绝、accuracy单位和方法标签拒绝、设置与数据指纹不匹配说明、完整五图十文件实际渲染、PNG尺寸/PDF头、源CSV字节不变、输出防覆盖、绘图区.8比例/颜色线型及active坐标、无第三方依赖help。另在/private/tmp/dpco-plots-visual-qa-*用明确的合成fixture导出五张PNG/PDF，并查看CE-step、accuracy-time及ms/step三种代表布局，无文字裁剪/颜色或比例问题；这些QA图不是用户实验数据或算法效果证据。Python AST/TOML、README命令shell语法及git diff --check通过。
- **证据边界与未完成：** 未取得用户服务器78000步的完整CSV，尚未生成真实长程实验图，不能用本轮fixture说明方法性能。训练CE/accuracy仍是观察噪声标签拟合，泛化与clean/corrupted曲线未在此入口新增；未新增time-to-target/阈值选择、跨seed置信区间或修改旧theorem probes。绘图参数不改变训练轨迹，无算法收敛或收益新结论。


### 2026-10-30 / DPCO-0022：绘图输出改为results/figures并将Filter改为橙色

- **用户授权依据：** 用户明确要求“把plot的输出结果放到result目录下，起名figures/ 与files等目录同目录级别。现在修改代码，并且把fliter变成橙色”。依据与既有files/runs/logs同级的要求，沿用实际项目根目录名results，目标为DPCO/results/figures，不另建单数result。
- **目的与具体改动：** plots/plot_dpco_training.py将新绘图任务输出根目录从results/files/plots改为results/figures，保留原时间戳或--output-name子目录结构。五张图的PNG/PDF、派生曲线CSV与plot_manifest.json统一写该任务目录；CLI帮助与使用文档同步。Filter颜色改为橙色#ff7f0e，保留虚线，Adam蓝实线和Flip红实线保持。旧路径/旧配色记录作为历史保留，以本条和当前代码为准。
- **涉及文件：** plots/plot_dpco_training.py、plots/README.md、docs/cifar100n.md、tests/test_dpco_training_plots.py及本文件。已有集成渲染检查更新为断言新目录与不再创建旧输出目录，颜色检查更新为橙色。没有修改训练源代码、计时/平滑公式、依赖或原始结果。
- **验证结果：** 新/旧绘图合计18项检查全部通过（0.634秒），包含新目录下实际生成五张PNG/PDF、源CSV不变、防覆盖及三种颜色/线型。git diff --check通过；只进行了授权变更所需的本地验证，无新算法测试或正式训练。
- **理论与证据边界：** 本次仅输出组织与视觉样式修改，无理论或实验性能新结论；渲染仍使用临时fixture，未读取服务器真实长程数据。已有results/files/plots产物保留，不自动搬移、覆盖或删除；用户同步脚本后使用原绘图命令即可生成到新位置。未连接服务器、提交或推送。


### 2026-10-30 / DPCO-0023：解释长程训练CE曲线、继承的40.2%噪声与Adam学习率下限

- **用户请求与授权边界：** 用户批注此前78000步命令，并提供train CE对wall-time/step两张图，认为未达预期，询问实际错误数据百分比及Adam是否必须把lr降至0。本轮只读核对、解释并按协议记录研究判断；不构成修改算法、训练/绘图配置或再次运行的授权。
- **图像证据：** 用户附件codex-clipboard-d627b634-cc8e-46de-a54e-9e3ae140adb0.png和codex-clipboard-77e1a4a7-3c65-4e97-be08-555058a96f66.png显示Adam和Filter后期训练CE接近0，Filter早期较慢，Flip在78000步附近仍约3.2；这些为读图近似值，未取得对应完整CSV或test/group曲线，不能凭此判定泛化优劣或去噪成功。曲线展示普通观察标签CE，不是signed训练替代目标；跨时间变化的训练batch/增强/BN指标也不是同一冻结数据评价。
- **噪声核对：** 选中命令通过--config继承configs/tpu/cifar100n/dpco_adam.toml的data.noise_type=noisy100、is_human=true、noise_percent=40.2；三组共享已发布human标签。先前prepared/data_ready记录为20100/50000即40.2%相对原标注不一致，另外29900即59.8%一致；并非只选40.2%样本训练、每批恰有40.2%错误、每轮重新改标或负alignment比例。noise_percent为0–100实际比例断言，不是人为生成指定百分比的旋钮；如需命令可读性，可显式追加--noise-type noisy100 --is-human --noise-percent 40.2，但不会改变这一数据版本。测试标签未额外注入噪声；原benchmark标注仍不是绝对语义正确的认证。服务器正式run应以其metadata/data_ready和实际CSV为最终证据，本轮未连接服务器核验文件。
- **学习率核对：** 重读src/training/learning_rates.py及CLI解析，命令指定峰值3e-4、cosine、min1e-6、T78000。第k次更新k=1..T的lr为min+(peak-min)/2*(1+cos(pi*(k-1)/(T-1)))；本地纯标量计算首步0.0003、接近中点0.0001505030107315586、末步0.000001。故本次已衰减约300倍，未衰减到0。Adam的moment自适应与外部global lr schedule是不同机制，算法本身不要求有限预算末步为0；原算法可用固定alpha，cosine可选0或正下限。特定理论中的渐近步长条件不能泛化成所有深度学习Adam训练必须最后一步0。核对原论文 https://arxiv.org/pdf/1412.6980 及Optax调度文档 https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html ，项目采用本地定义的cosine而非声称调用该库schedule。将floor设0并无当前Flip更好的证据，不将其当作解释表现差异的充分原因。
- **理论解释与下一步判据：** 原始带噪训练CE更低可能伴随记忆错误标签，不等同更好泛化；Flip较高亦可能来自抑制错误拟合或误伤有效样本，不能自动解释成抗噪有效。r_t^T g_i<0并不推出标签错误；raw noisy-gradient EMA可平滑波动但不自动消除标签偏差，Flip对误识别样本的反向处理比置零更强。之前200轮/beta_ref.99是待验证起点，不是已证实最优设置。下一步优先共同查看evaluation.csv官方test CE/top1、group_evaluation.csv的clean原标签/corrupted noisy/corrupted original指标及dpco_steps.csv的negative_count/valid_batch_size、reference cosine等；目前没有按C/N分别统计识别负alignment的现成功能，不宣称已验证噪声识别精度。不按这两张train图直接调参、延长训练或修改算法。
- **验证与涉及文件：** 使用完整批注参数对三组现有CLI执行dry-run，逐组确认noisy100/human/expected40.2及cosine/3e-4/min1e-6；标量公式核算如上。仅本文件追加，文档diff检查通过；没有跑训练测试、改动代码/配置/图片或原数据、启动正式训练、提交或推送。


### 2026-10-30 / DPCO-0024：分析导入的完整三组结果、正确标签误干预界与三个beta释义

- **用户请求与授权边界：** 用户已将服务器结果复制到项目根目录Results/，要求分析差异和DPCO未优于Adam的原因，并询问reference定义及“beta2=.999是否当前.999/历史.001”。本轮只读实证分析、生成研究报告/派生CSV/图表及追加本记录；没有修改程序源代码、配置、依赖或启动模型训练。开始时DPCO_dev.md已有上一轮未提交记录，予以保留。
- **数据与公平条件核对：** 读取Results/files及logs中三个长程run全部train_steps/evaluation/group_evaluation/summary/timing、两组dpco_steps与三份metadata。每组78000训练行、201次10000张test评价、603行分组评价，均completed。三组data/model/training/execution/seed/global_batch_size/evaluation/checkpoints完全一致，均seed42、B128、native ResNet34、FP32、四TPU、cosine3e-4→1e-6、Adam(.9,.999,1e-8)、L2=0、40.2%官方human噪声。记录的commit f89da2f5654a82fb119788752e291de7ceaeaecf、clean工作区、source SHA256和数据指纹均一致。源码diff核对当前dpco_history/updates/images与该commit无差异；训练状态/逐样本alignment未提供，不能因此排除所有潜在实现问题。
- **最终结果事实：** Adam/Filter/Flip最终官方test top1分别51.24%/51.20%/26.10%，test CE4.19585/3.73212109375/3.0038705078125，循环时间53.1137/59.3314/59.0950分钟，train_window加权27.5060/31.1781/31.1120ms/step。Filter top1仅低.04个百分点（总正确数差4），单seed不足认定比Adam差；test CE与top5反而较好。Flip比Adam低25.14个百分点，但test CE更低，须区分准确率、log loss与未测的校准指标，不能以任一项代替全部效果。
- **固定分组证据：** Adam/Filter/Flip正确标签组top1为100%/99.9967%/33.9097%；错标组对人工错标签top1为99.9602%/99.9502%/9.6866%；同一错标组对原标注top1为.0199005%/.0298507%/17.1542%，对应CE17.9084/14.7815/3.36735。故Filter最终仍几乎完全记住了错标签；Flip避免大量错标记忆，却严重欠拟合正确标签组，不能称为成功抗噪。相对原benchmark标注的分组不认证绝对语义正确。
- **时间演变与过拟合：** 错标组第一次达到50%错标签拟合：Adam27轮、Filter44轮；达到90%：47/63轮。第一次test top1达到50%：Adam15轮、Filter28轮，Flip未达。第30轮Filter错标原标签top1为33.31%，Adam12.55%，但最后两者均近零，说明Filter延缓而未阻止记忆。test事后峰值Adam51.82%(8580步约22轮)、Filter51.28%(12870步约33轮)、Flip26.10%(最后)，仅用于描述，不用于test选checkpoint。Adam最低test CE1.81309约16轮、Filter1.79427约30轮，最终更高，结合错标组体现长程过拟合；不能把最终train CE趋零当成去噪成功。
- **干预与reference统计：** Filter全程负alignment/贡献编辑比例44.1561%，Flip40.3385%；第101–150轮Filter50.4653%。reference/raw batch gradient平均cosine分别.123965/.144627，低相似度本身不证明reference错误，但没有提供与clean目标一致的证据。编辑后/原始方向范数比均值Filter.925812、Flip1.16616，故Flip不能简单归因于“正负抵消导致梯度范数消失”。Adam可抵消部分整体缩放，范数比不等于实际参数步长比。所有reference_gain最小为0符合代数构造，却没有带来Flip泛化优势。
- **离线误干预下界（新推导与验证条件）：** 利用先前本机已验证的/private/tmp/dpco-cifar-real/prepared/labels.npz，核对prepared manifest哈希1432cf...、labels.npz哈希与本次观察标签哈希4eeae1...逐项一致。按同一代码seed42/epoch-shuffle/drop-last重建200轮每批实际错标数n_t；共9984000次出现，错标4013550、正确标签5970450次。若日志负数a_t，则该批正确标签编辑数至少max(0,a_t−n_t)。逐批求和，Filter至少606666次正确标签出现被过滤（正确标签出现次数的10.1611%），Flip至少262987次被翻转（4.4048%）；相应噪声识别precision至多86.2389%/93.4700%。这些是条件于已核对批顺序的保守下/上界，不是实测precision/recall或独立图片占比，不识别具体被编辑的样本；真实误伤可能更多。它证明存在误干预，尚未证明该因素解释全部性能损失。
- **reference精确定义与纠正：** G_t为当前有效batch所有原始观察标签梯度均值，H_t=beta_ref H_{t-1}+(1−beta_ref)G_t，r_t=H_t/(1−beta_ref^t)，H0=0。本次metadata确认beta_ref=.99，故历史状态99%+当前原始梯度1%，不是beta2=.999。历史状态包含更早梯度且在各自旧参数处计算，不保存/重算旧batch；当前batch包含在reference。bias修正后r1=G1，r2约.49749G1+.50251G2；正标量修正不改分数符号。q_i=r_t^T g_i，负分数权重Filter0/Flip−1，其余1；D_t用原始B分母聚合。被过滤/翻转样本原始梯度仍通过G_t进入H，且过滤非永久删除。
- **Adam两个beta与理论边界：** beta1=.9：m_t=.9m_{t-1}+.1D_t；beta2=.999：v_t=.999v_{t-1}+.001(D_t⊙D_t)，再分别偏差修正、按m_hat/(sqrt(v_hat)+eps)更新。故用户当前/历史权重理解反了；v为编辑后方向的逐坐标二阶原始矩，不是reference、有符号方向或每样本平方梯度的平均。本次三个状态独立；beta_ref名义时间尺度约100步，beta2约1000步不是同一平均。依据代码dpco_history.py:66–81和updates.py:102–104/124–133及Adam原论文 https://arxiv.org/pdf/1412.6980 。raw带噪梯度EMA平滑波动不自动去偏、旧参数梯度滞后、正确难样本梯度冲突、−1强干预与Adam坐标预条件化均为合理机制假设，尚不能据当前汇总日志完成因果分解。r^T(D−G)≥0不保证clean目标或Adam实际更新下降。
- **产物与复核：** 在DPCO/results/files/cifar100n_review_20261030_07oqs01k/生成analysis.md、final_comparison.csv、epoch_metrics.csv、dpco_epoch_stats.csv、intervention_bounds.csv、analysis_manifest.json；同名results/figures目录保存accuracy_and_memorization和reference_and_interventions两套真实数据PNG/PDF，已视觉检查轴、标签和颜色。原始输入前后SHA256逐项相同，记录在manifest。统计核对全部78000步连续性、窗口总步数、元信息一致性与分组节点完整性，独立计算首次阈值到达轮数作交叉检查；doc diff检查通过，不运行训练或更改绘图程序。
- **未完成项与建议：** 单seed、未调优、官方test已监测；不能给出显著性、最优超参数或所有设置的算法失败结论。没有逐样本scores与checkpoint，无法重算实际混淆矩阵、clean目标梯度或实际更新投影。下一步优先拟定只读的C/N负alignment诊断和独立validation方案，再由用户授权实施与对照强度/时间尺度实验；本轮未实施、提交、推送或更改原始Results/。


### 2026-10-30 / DPCO-0025：重读组会，校正对教授小规模flip验证及正确标签编辑的解释

- **用户请求与授权边界：** 用户指出教授已在小数据集/MLP上做过小规模正面验证，要求重读新放入文件夹的两份组会MD并回忆原实验要求。本轮为阅读、理论解释和追加更正；不授权更改算法、配置、脚本或启动训练。开始时本文件已有未提交记录，完整保留。
- **涉及文件与核对：** 重读上级AGENTS.md、本目录AGENTS.md及近期开发记录；完整阅读项目根目录《教授周会_10.27_转写校对稿.md》《教授周会_10.27_新项目实验重点总结.md》，核对dpco_history.py:66–81与前轮派生final_comparison.csv。仅追加本文件与results/files/cifar100n_review_20261030_07oqs01k/analysis.md的第9节，不改原始会议记录或Results输入。
- **组会证据与边界：** 转写27:59–29:19提出batch mean的历史衰减signal；31:27–31:44明确确认换reference后继续flip；32:16–35:24先做noisy数据的controlled verification，观察clean loss低/下降与corrupted loss高，之后才scale up；33:40–33:52教授明确报告MNIST简单实验已完成，33:52–34:21建议先Adam并讨论flip频率。MLP背景由用户本轮补充，记录未给模型结构。两份MD没有教授实验代码、精确r、beta、噪声比例/机制、损失、初始化/归一化/LO细节，亦不能确认“每轮”是step还是epoch；因此当前含当前batch、raw EMA、beta_ref=.99、每step全参数flip的实现属于已授权原型，不能宣称严格复现其MNIST配方。
- **理论更正：** 对DPCO-0024及analysis.md中“正确标签误干预”“证明存在误干预”的解释追加收紧：逐批计数推导仅证明原标注一致样本被编辑，不能由标签正确推出其对给定reference的贡献一定有利，故不能证明贡献优化意义上的误判。Filter10.1611%/Flip4.4048%仍为正确标签样本出现次数被编辑的下界；相应precision上界只是在将负alignment作为错标识别预测时的指标，不是flip最优性的反证，也没有完成性能损失的因果分解。此前过强用语以本条追加更正为准，原数值及历史文本保留。
- **条件性教学推导：** 固定r及g_i，对w_i∈[-1,1]独立约束下最大化r^T(Σ_i w_i g_i/B)，每项为w_i q_i，故q_i>0取1、q_i<0取−1，q_i=0任意。这只是与当前实现一致、便于解释的子问题，不能代替录音未完整给出的教授定理及其假设。该局部子问题最优、r对关心的学习目标是否合适、多步Adam轨迹是否改善，是三个不同问题。教授报告的MNIST正面实验应保留为其条件下的实验证据；当前单seed的CIFAR结果既不能否定该报告，也不能证明普遍成功。
- **按教授目标重述实验：** 最终clean CE为Adam0.000167929、Filter0.000403484、Flip2.683918；corrupted观察错标签CE为0.001030090/0.001556779/3.963198。Flip的clean CE随训练也下降，不能称为完全没学；但同预算clean拟合显著弱于对照，尚未得到所希望的选择性改善。单看高corrupted loss或低整体train CE均不足判断。Filter最终test top1与Adam近似持平，不将两种干预笼统判为失败。
- **后续建议及未完成项：** 优先对齐教授小实验的r精确公式、原始/编辑梯度、评分参数范围、噪声设置、损失、Adam参数及flip频率，再做MNIST/MLP最小复现并逐步改变数据/模型以定位差异。此前三组是按用户后来确认的范围实现，未增加batch-only基线；历史signal净收益需另作消融，不能把原会后计划当作本轮新增实现授权。BN、真实噪声、多类别、reference滞后等仅为待验证候选因素；本轮无新因果归因。
- **验证结果：** 对照原文时间戳、当前reference实现和已有派生CSV，表内数值一致；文档diff检查通过。未运行模型测试、修改代码/配置/依赖、启动训练、提交或推送；教授最小实验的精确配方仍待取得与核验。


### 2026-10-30 / DPCO-0026：评估MNIST接入范围、CIFAR后续定位与历史reference的会议符合性

- **用户请求与边界：** 用户更正MLP可能记错，只确认教授说过MNIST，具体模型未知；要求评估MNIST所需代码改动、不做MNIST时如何继续CIFAR-100N，以及当前指数衰减reference是否满足教授要求。仅评估和理论澄清，未授权实现。本轮只追加开发记录，不改代码、配置、依赖或启动实验；DPCO-0025中MLP作为用户补充的描述以本条撤回其确定性，不能把MLP当作已知教授设置。
- **读取与保护：** 阅读两级AGENTS、最近记录、config.py、data/classification.py、data/images.py、data/cifar100n.py、models/resnet.py、tasks/classification.py、training/evaluation.py、training/launch.py、training/results.py、scripts/train_classification.py及相关EMA测试；核对会议27:59–29:19、31:27–31:44、33:40–34:21。已有未提交记录保留。一次读取dpco_history.py时工作目录少一层导致未找到文件；其路径已在上一轮核验，未据该失败推断文件缺失。
- **MNIST工作量评估：** 当前没有MNIST loader/模型预设，不能只改数据集名。原生28×28灰度接入需处理evaluation_batches与模型initialize的三通道硬编码；无BN小模型还需独立adapter/初始化，不能套用强制batch_stats的ResNet adapter。完整规范接入包括下载/校验/缓存、固定训练标签污染及原标签审计、数据分发/配置校验、模型选择、分组评价、训练/启动CLI与元信息、噪声身份/run命名、配置文档和必要测试。按独立小模型、复用现有结果/恢复体系的设计，粗估修改7–10个现有文件并新增5–8个文件（含测试、预设和文档），不是已实现或承诺精确diff；最终取决于模型、单通道方案和噪声接口。EMA/flip/filter、Adam、CE任务、通用训练循环和checkpoint主体可以复用，无需重写优化器。小CNN/MLP均只能是我们自主选择的验证模型，缺教授配方时称MNIST机制验证而非严格复现。
- **符合性结论：** 当前G_t为原始batch梯度均值，H_t=beta_ref H_{t-1}+(1-beta_ref)G_t，r_t=H_t/(1-beta_ref^t)，再依负alignment编辑并交Adam；这直接落实会议提出的batch mean、历史beta衰减、继续flip三个要点，属于符合明确构造方向的合理实现，不应因单次结果差认定偏离教授要求。它借鉴的是Adam式一阶历史累计，不要求把Adam二阶矩或预条件化也放入r。beta_ref=.99为我们具体选值，并非教授指定；正标量bias correction在非退化精确算术下不改变评分符号，不能把是否bias correction本身列作本次sign-only性能差异的主要解释。精确数值/频率和MNIST配置尚未知；实现符合方向不等于已达到受控验证预期。
- **继续CIFAR的建议（未执行）：** 保留当前三组和原始结果，先用已有clean/corrupted_noisy/corrupted_original轨迹评价选择性学习。首选在同一ResNet34上追加clean100对照，比较Adam与Flip是否在无训练标签污染时仍出现学习阻碍；当前CLI已支持clean100/0断言，无需实现新数据集。clean100下现有C/N分组仍按原human reference划分，组名不是当前训练的实际污染状态，clean训练分析须用原标签评价，不能混淆corrupted_noisy的反事实标签。随后固定flip=-1、Adam参数和其他条件，只比较beta_ref=.9/.99/.999；阶段性实验需共享总预算与lr调度，短预算cosine不能直接同原200轮前段视为完全同设置。新增干净对照或消融均建议从共同初始化新run，不修改现有run配置续训。ResNet18已可配置，可作后续模型敏感性而非教授模型复现。
- **诊断与噪声接口边界：** 如需定位机制，另行授权增加周期性C/N alignment分布/负数比例、贡献大小，以及固定clean目标梯度与实际Adam参数增量的内积；真值仅进入观察者，不进入训练reference或权重。现有dpco_steps总计数不足以恢复实际逐样本归属，旧observer在EMA/Adam配置被禁用且不直接适用，不直接打开旧probe。CIFAR当前noise_percent是实际噪声比例断言，no-is-human是按人类标签经验转移矩阵采样；两者都不是任意20%等污染强度的生成接口。自行引入污染比例旋钮需另行授权，并保留官方40.2%基准的原含义。
- **验证结果与未完成项：** 静态确认上述配置/数据/模型/评价硬编码和既有可调入口，文档diff检查通过；没有声称MNIST新路径运行通过或给出服务器训练时长。模型未知、教授配方未知、候选失败机制尚无因果确认；本轮没有修改源代码、生成正式配置、运行训练、提交或推送。


### 2026-10-30 / DPCO-0027：核对ResNet型号并提供三档reference beta串行后台命令

- **用户请求与边界：** 用户询问为何ResNet34、能否用50，并决定先固定Flip−1、Adam/模型/数据，仅比较beta_ref=.9/.99/.999，要求启动命令，由用户在服务器执行；随后另一台服务器的MNIST等研究是后续计划。本轮只核验入口、提供命令并追加记录，没有修改源代码/配置/脚本、连接服务器或启动正式训练，不据后续计划实现MNIST。
- **模型事实与解释：** 重读两级协议、最近记录、CIFAR模型与配置校验、训练launcher和导入metadata。三个既有长程run均为cifar_resnet34，并非resnet50。沿用34是为了保持本轮beta消融的模型与既有对照一致，不是理论上必须用34或已证实34优于50。ResNet50原则上可以开展此类实验，但当前CIFAR分支仅实现native32的cifar_resnet18/34；已有resnet50是ImageNet分支的模型入口，配置校验拒绝CIFAR+resnet50，不能仅换CLI参数。若以后引入CIFAR版50需另行授权适配并重建同模型对照。
- **命令方案（提供，未执行）：** 从服务器/home/jason_chia925_gmail_com/Project_DPCO/DPCO运行；使用绝对Python /home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python；外层nohup bash按.9/.99/.999顺序调用scripts/run_suite.py --wait --mode train，每组只有dpco_flip、seed42、negative_weight−1。明确cifar100n/cifar_resnet34、noisy100/human/actual40.2%、78000步/B128、cosine3e-4→1e-6、Adam(.9,.999,1e-8)、L2=0、eval390步、checkpoint1950步且开启保存。不使用resume，从共同seed重新开始；已有Adam作对照无需按reference beta重复训练Adam，三组beta均新建run且保留原.99结果。
- **后台与输出：** 外层stdin=/dev/null、stdout/stderr统一进入results/logs下mktemp生成的独立sweep子目录/sweep.log，保存sweep.pid；各beta独立launcher日志。建议用set -euo pipefail及管道tee同时记录launcher输出到每组日志和sweep.log；任意组退出失败则停止后续组。run_suite子进程本身也用nohup，保持逐run的files/runs/logs输出，并等待实际退出/TPU锁释放后进入下一beta。提供PID、ps与tail方式；这是终端断开保护，不声称机器重启恢复或自动续训。
- **验证结果：** 本地现有解释器对三档完整CLI分别dry-run并解析JSON，逐组断言model、EMA beta、负权重−1、Adam、78000步、human40.2和wait；检查子命令透传reference_beta/negative_weight及Adam三个参数，均通过。另通过resolve_config确认CIFAR+resnet50确实被拒绝。没有创建训练run或调用JAX/TPU执行。一次外层目录git status因外层非Git根返回失败，已在DPCO内正确检查，仅本开发记录有未提交改动。文档diff检查通过；没有新增理论性能结论，三档能否改善仍待用户实验。


### 2026-10-30 / DPCO-0028：区分官方ResNet34基线与本项目18/34的实现范围

- **用户请求与授权边界：** 用户追问本项目为何只支持CIFAR ResNet18/34，是否官方既定标准。本轮为只读源码/官方来源核验与解释；没有授权适配新模型。只追加本记录，不修改源代码、配置或训练状态。
- **来源与验证：** 读取两级AGENTS、最近记录、src/models/cifar_resnet.py、src/models/resnet.py、src/config.py及docs/cifar100n.md。浏览官方仓库 https://github.com/UCSC-REAL/cifar-10-100n ，其固定版本main.py https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/49df7d8a69e355470c77c1c2f2424916325a394b/main.py 明确model=ResNet34(num_classes)，并使用SGD；当前官方main分支模型文件 https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/models/resnet.py 同时定义ResNet18/34/50/101/152，其中18/34为BasicBlock，50/101/152为Bottleneck。固定版本models/resnet.py一次网页读取cache miss，未将失败页面当证据，后改用成功读取的main分支文件并明确版本范围。
- **解释与更正范围：** 官方公开训练入口采用ResNet34，作为本项目34模型选择的基线依据；不存在由这些来源得出的“CIFAR-100N只允许18或34”的要求，官方自身也提供50等模型定义。当前18/34限制来自本项目CIFAR专用模块只实现BasicBlock及工厂中的两个stage映射，并在配置层显式限制；18提供同一结构族的较小选项。ImageNet分支已有50/101的Bottleneck与7×7 stride2/maxpool入口，尚未接成CIFAR模型。此为本项目工程支持范围，不能表述为数据集或官方模型白名单，更不能从中推断34性能必优于50。当前JAX/Adam训练亦不等同完整官方PyTorch/SGD配方复现。
- **对当前实验的影响：** 已提供的beta消融继续固定cifar_resnet34，以仅改变reference beta；以后若研究50，应明确适配及模型定义并重建同模型对照。没有改动或重新启动用户可能正在执行的任务。
- **涉及文件、检查与未完成项：** 仅本文件追加；git diff --check通过。未运行算法测试、实现CIFAR50、启动正式训练、提交或推送。实际50模型的接入仍未实现，且本轮未讨论其性能保证。


### 2026-10-30 / DPCO-0029：按模型参数复用ResNet50并接入MNIST三组Adam实验

- **用户授权依据：** 用户明确要求“现在改代码，用参数来控制使用那个model，并且加入minist”。工作中用户进一步选择“小CNN，20%合成错标签”。授权涵盖模型选择、既有ResNet50复用、MNIST准备/加载/训练接入、必要配置/文档/检查；没有授权启动服务器正式训练、推送或改变EMA/flip公式。开始时仅本文件有未提交历史追加记录，全部保留。
- **模型选择与语义：** 新增src/models/classification.py，以明确模型名分发，训练入口不再按CIFAR数据集强行选择CIFAR工厂。--model resnet50/resnet101直接复用既有ImageNet结构（7×7 stride2及maxpool）；CIFAR分类头100类，原生32输入及原预处理保留。--model cifar_resnet18/34保留原结构，新增cifar_resnet50使用3×3 stride1入口并复用既有Bottleneck类；既有Bottleneck的stride位置及最后BN零scale初始化均保留，不宣称完整复刻官方PyTorch模型。配置与CLI允许显式选择这些模型；CIFAR预设默认34不变。MNIST另提供mnist_cnn、mnist_mlp，均无BN/dropout；默认CNN为Conv16/pool、Conv32/pool、Dense64、10类头，MLP为128/64两隐藏层。MNIST也可显式选择现有ResNet，输入通道由数据确定为1，不暗中改stem。
- **数据准备：** 新增src/data/mnist_prepare.py及scripts/prepare_mnist.py，提供plan/download/verify。沿用完整文件校验、TLS、准备锁与原子发布辅助函数；四份原始IDX gzip采用torchvision官方MNIST loader列出的HTTPS镜像及MD5（https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py），严格检查magic、60000/10000计数、28×28形状、长度与0–9标签范围；NumPy缓存另有SHA256 manifest及complete标记。缓存只含干净来源，下载/加载不依赖Torch或TensorFlow。现有坏文件不覆盖；已完成下载复用，单个.partial重试从头下载，不新增断点续传承诺。默认数据路径../data/mnist。
- **MNIST噪声定义与研究边界：** 新增src/data/mnist.py。--label-noise-percent是专属生成参数，0–100百分数；固定noise_seed先产生训练索引permutation，再生成全体1–9偏移，选择floor(N*p/100+.5)个索引，将标签偏移模10，保证被选标签改变且在另外9类均匀采样。完整数据20%恰为12000/60000；训练seed与noise_seed独立，三个方法共享固定标签，不每batch/epoch重生成。noise_percent继续为实际比例断言，CIFAR官方40.2%接口不改义。0%自动clean，其他自动noisy；只改变data_mode=clean亦清零MNIST生成比例。记录生成算法版本、实际比例、观察标签hash、分组与视图计数，配置/噪声变动导致不同数据指纹。使用合成污染属于新的机制验证，教授MNIST模型与配方仍未知，无理论/实证性能保证。
- **输入、评价与训练连接：** 原生28×28×1 float32/255，无随机裁剪/翻转。ImageBatchSource复用原shuffle/drop-last/恢复；evaluation_batches按reader.channels分配padding，原数据默认3不变。initialize新增可选channels=3；model_adapter同时支持BN状态和空状态小模型。训练batch仍仅images/观察labels/valid_mask，原标签和污染mask只用于固定审计。复用GroupedEvaluation，记录clean、corrupted_noisy、corrupted_original与不加噪的官方test；空组记录0及缺失指标，不用test选best。训练、timing、分组CE、EMA状态、Adam moments及checkpoint延续现有files/runs/logs协议，旧observer保持禁用。
- **配置与兼容：** 新增configs/tpu/mnist/dpco_adam.toml，用户选定CNN/20%/noise_seed0，探索性seed42/B128、Adam(.9,.999,1e-8)、lr.001/cosine至1e-6、beta_ref.99、L2=0、9360步（20个drop-last数据遍历）、eval468、latest2340；不是教授配方或已优化超参数。run_name包含模型、生成比例及noise seed。共享launcher、child与配置检查CLI透传--model、--label-noise-percent等；dataset切换清理不兼容标签配置。DataConfig新增字段默认0，旧恢复metadata缺失该字段时仅内存补0比较，不改写旧metadata。
- **涉及文件：** 新增src/data/mnist.py、mnist_prepare.py，src/models/classification.py、mnist.py，scripts/prepare_mnist.py，configs/tpu/mnist/dpco_adam.toml，docs/mnist.md，tests/test_mnist.py、test_model_selection.py；修改src/config.py、cli.py、data/classification.py、data/images.py、models/cifar_resnet.py、models/resnet.py、training/launch.py、training/evaluation.py、training/experiment.py、training/results.py，scripts/launch_tpu.py、train_classification.py，tests/test_cifar100n.py，README.md、docs/cifar100n.md及本文件。未修改src/methods/dpco_history.py、Adam更新公式或旧结果。
- **本地验证：** 定向运行test_model_selection、test_cifar100n、test_mnist、test_config、test_suite、test_classification、test_dpco_history、test_checkpoint_controls、test_optimizer_options、test_results，共82项CPU检查通过（33.377秒）。覆盖模拟四原始IDX文件完整60000/10000下载/解析、校验拒绝/缓存复用、固定噪声精确计数、单通道padding、固定分组CE独立NumPy核对、MNIST三组真实训练入口2步及checkpoint精确恢复；两种50完整模型eval_shape确认stem及2048维head，并用缩小Bottleneck骨干经真实入口训练/恢复。早期61项检查有1项沿用旧“CIFAR必须拒绝resnet50”断言失败，按新授权改成拒绝MNIST-only模型后，上述82项全部通过。
- **额外兼容与并行验证：** 对CIFAR旧metadata人为移除新增字段后，三组从p50恢复的state逐项精确一致且metadata字节不变，定向检查通过（4.483秒）。与HEAD的旧实现比较缩小ResNet及CIFAR BasicBlock，初始化参数/BN、train logits及更新BN逐项精确相同。全部15份TOML通过配置校验；CIFAR34/既有50/native50及MNISTCNN/MLP/既有50各三组dry-run通过，子命令参数回传一致；MNIST source plan通过。另在四个模拟CPU设备执行灰度CNN/EMA/Flip/Adam真实入口2步，分组评价padding与checkpoint计数正确；不把CPU模拟称为TPU实机验证。git diff --check通过。
- **文档、输出与未完成项：** docs/mnist.md提供绝对项目Python路径的nohup下载与串行三组训练命令、stdin=/dev/null、独立日志/PID和查看方式，说明新服务器需匹配execution拓扑；README与CIFAR文档说明两个50名称的结构区别。所有本地训练/下载检查均在临时fixture目录，未下载真实MNIST、启动正式服务器训练、改写原始Results、提交或推送。完整真实数据准备、TPU实机执行、长程收敛与方法优劣尚待用户验证；当前CIFAR beta消融不改模型或配置，不因本功能接入自动切换到50。


### 2026-10-30 / DPCO-0030：制定MNIST三方法配对机制验证参数与启动命令

- **用户请求与授权范围：** 用户要求给出一组合理MNIST参数，检验能否出现教授描述的DPCO效果。本轮读取两级协议、最近记录、MNIST预设/实现/文档和launcher，提供实验方案与nohup命令；仅追加本研究记录，不修改源代码、预设配置或启动服务器任务。
- **外部依据与边界：** 核对Adam原论文Algorithm1（https://arxiv.org/pdf/1412.6980）给出的通用起点alpha=.001、beta1=.9、beta2=.999、epsilon1e-8；核对Keras官方MNIST CNN示例（https://keras.io/examples/vision/mnist_convnet/）采用原生28×28×1、uint8/255、Adam/B128及15轮干净训练。Keras示例网络大小、dropout、validation划分与本项目不同，不能转用其99%性能或15轮结论作为本项目带噪DPCO保证。教授完整模型/噪声/超参数未知，本次是其选择性学习现象的机制验证，不称严格配方复现。
- **预先固定的首轮方案：** 使用用户已选择的mnist_cnn无BN/dropout模型、60000训练/10000官方test、20%精确固定对称错标（12000条，其余48000），noise_seed=0独立于训练seed。三方法adam/dpco_filter/dpco_flip，贡献权重分别全部保留/负align0/负align−1；三个训练seed42/43/44，共九个串行run，各seed的方法共享初始参数/数据顺序/污染图。B128，50个drop-last遍历=468×50=23400步，Adam(.9,.999,1e-8)、峰值lr.001/cosine至1e-6、L2=0、eval每468步、checkpoint每2340步并开启保存。50轮是为同时观察初期学习与后期错标记忆的预设探索预算，不是论文规定或保证充分收敛。不按test峰值早停选模型。
- **reference选择说明：** 本轮命令显式覆盖预设beta_ref=.99为.9，采用名义约10步的较短历史作首轮起点，以减少历史滞后的可能影响；这是设计判断而非MNIST性能证据或教授指定值。Adam beta1亦为.9，但reference与Adam moments仍为独立状态；不因此把两者合并。仅一个beta下的结果不能断言DPCO在所有历史设置中成功或失败；后续敏感性另行预先设计，不反复按官方test调参。
- **评价判据：** 优先查看group_evaluation.csv中固定clean原标签CE下降且学习未被明显牺牲；corrupted_noisy对错误标签CE较普通Adam维持较高、错误记忆较少；同时corrupted_original对原标签的CE/accuracy表现更好或保留有效学习，以区分选择性抑噪与整体欠拟合。官方test CE/top1为辅助泛化结果，报告固定最终预算和完整曲线，三训练seed做配对均值/波动；不能只凭整体train CE高或corrupted_noisy高判成功，也不要求clean训练CE必须低于已近零的过拟合Adam才承认选择性现象。固定noise_seed0的重复只覆盖该污染图下的初始化/顺序波动，不覆盖不同污染实现的方差；单组不出现现象不反驳教授其他条件下的报告。
- **启动与输出：** 提供绝对服务器Python /home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python，配置configs/tpu/mnist/dpco_adam.toml，显式模型/生成比例/实际20%断言/noise_seed与训练参数；--download在首次运行准备和校验缓存，后续复用。外层nohup、stdin=/dev/null、stdout/stderr到results/logs独立套件日志、后台&、保存PID并提供查看命令；run_suite --wait串行九任务并保留每run files/runs/logs与子进程日志。三方法套件不共用--negative-weight -1选项，以免与filter的0别名冲突。假定沿用现有四本地TPU拓扑，异机路径/拓扑需匹配实际环境，不隐式GPU/CPU回退。
- **验证与未完成项：** 本地完整CLI加--dry-run解析九份plan，逐一断言seed顺序、方法负权重None/0/−1、DPCO beta .9、CNN、23400步/B128、Adam/lr、20%生成与断言、noise_seed0、download、eval468、checkpoint2340及关键child参数透传，全部通过；核算floor(60000/128)=468及23400/468=50。文档diff检查通过。没有下载真实数据、运行模型测试或正式训练、改变预设、提交或推送；有效性结论待用户实际实验。


### 2026-10-30 / DPCO-0031：澄清accuracy最终目标与CE代理/贡献目标，调整MNIST评价优先级

- **用户请求与范围：** 用户指出不能只比较loss，认为本身优化目标是acc。本轮核对两级协议、最近记录、组会转写05:17–11:42及当前任务/EMA/评价源码，作概念澄清并记录研究评价优先级；未获得新代码修改授权，仅追加本文件，不改算法、命令或正在运行的任务。
- **会议与实现事实：** 组会09:42–10:48强调loss低未必效果好；10:48–11:17明确指出accuracy不可直接作为通常的可微目标，需要surrogate。当前tasks/classification.py计算逐样本CE及预测错误/accuracy统计；methods/dpco_history.py的g_i来自CE梯度，r是raw CE batch-gradient EMA，评分r^Tg_i决定权重；更新仍是编辑后梯度交Adam，未对accuracy求梯度。给定r的独立贡献子问题可写为max_{w_i∈[-1,1]} r^T(Σw_i g_i/B)，它不等于max accuracy，也没有保证多步test accuracy单调提高。0–1 top1在预测类别不变区域对参数局部为常数、通常梯度为0，在决策切换处可不连续/不可微，不能把现有CE梯度叫accuracy梯度。
- **解释与研究边界：** 最终任务追求更准确分类，可以把accuracy视为任务层面的优化目标；代码层面的可微代理、给定reference的贡献优化和最终评价指标必须区分。CE下降与accuracy提高不等价，正确类别置信度变化而argmax不变时CE可变、accuracy不变。DPCO也不保证是某个固定普通CE的标准下降法。各方法不同signed/加权训练目标值不宜直接排序，但同一固定数据/标签上的普通CE仍是可比的概率预测质量度量，无需丢弃。此前CIFAR Flip较低test CE伴随明显较差top1说明二者不能互代，不能据低CE宣称分类性能成功。
- **追加调整DPCO-0030评价表述：** 其“官方test CE/top1为辅助”作为整体算法优劣排序过弱，按用户强调的任务目的修正为：固定最终预算下干净官方test top1及三训练seed配对结果是总体分类效果主指标；不按test峰值选checkpoint或据test调参。教授controlled verification仍需分组机制证据：clean原标签accuracy保持良好，corrupted_original accuracy/CE说明是否保留真实类别学习，corrupted_noisy accuracy/CE说明错误标签记忆；低噪声标签accuracy本身也可能来自整体欠拟合，必须联合clean与test判断。train观察标签accuracy越高未必越好，因为含12000个错标签。只复现选择性现象与证明总体泛化优于Adam为不同结论，需分别报告。
- **现有输出与验证：** evaluation.csv已含val_top1（当前为官方test）、val_ce；group_evaluation.csv已含各组top1/ce；train_steps.csv已有train_top1。无需改动上一轮MNIST启动命令或新增记录功能，即可按上述优先级分析。静态核对字段与源码一致，文档diff检查通过；没有运行新测试/训练、提出新的性能保证、提交或推送。原标签仍仅用于观察，不进入训练reference或权重。


### 2026-10-30 / DPCO-0032：核实教授MNIST口头报告与flip的证据层级

- **用户请求与范围：** 用户要求再次回看组会，概括教授已做的MNIST结果、是否使用flip。本轮重读两级协议、近期记录及两份组会MD的相关段落，仅追加核对记录，不修改代码/配置、实验命令或原会议文件，不启动训练。
- **明确记录：** 转写31:27–31:44（第147行）对“reference借鉴Adam确定后，闭式解还是flip，所以保持flip去做？”明确回答“对”；这证明教授认可后续方法保留flip。32:30–33:26（第155/159行）提出controlled verification希望clean loss低、corrupted loss高；33:40–33:52（第163行）说“这些实验我已经做完了……在MNIST上已经跑完了”，确认其MNIST简化实验已经做过。33:52–34:21（第165/167行）讨论SGD交替flip与建议先Adam，34:35–35:10（第171/173行）布置discard/flip和不同signal对照。
- **上下文推断与限制：** 围绕flip及clean/corrupted受控行为讨论后报告MNIST完成，可合理推断其小实验与该贡献干预方向相关，flip很可能是其中操作；但“已完成MNIST实验”的一句话没有逐项列出本人实际使用的reference、flip频率、optimizer或baseline，不能将上下文推断升级为完整配方事实。尤其不能断言已验证的就是当前raw含当前batch EMA+每step整梯度Flip+Adam，或宣称已证明该组合在accuracy上优于Adam。记录也没有明确逐指标公布其MNIST成功程度、数值或曲线；“clean低、corrupted高”首先是明确实验要求，其已完成相应现象验证的理解来自语境。
- **未知结果与避免混淆：** MNIST具体模型、噪声类型/比例、beta、lr、训练轮数、clean/corrupted loss标签口径、实际accuracy及相对Adam提升幅度均未记录。建议先Adam不等于其本人已完成的全部MNIST实验必用Adam；后续引入历史signal不等于本人此前MNIST必用该历史版本。本文此前“小规模正面验证”的说法应按此口头报告和上下文推断理解，不能作为已经取得原始数值结果的表述。两份MD是同次会议的转写与整理，不算两次独立实验佐证。
- **验证结果及未完成项：** 已逐项核对原转写时间戳与总结中的来源注记，最终回答明确区分记录/推断/未记录；git diff --check通过。仅本文件追加，无代码修改、模型测试、训练、提交或推送；教授实际运行配方和结果仍需原始代码/图表或进一步信息才能确认。


### 2026-10-30 / DPCO-0033：读取MNIST九组CSV，分析train accuracy与选择性抑噪

- **用户请求与范围：** 用户导入Results/Minist结果，要求先读数据分析，更多关注train acc，初步认为Flip未达到教授预期，之后自行画图。本轮只读分析及生成研究报告/派生JSON，不修改训练/绘图源代码、配置或原始CSV，不启动训练。读取两级协议与近期记录，应用Spreadsheets技能的只读/科学数据分析规范，原始记录保留；初次外层git status失败后在DPCO内核对状态，已有本文件未提交记录完整保留。
- **输入完整性：** 目录只有51份CSV，无metadata/log/checkpoint或原标签缓存。九组为Adam/Filter/Flip×seed42/43/44，均summary completed，23400连续更新、2995200样本出现；合计210600行train_steps、459行evaluation、1377行固定分组、140400行dpco_steps。每步B128，lr逐项符合cosine.001→1e-6，51个评价节点为0至23400每468步，50个drop-last数据遍历；clean48000、corrupted12000、test10000。实际负权重日志0/−1、EMA计数随step正确；同seed三组的初始test及分组指标完全一致。目录标记CNN/20%对称污染/noise_seed0/beta_ref.9与命令一致，但不能仅由命名认证模型、代码版本、完整优化器设置、噪声指纹或硬件；无完整配置元信息的限制已明确记录。
- **最终accuracy（3训练seed均值）：** 末468步有效样本加权online观察标签train acc为Adam83.3561%、Filter80.2311%、Flip78.6425%。23400步冻结模型全训练集观察标签acc为83.3644%/80.2372%/78.6494%；clean原标签acc99.2611%/99.4340%/98.2139%；corrupted错误标签acc19.7778%/3.4500%/.3917%；corrupted原标签acc75.5139%/94.5361%/97.7889%；全训练集原标签acc94.5117%/98.4544%/98.1289%。固定最终test为93.3533±1.1705%、97.7333±.0643%、98.1267±.1401%（±样本标准差ddof=1，不是CI）。当前Flip相对Adam总体原标签训练和test均改善，不能概括为方法完全未产生预期抗噪效果；clean及全原标签训练acc的最优方法并非Flip。
- **train acc精确分解：** 对同一冻结模型，观察标签acc=.8×clean_acc+.2×corrupted_noisy_acc；原标签acc=.8×clean_acc+.2×corrupted_original_acc。Flip观察标签acc比Adam低4.7150个百分点，其中clean代价贡献−.8378个百分点、减少匹配错误标签贡献−3.8772个百分点，约82.2%的表面下降来自后者；这是指标恒等式分解，非因果归因。完美预测原标签的分类器在20%强制错标下观察标签acc为80%，这是参照值而非观察标签acc上限，记住错标签可以更高。用户train曲线较低的观感在观察标签口径成立，不能直接作为真实分类能力差的证据。
- **配对与阶段变化：** Flip最终test相对Adam在42/43/44种子分别+5.41/+3.57/+5.34个百分点（均值+4.7733），相对Filter+.23/+.61/+.34（均值+.3933）；三次方向一致，但不能由三训练seed宣称普遍显著性或跨污染图稳健性。第5轮平均test：Adam98.55%、Flip94.27%；第10轮98.5433%/95.6833%；第50轮93.3533%/98.1267%。Adam观察标签train acc持续升高同时corrupted原标签acc从第5轮98.2472%降至最终75.5139%，与后期增加错误记忆/泛化变差一致；Flip明显学习慢，clean最终较Adam低1.0472个百分点、较Filter低1.2201个百分点。仅描述事后均值曲线：Adam峰98.6567%在第8轮、Filter98.5167%在第15轮、Flip98.1367%在第47轮；不能用这些test峰值选模型。当前最终预算优势不证明优于有独立validation停止/正则化的Adam。
- **loss与教授机制：** 最终clean CE均值Adam/Filter/Flip=.17350/.22895/.31749；corrupted错误标签CE=2.29435/3.11023/3.74913；corrupted原标签CE=.79408/.44120/.34167。Flip保留较高正确类别accuracy、几乎不匹配污染标签且最终test改善，为选择性学习提供正面证据；同时clean CE与clean acc未最优，不声称全面超越或严格复现教授未知配方。全程负alignment平均Filter38.2880%、Flip41.8589%，明显不是只处理20%错标的分类器；正确标签编辑不自动等于贡献优化错误。Flip编辑方向范数/原始范数均值约2.0695，不能以梯度范数抵消消失解释变慢。无逐样本分数及checkpoint，不能识别precision/recall、每次编辑净收益或唯一失败/成功机制。
- **研究产物与来源：** 在results/files/mnist_review_20261030_fhcubve_/新增analysis.md、analysis_data.json，含最终/逐seed/逐轮accuracy、CE、配对差异、干预统计和来源链接；JSON保留459条派生逐轮记录、51份输入的相对路径/SHA256/行数与限制。未生成图表、XLSX或改原CSV；临时分析代码未写入项目源文件。报告表格结构已校验，原数据前后SHA256全部一致。
- **验证及后续边界：** 独立将group top1×sample_count还原整数正确数，重算全train两口径；将末轮逐batch top1×128还原整数后重算online acc，均与主计算一致。summary final字段逐项等于最终evaluation，组节点完整、lr全23400步、dpco计数与窗口总步数检查通过，git diff --check通过。后续优先画固定分组accuracy和test完整过程，同时标明online观察标签口径；如新增独立validation或历史/编辑消融需另行授权。没有运行模型、启动训练、提交或推送；不据此改掉Flip或追逐test调参。


### 2026-10-31 / DPCO-0034：澄清MNIST两套训练accuracy及教授clean subset与validation的区别

- **用户请求与范围：** 用户批注前轮accuracy表，询问为何Adam train acc高而Filter/Flip“全部训练集按原标签”更好，且教授要求clean loss是否指clean validation。重读两级协议、最近记录、组会原文及总结，核对当前评价/指标源码与已有MNIST分析报告。本轮只解释和追加记录，不修改源代码、配置、原始CSV或实验方案，不启动训练。
- **两个标签口径：** 本次60000张MNIST训练图中48000个标签保留、12000个标签被固定替换。online train_top1使用加噪后的实际训练标签，模型匹配错标签也计正确。“全部训练集按原标签”是冻结最终模型后，对同样60000张训练图片按加噪前原始MNIST标签作离线评价，由clean和corrupted_original的正确数合并，不是额外数据集、重新训练或训练reference使用真值，也不等同未见数据的泛化accuracy。clean组只指未污染的48000个训练样本，不能把60000张按原标签评价都称为clean subset。
- **指标核对与例子：** 一张原标签7、训练标签被改成2的图片，预测7在观察标签train acc中算错、在原标签accuracy中算对；预测2则相反。冻结最终模型的观察标签acc=.8*clean_acc+.2*corrupted_noisy_acc，原标签acc=.8*clean_acc+.2*corrupted_original_acc。Flip两者约78.65%/98.13%；Adam83.36%/94.51%，排名不同源于答案标签不同，不是数据矛盾。前轮表第一行使用末468步online均值（Flip78.64%），与最终模型完整评价78.65%的小差异另来自时间口径；本轮明确说明两点。
- **教授原话与未知划分：** 转写32:30–33:26及34:25–34:35要求clean subset loss低/下降、corrupted subset loss高，未明确说必须为独立clean validation，也没有完整规定clean subset来自train还是held-out集。当前以污染训练集内未污染48000样本实现，是具体落实选择，不应冒充会议已指定。全文唯一validation出现在15:43–16:24的DIVE reference来源介绍，不能据此推成此次实验必须用clean validation。独立validation用于调参/早停是助手额外建议，应与教授明确布置区分。
- **按教授clean-loss要求收紧评价：** 最终clean CE为Adam.17350、Filter.22895、Flip.31749，clean accuracy为99.2611%/99.4340%/98.2139%。因此就“同样未污染训练子集上Flip的loss是否低于两个对照”而言，当前最终结果没有胜出；Flip的clean loss下降且错标签记忆较少、最终test准确率更高属于其他明确观察，不能替代clean loss比较。教授原话“低/降低”未给出精确数值阈值或普遍相对优势承诺，不把末点比较扩展为整个理论被否定。此前“部分机制现象得到支持”不能解释成已经证明教授要求的全部比较均成立。
- **验证与未完成项：** 核对noisy_classification_evaluation三组均使用split=train、primary使用官方test，以及train_top1来自观察标签预测。现有evaluation.csv的val_top1为历史字段名，当前实际含义是官方test，未单独留出clean validation；其命名不能证明存在验证集。数值沿用前轮已独立复核的报告，无新增CSV处理或模型测试；git diff --check通过。仅本文件追加，未绘图、训练、提交或推送。


### 2026-10-31 / DPCO-0035：讨论clean训练CE三图数据来源与原标签accuracy的定期评价

- **用户请求与边界：** 用户拟画clean训练子集CE对step、对wallclock，以及全部训练集原标签accuracy对step，明确先讨论、不改代码，并询问原标签acc是否随每一步validate。本轮重读两级协议、近期记录，核对experiment.evaluate、评价触发、evaluation/recording字段及当前MNIST实际CSV，仅追加研究记录；不修改训练/绘图代码、配置或原始结果，不生成图。
- **评价频率与含义：** MNIST本次记录节点为0、468、936、…、23400，共51次，每次对冻结的当前参数作完整固定分组前向评价，不更新模型参数、Adam moments、EMA reference或BN状态；不是每个训练step都对60000张原标签做评价。每步记录的train_steps.train_top1是当前batch按观察标签的online指标，不能替代原标签训练accuracy。此处validate指训练数据审计评估，不代表独立validation集。
- **整体原标签accuracy重建：** group_evaluation每节点记录clean的48000张及corrupted_original的12000张完整原标签评价；二者分组互斥且覆盖60000张。按实际sample_count加权：(n_C*acc_C+n_N*acc_N_original)/(n_C+n_N)，本次为.8*acc_C+.2*acc_N_original，再乘100画百分数。整体值由已有分组准确合并，不是现有CSV里直接名为original_train_acc的一列，也无需重跑训练/模型。不得将corrupted_noisy误并入此图，亦不得把原标签评估理解为模型训练用原标签。
- **三张拟议图：** ①横轴记录step，纵轴group=clean的普通CE均值；②相同clean CE，横轴训练循环累计wall-clock分钟；③相同step节点，纵轴上述全训练集原标签accuracy。三者是训练集审计曲线，不冒称验证/测试集曲线；评价样本和标签固定，CE不使用signed/贡献权重。已有数据只能画51个真实节点及连接线，不插值宣称每步全量评估；当前先保留原始节点，平滑如后续加入应按评价点计并明确窗口，不照搬每步train曲线的100点窗口。
- **时间来源与范围：** group_evaluation没有elapsed_wall_s，需要同一run按step与evaluation.csv严格一一对应，取后者elapsed_wall_s/60。experiment.evaluate在同一state完成primary及全部group评价后记录该elapsed，表示这批评价指标可获得的累计时间，避免以train_steps的host dispatch时间冒充相应全量评价完成时间。横轴包括循环内JIT、评价、保存及训练开销，不包括进入实验循环前的TPU启动/下载准备/模型初始化；step0实际评价完成时间可大于0，不私自逐组平移清除开销。若要active训练时间需另用timing的train_window累计，属于另一时间口径，不能混标wallclock。
- **多seed统计建议与验证：** step图可在相同节点对42/43/44求均值±样本标准差；各seed墙钟节点不同，后续实现需明确采用各seed原时间线、平均节点时间还是同时间网格重采样，不能把同step误称同wallclock。静态核对当前Adam42的CSV为153条group与51条evaluation，字段/节点与上述一致；无需新测试或正式运行。git diff --check通过；未实现新图、改算法、启动训练、提交或推送。


### 2026-10-31 / DPCO-0036：讨论四张分组评价图与三seed显式路径接口

- **用户请求与授权边界：** 用户在上一轮明确先讨论、不改代码的基础上，补充全训练集原标签accuracy对wallclock图，询问现有plots脚本如何传入每方法三个seed目录，并贴出原nohup命令。本轮只读现有入口和协议、提出待实现接口并追加本记录；未将“可以接入”的讨论解释为新增绘图功能的明确代码授权。
- **现有能力核对：** plots/plot_dpco_training.py目前每方法一个--adam-dir/--flip-dir/--filter-dir字符串，读取train_steps.csv与timing.csv，画观察标签下的online训练指标；不读取group_evaluation.csv，也不汇总seed。当前argparse默认为store，重复同一参数只保留最后一个值，不能直接复制三次旧参数后声称已经画了三seed。路径解析现支持绝对路径、相对DPCO根目录路径和results/files下单个run名。Git起始只有本文件的历史追加变动，保留全部原文。
- **建议接口（尚未实现）：** 保留三个单数参数名，改为可重复收集显式run路径，每方法传42/43/44三个目录；单次传入仍兼容原命令。不按最新时间或最佳结果自动选run，不要求搬动或重命名原始目录。建议增加--plot-set clean-original选择新的四图组，保留旧训练指标入口；--eval-smooth-window 1按评价节点计且默认不平滑，独立于旧--smooth-window按训练更新计的定义。上述新选项及多路径语义在当前代码不可用，给用户的参数片段是拟定接口，不是可立即运行的命令。
- **四图定义和输入：** clean_train_ce_vs_step/time与original_train_accuracy_vs_step/time，纵轴分别为固定clean训练子集普通CE、全训练集原标签accuracy百分数，横轴为step或训练循环wall分钟。每run读取group_evaluation.csv，按sample_count合并clean与corrupted_original的top1，再与同run的evaluation.csv按step严格匹配elapsed_wall_s/60，不使用corrupted_noisy或train_top1替代。当前MNIST每run为51个真实评价节点，无需重训练；原标签指标是训练集审计而非独立validation。
- **三seed统计建议：** 每seed先独立重建完整原标签accuracy，再对同step取等权均值及ddof=1样本标准差。墙钟图建议在该方法三个seed的实际时间共同覆盖区间上对记录点作分段线性插值后计算等权均值及样本标准差；每处统计保持n=3，不外推、不把训练时长归一化成相同终点，明确插值只服务对齐而非新增测量。可保留三个seed原始淡线帮助核对；标准差阴影仅描述这三个训练seed的离散度，不是置信区间、不包含未运行的新噪声划分。数据/噪声/方法/seed配置核查应尽量使用metadata，缺失时说明目录名与显式参数的证据限制；不按输入顺序自动断言seed配对。
- **输出与未完成项：** 拟沿用Adam蓝、Flip红、Filter橙虚线、5×4英寸与axes高宽比.8，PNG/PDF及派生曲线和manifest输出DPCO/results/figures/<comparison_name>/，保留原始CSV与既有产物。原nohup后台包装在实现后可沿用，但当前没有执行依赖安装、绘图或服务器任务。仅静态接口核对与文档diff检查，不运行算法或渲染测试；待明确授权后再实现、验证并提供可执行九run命令。


### 2026-10-31 / DPCO-0037：实现多seed固定分组四图，并用MNIST真实结果验证

- **用户授权依据：** 用户明确要求“按照沟通来进行代码修改”，授权落实DPCO-0035/0036讨论的四图、显式多run输入、三seed统计、评价点平滑和原有绘图兼容。授权不扩展为训练算法、评价数据生产或实验超参数变更。本轮先读两级协议及近期记录；起始只有本文件历史追加改动，全部保留。
- **涉及文件：** 修改plots/plot_dpco_training.py、plots/README.md，新增tests/test_dpco_evaluation_plots.py，追加本文件；真实派生图保存在下述results/figures新目录。没有修改src、configs、依赖声明、既有结果或历史绘图入口scripts/plot_training_results.py。
- **入口与兼容：** --adam-dir/--flip-dir/--filter-dir改为可重复收集路径。新增--plot-set clean-original与--eval-smooth-window（默认1，不平滑），旧命令默认--plot-set training仍每方法一个run、生成原五图；旧模式传多run明确报错，避免静默只用最后一个。路径解析仍支持绝对路径、相对DPCO根路径及results/files下单个run名；重复实际路径拒绝，输入选择完全显式。
- **指标与输入校验：** 新模式只读group_evaluation.csv与evaluation.csv，不需要train_steps/timing。clean普通CE取未污染训练子集；全训练集原标签accuracy按clean和corrupted_original的实际sample_count加权再乘100，排除corrupted_noisy，比例不硬编码。要求label_source=original、分组完整且唯一、同step样本出现数精确匹配、step/samples_seen/wall严格递增、固定组大小一致、有限数值及accuracy为0–1比例。clean为空报错；显式零样本corrupted允许空指标，不伪造缺组。各比较run的固定组大小必须一致。同方法重复run的step/samples_seen网格必须完全相同，拒绝因缺失节点而悄悄减少参与数。
- **统计、时间与平滑：** 每run先重建指标，再在run内按最近E个评价节点作后向等权平滑（前缀不足时用已有记录），跨run等权取均值和ddof=1样本标准差，阴影非CI；单run标准差导出约定0，不表示不确定性已估计。Step图仅使用真实节点。时间图取评价完成时elapsed_wall_s/60，同方法所有run时间范围的交集内，以记录时间并集及交集端点为网格，逐run分段线性插值后统计，保持每处全部run参与；无外推、时长归一化或虚构更密的真实评价。原始淡线保留各run完整实际记录范围，--no-raw可隐藏。保留非零step0时间，包含循环内训练/JIT/评价/保存，排除循环前准备初始化；新模式只接受wall。旧--smooth-window在新模式不使用。样本标准差带完整显示，未在0/100边界裁剪。
- **元信息与可追溯性：** 有对应metadata时核验方法/Adam/EMA/负权重并使用配置seed；从标准run名读出的seed标为directory_name_unverified，元信息与目录seed矛盾或同方法重复已识别seed报错。元信息缺失、身份未认证、重复数/seed集合或配置/数据指纹不一致明确写入提示和manifest；不同训练seed不会再被误当不匹配训练设置。没有从输入顺序虚构seed或进行配对显著性推断。导出evaluation_runs.csv、evaluation_step_curves.csv、evaluation_time_curves.csv以及最终完成manifest，保留源文件SHA256、样本数、实际节点/seed来源、插值/平滑/SD定义、共同时间范围和脚本哈希；输入改变时拒绝完成manifest。
- **绘图与运行文档：** 新增clean_train_ce_vs_step/time和original_train_accuracy_vs_step/time，各PNG/PDF，仍为5×4英寸、PNG220DPI、axes高宽比.8，Adam蓝实线、Flip红实线、Filter橙虚线。输出results/figures/<新名称>/，拒绝覆盖。README说明两种模式、九个精确run名和完整服务器nohup命令，明确base可指向另一CSV根目录，提供日志/PID/查看方式与绝对服务器Python；没有擅自运行该服务器命令。
- **验证过程与结果：** 新增7项有独立数值预期的分组/多seed测试，加既有8项DPCO绘图及10项历史绘图，共25项全部通过（9.672秒）。覆盖非等大小分组的原标签权重、评价平滑、空corrupted、ddof=1、共同时间区间与插值边界、缺组/重复/标签口径/单位/非有限/样本数/时间倒退拒绝、重复seed/网格不齐/时间无交集、元信息身份、九目录真实CLI渲染、四PNG/PDF、导出行数/seed、输出防覆盖、源字节不变、颜色比例、原五图兼容及无第三方依赖help。初轮环境缺matplotlib，另有测试fixture目录编号被误作seed导致身份检查失败；已修正fixture名称并复用已有/private/tmp/sage-plotting-venv，不安装或修改依赖。Python AST、README两段bash -n及git diff --check通过。
- **真实MNIST产物与独立复核：** 读取外层Results/Minist/files九run，在results/figures/mnist_clean_original_3seeds_20261031_114630_957697/生成四PNG、四PDF、三CSV及manifest。逐张查看四图，无标签/图例裁剪，颜色与比例正确。独立将源top1×组样本数还原整数正确数，复核全部459个实际评价节点的整体原标签accuracy及时间、153个step汇总点的均值/样本标准差；18个输入CSV的当前SHA256与绘图前记录完全一致。末节点Adam/Flip/Filter clean CE=.17350320095/.31749141439/.22894647217，原标签train acc=94.5116667%/98.1288889%/98.4544444%，与既有分析一致。每run51评价点，三seed阴影不增加独立观测或新污染图证据。
- **理论边界与未完成项：** 本次只改变离线画图及其派生数据，无训练目标、reference、翻转逻辑或泛化保证新结论。clean子集与全部训练集原标签评价均非独立validation；真实输入无metadata，方法/seed/模型配置认证限制已写入manifest。没有启动训练、重新生成噪声、覆盖旧产物、提交或推送；用户同步修改后的代码到服务器后可运行README命令。未实现多seed旧online训练五图或active时间分组图，超出本轮约定范围。


### 2026-10-31 / DPCO-0038：解释MNIST Flip学习较慢与固定终点抗噪收益

- **用户请求与边界：** 用户报告服务器生成同样曲线，观察Flip学习较慢、最终可剔除noise影响并高于普通Adam。本轮核对两级协议、近期记录与既有MNIST分析报告，澄清观察和机制推断；只追加本记录，不修改代码、配置、数据或图，不启动任何任务。起始Git工作区干净。
- **当前数据支持：** 本次MNIST小CNN、固定20%对称强制错标、50轮、三个训练seed的设定下，Flip前期clean学习较慢，固定最终预算下全训练集原标签accuracy为98.1289%、Adam94.5117%，官方干净test为98.1267%、Adam93.3533%。污染训练子集按错误标签准确率Flip.3917%、Adam19.7778%；同一批图片按原标签准确率Flip97.7889%、Adam75.5139%。这些联合现象支持“较少记忆错误标签并保留真实类别学习”的抗噪解释，用户的主要趋势判断成立。
- **对剔除表述的限定：** 当前Flip没有删除污染样本，也没有获得标签真值来决定权重；它翻转负alignment样本梯度。负alignment不是错误标签的可靠等价标记，现有聚合结果不能证明每次操作精准识别错标、全部噪声影响被消除，或该机制具有普遍保证。研究表述宜为“抑制错误标签记忆/减轻噪声影响”，其中作用机制仍是由联合指标支持的解释而非已完成逐样本因果识别。
- **收益与代价共同报告：** clean未污染训练子集最终accuracy为Flip98.2139%、Adam99.2611%，clean CE为Flip.31749、Adam.17350，不能将整体原标签收益说成clean子集也全面胜出。Filter全训练集原标签98.4544%略高于Flip98.1289%，但最终test为97.7333%低于Flip98.1267%；评价指标决定排序，不能泛称Flip在三组所有指标最好。Adam早期三seed平均test曲线峰约98.6567%，高于Flip平均曲线峰约98.1367%；当前同预算终点优势主要体现后期较少噪声记忆，不证明优于有独立validation选停止时间的Adam，更不能用test峰值选模型。
- **验证与后续：** 本轮数值引用先前完整CSV及独立整数正确数复核结果，已对照analysis.md中的终点、分组及过程统计，没有重新计算或重画；git diff --check通过。结果限定于当前数据、污染图、模型及训练预算；教授未知实验配方的严格复现与更大数据集收益仍未完成。若后续需展示机制，可在已有记录上补充corrupted_noisy和corrupted_original曲线，但本轮没有实施新增绘图或改变实验。


### 2026-10-31 / DPCO-0039：仅比较imagenet目录中CIFAR-100N的reference beta消融

- **用户请求与范围：** 用户新增results/imagenet/files，要求只比较files，分析beta调整和后续方向，询问相较MNIST失败是否因为40.2%噪声太高。本轮按两级协议读取近期记录，应用Spreadsheets只读科学分析规范；只读该files内35个CSV，没有读取同级logs/metadata/runs，没有连接服务器、修改训练或绘图代码、改变参数或启动新实验。开始时本文件已有上轮未提交追加，完整保留。
- **首先纠正对象和beta口径：** 实际六run均命名cifar100n_noisy_cifar_resnet34，固定分组29900/20100、train50000/test10000，故本轮讨论CIFAR-100N而非ImageNet；目录外壳名字不改变数据含义。run名标记Adam、Filter ref beta=.99、原始/复跑Flip ref beta=.99、Flip ref beta=.9/.999；这里是reference beta消融，不能混同Adam二阶矩beta2。用户要求仅files，故未借现存元信息认证命名中的架构/beta/seed；CSV本身没有Adam beta2等完整配置。两种大小写Results/results在本机为同一文件对象，未重复计数或挪动目录。
- **输入与过程核对：** 六run均summary completed，78000连续更新、B128、9984000样本出现，201评价节点为0至78000每390步，每次clean/corrupted_noisy/corrupted_original三组。nominal200个drop-last遍历对应epoch_progress199.68。每个更新lr逐项符合cosine .0003到.000001；分组数、标签来源、样本出现数、有限值、整数正确数、summary终点、dpco权重/计数与timing窗口连续覆盖核对通过。初始test及分组数值完全一致；35个CSV前后SHA256不变。只有seed42，两个ref beta=.99的所有非计时训练、测试、分组指标完全相同，不能当作两个独立训练seed或计算有效标准差。
- **固定78000步终点（accuracy均为百分数）：** Adam test51.24、clean100、N错误标签99.9602、N原标签.01990、全训练原标签59.808；Filter .99为51.20、99.9967、99.9502、.02985、59.810；Flip .9为48.66、99.9465、99.8408、.04975、59.788；Flip .99原始与复跑均为26.10、33.9097、9.6866、17.1542、27.174；Flip .999为20.81、26.9699、8.6070、13.8806、21.708。全原标签acc由两个互斥组的整数正确数独立合并，而非观察标签train_top1。Filter最终较Adam仅差.04个百分点（10000测试图的4张），不能描述为两种方法都最终远低于Adam。Flip .9比.99提高22.56个百分点、比.999提高27.85个百分点，但仍比Adam低2.58个百分点。
- **与MNIST机制的差别：** .9虽然恢复大部分分类能力，但最终几乎完全记住污染标签，N原标签只10/20100正确；未复现MNIST中N原标签高、N错误标签低的组合。高beta两组不匹配错误标签的同时，clean也仅学到33.91%/26.97%，说明当前预算下有明显学习受阻与选择性不足，不能直接称抗噪成功。Adam和Filter全训练原标签约59.8%与其记住观察标签一致；不能因其普通训练loss接近0就当作已解决错标影响。
- **过程与reference证据：** 第10轮test：Adam48.58、Filter34.79、Flip .9 16.92、Flip .99 4.84、Flip .999 5.06；第50轮分别47.74、47.94、50.04、9.04、8.79。Adam事后最高test51.82在第22轮，Flip .9同为51.82在第47轮；峰值只是描述，不能按test选checkpoint或调参。Flip .9/.99/.999全程ref/raw余弦均值约.4029/.1446/.0484，负alignment比例33.329%/40.338%/43.835%，方向/raw范数比1.0411/1.1662/1.1672；余弦对象是当前带噪batch而非clean oracle，范数未经过Adam。结果支持测试reference时效性/当前batch占比/编辑强度假设，尚不能隔离唯一机制；不是梯度范数接近0的直接证据。
- **理论解释补充：** EMA记忆尺度约1/(1-beta)，.9/.99/.999约10/100/1000步，为教学近似而非硬窗口。更大beta可降平稳随机方差，但可能增加表示/参数非平稳变化下的滞后。原始梯度EMA仍含错误标签梯度，不自动产生clean参考方向；当前batch被纳入reference时，H_t^Tg_i展开含(1-beta)||g_i||^2/B的正自项，因此减小beta既改变历史平滑，也增强self成分，不能仅由.9表现变好证明历史reference更准确识噪。r^TD的贡献目标也不自动保证经Adam后的clean CE或accuracy改善。
- **40.2%因果边界与外部依据：** 本批20100/50000为参考标签与original不一致，按run名noisy100 human用于训练；不是已认证的绝对语义错标比例。MNIST20%均匀强制错标与CIFAR100N40.2%human噪声，同时改变类别数、图像、模型、BN/增强和噪声结构，无法归结单一噪声率。核对官方原论文 https://arxiv.org/html/2110.12088v2 Table1及§3.2/4.2，fine噪声40.20%、human具有instance-dependent结构且不同于常用合成模式；官方发布 https://github.com/UCSC-REAL/cifar-10-100n 亦提供human与经验类别转移矩阵合成对照。不据此宣称40.2%不可学习或某方案已有收益保证。
- **后续优先级建议（讨论，未执行）：** 优先补充负alignment在clean/N中的计数与分布，获取P(q<0|C)、P(q<0|N)、P(N|q<0)，真值只审计、不作训练信号；当前总负数不能回推识噪精度。固定ref beta=.9，复用已有w=-1、增加w=0/−.25与Adam小对照，检查能否兼顾clean学习和N原类别保留，而非盲目扩beta网格。为区分噪声率和任务，现有MNIST接口可比较20%与40.2%；现有CIFAR clean100可作零训练噪声控制。CIFAR现支持--no-is-human经验类别转移合成控制，实际比例随采样变动，非均匀对称20%生成器；更多同图像的固定20/40.2%对称噪声须另授权实现，noise-percent仍只是实际比例断言。若诊断显示早期误翻有效信号，再考虑明确的Adam预热后启用Flip、EMA持续累积及状态保留策略，或冻结Adam尺度的reference评分变体，这些不是当前代码已验证能力。新增策略和诊断均不因本轮分析自动获得代码修改授权。
- **研究产物与验证结果：** 在results/files/cifar100n_beta_review_20261031_gpo7zrz9/保存analysis.md、analysis_data.json，包含六run完整201节点轨迹、逐阶段干预统计、终点/过程/峰值表、非计时重复检测、35原始CSV链接/行数/哈希、理论与未来实验边界；没有新增xlsx、图或代码文件。文档数字与JSON逐项对照，源哈希再次复核不变，git diff --check通过。未完成项是新对照实验、分组alignment诊断、多seed稳健性及真实ImageNet结果，而非把当前CIFAR结果当作ImageNet证据。


### 2026-10-31 / DPCO-0040：澄清污染子集原标签accuracy的分母及整体排序

- **用户请求与边界：** 用户批注DPCO-0039终点表，询问Flip ref beta=.99的17.15%是否高于Adam的.02%，以及是否意味着Adam按原标签总体准确率只有.02%。本轮只解释并追加研究记录，不修改源代码、配置、图或输入CSV；起始已有本文件未提交追加，保留原文。
- **重新定位与数值核对：** 原先results/imagenet/files已不在原位置；通过文件清单定位到外层Results/Cifar100N/files（本轮未移动目录），读取Adam与Flip .99复跑的group_evaluation.csv中step78000行。污染子集均为20100张，按原标签正确数量Adam4、Flip3448，分别.0199005%/17.1542289%；clean子集29900张，正确数量Adam29900、Flip10139，即100%/33.909699%。同一污染子集按human错误标签正确数量分别20092/1947，即99.960199%/9.686567%。
- **解释与证据范围：** 用户对“只在污染20100张训练图片、按原标签评价”这一项的比较理解正确，不能把它推广为整个数据集或测试集的Adam准确率。示例原标签狗、观察标签被改猫时，预测猫在观察标签评价算对、按原标签算错；同一训练图片被记住错误标签与新测试图片分类表现是不同口径。未污染29900张按原标签正确数与污染组相加，得到全50000张原标签accuracy：Adam29904/50000=59.808%，Flip13587/50000=27.174%；对应独立官方test为51.24%/26.10%（沿用前轮已核对evaluation终点）。
- **研究判断及验证：** Flip在污染子集保留原类别方面好于Adam这一有限观察成立，同时其clean组明显受损，故未转化为整体准确率优势；17.15%不能单独证明已经有效识别噪声或算法整体更好。整数正确数合并与此前完整分析一致，git diff --check通过；没有新实验、训练、绘图、提交或推送。先前报告源路径保留为历史记录，本条记录当前实际位置。


### 2026-10-31 / DPCO-0041：讨论以reference beta=.9为起点补齐Filter及分阶段细化beta

- **用户请求与范围：** 用户认为Flip beta=.9的clean accuracy99.95%是有效起点，希望平衡clean和污染样本原标签表现，询问能否先关注beta=.9的Flip/Filter、是否应在.9与.99之间微调。本轮按协议阅读近期记录，仅提出具体实验设计、澄清判据并追加本条，不修改代码、配置、启动脚本或原数据，不运行训练。已有未提交开发记录保留。
- **起点判断：** .9下clean99.9465%证明当前设置能够充分拟合未污染训练样本，但N原标签.04975%、N错误标签99.8408%，所以“可学习”与“抗噪有效”仍需区分。现有Filter仅有ref beta=.99，不能替代同beta=.9的Filter；补齐.9两种干预对照具有辨识价值，不承诺Filter一定改善。clean准确率接近饱和时，仍需同时观察普通CE、学习速度及N原类别保留，不能只比较末点99.95%。
- **分阶段小实验建议（未执行）：** 第一阶段固定模型、数据/噪声图、B、lr调度、Adam参数、总步数、评价协议与seed，比较beta_ref={.9,.95} × negative_weight={−1,0}，并保留Adam；已有.9 Flip及匹配Adam可复用，因此seed42需要新增.9 Filter、.95 Flip、.95 Filter三run。先用这一小对照判断增大历史权重是否开始改善N原标签acc而clean仍学得好。若.95呈现有益趋势，再补.98 Flip/Filter；.99已有结果作历史锚点，.999暂不重复。不将.9到.99之间默认存在成功/最优点；若中间设置只破坏clean学习而不形成N原类别保留，优先审计reference区分力/贡献强度，而非继续密扫。
- **为何中间值值得测：** 教学近似记忆尺度1/(1-beta)：.9约10步、.95约20步、.98约50步、.99约100步，故.9与.99相隔的是十倍时间尺度，不能因beta小数相近视为微小变化。该尺度不是硬窗口或独立有效样本数；beta调节历史平均、时效性和当前batch自项，不是直接控制clean/noisy两组权重，效果不保证连续、单调或线性插值。
- **量化权衡与选择边界：** 本次全训练原标签A=.598*A_C+.402*A_N，其中A_N按污染子集原标签而非human错标签评价；若A_C减少d个百分点，A_N需增加超过(.598/.402)*d≈1.49d个百分点才使该总体训练accuracy提高。这是固定分组指标恒等式，并非测试泛化保证或用真值定义训练reference。对候选同时报告clean CE/acc、N原标签CE/acc、N错误标签acc及完整训练过程；机制诊断不以总负alignment比例冒充噪声识别精度。seed42小扫描只作探索，候选还需其他seed和独立验证选择，避免据官方test反复选参数。
- **验证及未完成项：** 数值沿用DPCO-0039/0040已复核记录，本轮无新增CSV分析、模型检查或绘图；git diff --check通过。所有上述run均是建议，未发起、未改变正在运行实验，也未把讨论当作代码或训练授权。


### 2026-10-31 / DPCO-0042：提供一次后台启动三个CIFAR-100N补充实验的命令

- **用户请求与边界：** 用户批注“只需要新增3个run：.9 Filter、.95 Flip、.95 Filter”，明确要求一次性启动命令。本轮提供可复制服务器命令、核对参数并追加记录，不新增或修改脚本/配置/算法文件，不替用户连接服务器或启动正式训练。此前本文件未提交追加完整保留。
- **命令组织：** 从/home/jason_chia925_gmail_com/Project_DPCO/DPCO运行，单个nohup bash后台任务内设set -euo pipefail，显式Python /home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python。第一条run_suite调用--methods dpco_filter --reference-beta .9；第一组完成退出后，第二条调用--methods dpco_flip dpco_filter --reference-beta .95。两者均--mode train --wait --seeds42，合计正好三个run，顺序为.9 Filter→.95 Flip→.95 Filter，无并发抢TPU；任一失败停止后续。别名自动确定negative_weight=0/−1，不向混合Flip/Filter套件共传一个冲突权重。
- **固定设置与产物：** configs/tpu/cifar100n/dpco_adam.toml；显式dataset=cifar100n、model=cifar_resnet34、noisy100/human/noise_seed0、actual40.2%断言且no-download复用既有数据，steps78000、B128、cosine3e-4→1e-6、Adam(.9,.999,1e-8)、L2=0、eval390步、checkpoint1950步且开启保存。不传resume，从共同seed初始化开始新run，继承原TPU拓扑与已关闭的旧probes。顶层stdout/stderr独立写results/logs/cifar100n_beta090_095_<时间戳>.log，stdin=/dev/null，&后台启动，保存外层PID到同名.pid并输出PID/日志；每run沿用results/files、runs、logs自动唯一目录，旧结果不覆盖。
- **验证结果：** 用本地依赖轻量run_suite --dry-run解析两次入口，合计三计划精确为(beta,weight)=(.9,0)、(.95,−1)、(.95,0)，核对模型、数据、步数、batch、Adam子进程数值参数、noise断言、checkpoint、评价及wait均符合预期；dry-run不保留run目录、不接触TPU或数据。完整拟发服务器命令bash -n通过。首次数值参数检查因epsilon字符串1e-08与1e-8格式不同失败，改为按数值核验后通过；未修改程序。git diff --check通过。
- **理论与未完成项：** 本轮没有新实验结果或收益保证，仅落实已讨论的小消融命令。服务器真实运行由用户执行，数据路径/四本地TPU环境沿用此前实验，结果仍待训练完成后比较clean及污染组原标签行为。没有安装依赖、下载数据、提交或推送。


### 2026-10-31 / DPCO-0043：核对ImageNet-1K既有支持与ImageNet受控标签噪声路线

- **用户请求与授权边界：** 用户已做MNIST和CIFAR-100N，询问ImageNet能否加噪、下一步选择ImageNet-100还是1K，并认为当前只接入普通100。本轮按协议只读代码、检索原始资料和讨论实验路线；仅追加本记录，不修改数据/训练/绘图源代码、配置、脚本或依赖，不准备数据或启动训练。起始本文件已有未提交研究追加，保留全部历史。
- **现有能力核对：** src/config.py已有imagenet100/imagenet1k，num_classes100/1000；src/data/imagenet_hf.label_mapping在1K使用0至999恒等映射，在100使用固定类表；共享prepare可完整缓存/校验100和1K，src/data/classification.open_training_images已有两分支，共享ResNet/Adam/EMA DPCO训练。现成DPCO preset为configs/tpu/dpco/imagenet100_resnet50.toml，但通过--dataset imagenet1k可解析为1K。因此当前“只有100 preset”和“底层不支持1K”应区分，不能说1K必须从零接入。
- **明确缺失：** PreparedImages缓存读取只有既有单标签get接口，没有ImageNet原标签/观察标签双视图、加噪标签映射、clean/corrupted分组。现CLI/config的label-noise-percent仅允许MNIST；ImageNet非clean模式在train_classification显式拒绝，noise-percent不是ImageNet生成器。固定三组评价目前只在MNIST/CIFAR分支组装。故干净1K入口存在，不等于带噪1K和现有四图所需分组指标已可用；不提供会默默保持clean的数据命令。
- **规模与预算：** 项目当前固定ImageNet-100类表预期train126689、validation5000，不能泛称所有ImageNet-100拆分都同样大小；1K预期train1281167、validation50000，和官方HF数据卡一致。相同输入尺寸/模型、batch与遍历数，1K约10.1倍样本/更新预算，非已测量的10.1倍wallclock保证。当前prepare磁盘预估100约19e9、1K约170e9字节另3e9保留，是代码保守检查而非真实下载体积/可用空间认证；100流式筛选也可能扫描较大1K来源，不能承诺下载成本仅十分之一。没有检查服务器数据缓存、授权、磁盘或设备性能。
- **噪声设计建议（待授权实现）：** 第一优先为固定ImageNet-100原训练图和原标签、随机初始化ResNet50，比较0%基线与20%固定对称强制错标；明确选中round(p*N)个训练样本，从另外K−1类均匀选错误类别，noise_seed独立于训练seed，跨方法和epoch复用同一映射；验证集不额外加噪。100子集应先按固定manifest映射为0至99，再在这100类内改标，不能产生分类头之外的标签。保存原标签/观察标签/污染mask、实际噪声率及映射哈希，作为原缓存上只读标签视图/sidecar，保证恢复和跨方法配对一致。沿用clean、corrupted_noisy、corrupted_original三组及全训练原标签评价。初步有稳定机制收益再做40%及类别依赖/语义近邻噪声，后者需预先固定有意义类对，不能把随意相邻class id当语义噪声。原标签只是数据集基准标签，不认证其语义绝对无误。
- **100与1K定位：** 100先解决224分辨率/ResNet50上机制可学习、识别/编辑是否具有选择性，便于补对照和多seed；1K用于在候选固定后验证类别/样本规模扩展。规模更大提供不同范围证据，不保证方法更好或clean/noisy梯度更易区分；当前CIFAR问题不因升1K自动解决。建议100先固定噪声/模型/训练预算，与Adam/Filter/Flip对比，随后在1K做少量固定候选；相同标签视图方案可共享，主要新增工作在数据标签、配置校验、分组评价及指纹，不需重写ResNet/Adam/DPCO。若1K采用固定分层审计子集降评价开销，必须标明估计范围并按总体组权重合并，不把子集accuracy冒称完整训练集全量测量。
- **已核对公开基准及区别：** CNWL官方 https://google.github.io/controlled-noisy-web-labels/ 提供Blue Mini-ImageNet合成噪声、Red Mini-ImageNet真实web噪声、0–80%多等级及共同clean validation；它们是另一ImageNet衍生数据，不能等同项目当前固定100类图像拆分。WebVision原论文 https://arxiv.org/abs/1708.02862 为对应ILSVRC1000概念的超过240万web图片及人工验证/测试，可作为后续真实web噪声评估；它不是将原1K训练图逐个替换标签的同一数据，不能默认所有训练图都有完整原标签可用于同样C/N原类别审计。官方数据卡 https://huggingface.co/datasets/ILSVRC/imagenet-1k 核实1K规模。ImageNet-C官方 https://github.com/hendrycks/robustness 是图像corruption鲁棒性评价，非本项目当前研究的训练标签错标协议。未声称存在已接入/统一官方的ImageNet-N human40.2%接口，也未承诺URL图像当前全部可获取。
- **验证结果与未完成项：** 本地对ImageNet DPCO preset加--dataset imagenet1k --model resnet50 --methods adam dpco_filter dpco_flip --seeds42 --mode train --dry-run，三计划均正确解析1000类、clean、Adam及DPCO EMA(.9)权重0/−1，未读取图像/下载/访问TPU或创建run。此为配置/分派静态验证，未验证服务器真实完整1K训练成功。代码核对与git diff --check通过；本轮没有新增实验结果、改动模型/噪声或给出未经测量的收敛/时间保证。


### 2026-10-31 / DPCO-0044：澄清CNWL作者官方基准与ImageNet原项目标签噪声协议

- **用户请求与边界：** 用户批注CNWL Blue/Red说明，询问该论文加噪方式是否有ImageNet官方加噪方式。本轮只读协议、近期记录并查原论文与官方项目网页，解释发布主体及噪声定义；仅追加本条，不修改代码、配置、原始数据、脚本或正在运行实验。已有未提交文档记录保留。
- **官方一词的区分：** ImageNet/ILSVRC原项目提供原始图片、标注和挑战评价资料；在本轮可检索的官方发布中未查到统一指定20%/40%训练标签噪声生成协议。此为检索范围内结论，不将未查到写成不存在任何ImageNet衍生噪声基准。CNWL为Jiang等ICML2020论文作者发布的公开噪声基准，是作者官方发布，不是ImageNet团队制定的标准；CIFAR-100N的“官方数据/接口”同样指其噪声基准作者发布，不能把二者与母数据集创建方混同。
- **原论文噪声定义核对：** https://proceedings.mlr.press/v119/jiang20c/jiang20c.pdf §3.1及脚注2明确Blue采用对称标签翻转，每样本以p概率变成另外K−1个类别之一，排除原标签；转移概率对角1−p、非对角p/(K−1)。p表示标签改错概率而非从含原标签的全部K类重新抽样的概率；有限样本Bernoulli抽样的实际计数仍需核验，不能把本项目拟用round(p*N)固定总量选择与论文作者发布的具体split视为逐样本完全相同。Red保持类别标签、将所选干净图片替换为已人工核实不符合该类别的web图片，可能包含既有类表之外的内容，不是仅把同一张ImageNet图片标签改错。
- **公开协议与迁移边界：** 原论文Table1给出Blue/Red Mini-ImageNet均100类，Blue60000训练、Red50000训练、共同5000验证，多档p=0,5,10,15,20,30,40,50,60,80%；这不是本项目固定ImageNet-100的126689训练图，也不是全ImageNet-1K。作者项目 https://google.github.io/controlled-noisy-web-labels/ 及download页提供标注/分割说明。若要严格benchmark优先复用其发布split；若在本项目100/1K上采用相同对称噪声模型，应表述为“采用CNWL/相关论文的噪声模型在ImageNet上构造受控实验”，同时公开类表、采样、seed、映射和实际比例，不能称ImageNet官方噪声或现成Blue数据集的精确复现。
- **来源与验证：** 核对PMLR论文入口 https://proceedings.mlr.press/v119/jiang20c.html 、上述原PDF与CNWL作者官网。ImageNet官方网页直接访问超时，使用搜索返回的官方ILSVRC2012下载页 https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php 与 https://www.image-net.org/download-images.php 的发布内容作有限核查；没有编造官方背书、默认噪声标准或声称完成全网站审计。未运行代码/下载数据或生成新研究结果；git diff --check通过。


### 2026-10-31 / DPCO-0045：核实近期顶会对CNWL基准及衍生设置的实际使用

- **用户请求与授权边界：** 用户询问CNWL加噪方式是否被近两年顶会论文使用。本轮检索正式会议论文入口并核对实验段落，追加研究记录；没有获得或使用新的代码修改授权，不修改源代码、配置、脚本、依赖或实验数据，不启动下载/训练、提交或推送。保留本文件已有未提交记录。
- **2026年明确实例：** CARE: Class-Adaptive Expert Consensus for Reliable Learning with Long-Tailed Noisy Labels，ICML2026，PMLR306:68314–68334，正式入口 https://proceedings.mlr.press/v306/li26cy.html 。作者全文 https://arxiv.org/html/2605.23254v1 §3.1明确以Jiang等2020（参考文献即CNWL论文）的mini-ImageNet为基础，经指数下采样形成长尾分布并注入真实web噪声，命名ImageNet-LTN^r；Table4使用NR=.4、IF=10/100。此为CNWL来源/真实web噪声路线的衍生实验使用，不是仅在related work引用，也不能冒称未经改变的原始平衡Red split。主设置为预训练CLIP ViT-B/16+AdaptFormer、SGD微调20epoch，与本项目随机初始化ResNet+Adam的准确率不能直接横比。正式PMLR PDF的raw GitHub链接被浏览工具拒绝octet-stream，实验细节据作者arXiv正文、会议归属据PMLR正式条目交叉核验。
- **较早明确实例（单列2024，不能冒充2025–2026）：** CVPR2024 Learning with Structural Labels for Learning with Noisy Labels，官方PDF https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Learning_with_Structural_Labels_for_Learning_with_Noisy_Labels_CVPR_2024_paper.pdf 的§4.1明确使用CNWL Red Mini-ImageNet，20/40/60/80%噪声并缩放至32×32；从官方PDF的搜索提取正文核实，直接open曾返回403。ECCV2024 An accurate detection is not all you need to combat label noise in web-noisy datasets，正式入口 https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6511_ECCV_2024_paper.php ，作者全文 https://arxiv.org/html/2407.05528v1 §4.4、§6.1明确CNWL20/40/60/80% web噪声、替换图片的协议、32×32 PreActResNet18训练；作者官方代码 https://github.com/PaulAlbert31/LSA 同时列出CNWL下载与训练。两篇均为2024会议背景证据，按当前2026-10-31不归入2025–2026年份范围。
- **结论与证据边界：** 可确认CNWL真实web噪声基准及其衍生设置近期仍有顶会实际采用，不能从一个2026衍生实例推断它是2025–2026最主流或每篇直接复用原始split。本轮未核实2025–2026直接使用未经改变的Blue Mini-ImageNet的正式顶会实例；不将普通few-shot mini-ImageNet、像素Gaussian noise、仅引用CNWL、under-review ICLR2026稿件或2025公布的旧NCR专利当作该证据。未查到不等于证明不存在。
- **对DPCO的研究含义：** Red可作为真实web噪声外部验证；受控机制诊断仍优先能保留每张图原标签/污染mask的合成错标设置，以便复用clean训练子集loss及污染组原标签accuracy审计。Red的替换图可能含类表外样本，不能默认每个污染样本都有100类内的完整原标签。若选择CNWL用于可比benchmark，需固定作者split、分辨率、初始化/预训练及评价协议；若在项目现有ImageNet100/1K上自行加噪，应明确称自建受控设置而非原始CNWL复现。此为实验设计判断，不构成效果保证或实现授权。
- **验证及未完成项：** 按正式入口和实验段落区分会议年份、数据实际使用与衍生协议；未进行数据可下载性或官方代码运行验证，未新增算法实验证据。文档差异检查通过；是否接入CNWL或自建ImageNet噪声仍待用户决定并另行授权。


### 2026-10-31 / DPCO-0046：核对CARE公开数据构造实现及mini-ImageNet与项目100/1K的区别

- **用户请求与授权边界：** 用户批注CARE ICML2026，询问是否公开加噪方式、能否采用、mini-ImageNet与普通100/1K的差别。本轮阅读两级协议、近期记录、作者论文/仓库及项目ImageNet加载计数；仅追加研究解释，不修改代码、配置、脚本、依赖、数据或启动正式任务。已有未提交记录保留。
- **比上一轮更具体的公开实现证据：** 通过GitHub连接器读取qwq123-study/CARE的datasets/lt_mini_imagenet.py（blob160991e9aa04bf70621393fb7cb3f15defc41771）、datasets/mini_imagenet.py及configs/data/mini_imagenet_ir100_nr40.yaml、mini_imagenet_ir10_nr40.yaml。作者论文 https://arxiv.org/html/2605.23254v1 §3.1与Table4公开长尾+web噪声和40%、IF10/100；代码提供更具体、可审计的构造流程，但本轮未执行或验证数据下载可用性。
- **流程与追加澄清：** LT_Dataset读取split/blue_noise_nl_0.0为干净训练池；读取red_noise_nl_0.8，并按image id不匹配^n.*筛出web图，使用其观察标签组成噪声池并打乱。干净组按n_c=floor(n_max*a^(c/(K-1)))、K100指数采样，配置IF10对应a=.1、IF100对应a=.01；论文的IF为头尾数量比，代码imb_factor为其倒数。设保留干净图M，追加噪声图数量int(M/(1-p)-M)，p=.4时约为2M/3，使最终混合集合噪声占比约40%；不是把M张内40%随机改标签。red_noise_nl_0.8是噪声候选来源，不代表最终80%。若候选池不足，代码切片不会自动补齐，应核验实际样本数和比例；比例为总体目标，不保证各类同为40%。该实现是先长尾下采样、再追加web图，不能直接等同原始Red保持总量的替换流程；本条细化DPCO-0045的高层描述，保留原文。
- **可采用范围与可复现边界：** 可复用其数据构造设置来比较Adam/DPCO Filter/Flip，并引用CARE及CNWL；复用数据不要求采用CARE的CLIP多专家纠标算法。主实验CARE使用预训练CLIP/AdaptFormer，不能把其准确率直接与随机初始化ResNet对照。公开代码依赖已有split/all_images/validation_all布局及本机绝对class name路径，没有据README自动下载CIFAR的说明推断mini-ImageNet也能一键下载。采样含set转换及默认random_seed0，严格配对需保存实际样本清单/顺序、标签、seed、版本和实际比例；本轮未认证跨进程逐图复现。原标签字段返回同一label不能被当成web图正确类别oracle。
- **数据区别：** CNWL官网 https://google.github.io/controlled-noisy-web-labels/ 指向的mini工具 https://github.com/yaoyao-liu/mini-imagenet-tools 描述经典mini为100类、每类600、总60000、常用84×84，并说明few-shot的64/16/20类拆分；这不等于CNWL/CARE普通100类分类评价。CNWL原论文 https://proceedings.mlr.press/v119/jiang20c/jiang20c.pdf Table1的Blue训练60000、Red50000、共同validation5000；CARE衍生训练量随长尾采样和追加比例变化，不能写死60000或50000。项目src/data/imagenet_hf.py固定类表ImageNet100预期train126689/val5000，100类映射由manifest指定；1K为train1281167/val50000，和 https://huggingface.co/datasets/ILSVRC/imagenet-1k 一致。普通ImageNet100的类表是具体实验约定，不能只因同为100类就认为与mini相同或图像/分割可交换。普通100/1K保留原图尺寸，224通常是训练裁剪设置而非原数据统一分辨率；mini也可重采样/放大但不能恢复已丢失细节。mini默认不自带标签噪声，噪声来自Blue/Red或后续构造。
- **DPCO路线判断与限制：** 若复现CARE数据设置，应完整使用其40% web噪声及IF10/100，保持同数据跨方法配对。若当前只想隔离抗噪机制，先做类别较平衡的CNWL Red或已有ImageNet100上的受控合成错标，可避免同时引入长尾造成混淆；这些是不同实验，不能统一称CARE精确复现。CNWL100类web池不能未经对应类映射就当完整1K噪声池；100/1K若采用随机翻标签，应称相应合成噪声协议而非CARE真实web噪声。Red/web图可有类表外内容，clean loss与干净val可用，污染图全量100类内原标签acc不保证有定义。
- **验证与未完成项：** 作者四份源码/配置只读核验，核对项目计数，无运行作者程序、下载数据或创建新实验；git diff --check通过。尚需用户选择机制诊断还是CARE长尾真实web噪声协议，具体接入及数据可用性验证均未实施，本轮讨论不构成代码授权。


### 2026-10-31 / DPCO-0047：按用户自然噪声目标澄清CARE沿用CNWL的范围

- **用户请求与偏好：** 用户明确看重CNWL自然噪声，询问CARE是否沿用自然噪声。本轮只核对和解释并追加本记录，无代码修改授权，不修改程序、配置、数据，不下载或训练。后续数据方案优先围绕用户的真实web错标目标，不继续默认要求以自行均匀随机改标替代该目标。
- **核对证据：** 重读 https://arxiv.org/html/2605.23254v1 §3.1及 https://google.github.io/controlled-noisy-web-labels/ ，通过GitHub连接器再次读取作者datasets/lt_mini_imagenet.py第24–70行，blob160991e9aa04bf70621393fb7cb3f15defc41771。CARE的mini-ImageNet实验确实使用CNWL Red的web错标图片和既有观察标签，在干净部分长尾采样后追加噪声图片，不是重新均匀随机生成错误类别；其CIFAR合成实验另有对称/非对称/joint噪声，不能把所有CARE实验统称自然噪声。
- **核心解释：** 随机抽取哪些既有自然错标样本进入训练集，与随机生成样本的错误标签是不同操作。前者控制比例/组成而仍沿用自然来源的错误；人工控制比例及长尾后也不应称原始未筛选网络分布。自行组装数据不必等于均匀随机翻标签，CARE就是复用自然错标池的例子。
- **路线调整与边界：** 若用户主要验证DPCO对自然噪声的效果，优先直接采用CNWL Red官方split、共同干净验证集及同split的Adam/Filter/Flip对照；20/40%可作为建议起点，并不要求先做合成错标，也不需因引用CARE而同步引入长尾或其CLIP纠标方法。CARE长尾版本可作为后续额外难度，若精确采用则保持其样本构造/IF与实际比例。上述建议未实施且不保证DPCO性能改善；web噪声样本仍不保证具备闭集正确类别标签。
- **验证及未完成项：** 文献与作者源码只读交叉核对，git diff --check通过。自然噪声数据的下载可用性、具体split选择与接入仍未执行；未新增实验结果、提交或推送。


### 2026-10-31 / DPCO-0048：按CNWL作者发布划分接入Red Mini-ImageNet自然噪声训练

- **用户明确授权与目的：** 用户批注CNWL自然错标及可控比例，要求“按照cnwl的mini-imagenet数据集的论文官网的方式，将该噪声数据集加入训练……按照论文作者定义的数据情况，修改接入现在的dpco的训练代码，并将这个cnwl的数据集的基本描述计入开发训练md文档”。本轮授权覆盖数据准备/加载、必要配置与CLI、共享训练评价接入和文档/验证；不扩展为CARE长尾算法、Blue随机错标、DPCO/Adam公式变更或正式服务器训练。先读两级AGENTS及最新记录；起始只有本文件已有历史追加，完整保留，无提交/推送。
- **数据基本描述和前文精度补充：** 来源Jiang、Huang、Liu、Yang，ICML2020，Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels。官网 https://google.github.io/controlled-noisy-web-labels/index.html 、下载说明 https://google.github.io/controlled-noisy-web-labels/download.html 、论文 https://proceedings.mlr.press/v119/jiang20c.html 。Red Mini-ImageNet是100类、84×84 RGB的真实web检索错标基准，原论文以人工3–5人核验的不符合检索类别图片替换部分干净图，按作者发布的固定split控制0/5/10/15/20/30/40/50/60/80%档位。Blue是对称合成错标，非本轮目标；不自行改标、不引入CARE长尾、不将本项目另一份ImageNet100/1K视为同一数据。所有档位共用5000张clean_validation。补充DPCO-0044至0046中的“Red50000”概括：官方TFDS目录给出的精确训练数为0/5/10/15/20/60档50000，30档49985、40档50010、50档49962、80档50008，不能为凑整数比例调整发布成员；本轮按该公开基数校验并记录加载后的实际错标率。来源 https://www.tensorflow.org/datasets/catalog/controlled_noisy_web_labels 。
- **作者/官方接口语义核对：** 读取官方TFDS builder（tensorflow/datasets的controlled_noisy_web_labels.py，blob bbbc776dca8a8e704a8651831d0e1fd700665419）、其dummy标注schema及发布说明；采用image_id/class_id两列split、mini-imagenet-annotations.json的data列表/每图record列表及image/id、image/uri、image/class/label、image/class/label/is_clean字段。类别名接受官网class_names.txt与TFDS class_name.txt变体，保留原行号标签0–99。干净图原标签只用于确认Red未改变其类别；web图确认来自已发布错误标注，训练时仅使用split观察标签。metadata来源文件内容SHA256用于配对/追溯，官网未提供本轮可核实的固定zip校验签名，不能把本地哈希当作作者真实性签名。
- **新增文件：** src/data/cnwl_prepare.py（官方元数据、ImageNet/Web图片准备及按档位不可变缓存）；src/data/cnwl.py（只读数据/增强和audit分组）；scripts/prepare_cnwl.py（plan/prepare/verify）；configs/tpu/cnwl/dpco_adam.toml；run_cnwl_suite.sh；docs/cnwl.md（完整基本描述、布局、指标、nohup命令和限制）；tests/test_cnwl.py。修改src/config.py、src/data/classification.py、src/training/launch.py、src/cli.py、scripts/launch_tpu.py、scripts/train_classification.py、src/training/evaluation.py、src/training/results.py、pyproject.toml、README.md及本文件。未修改既有原始结果、模型实现或方法更新公式。
- **准备与读取实现：** 数据名cnwl，Red-only，按作者split保留行顺序/ID/观察标签；严格检查100类名、clean Blue0来源、web错误标注、train/val分离、唯一ID及公开基数，拒绝不完整/自定义划分冒充官方。原始ImageNet训练tar内每类按JPEG数字编号排序，再用mini ID后八位的1-based位置取图，不能把mini序号直接当原图文件名后缀；验证图片按原validation ID匹配。也支持已合法取得且保留官方ID的clean_images/all_images、noisy_images与validation_all目录。按官方TFDS采用OpenCV color decode、INTER_AREA84、JPEG quality72；运行时训练用ImageNet归一化及随机resized-crop/flip，干净评价/组审计使用完整84、无随机增强。只声明数据及预处理兼容，不声称复现MentorMix训练超参数/模型/方法。
- **下载、失败和恢复边界：** 默认本地源，显式--download才抓取官网zip及当前split需要的web URI，不自动下载需要同意ImageNet条款的train/val tar，不自动镜像/重新检索/替图。并行下载有界，全部图片缺失/解码失败有明确报错与失败ID清单，不静默删样本或生成随机标签；SQLite暂存按已完成行续做、成功文件复用，单个中断HTTP请求重下而非字节续传。metadata解压只读取指定成员，避免路径越界；中断后能从已有zip补齐元数据。完成后存prepared/red_20或red_40等独立目录，manifest含源metadata哈希、归档位置/大小/mtime、类别、实际计数、处理方式、成品DB哈希和completion标记，训练只读并校验DB哈希；每run数据fingerprint纳入恢复检查。原始源文件可能随web变化，成品哈希保证当前实际输入可追溯，不保证所有历史URI长期可用。
- **参数和启动贯通：** data.cnwl_noise_percent / --cnwl-noise-percent选择官方整数档位，默认20；已有--noise-percent仍是实际率断言，--label-noise-percent仍MNIST-only。固定Red要求noise_type=red、is_human=true（人工核验发布数据，观察标签仍源自web）、noise_seed0；0档clean，其余noisy，CLI同步mode。父套件/子进程/配置检查入口均传递新参数，run名包含red20_official_split等档位。支持既有resnet50/101及native-stem cifar_resnet18/34/50，输入固定84；默认ResNet50随机初始化、batch128、Adam3e-4 cosine→1e-6、78000步、评价390、latest1950、reference beta=.9，Filter0/Flip−1，均是沿用探索预设而非经过CNWL调优或论文最优配置。非CNWL的as_dict不添加新参数，保留原有metadata/恢复配置身份；切换数据集重置专属噪声参数，不改变原数据算法。
- **评价与研究边界：** 沿用train_steps.csv/timing.csv/evaluation.csv及checkpoint/恢复。group_evaluation.csv只含clean（已知原标签）和corrupted_noisy（web观察标签）两组，空组显式空指标。is_clean只划分审计组，不进入batch、不用于reference/编辑选择。网络替换图可来自100类外，正确闭集类别未知，因此不生成corrupted_original或全部训练集原标签acc，不用观察标签冒充真值。主评价为官方共同clean_validation，沿用不自动存best的协议；clean train loss与验证acc可用于研究，未把训练audit称独立validation。旧training绘图可用；原clean-original四图依赖完整真值三组，不能直接用于CNWL，新增CNWL专用图不属于本轮实现。
- **依赖与输出：** 新增cnwl可选OpenCV依赖并纳入test extra，运行时JAX训练不依赖TensorFlow/PyTorch；data extra供准备阶段NumPy/Pillow/certifi。为本地验证仅在/private/tmp/dpco-cnwl-testdeps装OpenCV4.14，不改用户全局或服务器环境。docs/cnwl.md给出安装、准备和三组训练的nohup后台命令，stdin/dev/null、独立results/logs日志及PID/查看方式、显式项目Python；代码开发没有执行服务器正式命令。准备报告位于results/files/cnwl_prepare_<时间>；训练继续files/runs/logs三级同run名。
- **验证结果：** 55项相关测试通过（含9项CNWL测试及原config/suite/MNIST/CIFAR测试），共约18秒；本地原始tar与canonical图片两种准备路径逐图像素/标签一致、数字排序而非字典序/文件名后缀、metadata zip读取及中断补齐、错误标签/重复ID/越界/篡改拒绝、web URI下载模拟及403报错、缺图不发布与299行暂存后仅重做1行、nominal选择与actual断言区分、训练batch排除audit真值、分组CE与独立NumPy核对、3种方法真实共享runner/updater/EMA/CSV/checkpoint及p50到final逐项精确恢复。训练集成测试使用CPU小模型和明确合成夹具验证通路，不声称ResNet50 TPU性能或真实自然噪声学习结果。首轮一个CE断言仅差5.72e-7浮点累加误差，改成显式float32合理容差后通过；没有调整算法。shell bash -n、官方40档prepare plan（50010/5000）、三组Red20 wrapper dry-run、CLI父子传递和git diff --check通过。
- **验证产物与真实数据限制：** results/files/cnwl_integration_check_20261031_155223/保存validation.json和three_method_plan.json，results/logs/同名目录保存tests.log。官网zip通过浏览和获批curl实测均HTTP403，故未完成真实标注包/全量图片获取或完整Red准备；原始ImageNet归档也未在本次任务获取。人工取得原官方材料后可按docs/cnwl.md运行准备，旧web失效图仍须恢复同一图片，不能静默删除后宣称原benchmark。未启动正式训练、未测TPU、未产生CNWL研究准确率，不作DPCO收益保证。代码接入及本地验证完成，后续真实源可用性/服务器准备与实验结果尚待实际执行。
