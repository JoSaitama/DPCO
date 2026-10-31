# DPCO 训练与固定分组评价曲线

`plot_dpco_training.py` 是独立的离线绘图入口，只需标准库和 Matplotlib，不导入 JAX、Flax、Optax 或 TPU。它不会修改训练CSV或checkpoint，也不会启动训练。原 `scripts/plot_training_results.py` 的历史绘图接口保持不变。

默认 `--plot-set training` 沿用原有五图模式，显式选择三个不同目录，每个目录需要完整、已刷盘的 `train_steps.csv` 和 `timing.csv`。新增 `--plot-set clean-original` 模式读取固定训练分组评价，每个方法可重复传入多个seed目录，详见下文。两种模式均支持绝对路径、相对DPCO根目录的路径，或 `results/files/` 下的单个run目录名；不自动寻找“最新”或“最好”的运行。若有对应的 `results/logs/<run_name>/metadata.json`，会核验方法标签，并报告训练设置差异；仅复制所需CSV也可以绘图，但会注明无法核对元信息。

在服务器DPCO根目录执行下面整段。替换三处 `YOUR_*_RUN` 为实际目录名。安装绘图依赖和绘图均在同一个nohup任务内顺序运行，日志/PID位于results/logs。

```bash
cd /home/jason_chia925_gmail_com/Project_DPCO/DPCO
mkdir -p results/logs
export DPCO_PYTHON="$PWD/../.venv/bin/python"
log="results/logs/plot_dpco_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c '
  set -e
  "$DPCO_PYTHON" -m pip install -e ".[plot]"
  exec "$DPCO_PYTHON" -u plots/plot_dpco_training.py "$@"
' _ \
  --adam-dir YOUR_ADAM_RUN \
  --flip-dir YOUR_FLIP_RUN \
  --filter-dir YOUR_FILTER_RUN \
  --smooth-window 100 \
  --time-axis wall \
  > "$log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid  LOG=$log"
tail -n 30 "$log"
```

本地可通过项目解释器运行同一Python入口；`--help`无需安装绘图或训练依赖。

## 输出

默认输出到 `results/figures/dpco_training_<时间戳>/`，`figures/` 与 `files/`、`runs/`、`logs/` 同级。可用 `--output-name experiment_a_w100` 指定该目录名；已有目录一律拒绝覆盖。图片、派生曲线CSV和绘图清单均保存在该任务目录。更换平滑参数时使用新的输出名，原CSV、旧图片与历史结果均保留；此前生成在 `results/files/plots/` 的结果不会自动搬移。

| 文件名（默认各保存PNG和PDF） | 横轴 | 纵轴 |
| --- | --- | --- |
| train_ce_vs_step | 完成更新次数step | 普通训练CE |
| train_accuracy_vs_step | step | 训练top-1，百分数 |
| train_ce_vs_time | 累计分钟 | 普通训练CE |
| train_accuracy_vs_time | 累计分钟 | 训练top-1，百分数 |
| ms_per_step_vs_step | 窗口末step | 训练窗口平均ms/step |

另外保存 `step_curves.csv`、`timing_curves.csv` 和最后写入的 `plot_manifest.json`，记录曲线数据、平滑参数、输入文件SHA256、脚本SHA256、方法颜色/线型、计时说明、加权平均ms/step和设置差异。仅在全部绘图和输入未变检查成功后生成manifest；没有manifest的输出目录可能是不完整任务。绘图前后输入文件改变会报错，请对完成的run或一致的文件快照绘图。

所有图画布为宽5英寸、高4英寸，绘图区高/宽固定0.8。PNG默认220 DPI，可用 `--dpi` 修改；`--formats png` 可只输出PNG。Adam为蓝实线、Flip为红实线、Filter为橙色虚线（`#ff7f0e`）。保留线性纵轴和原始异常值，不自动隐藏前期或裁剪高CE。

## 平滑与时间定义

- `--smooth-window W` 默认100，必须为正整数。Step图按最近W次更新的有效样本数加权平均，前缀不足W时用已有数据；`1`关闭额外平滑。
- 时间图以 `event=train_window` 的已完成窗口为基本观测单位，CE与accuracy对窗口覆盖的训练批按有效样本数聚合。平滑时往前纳入完整窗口，直到覆盖至少W次更新（开头不足时用已有窗口）；不拆分窗口，也不为每步插值一个精确完成时间。因此 `W=1` 仍显示原始窗口均值，时间图不是逐步设备完成轨迹。
- ms/step的原始值为 `1000 * duration_s / window_steps`；平滑值为纳入窗口的 `1000 * sum(duration_s) / sum(window_steps)`，不是对不同长度窗口的单步均值做简单平均。CSV中的`smoothing_steps`记录实际平滑跨度。
- 默认 `--raw` 同时画淡色未平滑线，可用 `--no-raw` 隐藏。原始数据始终保留在输出CSV；所有方法使用同一个W。相邻窗口长度不同会使时间图的实际平滑跨度略有不同，不能称为严格等时长平滑。
- `--time-axis wall` 默认使用timing窗口末的 `elapsed_wall_s / 60`。这是训练循环累计墙钟时间，包括循环内初始评价、JIT和先前评价/保存开销，不包括循环前的数据准备与模型初始化。窗口指标是区间平均值，横坐标是区间完成端点。窗口末正在开始的评价或checkpoint尚未发生，因此曲线末点不等于整个进程最终结束时间。
- `--time-axis active` 仅累计train_window的duration，排除独立的编译、评价及checkpoint阶段；它仍包括训练窗口中的输入处理和设备同步，不能称为纯TPU kernel时间。
- 不使用train_steps中的host异步派发时间戳作为逐步完成耗时，也不将所有timing事件的duration相加。完整train_window覆盖、step连续、时间单调与数值/单位会校验，缺失数据不自动补零或外推。

以上五图的训练CE/top-1对应当前观察标签；在CIFAR-100N中它们衡量带噪标签的拟合，不是独立测试或去噪效果。此模式保持每方法一个run；多个seed请使用以下分组评价模式。

## clean loss 与全部训练集原标签 accuracy：多 seed 四图

`--plot-set clean-original` 每个run只需 `group_evaluation.csv` 和 `evaluation.csv`，不要求逐步训练CSV或timing文件。每种方法的路径参数可重复出现，旧的单次传入仍然可用，例如：

```text
--adam-dir /absolute/path/adam_seed42 --adam-dir /absolute/path/adam_seed43 --adam-dir /absolute/path/adam_seed44
--flip-dir /absolute/path/flip_seed42 --flip-dir /absolute/path/flip_seed43 --flip-dir /absolute/path/flip_seed44
--filter-dir /absolute/path/filter_seed42 --filter-dir /absolute/path/filter_seed43 --filter-dir /absolute/path/filter_seed44
--plot-set clean-original --eval-smooth-window 1 --time-axis wall
```

原目录无需移动或更名。仅复制CSV时，方法依据显式参数，seed可从项目标准run名推断并标为未经元信息认证；元信息存在时优先使用配置seed并检查与目录标识一致。同方法重复seed、任意重复路径均拒绝。不能把同一seed的两次运行当作两个独立训练seed；没有可识别seed的自定义目录会保留为显式重复运行并提示身份不可核实，不按参数顺序猜测seed。

| 文件名（PNG和PDF） | 横轴 | 纵轴 |
| --- | --- | --- |
| clean_train_ce_vs_step | 真实评价step | 未污染训练子集普通平均CE |
| clean_train_ce_vs_time | 训练循环累计wall分钟 | 同一clean CE |
| original_train_accuracy_vs_step | 真实评价step | 全训练集按原标签top-1，百分数 |
| original_train_accuracy_vs_time | 训练循环累计wall分钟 | 同一原标签accuracy |

原标签accuracy对每个run、每个评价节点按样本数重建：

```text
100 * (n_clean * top1_clean + n_corrupted * top1_corrupted_original)
    / (n_clean + n_corrupted)
```

只用 `label_source=original` 的 `clean`、`corrupted_original` 两组，绝不使用 `corrupted_noisy` 或观察标签的 `train_top1` 替代。本次MNIST分别48000/12000，因此权重为0.8/0.2，代码不硬编码该比例。这是训练集审计评价，不是独立validation/test；原标签不进入模型训练。clean为空时拒绝画其未定义loss；corrupted组可以显式为零样本，此时其空指标不参与计算。

- **Step汇总：** 同方法重复run必须有完全一致的step/samples_seen节点。先对各run重建指标，再对run等权计算均值和样本标准差（ddof=1）；不同方法的节点可以不同，但会提示。固定组大小必须在各节点和各run间一致；缺组、重复节点、错误标签来源、非有限值、百分数误当比例或时间倒退会报错，不自动补零。
- **时间汇总：** 同一run按step和samples_seen严格匹配evaluation时间，取 `elapsed_wall_s/60`。各方法分别取所有重复run的共同覆盖时间区间，使用区间内各run记录时间的并集及区间端点作为网格，逐run分段线性插值后等权求均值和标准差。每个汇总点保持全部run参与；不外推、不缩放各run时长。插值是显示对齐，不是增加真实评价频率；曲线末端按共同覆盖截取，不能当作最慢seed的完成时间。
- **误差带：** 主线为均值，阴影为±1样本标准差，表示已有训练seed间的波动，不是置信区间，也不包含新噪声划分的不确定性。单run时标准差导出约定为0，不代表已估计不确定性。阴影不在0/100边界处截断，图轴自动容纳完整阴影；均值指标仍应在物理范围内。
- **原始曲线：** 默认淡线显示每个run在自己真实记录点上的完整原始轨迹；粗线和误差带采用上述汇总范围。`--no-raw`只隐藏淡线，不删除CSV数据。
- **平滑：** `--eval-smooth-window E` 默认1，不平滑。大于1时，先在每个run内对最近E个评价节点作后向等权平均，开头使用已有前缀，再跨run汇总。单位是评价节点，独立于旧 `--smooth-window`（本模式忽略后者）。本次51点建议先用1。
- **时间边界：** 时间记录在同一模型完成primary与全部group评价后，包含训练循环内的训练、JIT、评价及保存开销，排除循环前下载、数据准备与模型初始化；保留非零step0时间。本模式仅支持 `--time-axis wall`，不会将其他计时口径混入。

另存 `evaluation_runs.csv`（各run真实节点、原始/平滑指标、组样本数、seed来源）、`evaluation_step_curves.csv` 和 `evaluation_time_curves.csv`（均值、标准差、run_count），以及最后写入的 `plot_manifest.json`。清单记录来源哈希、统计/插值/平滑定义、各方法共同时间范围和元信息限制。输出仍在 `results/figures/<新名称>/`，不覆盖已有结果。

### 本次 MNIST 三个 seed 的服务器命令

下面精确列出已导入的九个run名；假定服务器原结果保留在 `results/files/`。如果服务器CSV在另一位置，只修改 `base`，不改run名。日志中出现缺少metadata的提示是便携CSV导出的已知限制，不表示缺少训练指标。

```bash
cd /home/jason_chia925_gmail_com/Project_DPCO/DPCO
mkdir -p results/logs
base="$PWD/results/files"
prefix="mnist_noisy_mnist_cnn"
log="results/logs/plot_mnist_clean_original_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c '
  set -e
  py="/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python"
  "$py" -m pip install -e ".[plot]"
  exec "$py" -u plots/plot_dpco_training.py "$@"
' _ \
  --adam-dir "$base/${prefix}_adam_42_0.001_symmetric20_noise_s0_cosine_20261030_062122" \
  --adam-dir "$base/${prefix}_adam_43_0.001_symmetric20_noise_s0_cosine_20261030_063925" \
  --adam-dir "$base/${prefix}_adam_44_0.001_symmetric20_noise_s0_cosine_20261030_065727" \
  --flip-dir "$base/${prefix}_dpco_ema_b0.9_w-1_adam_42_0.001_symmetric20_noise_s0_cosine_20261030_063321" \
  --flip-dir "$base/${prefix}_dpco_ema_b0.9_w-1_adam_43_0.001_symmetric20_noise_s0_cosine_20261030_065122" \
  --flip-dir "$base/${prefix}_dpco_ema_b0.9_w-1_adam_44_0.001_symmetric20_noise_s0_cosine_20261030_070927" \
  --filter-dir "$base/${prefix}_dpco_ema_b0.9_w0_adam_42_0.001_symmetric20_noise_s0_cosine_20261030_062714" \
  --filter-dir "$base/${prefix}_dpco_ema_b0.9_w0_adam_43_0.001_symmetric20_noise_s0_cosine_20261030_064514" \
  --filter-dir "$base/${prefix}_dpco_ema_b0.9_w0_adam_44_0.001_symmetric20_noise_s0_cosine_20261030_070322" \
  --plot-set clean-original \
  --eval-smooth-window 1 \
  --time-axis wall \
  > "$log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid  LOG=$log"
tail -n 30 "$log"
```
