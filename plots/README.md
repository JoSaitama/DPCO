# DPCO 三组训练与效率曲线

`plot_dpco_training.py` 是独立的离线绘图入口，只需标准库和 Matplotlib，不导入 JAX、Flax、Optax 或 TPU。它不会修改训练CSV或checkpoint，也不会启动训练。原 `scripts/plot_training_results.py` 的历史绘图接口保持不变。

显式选择三个不同目录，每个目录需要完整、已刷盘的 `train_steps.csv` 和 `timing.csv`。支持绝对路径、相对DPCO根目录的路径，或 `results/files/` 下的单个run目录名；不自动寻找“最新”或“最好”的运行。若有对应的 `results/logs/<run_name>/metadata.json`，会核验方法标签，并报告训练设置差异；仅复制两个CSV也可以绘图，但会注明无法核对元信息。

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

默认输出到 `results/files/plots/dpco_training_<时间戳>/`。可用 `--output-name experiment_a_w100` 指定该目录名；已有目录一律拒绝覆盖。更换平滑参数时使用新的输出名，原CSV、旧图片与历史结果均保留。

| 文件名（默认各保存PNG和PDF） | 横轴 | 纵轴 |
| --- | --- | --- |
| train_ce_vs_step | 完成更新次数step | 普通训练CE |
| train_accuracy_vs_step | step | 训练top-1，百分数 |
| train_ce_vs_time | 累计分钟 | 普通训练CE |
| train_accuracy_vs_time | 累计分钟 | 训练top-1，百分数 |
| ms_per_step_vs_step | 窗口末step | 训练窗口平均ms/step |

另外保存 `step_curves.csv`、`timing_curves.csv` 和最后写入的 `plot_manifest.json`，记录曲线数据、平滑参数、输入文件SHA256、脚本SHA256、方法颜色/线型、计时说明、加权平均ms/step和设置差异。仅在全部绘图和输入未变检查成功后生成manifest；没有manifest的输出目录可能是不完整任务。绘图前后输入文件改变会报错，请对完成的run或一致的文件快照绘图。

所有图画布为宽5英寸、高4英寸，绘图区高/宽固定0.8。PNG默认220 DPI，可用 `--dpi` 修改；`--formats png` 可只输出PNG。Adam为蓝实线、Flip为红实线、Filter为蓝虚线。保留线性纵轴和原始异常值，不自动隐藏前期或裁剪高CE。

## 平滑与时间定义

- `--smooth-window W` 默认100，必须为正整数。Step图按最近W次更新的有效样本数加权平均，前缀不足W时用已有数据；`1`关闭额外平滑。
- 时间图以 `event=train_window` 的已完成窗口为基本观测单位，CE与accuracy对窗口覆盖的训练批按有效样本数聚合。平滑时往前纳入完整窗口，直到覆盖至少W次更新（开头不足时用已有窗口）；不拆分窗口，也不为每步插值一个精确完成时间。因此 `W=1` 仍显示原始窗口均值，时间图不是逐步设备完成轨迹。
- ms/step的原始值为 `1000 * duration_s / window_steps`；平滑值为纳入窗口的 `1000 * sum(duration_s) / sum(window_steps)`，不是对不同长度窗口的单步均值做简单平均。CSV中的`smoothing_steps`记录实际平滑跨度。
- 默认 `--raw` 同时画淡色未平滑线，可用 `--no-raw` 隐藏。原始数据始终保留在输出CSV；所有方法使用同一个W。相邻窗口长度不同会使时间图的实际平滑跨度略有不同，不能称为严格等时长平滑。
- `--time-axis wall` 默认使用timing窗口末的 `elapsed_wall_s / 60`。这是训练循环累计墙钟时间，包括循环内初始评价、JIT和先前评价/保存开销，不包括循环前的数据准备与模型初始化。窗口指标是区间平均值，横坐标是区间完成端点。窗口末正在开始的评价或checkpoint尚未发生，因此曲线末点不等于整个进程最终结束时间。
- `--time-axis active` 仅累计train_window的duration，排除独立的编译、评价及checkpoint阶段；它仍包括训练窗口中的输入处理和设备同步，不能称为纯TPU kernel时间。
- 不使用train_steps中的host异步派发时间戳作为逐步完成耗时，也不将所有timing事件的duration相加。完整train_window覆盖、step连续、时间单调与数值/单位会校验，缺失数据不自动补零或外推。

训练CE/top-1对应当前观察标签；在CIFAR-100N中它们衡量带噪标签的拟合，不是独立测试或去噪效果。此入口只读指定两类CSV；泛化与固定clean/corrupted机制曲线需另行读取evaluation/group_evaluation数据并明确口径。
