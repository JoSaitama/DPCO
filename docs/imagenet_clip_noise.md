# ImageNet100：冻结 CLIP ViT-L/14 生成固定比例标签噪声

这是本项目定义的**模型诱导、实例依赖合成噪声**，不是自然人工噪声，也不是 AAAI 论文的原样复现。CLIP 只在离线准备阶段前向推理；训练的模型仍是随机初始化的 ResNet50。

## 固定协议

- OpenAI 原版 `ViT-L/14`、224像素官方预处理；官方权重 SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。下载和本地权重均必须通过该校验。
- 官方实现固定 commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`。不使用同名的其他来源权重。记录实际源码哈希、安装来源、软件版本、推理设备与 batch size。
- 类别文本取现有 HF 缓存的 `original_label_names`，按 manifest 的局部类别索引排序。默认单一模板 `a photo of a {class_name}.`，可以用 `--prompt` 指定并记录。概率只在本数据集100个候选类别内归一化。
- 对原标签为 y 的样本，候选错误类 j 为最大概率的非 y 类；难度 d = p(j) - p(y)。使用权重 exp(d/tau) 做不放回抽样，默认 `--sampling-temperature 0.2`。它不是人类错误概率，也不是 DPCO/Adam 的 beta。
- tau=0.2是本项目原型默认值，尚未由真实ImageNet实验确认最优；低难度样本仍可能被抽中，抽样权重不等于最终边际入选概率。若没有足够合理的候选错误，不能仅以达到20%为由宣称噪声自然。
- 实现采用 PCG64 随机数和 Gumbel top-k 全排序，再选前 `floor(N*percent/100+0.5)` 条，全部改为 j。相同预测、seed、温度可重建同一标签；同比例的不同训练 seed 复用同一份标签。相同 noise seed/温度下10%与20%污染集合嵌套。改变 noise seed 会改变抽样，错误目标仍固定。
- 比例是**全局**比例，不保证每类相同比例。通过 `summary.json` 和 `transition.csv` 检查各类分布。126689张的20%取25338条，实际约20.000158%；记录实际值，不假称数学上恰为20%。
- 支持复用同一模型到1K：改用1K配置和对应的已准备缓存即可。候选类别、概率和错误目标随之变化，需要重新评分，不能直接沿用100类标签。

固定权重/seed仍不保证不同设备、库版本的浮点推理逐位相同。**固定并共享发布后的标签文件及 manifest SHA256** 才是实验复现的依据。训练恢复会检查数据指纹；训练代码不会重新运行 CLIP 或改标。

## 服务器准备命令（20%示例）

先确保现有 ImageNet100 `images.sqlite` 已完整准备好。本脚本读取现有缓存，**不下载 ImageNet 图片**。默认根据配置/DataPlan指纹定位缓存；也可用 `--image-db /absolute/path/images.sqlite` 明确指定。训练仍会核对该缓存是否与训练配置完全匹配。

```bash
cd /home/jason_chia925_gmail_com/Project_DPCO/DPCO
mkdir -p results/logs
log="results/logs/imagenet100_clip_noise20_$(date +%Y%m%d_%H%M%S).log"

nohup bash -c '
  set -euo pipefail
  py="/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python"
  "$py" -m pip install "torch>=2.2,<3" "torchvision>=0.17,<1" \
    --index-url https://download.pytorch.org/whl/cpu
  "$py" -m pip install -e ".[clip]"
  exec "$py" -u scripts/prepare_imagenet_clip_noise.py \
    --stage all \
    --config configs/tpu/dpco/imagenet100_resnet50.toml \
    --data-dir /home/jason_chia925_gmail_com/Project_DPCO/data \
    --output-dir results/files/imagenet100_clip_l14_noise20_s0 \
    --label-noise-percent 20 --noise-seed 0 \
    --sampling-temperature 0.2 --batch-size 32 \
    --device cpu --download-weights \
    --audit-count 100 --audit-seed 42
' > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid LOG=$log"
tail -n 30 "$log"
```

`--device cpu` 可在现有TPU服务器的CPU上运行，但**不会利用TPU**，全量L/14推理可能很慢；有 NVIDIA GPU 的准备机可用 `--device cuda`。GPU版PyTorch应按该机驱动环境安装，本项目不自动选择CUDA运行时，也不安装PyTorch/XLA。准备可在另一机器完成，再传输生成目录到训练服务器；无需把CLIP放进JAX训练进程。

该命令会安装可选依赖和下载大型模型权重，执行前应使用希望安装的项目环境。生成脚本本身不训练模型。默认权重位于 `../data/models/openai_clip/`；`--checkpoint` 可以指定已下载的同一官方权重。官方权重下载器**不承诺断点续传**。标签评分按已提交 batch 续跑，下载中断和评分中断要区分。

如果评分中断，保持全部参数及输出目录不变，重新用上述 nohup 命令启动；已完成的评分 batch 会跳过。更换模型、prompt、来源、推理设备、batch size 或软件身份时会拒绝沿用旧评分缓存，需要新的 `--scores-dir`。已发布标签不会覆盖；同一请求会校验后复用。发布文件的瞬间若异常中断、留下文件但没有完整 manifest，则需使用新 output-dir 并复用原 scores-dir，不能将半成品当成完成。

## 只改变噪声比例，不重复运行CLIP

```bash
mkdir -p results/logs
log="results/logs/imagenet100_clip_noise10_$(date +%Y%m%d_%H%M%S).log"
nohup /home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  -u scripts/prepare_imagenet_clip_noise.py --stage generate \
  --scores-dir results/files/imagenet100_clip_l14_noise20_s0/clip_scores \
  --output-dir results/files/imagenet100_clip_l14_noise10_s0 \
  --label-noise-percent 10 --noise-seed 0 --sampling-temperature 0.2 \
  > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid LOG=$log"
tail -n 30 "$log"
```

## 文件和人工抽查

所有本次准备、检查输出放在 `results/files/<noise_name>/`，后台日志/PID放在 `results/logs/`。

| 文件 | 内容 |
|---|---|
| `clip_scores/scores.sqlite` | 分batch提交的预测缓存，可中断重跑、供多个比例/seed复用 |
| `noise_manifest.json` | 版本、来源指纹、完整模型身份、生成参数、文件哈希、目标和实际率 |
| `labels.npz` / `labels.csv` | sample ID、原标签、训练污染标签、mask、候选类、概率、难度、原缓存source_index和图片字节SHA256 |
| `summary.json` | 每类总数/改标数、整体实际率、CLIP本身自然top1错误率 |
| `transition.csv` | 实际原类→污染类的计数 |
| `audit/index.html` | 图片与原类别/训练类别/候选错误类/概率的离线抽查页 |
| `audit/review.csv` | 同一抽样清单，预留 `human_plausible` / `notes` 供人工记录 |
| `audit/audit.json` | 抽查seed、样本ID及被审查manifest哈希 |

`--audit-count 100` 表示从已污染/未污染两组**各均匀抽取最多100张**，默认共200张；不是只挑最容易解释的例子。可把整个 audit/ 下载到Mac，浏览器打开 index.html。页面展示原图缩略图；CLIP实际评分使用官方中心裁剪，二者视野可能不同。分组均匀抽样不等于全体样本均匀抽样，不能未经权重修正用这200张估计总体占比。

可用 `--stage audit --output-dir ... --data-dir ... --audit-count 200 --audit-seed 43 --audit-dir results/files/<noise_name>/audit_s43` 再抽一组；此类长任务同样按上述 nohup 日志/PID模板启动。已有抽查目录拒绝覆盖，保护手工评语。人工意见用于检验生成协议的合理性；本实现不会读取评语后暗中修改训练标签。若决定修订协议，应另生成版本并留存旧版本，而不是按DPCO收益改噪声。

完成后可前台运行短检查：

```bash
/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  scripts/prepare_imagenet_clip_noise.py --stage verify \
  --output-dir results/files/imagenet100_clip_l14_noise20_s0
```

## 接入现有Adam训练套件

在原有 `scripts/run_suite.py` 的 nohup 命令中增加：

```text
--config configs/tpu/dpco/imagenet100_resnet50.toml
--methods adam dpco_filter dpco_flip
--noise-manifest results/files/imagenet100_clip_l14_noise20_s0/noise_manifest.json
```

`--noise-manifest` 自动选择 `mode=noisy, noise_type=clip_idn, is_human=false`。此阶段不要传生成用的 `--label-noise-percent`、`--noise-seed`；原比例与seed从不可变manifest读取，训练seed只控制初始化/数据顺序。`--noise-percent`仍是对**实际**错标率的断言，不负责生成标签；因整数取整通常不能直接断言20.000000%。保持现有ResNet50、Adam、reference beta和训练预算参数；本功能不自动启动或选定训练预算。

训练 `ClassificationBatch` 仍只有图片、污染标签和valid mask。原标签/mask只供离线生成和诊断，不进入梯度、EMA reference或Adam状态。

每个评价节点继续输出 `evaluation.csv`（干净验证集）和 `group_evaluation.csv`：

- `clean / original`：未改标训练子集CE、top1/top5；
- `corrupted_noisy / reference`：污染子集按训练错误标签评价；
- `corrupted_original / original`：污染子集按原标签评价；
- `all_original / original`：上述两个原标签组按样本数加权，得到完整训练集原标签CE、acc，**不额外重复推理**。

训练step中的train acc仍按污染标签；它不能替代 `all_original`。评价使用无随机增强的eval模式，能直接接入既有 `plots/plot_dpco_training.py` 的 evaluation 模式，绘制clean CE和完整训练集原标签acc随step/时间变化。分组全量评价增加时间成本，所有方法要使用相同评价频率。旧定理probe/实时机制observer在这一新噪声协议中暂禁用，避免沿用不适用的理论假设；已有分组诊断与绘图保留。

ImageNet的best checkpoint仍按独立干净validation的准确率选择，不能按训练集原标签指标挑选；这里的“原标签”指基准标注，不保证每张图片的语义真值都正确。

参考：[OpenAI官方CLIP接口与预处理](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py)。本地测试使用小型合成图片、模拟预测器和微型ResNet，不能代替真实L/14与ImageNet100的资源/噪声合理性检查。
