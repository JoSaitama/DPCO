# CNWL Red Mini-ImageNet：自然网络标签噪声

来源是 Jiang 等，ICML 2020，*Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels*。

- 作者官网：https://google.github.io/controlled-noisy-web-labels/index.html
- 标注/划分下载说明：https://google.github.io/controlled-noisy-web-labels/download.html
- 原论文：https://proceedings.mlr.press/v119/jiang20c.html
- 官方 TFDS 数据说明：https://www.tensorflow.org/datasets/catalog/controlled_noisy_web_labels

本接入选择 **Red Mini-ImageNet**，100 类，84×84 RGB，标签来自真实网络检索错误。
作者用 3–5 人核验图片是否符合观察类别，并用这些真实错标图片替换干净图片，控制噪声比例。
我们直接使用作者发布的 `red_noise_nl_*` 清单、类别编号和行顺序，不随机生成标签，不进行 CARE 长尾采样。
Blue 是另一套合成错标基准，本入口不提供 Blue 选项。

## 档位和实际比例

`--cnwl-noise-percent` / TOML `data.cnwl_noise_percent` **选择官方划分**，只能取
`0,5,10,15,20,30,40,50,60,80`，默认 20。
`--noise-percent` 仍是已有的**实际比例断言**，不选 split，不产生噪声。
`--label-noise-percent` 是 MNIST 合成器，不能用于 CNWL。
固定 Red 划分不使用 noise seed：必须 `noise_type=red`、`is_human=true`、`noise_seed=0`。
这里沿用的 `is_human` 开关表示使用人工核验的发布数据；观察标签仍来自 web 检索，不是人工随机改标。
0 档对应 `mode=clean`，其余对应 `mode=noisy`；CLI 切换档位时自动同步。

论文把 Red 训练规模概括为 50,000，官方 TFDS 给出的具体计数如下，代码保留这些差异：

| 档位（%） | 训练图数 |
|---|---:|
| 0、5、10、15、20、60 | 50,000 |
| 30 | 49,985 |
| 40 | 50,010 |
| 50 | 49,962 |
| 80 | 50,008 |

所有档位共享 5,000 张干净 `clean_validation`。实际错标率由选中清单中的 web 错标样本数计算，
记录为 `actual_noise_percent`，不能为了凑成精确 40% 而删图/补图。
这些是 100 类普通分类训练/评价，不是经典 mini-ImageNet 的 64/16/20 类 few-shot 任务划分，
也不是项目另一份固定类表的 ImageNet-100 或完整 ImageNet-1K。

## 准备来源

需要用户依 ImageNet 访问条款取得原始训练/验证图片。CNWL 官网提供的是 URI、标注和划分，
不保证所有旧 web URI 现在仍能访问。本次开发核查时标注 zip 返回 **HTTP 403**；
程序保留官方下载尝试，也支持手动放入合法取得的原发布 zip/解压目录，不自动切换镜像。

`--manual-dir` 默认是 `--data-dir/raw`，布局沿用官方 TFDS：

```text
manual/
  dataset_no_images.zip             # 或下面的解压目录
  dataset_no_images/
    mini-imagenet-annotations.json
    mini-imagenet/
      class_names.txt               # 也接受官方变体 class_name.txt
      split/
        blue_noise_nl_0.0
        red_noise_nl_0.0
        red_noise_nl_0.05
        ...
        red_noise_nl_0.8
        clean_validation
  ILSVRC2012_img_train.tar
  ILSVRC2012_img_val.tar
  noisy_images/
    <官方 image/id>.jpg
```

可用现有官方命名图片代替扫描 tar：干净训练图放 `clean_images/<mini-id>.jpg`，
验证图放 `validation_all/<validation-id>.jpg`。也兼容 `all_images/<mini-id或web-id>.jpg`，
前提是图像确实对应作者 ID；不会把任意当前 ImageNet-100 图片当作 mini 数据。
web 图也可位于 `dataset_no_images/noisy_images/`。

**原始 ImageNet 训练文件名不能简单改后缀。** mini ID 后八位是该 synset 中按原图数字编号排序后的
1-based 顺序。准备代码按官方 TFDS 的排序/位置语义从 class tar 找图；验证图按发布清单 ID 匹配。
来源像素使用 OpenCV `INTER_AREA` 缩放到 84×84，并以 JPEG quality=72 编码，和官方 TFDS 处理一致。
运行时训练使用 ImageNet mean/std、随机 resized crop/水平翻转；分组评价/验证使用完整 84×84，
不加随机增强。这是项目的 Adam/DPCO 比较设置，不声称复现 MentorMix 的优化器、模型或整套增强。

只有准备阶段需要额外 OpenCV；训练仍用现有 JAX/Flax，不导入 TensorFlow/PyTorch。
安装示例（服务器长任务统一后台）：

```bash
cd /home/jason_chia925_gmail_com/Project_DPCO/DPCO
mkdir -p results/logs
py=/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python
log="results/logs/cnwl_install_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" -m pip install -e '.[data,cnwl]' > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid LOG=$log"
tail -f "$log"
```

先检查计划，不下载：

```bash
/home/jason_chia925_gmail_com/Project_DPCO/.venv/bin/python \
  scripts/prepare_cnwl.py --stage plan --cnwl-noise-percent 20
```

在原始文件已位于 `../data/cnwl/raw` 的前提下，准备 20% 官方划分：

```bash
log="results/logs/cnwl_red20_prepare_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" -u scripts/prepare_cnwl.py \
  --stage prepare --data-dir ../data/cnwl --cnwl-noise-percent 20 \
  --download --workers 8 > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid LOG=$log"
tail -f "$log"
```

可用 `--manual-dir /已取得的官方源文件目录` 指向已有源文件，避免复制 ImageNet 大归档。
`--download` 仅允许抓取官网标注包和当前档位需要的 web URI；不自动下载需同意条款的 ImageNet。
没有该参数时完全使用本地源。图片缺失/解码失败时保留已完成数据，输出失败清单并报错；
不静默跳过失效 URL、不补别的图片、不重新搜索。取得缺失原图后重跑同一命令即可重用已缓存行。
这是**文件/样本级续做**；单个被中断 HTTP 文件从头下载，不承诺字节续传或 URL 永久有效。

缓存按档位独立存储于 `../data/cnwl/prepared/red_20/`、`red_40/` 等。
manifest 保存所读官方元数据 SHA256、源归档位置/大小/mtime、类别名、实际计数、处理方式和成品数据库 SHA256。
作者未在下载页给出 zip 固定校验值；记录本地元数据内容哈希用于可追溯/配对，不冒充作者发布的签名。
完整缓存有 completion 标记；训练重读时验证哈希，恢复状态同时检查数据 fingerprint。
准备报告写到 `results/files/cnwl_prepare_<时间>/`，终端日志在上面的 `results/logs/`。

## 训练和指标

`configs/tpu/cnwl/dpco_adam.toml` 默认 Red20、随机初始化 ResNet50、Adam、reference beta=.9，
batch128，lr=3e-4 cosine→1e-6，78,000 步，390 步评价一次，1,950 步保存 latest。
这些是可改的探索预设，不是已验证最优参数；更换档位后按实际样本数解释 epoch。
既有 `--model` 可切换 resnet50/101 或 native-stem cifar_resnet18/34/50，输入仍保持 84×84。

先做无训练的参数检查：

```bash
bash run_cnwl_suite.sh --dry-run --cnwl-noise-percent 20
```

确认准备完成后，可由用户执行以下正式三组训练命令：

```bash
log="results/logs/cnwl_red20_suite_$(date +%Y%m%d_%H%M%S).log"
nohup "$py" -u scripts/run_suite.py \
  --config configs/tpu/cnwl/dpco_adam.toml \
  --methods adam dpco_filter dpco_flip --seeds 42 \
  --cnwl-noise-percent 20 --no-download --wait \
  > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
echo "PID=$pid LOG=$log"
tail -f "$log"
```

三组顺序执行，共用固定作者 split；权重仍为 Filter=0、Flip=−1，可用 `--method dpco --negative-weight ...` 自定义。
没有修改 raw-gradient EMA reference 或 Adam 更新公式。训练产物继续保存到
`results/files/<run>/`、`results/runs/<run>/`、`results/logs/<run>/`，run 名包含 `red20_official_split`。

- `evaluation.csv`：共同干净验证集的 CE/top1/top5；沿用不自动选择 best checkpoint 的协议。
- `group_evaluation.csv`：固定 clean 训练子集（原标签可知）的普通 CE/acc，以及 web 错标训练子集按观察标签的 CE/acc。
- `train_steps.csv`：实际训练 batch 按观察标签计算的 online CE/acc。
- `timing.csv`、checkpoint、恢复和各方法优化统计沿用现有流程。

`is_clean` 只划分评价组，**不进入训练 batch、不用于 reference 或样本编辑**。
web 图片可能来自 100 类之外，作者核验的是“观察标签是否正确”，没有给出全量闭集正确类别。
因此不输出虚构的 `corrupted_original` 或全训练原标签 acc。
旧 `plots/plot_dpco_training.py --plot-set training` 可画观察标签训练指标；
`--plot-set clean-original` 要求完整原标签三组，不能用于该 CNWL 数据，不应把缺失真值补成观察标签。

本次开发的测试使用明确标为合成的本地小夹具验证数据通路、标签排除、三种更新器及恢复；
尚未完成真实 Red 全量下载或 TPU 正式训练。
