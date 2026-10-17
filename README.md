# SAGE

用于比较 SGD、SAGE、DTV 和随机翻转对照的模块化实验代码。

## 当前状态

已按 task → signal/reference → method → trainer 拆分，并实现可运行的分类 task、
loss/error reference、普通/LOO alignment、六种方法配置及单步训练。
更新方向按原有效 batch 大小 B 求平均，保留梯度原始强度；trainer 不二次平均。
alignment 方法当前显式保存逐样本梯度，仅用于小批量正确性验证，不能用于宣称大模型效率。
尚未迁入 Flax ResNet、ImageNet 读取、低内存评分、TPU 分布式训练或定理观测实现。
本仓库目前不能直接启动 ImageNet 训练，也没有训练结果。

## 方法边界

训练器只编排一次更新，不根据方法名称编写条件分支：

```text
trainer.step
    -> method.compute_direction(task, params, batch, states, key)
         -> task losses / feedback
         -> signal + reference (only when needed)
         -> correction weights
         -> direction = sum_i(w_i * g_i) / valid_B
    -> updater: params - learning_rate * direction
```

- `tasks/`：定义逐贡献 loss、有效 mask、任务反馈以及显式模型状态；不按数据集分支。
- `signals/`：提供 prediction error 或原始梯度内积，不决定 filter/flip。
- `references/`：提供 loss/error 子集的共同或逐样本 LOO 参考梯度，不决定 filter/flip。
- `methods/`：调用上述组件，决定权重并返回完整方向；不修改参数、不乘学习率。
- `training/trainer.py`：调用注入的 method 与 updater，推进训练状态；没有方法名分支。
- `probes/`：仅保留扩展位置；观测字段、频率和格式需先确认，本次没有记录实现。

SGD 和 random_flip 都跳过 reference/alignment；不会为统一接口支付这部分成本。
`reference.name="error"` 是错误样本 surrogate-gradient reference 的候选实现，
不是对离散 0/1 error 求梯度，也不代表 SAGE-E 的全部最终定义已由实验确定。
默认 `reference.name="loss"` 对应当前有效 batch mean。

## 方法与 reference

设 `q_i` 为同一 reference 的 alignment score。

| `method.name` | 非负分数 | 负分数 |
| --- | --- | --- |
| `sgd` | +1 | +1 |
| `sage` | +1 | -1 |
| `dtv` | +1 | 0 |
| `random_flip` | 以概率 p 取 -1，否则 +1 | 以概率 p 取 -1，否则 +1 |

SAGE/DTV 的零分数保留 +1；random_flip 不看 score，独立翻转整个有效 batch。
所有方法除以原有效 B，不除以权重和或保留数。p=0 的 random 是 SGD，p=1 是全部贡献翻转，非 SAGE。
`sage_loo`/`dtv_loo` 排除被评分样本自身；reference 来源仍独立配置为 loss/error。
error reference 下，仅在样本属于参考子集时才减掉自身；子集外样本仍参考整个错误集合。
空 reference 或 singleton LOO 通过 has_reference 显式标记，目前沿用旧骨架的“不修正”回退；
这不是新的收敛保证，正式论文中的空集合策略仍需明确。LOO 不直接继承共同 reference 的 Delta 定理。
此处 DTV 为指定 reference 下的过滤方向，不声称复现所有原论文训练细节。

## 布局

```text
SAGE/
  main.py                目前只支持显式 --check-config，不伪装成训练入口
  configs/               ImageNet100/1k × ResNet50/101 配置
  src/
    config.py
    cli.py
    data/                统一 ClassificationBatch；未接 ImageNet reader
    models/              预留 Flax 模型适配
    tasks/               base.py、classification.py
    signals/             prediction_error.py、gradient_alignment.py
    references/          batch_mean.py、error_subset.py
    methods/             sgd.py、sage.py、dtv.py、random_flip.py、base.py
    training/            trainer.py、updates.py、randomness.py
    probes/              待确认观测需求后开发
  tests/                 配置、数值方向、reference、分类与 trainer 测试
```

## 本地检查

配置检查只需 Python 3.11+；数值测试需要 CPU JAX，可通过 test extra 安装。
已在临时 CPU 环境验证 JAX 0.11.2；这不是 TPU/Flax 全栈兼容性认证。
训练依赖尚未锁定，不创建未经验证的 lock 文件；未来上游模型仍须固定 commit。
配置检查不验证数据文件存在，也不检查当前设备；ImageNet-100 的实际类别清单尚未选定。
选定后将清单放到可版本控制的 `configs/manifests/` 中，图像数据本身不入 Git。

```bash
python main.py --check-config --config configs/imagenet100_resnet50.toml
python main.py --check-config --config configs/imagenet100_resnet50.toml --method sage_loo --reference loss
python main.py --check-config --config configs/imagenet100_resnet50.toml --method random_flip --flip-probability 0.25
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

未来在已配置的 TPU 环境安装依赖后才进行训练集成：

```bash
python -m pip install -e '.[train,tpu]'
```

这是安装示例，尚未在本项目 TPU 上执行或验证。
现阶段只有 SGD/SAGE 是启动实验对照；预留 DTV/random 不等于现在启动所有消融。

## 多设备约定

global batch 是 reference 和贡献聚合的定义单位，不能静默改为逐设备 reference。
在多设备上对有效贡献之和与有效数量分别做全局归约，再相除；不平均不等长分片的局部均值。
错误 reference 也必须按全局错误样本数量归一化。
随机 key 由调用方按 seed、step、设备/样本定义显式生成；不能在每个设备重复使用同一个 key。
当前方法实现以单设备完整逻辑 batch 为前提，不包含生产分布式归约。
`ClassificationTask` 的 model_fn 是纯函数，返回 logits 和候选 model_state；trainer 只提交一次。
接入训练态 BatchNorm 时还需处理跨样本耦合，不能把本次线性模型测试当作该问题已经解决。
任务必须提供至少一个有效样本；未知记录字段不自动采集，没有任何 CSV/JSONL 写入代码。
