# HEVA 实验项目

**HEVA** (High-Entropy Visual Attention) 验证高熵生成 token 是否真的"看图"（attend to 视觉 token），并实现两种解码干预方法（**DLA**、**CAD**）来引导模型关注视觉信息。

---

## 目录

- [项目结构](#项目结构)
- [核心方法](#核心方法)
- [四种推理模式 (NV 设置)](#四种推理模式-nv-设置)
- [快速开始](#快速开始)
- [核心脚本说明](#核心脚本说明)
- [HEVA 指标定义](#heva-指标定义)
- [输出结果结构](#输出结果结构)
- [环境与硬件](#环境与硬件)
- [文档索引](#文档索引)
- [可复现性](#可复现性)
- [验证检查点](#验证检查点)

---

## 项目结构

```
heva-pre/
├── README.md                    # 本文件
├── AGENTS.md                    # 给 AI agent 的项目说明
├── OBS_UPLOAD_NOTES.md          # OBS 上传注意事项
│
├── doc.md                       # HEVA 原始技术规范
├── doc2.md                      # CAD/DLA 设计公式文档
│
├── data/
│   ├── loader.py                # 数据加载器（VisuRiddles 等 12 个数据集）
│   └── perturbations.py         # 图像扰动函数
│
├── metrics/
│   ├── inference.py             # Qwen3-VL 推理模块（含 monkey-patch）
│   ├── heva.py                  # HEVA 计算 + DLA 反向因果路径
│   └── context_aware_logits_processor.py  # CAD logits processor
│
├── analysis/
│   ├── statistics.py            # 统计分析函数
│   └── plots.py                 # 可视化函数
│
├── 1_run_inference.py           # 基础推理（仅捕获）
├── 2_run_inference_heva_force.py  # HEVA-only 推理
├── 3_run_inference_trace.py     # 主流水线：trace + DLA + CAD
├── 4_run_inference_single.py    # 单样本推理
├── 7_statistics.py              # 统计分析入口
│
├── devp.py / devp2.py / devp3.py  # Debug 脚本
│
├── NV0-2B.sh                    # 标准推理（capture only）
├── NV1-2b.sh                    # DLA 全 token 干预
├── NV3-2b.sh                    # DLA 高熵 token 干预
├── NV4-2b.sh                    # DLA + CAD 联合干预
├── NV-V-2b.sh                   # CAD-only 干预
├── start.sh / start-NV.sh       # 旧启动脚本
│
├── docs/                        # 详细方法文档
│   ├── CAD.md                   # CAD 详细说明
│   ├── DLA.md                   # DLA 详细说明
│   ├── CAD_DLA.md               # CAD + DLA 合并版
│   ├── REVIEW.md                # 文档审查报告
│   ├── index.md                 # HEVA 脚本总览
│   ├── run_inference_trace.md   # trace 脚本说明
│   └── run_inference_heva_force.md  # heva_force 脚本说明
│
├── results/                     # 实验结果（git ignored）
└── .sisyphus/                   # Sisyphus 工作目录
```

---

## 核心方法

### HEVA (High-Entropy Visual Attention)

高熵生成 token 对视觉 token 的平均注意力质量。验证模型在"不确定"时是否真的"看图"。

### DLA (Direct Logits Attribution)

**视觉错误纠正机制**。对每个候选 token，沿反向因果路径 `z @ W_O @ W_U` 计算 `||h||`（头输出的模长），保留 top-k//2 候选，重新采样。

**关键技术决策**: 用 **raw dot product** 而非 cosine。`||head_output||` 本身就是视觉依赖信号（关注视觉 token 的 head 模长更大），cosine 归一化会把这个信号抹掉（仅剩 ~0.03 噪声）。详见 [`docs/DLA.md`](docs/DLA.md)。

### CAD (Context-Aware Decoding)

**减法式 logits 处理器**。在熵高时，根据最后一层 attention 找出"上下文头"（关注视觉 token 的头），用 `z @ W_O @ W_U` 计算每个候选的支持度，过滤掉 top-k 中支持度低的候选（设为 `-inf`）。

**重要**: CAD 是**减法**的——只删除 token，不增强分数。详见 [`docs/CAD.md`](docs/CAD.md)。

### DLA + CAD 联合

DLA 选出 `dla_candidates`，CAD 选出 `cad_candidates`，**取交集**作为最终候选。交集为空时回退到 `dla_candidates`。

---

## 四种推理模式 (NV 设置)

通过 `3_run_inference_trace.py` 的 CLI 参数控制：

| 模式 | 脚本 | `--use_attention_guidance` | `--use_context_aware` | `--dla_entropy_threshold` | 用途 |
|------|------|:---:|:---:|:---:|---|
| **NV0** | `NV0-2B.sh` | `false` | `false` | — | 基线：仅捕获数据，无干预 |
| **NV1** | `NV1-2b.sh` | `true` | `false` | `-10` | DLA-only：所有 token 都评估 |
| **NV3** | `NV3-2b.sh` | `true` | `false` | `1.3` | DLA-only：仅高熵 token 评估 |
| **NV-V** | `NV-V-2b.sh` | `false` | `true` | — | CAD-only：仅 CAD 过滤 |
| **NV4** | `NV4-2b.sh` | `true` | `true` | `1.3` | DLA + CAD：两者交集 |

> NV1-2b.sh 用 `--dla_entropy_threshold -10`（负无穷）让 DLA 在每一步都生效。
> NV-V-2b.sh 用 `--ctx_entropy_threshold 1.3`（默认 5.0 对 THK 模型过高，CAD 不触发）。

---

## 快速开始

### 1. 激活环境

```bash
conda activate PyTorch-2.1.0
```

### 2. 切换到项目目录

```bash
cd /w0rk5pace/aaworks/heva-pre
```

### 3. 选择并运行一个实验

```bash
# 方式 A: 直接用 shell 脚本（推荐）
bash NV0-2B.sh                    # 基线推理
bash NV1-2b.sh                    # DLA 全 token
bash NV3-2b.sh                    # DLA 高熵
bash NV-V-2b.sh                   # CAD-only
bash NV4-2b.sh                    # DLA + CAD

# 方式 B: 手动指定参数
ASCEND_RT_VISIBLE_DEVICES=0 python 3_run_inference_trace.py \
    --exp_name exp001 \
    --model_path /w0rk5pace/aaworks/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct \
    --dataset VisuRiddles \
    --num_samples 50 \
    --use_attention_guidance true \
    --use_context_aware false \
    --dla_entropy_threshold 1.3
```

### 4. 统计分析

```bash
python 7_statistics.py
```

---

## 核心脚本说明

| 脚本 | 用途 | 特点 |
|------|------|------|
| `1_run_inference.py` | 基础推理 | `--use_context_aware` 是布尔 flag（无值） |
| `2_run_inference_heva_force.py` | HEVA 离线捕获 | 不支持 CAD，仅捕获 `gen_entropy`、`gen_vattn` |
| `3_run_inference_trace.py` | **主流水线** | `--use_context_aware` 是字符串 (`"true"`/`"false"`)；支持 DLA + CAD |
| `4_run_inference_single.py` | 单样本推理 | 调试用 |
| `7_statistics.py` | 统计分析 | 算 accuracy、HEVA-correctness 相关性 |
| `devp.py` / `devp2.py` / `devp3.py` | Debug 脚本 | devp3.py 包含 CAD 测试 |

> **布尔参数陷阱**:
> - `1_run_inference.py`: `--use_context_aware`（flag，无值）
> - `3_run_inference_trace.py`: `--use_context_aware true` 或 `false`（**字符串**）

---

## HEVA 指标定义

```
HEVA = (1/|S|) × Σ v_t  for t in S

其中:
  S  = top-α% 高熵生成 token 集合
  v_t = token t 对所有视觉 token 的注意力之和
```

### 正常范围

| 样本类型 | HEVA 范围 |
|---------|-----------|
| 完全语言 shortcut | 0.01 ~ 0.05 |
| 视觉依赖 | 0.1 ~ 0.3 |
| 极端视觉依赖 | > 0.4 |

> ⚠️ **如果 HEVA > 0.8（对所有样本）**：视觉 token 索引很可能算错了。

---

## 输出结果结构

```
results/{exp_name}/{dataset}/
├── pkls/
│   ├── {idx}_gen_entropy.pkl      # (gen_tokens_num, batch_size)
│   ├── {idx}_gen_vattn.pkl        # (gen_tokens_num, batch_size, visual_token_num)
│   ├── {idx}_attn_acc_input.pkl   # (gen_tokens_num, batch_size)
│   ├── {idx}_attn_acc_visual.pkl
│   └── {idx}_gen_zs.pkl           # (num_layers, batch, seq, heads, d_head) — 仅 trace
├── {sample_id}_meta.json          # ground truth, predicted, correctness, token counts
├── exp_config.json
└── index.json
```

---

## 环境与硬件

### 硬件

- **NPU (Huawei Ascend)**: `ASCEND_RT_VISIBLE_DEVICES=0 python ...`
- **NVIDIA GPU**: `CUDA_VISIBLE_DEVICES=X python ...`
- ⚠️ **不要使用 gpu0**（详见 `OBS_UPLOAD_NOTES.md`）

### 软件

- `conda activate PyTorch-2.1.0`
- Python 3.x
- 依赖: torch, transformers, numpy, scipy, sklearn, matplotlib, seaborn, pandas, pillow, tqdm

### 模型路径

| 模型 | 路径 |
|------|------|
| Qwen3-VL-2B-Instruct | `/w0rk5pace/aaworks/Downloads/Models/Qwen/Qwen3-VL-2B-Instruct` |
| Qwen3-VL-2B-Thinking | `/w0rk5pace/aaworks/Downloads/Models/Qwen/Qwen3-VL-2B-Thinking` |

### 数据路径

- 数据根目录: `/w0rk5pace/aaworks/datas/`
- 支持的数据集: `VisuRiddles, RAVEN, MARVEL, LogicVista, PuzzleVQA, AlgoPuzzleVQA, AI2D, RealWorldQA, MMMU, MMMU_Pro, MathVista, MathVision`

### 上传到 OBS

```bash
/w0rk5pace/aaworks/obsutil_linux/obsutil cp [file] obs://lixiang01/
```

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/CAD.md`](docs/CAD.md) | CAD 详细算法 + 代码 walkthrough |
| [`docs/DLA.md`](docs/DLA.md) | DLA 详细算法 + z-capture monkey-patch + raw_dot 原理 |
| [`docs/CAD_DLA.md`](docs/CAD_DLA.md) | CAD + DLA 合并版（推荐先读） |
| [`docs/REVIEW.md`](docs/REVIEW.md) | 文档审查报告（CAD/DLA vs 源码） |
| [`docs/index.md`](docs/index.md) | HEVA 脚本总览（旧版） |
| [`docs/run_inference_trace.md`](docs/run_inference_trace.md) | trace 脚本详细说明 |
| [`AGENTS.md`](AGENTS.md) | 给 AI agent 的项目说明 |

---

## 可复现性

✅ **Seed 固定** — 所有推理脚本都有 `set_seed(seed)`：
- `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`

✅ **`do_sample=True`** — 所有脚本默认采样模式，DLA 内部也用 `torch.multinomial`

✅ **`attn_implementation="eager"`** — 必需，SDPA/flash attention 会破坏 attention 捕获

```bash
# 默认 seed=42
python 3_run_inference_trace.py --seed 42 ...
```

---

## 验证检查点

根据 `doc.md`，需要验证：

1. **Attention 归一化**: `attn_t.sum(dim=-1) ≈ 1`
2. **视觉 token 索引正确**: 打印 `input_ids` 确保是图像 embedding 区间（`151643` = `<|image_pad|>`）
3. **HEVA 数值范围**（见上表）

---

## 已知 Quirks

1. **熵计算**: 部分地方用 `-sum(p * log(p))`（不是 `+1e-9`），均匀分布可能产生 NaN
2. **DLA vicious cycle**（已修复）: 在高熵 step，top-40 被标点占满 → DLA 重排序 + multinomial 在 20 个标点里采样 → 上下文腐化 → 模型继续预测标点
   - **解决**: DLA 用 raw_dot（非 cosine）让 `||h||` 排序信号有 0.27 范围（cosine 仅 0.002）
3. **CAD 默认阈值 5.0 对 THK 模型过高**: entropy 极少超过 2.7，CAD 不触发。用 `ctx_entropy_threshold=1.3`
4. **3_run_inference_trace.py 必须用 left-padding**（batch>1 时）: right-padding 会破坏 next-token 预测
5. **CAD 是减法**: 设 `scores[b, drop_mask] = -inf`，不会增强任何 token
6. **`metrics/inference.py:12` 的 monkey-patch**: import 时立即替换 `Qwen3VLForConditionalGeneration._sample`，**不要移动或包装它**
7. **⚠️ DLA 和 CAD 在 2B 模型上**目前**降准确率**（详见 `docs/DIAGNOSIS.md`）:
   - DLA thr=1.3: THK -4.66%, Instruct -5.50%
   - CAD thr=1.3: THK -1.66%, Instruct -2.33%
   - **根因**: 阈值 1.3 触发太频繁（~20% steps）→ 扰动累积 → vicious cycle（生成卡 8192）
   - **修复**: 跑高阈值实验 `NV3-2b-thr2.0.sh` / `NV3-2b-thr2.5.sh` / `NV-V-2b-thr2.0.sh` / `NV-V-2b-thr2.5.sh`

---

## 已知问题: DLA / CAD 准确率下降 (2026-06-03)

详见 [`docs/DIAGNOSIS.md`](docs/DIAGNOSIS.md) 完整诊断报告。

**关键数据**:
- 基线 (NV0) THK 0.5083 / Instruct 0.4483
- DLA (NV3 thr=1.3) THK 0.4617 (-4.66%) / Instruct 0.3933 (-5.50%)
- CAD (NV-V thr=1.3) THK 0.4917 (-1.66%) / Instruct 0.4250 (-2.33%)

**三个根本原因**:
1. 阈值 1.3 ≈ p75,触发 ~20% steps,对 1500-token 生成是 300+ 次扰动
2. DLA 把低熵 step 推成高熵 step,触发更多 DLA → vicious cycle (8192 上限)
3. `||h||` 信号是置信度,不是视觉依赖。DLA top-20 ≈ 模型 top-20

**任务类型影响**:
- **直接视觉查询** (AI2D, RealWorldQA): DLA 帮 (+1~3%)
- **多步推理** (LogicVista, MARVEL): DLA 大幅伤害 (-9% 到 -17%)

**已创建的修复脚本**:
- `NV3-2b-thr1.7.sh` / `NV3-2b-thr2.0.sh` / `NV3-2b-thr2.5.sh` — DLA 高阈值
- `NV-V-2b-thr1.7.sh` / `NV-V-2b-thr2.0.sh` / `NV-V-2b-thr2.5.sh` — CAD 高阈值

---

## 联系方式

详见 `AGENTS.md`。
