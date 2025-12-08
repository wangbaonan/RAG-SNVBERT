# 📋 V18 推理路径配置总结

## ⚠️ 关键修正

根据 `run_v18_embedding_rag.sh` 训练脚本，已修正所有路径和参数！

---

## ✅ 正确的配置（已修正）

### 1. 模型架构参数（Critical!）

**从训练脚本获取**：
```bash
DIMS=384         # ✓ 正确
LAYERS=12        # ✓ 修正（之前错误写成 6）
HEADS=12         # ✓ 修正（之前错误写成 8）
```

**验证方法**：
```bash
# 查看训练脚本中的参数
grep -E "dims|layers|attn_heads" run_v18_embedding_rag.sh

# 应该看到:
# --dims 384 \
# --layers 12 \
# --attn_heads 12 \
```

---

### 2. 数据路径（已根据训练脚本修正）

#### Reference Panel（用于构建 FAISS 索引）

```bash
REF_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_split.h5"

REF_PANEL_INFO="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_panel.txt"
```

**说明**：
- 使用训练集作为 Reference Panel
- 与训练时的 `--train_dataset` 和 `--train_panel` 一致

#### Target Dataset（待填补的数据）

```bash
TARGET_DATASET="/path/to/your/target.vcf.gz"  # TODO: 修改为实际路径
TARGET_PANEL="/path/to/your/target_panel.txt"  # TODO: 修改为实际路径
```

**支持格式**：
- ✅ VCF: `.vcf`, `.vcf.gz`
- ✅ H5: `.h5`

**Panel 格式**：
```
sample_0  EUR
sample_1  EAS
sample_2  AFR
...
```

#### Frequency 和 Mapping Files

```bash
# Frequency data
FREQ_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy"

# Type to index mapping
TYPE_PATH="data/type_to_idx.bin"  # 相对路径（从项目根目录）

# Population to index mapping
POP_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin"

# Position to index mapping
POS_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin"
```

**说明**：
- 所有路径与训练脚本完全一致
- `TYPE_PATH` 使用相对路径（与训练脚本一致）

---

### 3. 输出路径

```bash
OUTPUT_DIR="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/infer_output_v18"
```

**输出文件**：
```
OUTPUT_DIR/
├── imputed.vcf                    # 填补后的 VCF 文件
├── inference_log.txt              # 推理日志
└── faiss_indexes_infer/           # FAISS 索引（临时）
```

---

### 4. 模型 Checkpoint

```bash
CHECK_POINT="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep11"
```

**说明**：
- 使用训练好的最佳模型
- 通常是验证集 F1 最高的 epoch
- 与 `--output_path` 目录一致

---

## 🔍 路径对比（训练 vs 推理）

| 用途 | 训练路径 | 推理路径 | 说明 |
|------|---------|---------|------|
| **Reference Panel** | `--train_dataset` | `REF_PANEL` | ✅ 一致 |
| **Frequency** | `--freq_path` | `FREQ_PATH` | ✅ 一致 |
| **Type Mapping** | `--type_path` | `TYPE_PATH` | ✅ 一致 |
| **Pop Mapping** | `--pop_path` | `POP_PATH` | ✅ 一致 |
| **Pos Mapping** | `--pos_path` | `POS_PATH` | ✅ 一致 |

---

## ⚠️ 常见错误

### ❌ 错误 1: 模型架构参数不一致

**错误配置**:
```bash
DIMS=384
LAYERS=6   # ❌ 错误！训练时是 12
HEADS=8    # ❌ 错误！训练时是 12
```

**正确配置**:
```bash
DIMS=384
LAYERS=12  # ✅ 正确
HEADS=12   # ✅ 正确
```

**后果**: 加载 checkpoint 失败
```
RuntimeError: size mismatch for transformer_blocks.0.attention.W_q.weight
```

### ❌ 错误 2: TYPE_PATH 使用绝对路径

**错误配置**:
```bash
TYPE_PATH="/cpfs01/.../data/type_to_idx.bin"  # ❌ 绝对路径
```

**正确配置**:
```bash
TYPE_PATH="data/type_to_idx.bin"  # ✅ 相对路径（与训练一致）
```

**原因**: 训练脚本使用相对路径，推理也应该一致

### ❌ 错误 3: Target Dataset 必须是 H5

**误解**: "Target Dataset 只能是 H5 格式"

**真相**: ✅ **支持 VCF 和 H5 格式**

```bash
# VCF 格式（推荐，最常用）
TARGET_DATASET="/path/to/target.vcf.gz"

# H5 格式（也支持）
TARGET_DATASET="/path/to/target.h5"
```

**说明**: `EmbeddingRAGInferDataset` 继承自 `InferDataset`，自动支持 VCF 和 H5

---

## 📝 完整配置示例

### 示例 1: 使用 VCF 格式的 Target

```bash
# 模型
CHECK_POINT="/cpfs01/.../output_v18_embrag/rag_bert.model.ep11"
DIMS=384
LAYERS=12
HEADS=12

# Reference Panel
REF_PANEL="/cpfs01/.../train_split.h5"
REF_PANEL_INFO="/cpfs01/.../train_panel.txt"

# Target Dataset (VCF)
TARGET_DATASET="/cpfs01/.../my_target_data.vcf.gz"  # ← VCF 格式
TARGET_PANEL="/cpfs01/.../my_target_panel.txt"

# Mapping files
FREQ_PATH="/cpfs01/.../Freq.npy"
TYPE_PATH="data/type_to_idx.bin"
POP_PATH="/cpfs01/.../pop_to_idx.bin"
POS_PATH="/cpfs01/.../pos_to_idx.bin"

# Output
OUTPUT_DIR="/cpfs01/.../infer_output_v18"
```

### 示例 2: 使用 H5 格式的 Target

```bash
# ... 其他配置相同 ...

# Target Dataset (H5)
TARGET_DATASET="/cpfs01/.../my_target_data.h5"  # ← H5 格式
TARGET_PANEL="/cpfs01/.../my_target_panel.txt"

# ... 其他配置相同 ...
```

---

## ✅ 验证检查清单

### 启动前验证

```bash
# 1. 检查模型 checkpoint 存在
ls -lh ${CHECK_POINT}

# 2. 检查 Reference Panel 存在
ls -lh ${REF_PANEL}
ls -lh ${REF_PANEL_INFO}

# 3. 检查 Target Dataset 存在
ls -lh ${TARGET_DATASET}
ls -lh ${TARGET_PANEL}

# 4. 检查 Mapping files 存在
ls -lh ${FREQ_PATH}
ls -lh ${TYPE_PATH}
ls -lh ${POP_PATH}
ls -lh ${POS_PATH}

# 5. 检查输出目录可写
mkdir -p ${OUTPUT_DIR} && touch ${OUTPUT_DIR}/test.txt && rm ${OUTPUT_DIR}/test.txt
```

### 参数验证

```bash
# 检查训练参数
grep -E "dims|layers|attn_heads" run_v18_embedding_rag.sh

# 应该看到:
# --dims 384
# --layers 12
# --attn_heads 12
```

---

## 🎯 快速修改指南

**只需修改 2 个参数**:

```bash
# 1. Target Dataset（你的待填补数据）
TARGET_DATASET="/path/to/your/target.vcf.gz"  # ← 修改这里

# 2. Target Panel（你的样本信息）
TARGET_PANEL="/path/to/your/target_panel.txt"  # ← 修改这里
```

**其他参数都已修正，无需修改！**

---

## 📚 相关文档

- **快速开始**: [QUICK_START_V18_INFER.md](QUICK_START_V18_INFER.md)
- **详细指南**: [V18_INFERENCE_GUIDE.md](V18_INFERENCE_GUIDE.md)
- **训练脚本**: [run_v18_embedding_rag.sh](run_v18_embedding_rag.sh)

---

## 🎉 总结

### 核心修正

1. ✅ **模型架构参数**: `LAYERS=12, HEADS=12`（不是 6 和 8）
2. ✅ **所有路径**: 与训练脚本完全一致
3. ✅ **VCF 支持**: Target Dataset 可以是 VCF 或 H5

### 使用方法

```bash
# 1. 拉取代码
git pull origin main

# 2. 修改配置（只需修改 TARGET_DATASET 和 TARGET_PANEL）
vim run_infer_embedding_rag.sh

# 3. 启动推理
bash run_infer_embedding_rag.sh
```

**现在可以正确使用 V18 推理了！🚀**
