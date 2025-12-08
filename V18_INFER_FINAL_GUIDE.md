# 🎯 V18 推理最终操作指南（已修正所有问题）

## ✅ 最新修正（重要！）

已根据训练脚本修正所有配置：
1. ✅ **模型参数**: `LAYERS=12, HEADS=12`（之前错误写成 6, 8）
2. ✅ **所有路径**: 与训练脚本完全一致
3. ✅ **VCF 支持**: Target Dataset 可以是 VCF 或 H5

---

## 🚀 三步启动（2 分钟）

### 第一步：拉取最新代码

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

git pull origin main
```

**预期输出**：
```
Updating 06e9da2..e581975
Fast-forward
 run_infer_embedding_rag.sh    | 修正路径和参数
 QUICK_START_V18_INFER.md      | 更新文档
 V18_INFER_PATH_CONFIG.md      | 新增路径配置总结
 3 files changed, 331 insertions(+), 16 deletions(-)
```

### 第二步：修改配置（只需 2 个参数！）

```bash
vim run_infer_embedding_rag.sh
```

**只需修改这 2 个参数**：

```bash
# 1. Target Dataset（你的待填补数据）
TARGET_DATASET="/path/to/your/target.vcf.gz"  # ← 修改为你的 VCF 文件路径

# 2. Target Panel（你的样本信息）
TARGET_PANEL="/path/to/your/target_panel.txt"  # ← 修改为你的 Panel 文件路径
```

**其他参数已全部修正，无需修改！**

### 第三步：启动推理

```bash
# 给脚本执行权限
chmod +x run_infer_embedding_rag.sh

# 启动推理
bash run_infer_embedding_rag.sh
```

---

## 📊 正确的配置（已修正）

### 模型架构参数

```bash
DIMS=384         # ✅ 正确
LAYERS=12        # ✅ 已修正（之前错误写成 6）
HEADS=12         # ✅ 已修正（之前错误写成 8）
```

**来源**: 从 `run_v18_embedding_rag.sh` 训练脚本获取

**验证方法**:
```bash
grep -E "dims|layers|attn_heads" run_v18_embedding_rag.sh
# 应该看到: --dims 384 \ --layers 12 \ --attn_heads 12 \
```

### 数据路径（已修正）

**Reference Panel**（已修正路径）:
```bash
REF_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_split.h5"
```

**Target Dataset**（支持 VCF）:
```bash
TARGET_DATASET="/path/to/your/target.vcf.gz"  # ✅ VCF 格式
# 或
TARGET_DATASET="/path/to/your/target.h5"      # ✅ H5 格式
```

**Mapping Files**（已修正）:
```bash
FREQ_PATH="/cpfs01/.../maf_data/Freq.npy"
TYPE_PATH="data/type_to_idx.bin"              # ← 相对路径（与训练一致）
POP_PATH="/cpfs01/.../maf_data/pop_to_idx.bin"
POS_PATH="/cpfs01/.../maf_data/pos_to_idx.bin"
```

---

## 📝 Target Panel 格式

**文件格式** (`target_panel.txt`):
```
sample_0  EUR
sample_1  EAS
sample_2  AFR
sample_3  SAS
sample_4  AMR
...
```

**说明**:
- 第一列: 样本 ID（与 VCF 中的样本名对应）
- 第二列: 人群标签（EUR, EAS, AFR, SAS, AMR 等）
- 分隔符: 空格或制表符

---

## 🔍 完整路径验证

### 验证所有文件存在

```bash
# 切换到项目目录
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 1. 模型 checkpoint
ls -lh /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep11

# 2. Reference Panel
ls -lh data/train_val_split/train_split.h5
ls -lh data/train_val_split/train_panel.txt

# 3. Target Dataset（你的文件）
ls -lh /path/to/your/target.vcf.gz
ls -lh /path/to/your/target_panel.txt

# 4. Mapping files
ls -lh maf_data/Freq.npy
ls -lh data/type_to_idx.bin
ls -lh maf_data/pop_to_idx.bin
ls -lh maf_data/pos_to_idx.bin
```

**如果任何文件不存在，检查路径是否正确。**

### 验证训练参数

```bash
# 查看训练脚本中的参数
grep -E "dims|layers|attn_heads" run_v18_embedding_rag.sh

# 应该看到:
# --dims 384 \
# --layers 12 \
# --attn_heads 12 \
```

---

## ⏱️ 预期时间和行为

### 推理流程

```
Step 1: Loading Vocabulary (10 秒)
  ✓ Vocab size: 2519

Step 2: Loading V18 Model (30 秒)
  - Architecture: dims=384, layers=12, heads=12
  ✓ Model loaded successfully

Step 3: Creating EmbeddingRAGInferDataset (15-20 分钟)
  - Building FAISS indexes with Imputation Masking...
  预编码推理窗口: 100%|████████████████| 50/50 [15:00<00:00]
  ✓ 推理索引构建完成!

Step 4: Creating DataLoader (5 秒)
  ✓ DataLoader created: 63 batches

Step 5: Starting Inference (5-10 分钟)
  Imputing: 100%|████████████████████████| 63/63 [05:30<00:00]
  ✓ Inference completed

Step 6: Generating Imputed VCF (30 秒)
  ✓ VCF file generated

✓ V18 Inference Completed Successfully!
Total time: ~20-30 分钟
```

### 时间估算

| 阶段 | 时间 | 说明 |
|------|------|------|
| 模型加载 | 30 秒 | 一次性 |
| **索引构建** | **15-20 分钟** | **首次必需** |
| 推理 | 5-10 分钟 | 1000 samples |
| VCF 生成 | 30 秒 | 一次性 |
| **总计（首次）** | **20-30 分钟** | 包含���引构建 |
| **总计（后续）** | **5-10 分钟** | 复用索引 |

---

## 📂 输出文件

```
OUTPUT_DIR/
├── imputed.vcf                    # 填补后的 VCF 文件 ← 主要输出
├── inference_log.txt              # 推理日志
└── faiss_indexes_infer/           # FAISS 索引（临时，可删除）
    ├── index_0.faiss
    ├── index_1.faiss
    └── ...
```

### 使用填补后的 VCF

```bash
# 1. 查看前几行
head -20 ${OUTPUT_DIR}/imputed.vcf

# 2. 统计填补位点数
grep -v "^#" ${OUTPUT_DIR}/imputed.vcf | wc -l

# 3. 排序（可选）
bcftools sort ${OUTPUT_DIR}/imputed.vcf -Oz -o ${OUTPUT_DIR}/imputed.sorted.vcf.gz

# 4. 索引（可选）
bcftools index ${OUTPUT_DIR}/imputed.sorted.vcf.gz

# 5. 质量检查（可选）
bcftools stats ${OUTPUT_DIR}/imputed.sorted.vcf.gz > ${OUTPUT_DIR}/stats.txt
```

---

## ⚠️ 常见问题

### ❌ 错误 1: 加载 checkpoint 失败

**错误信息**:
```
RuntimeError: size mismatch for transformer_blocks.0.attention.W_q.weight
```

**原因**: 模型架构参数不一致

**解决方法**: 确认使用正确的参数
```bash
# 查看训练参数
grep -E "dims|layers|attn_heads" run_v18_embedding_rag.sh

# 确认推理脚本中的参数一致
grep -E "DIMS|LAYERS|HEADS" run_infer_embedding_rag.sh
```

### ❌ 错误 2: CUDA Out of Memory

**错误信息**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方法**: 降低 Batch Size
```bash
# 在 run_infer_embedding_rag.sh 中修改
BATCH_SIZE=8   # 从 16 降到 8
```

### ❌ 错误 3: 找不到文件

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方法**: 验证路径
```bash
# 逐个检查文件是否存在
ls -lh ${CHECK_POINT}
ls -lh ${REF_PANEL}
ls -lh ${TARGET_DATASET}
ls -lh ${FREQ_PATH}
```

### ❌ 错误 4: VCF 格式不支持

**误解**: "Target Dataset 必须是 H5 格式"

**真相**: ✅ **完全支持 VCF 格式**

```bash
# VCF 格式（推荐）
TARGET_DATASET="/path/to/target.vcf.gz"

# H5 格式（也支持）
TARGET_DATASET="/path/to/target.h5"
```

---

## 📚 相关文档

| 文档 | 功能 | 推荐 |
|------|------|------|
| [V18_INFER_PATH_CONFIG.md](V18_INFER_PATH_CONFIG.md) | 路径配置总结 | ⭐⭐⭐⭐⭐ |
| [QUICK_START_V18_INFER.md](QUICK_START_V18_INFER.md) | 快速开始（5 分钟） | ⭐⭐⭐⭐⭐ |
| [V18_INFERENCE_GUIDE.md](V18_INFERENCE_GUIDE.md) | 详细指南 | ⭐⭐⭐⭐ |

---

## ✅ 检查清单

### 启动前

- [ ] 已拉取最新代码 (`git pull origin main`)
- [ ] 已修改 `TARGET_DATASET`（你的 VCF 文件路径）
- [ ] 已修改 `TARGET_PANEL`（你的 Panel 文件路径）
- [ ] 已验证所有文件存在（上方验证命令）
- [ ] 已确认模型参数正确（LAYERS=12, HEADS=12）
- [ ] GPU 可用 (`nvidia-smi`)

### 完成后

- [ ] `imputed.vcf` 文件已生成
- [ ] 文件大小合理（非空）
- [ ] 日志中无错误信息
- [ ] 填补的基因型数量符合预期

---

## 🎯 快速命令总结

```bash
# 1. 拉取代码
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main

# 2. 验证文件
ls -lh data/train_val_split/train_split.h5
ls -lh /path/to/your/target.vcf.gz

# 3. 修改配置（只需 2 个参数）
vim run_infer_embedding_rag.sh
# 修改 TARGET_DATASET 和 TARGET_PANEL

# 4. 启动推理
chmod +x run_infer_embedding_rag.sh
bash run_infer_embedding_rag.sh

# 5. 等待完成（20-30 分钟）
# 输出: OUTPUT_DIR/imputed.vcf
```

---

## 🎉 总结

### 核心修正

1. ✅ **模型参数**: `LAYERS=12, HEADS=12`（已修正）
2. ✅ **所有路径**: 与训练脚本一致（已修正）
3. ✅ **VCF 支持**: 明确支持（已说明）

### 用户操作

**只需修改 2 个参数**:
- `TARGET_DATASET`: 你的待填补 VCF 文件
- `TARGET_PANEL`: 你的样本信息文件

**其他参数已全部修正，无需修改！**

### 预期结果

- **时间**: 20-30 分钟（首次）/ 5-10 分钟（后续）
- **输出**: 完整的填补后 VCF 文件
- **格式**: 标准 VCF 格式，可直接使用

**现在可以正确使用 V18 进行基因型填补了！🚀**
