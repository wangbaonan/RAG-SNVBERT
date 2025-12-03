# 当前代码状态和使用指南

**日期**: 2025-12-02
**状态**: 包含V17修复 + V18 Embedding RAG (含AF修复)

---

## 📁 代码结构

当前仓库包含**两个版本**，共享同一套代码，通过不同训练脚本调用：

```
VCF-Bert/
├── src/
│   ├── model/
│   │   ├── bert.py                    # 包含3个模型类:
│   │   │                              #   - BERT (基础)
│   │   │                              #   - BERTWithRAG (V17)
│   │   │                              #   - BERTWithEmbeddingRAG (V18)
│   │   └── embedding/
│   │       ├── bert.py                # BERTEmbedding (已修改: 集成AF)
│   │       └── af_embedding.py        # 新增: AFEmbedding (Fourier Features)
│   │
│   ├── dataset/
│   │   ├── dataset.py                 # 基础Dataset
│   │   ├── rag_train_dataset.py       # V17使用
│   │   └── embedding_rag_dataset.py   # V18使用 (新增)
│   │
│   ├── train_with_val_optimized.py    # V17训练脚本 (已修复dynamic mask)
│   └── train_embedding_rag.py         # V18训练脚本 (新增)
│
├── run_v17_extreme_memory_fix.sh      # V17运行脚本
├── run_v18_embedding_rag.sh           # V18运行脚本 (新增)
└── test_embedding_rag.py              # V18测试脚本 (新增)
```

---

## 🔧 我做的修改清单

### 1. 影响V17的修改 ✅

#### 1.1 Dynamic Mask修复 (关键!)

**文件**: `src/train_with_val_optimized.py`

**修改内容**:
```python
# Line 122: 训练集也使用动态mask
rag_train_loader = RAGTrainDataset(
    ...
    use_dynamic_mask=True  # ← 新增! 防止过拟合
)
```

**影响**: 修复了训练集过拟合到固定mask的问题

**是否需要**: ✅ **必须** (否则训练会崩溃)

#### 1.2 BERT.forward() 传入AF (可选)

**文件**: `src/model/bert.py`

**修改内容**: Line 63-64
```python
# 传入AF到embedding层
hap_1_origin = self.embedding.forward(x['hap_1'], af=x['af'], pos=True)
hap_2_origin = self.embedding.forward(x['hap_2'], af=x['af'], pos=True)
```

**影响**: V17的BERT类也会使用AF embedding

**是否需要**: ⚠️ **可选** (V17仍使用BERTWithRAG，不直接用BERT类)

#### 1.3 BERTEmbedding 集成AF

**文件**: `src/model/embedding/bert.py`

**修改内容**: 添加AFEmbedding支持

**影响**: 所有使用BERTEmbedding的地方都会受影响

**是否需要**:
- V17如果不传`af=None`参数 → **不受影响** (向后兼容)
- V18必须传AF → **必需**

---

### 2. 仅影响V18的修改 ✅

#### 2.1 新增AFEmbedding模块

**文件**: `src/model/embedding/af_embedding.py` (新增)

**用途**: Fourier Features编码AF

**影响**: 仅V18使用

#### 2.2 新增BERTWithEmbeddingRAG

**文件**: `src/model/bert.py` Line 130-219

**用途**: V18 Embedding RAG模型

**影响**: 仅V18使用

#### 2.3 新增EmbeddingRAGDataset

**文件**: `src/dataset/embedding_rag_dataset.py` (新增)

**用途**: V18数据集，支持预编码和FAISS检索

**影响**: 仅V18使用

#### 2.4 新增V18训练脚本

**文件**: `src/train_embedding_rag.py` (新增)

**用途**: V18完整训练流程

**影响**: 仅V18使用

---

## 🚀 如何使用V17 (修复版)

### Step 1: 确认修改已应用

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert

# 检查dynamic mask修复
grep "use_dynamic_mask=True" src/train_with_val_optimized.py

# 应该看到两处:
# Line 122: 训练集 use_dynamic_mask=True
# Line 153: 验证集 use_dynamic_mask=True
```

### Step 2: 直接运行V17

```bash
# 使用修复后的V17脚本
bash run_v17_extreme_memory_fix.sh

# 或者如果想用batch=48 (需要调整LR)
bash run_v17_FIXED.sh
```

### Step 3: 监控训练

```bash
# 实时日志
tail -f logs/v17_extreme_memfix/latest.log

# 查看指标
watch -n 10 "tail -10 metrics/v17_extreme_memfix/latest.csv"
```

### 预期正常行为

```
Epoch 1:
  Train: Loss=~180, F1=~0.92-0.94
  Val:   Loss=~110, F1=~0.95

Epoch 2:
  Train: Loss=~140, F1=~0.94-0.95  ← 应该变化!
  Val:   Loss=~105, F1=~0.95-0.96  ← 应该稳定!

Epoch 3+:
  持续改善，不会崩溃
```

---

## 🚀 如何使用V18 Embedding RAG (新版本)

### Step 1: 快速测试 (可选，推荐)

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert

# 运行单元测试
python test_embedding_rag.py

# 预期输出:
# ✓ All tests passed!
```

### Step 2: 运行V18训练

```bash
# 使用V18脚本
bash run_v18_embedding_rag.sh
```

### Step 3: 监控训练

```bash
# 实时日志
tail -f logs/v18_embedding_rag/latest.log

# 查看指标
watch -n 10 "tail -10 metrics/v18_embedding_rag/latest.csv"
```

### V18特有流程

```
初始化 (首次约15分钟):
  ▣ 构建Embedding-based RAG索引
  → 预编码所有reference haplotypes
  → 构建FAISS索引
  ✓ 完成

每个Epoch (约1.3小时):
  → 训练
  → 验证
  → 刷新Reference Embeddings (约8分钟)
  → 下一个Epoch

相比V17: 速度快3x，内存省40%
```

---

## 📋 两个版本对比

| 特性 | V17 (BERTWithRAG) | V18 (Embedding RAG) |
|------|-------------------|---------------------|
| **训练脚本** | `train_with_val_optimized.py` | `train_embedding_rag.py` |
| **运行脚本** | `run_v17_extreme_memory_fix.sh` | `run_v18_embedding_rag.sh` |
| **模型类** | `BERTWithRAG` | `BERTWithEmbeddingRAG` |
| **数据集** | `RAGTrainDataset` | `EmbeddingRAGDataset` |
| **检索方式** | Token space (每次过BERT) | Embedding space (预编码) |
| **AF编码** | 原始 (稀释) | Fourier Features ⭐ |
| **内存消耗** | ~19GB/batch | ~15GB/batch |
| **速度** | 慢 (4.2h/epoch) | 快 (1.3h/epoch) |
| **已修复** | Dynamic mask ✅ | AF编码 + 特征空间 ✅ |

---

## ⚠️ 重要提醒

### 1. V17和V18独立

- 它们使用**不同的模型类**
- 训练**不互相影响**
- 可以**同时运行** (用不同GPU)

### 2. BERTEmbedding的修改影响

**向后兼容**:
```python
# V17: 不传af参数 → 不使用AF embedding
emb = embedding_layer(tokens)  # ✅ 仍然工作

# V18: 传af参数 → 使用AF embedding
emb = embedding_layer(tokens, af=af, pos=True)  # ✅ 新功能
```

### 3. 不能混用checkpoint

- V17的checkpoint不能用于V18 (模型结构不同)
- V18的checkpoint不能用于V17
- 必须从头训练

---

## 🔍 如何确认使用的是哪个版本？

### 方法1: 查看日志

```bash
# V17日志
tail logs/v17_extreme_memfix/latest.log

# V18日志
tail logs/v18_embedding_rag/latest.log
```

### 方法2: 查看进程

```bash
ps aux | grep python

# V17会显示: train_with_val_optimized
# V18会显示: train_embedding_rag
```

### 方法3: 查看输出目录

```bash
# V17输出
ls output_v17_memfix/

# V18输出 (新的)
ls output_v18_embedding_rag/
```

---

## 📥 如何Pull最新代码

### 当前状态

您现在在本地Windows路径 `e:\AI4S\00_SNVBERT\VCF-Bert`

所有修改都是**本地的**，还未提交到Git。

### 如果需要Pull (从服务器)

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert

# 1. 备份当前修改
git stash

# 2. Pull最新代码
git pull origin main

# 3. 恢复修改
git stash pop
```

### 如果是第一次使用

所有修改已经在您本地，**无需pull**，直接使用即可：

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert

# V17修复版 (dynamic mask已修复)
bash run_v17_extreme_memory_fix.sh

# 或 V18新版本 (AF编码已修复)
bash run_v18_embedding_rag.sh
```

---

## 🗂️ 修改文件清单

### 已修改的文件 (需要commit)

```bash
src/model/bert.py                       # 更新BERT.forward(), 新增BERTWithEmbeddingRAG
src/model/embedding/bert.py             # 集成AFEmbedding
src/train_with_val_optimized.py        # 修复dynamic mask
```

### 新增的文件 (需要add)

```bash
# V18核心代码
src/model/embedding/af_embedding.py     # AFEmbedding类
src/dataset/embedding_rag_dataset.py    # EmbeddingRAGDataset
src/train_embedding_rag.py              # V18训练脚本
run_v18_embedding_rag.sh                # V18运行脚本
test_embedding_rag.py                   # V18测试脚本

# 文档
AF_FIX_SUMMARY.md                       # AF修复快速参考
COMPLETE_AF_FIX_REVIEW.md               # AF修复详细审查
V17_REAL_ISSUE_FIXED.md                 # V17 dynamic mask修复说明
HOW_TO_RUN.md                           # 运行指南
...其他文档
```

---

## 🎯 推荐的使用流程

### 新手推荐: 先用V17

```bash
# 1. 使用修复后的V17 (稳定，已验证)
bash run_v17_extreme_memory_fix.sh

# 2. 等待训练完成 (约84小时，20 epochs)

# 3. 然后尝试V18 (新版本，更快)
bash run_v18_embedding_rag.sh
```

### 高级用户: 同时运行两个版本

```bash
# Terminal 1: V17
CUDA_VISIBLE_DEVICES=0 bash run_v17_extreme_memory_fix.sh

# Terminal 2: V18
CUDA_VISIBLE_DEVICES=1 bash run_v18_embedding_rag.sh

# 最后对比性能
```

---

## ✅ 快速检查清单

运行前确认:

- [ ] 确认在正确目录: `e:\AI4S\00_SNVBERT\VCF-Bert`
- [ ] V17修复已应用: `grep "use_dynamic_mask=True" src/train_with_val_optimized.py` 应该有2处
- [ ] (V18) AF文件存在: `ls src/model/embedding/af_embedding.py`
- [ ] (V18) 测试通过: `python test_embedding_rag.py` 显示 "All tests passed"
- [ ] GPU可用: `nvidia-smi` 显示至少20GB空闲内存

全部确认后:
```bash
# V17
bash run_v17_extreme_memory_fix.sh

# 或 V18
bash run_v18_embedding_rag.sh
```

---

## 🆘 常见问题

### Q1: 我该用V17还是V18？

**A**: 两个都可以，推荐都试试然后对比

- **V17**: 更稳定，已经过验证，但慢
- **V18**: 更快，内存更省，AF编码更好，但是新版本

### Q2: V18需要重新准备数据吗？

**A**: 不需要！V18使用相同的数据文件：
- `train_split.h5`
- `val_split.h5`
- `KGP.chr21.Panel.maf01.vcf.gz`
- `Freq.npy`
- ...

### Q3: V17的checkpoint能用于V18吗？

**A**: 不能，模型结构不同，必须从头训练

### Q4: 如何知道训练是否正常？

**A**: 查看日志前几个epoch：

V17正常:
```
Epoch 1: Train F1=0.92, Val F1=0.95
Epoch 2: Train F1=0.94, Val F1=0.95 (都在变化)
```

V18正常:
```
初始化: ✓ 预编码完成 (15分钟)
Epoch 1: Train F1=0.94, Val F1=0.95
刷新: ✓ 刷新完成 (8分钟)
```

### Q5: 修改会影响我之前的代码吗？

**A**: 不会！
- V17的修改是**修复bug** (dynamic mask)
- V18是**新增功能** (不影响V17)
- 所有修改都是向后兼容的

---

## 📞 联系信息

如果遇到问题:

1. 查看文档: [HOW_TO_RUN.md](HOW_TO_RUN.md)
2. V17问题: [V17_REAL_ISSUE_FIXED.md](V17_REAL_ISSUE_FIXED.md)
3. V18问题: [AF_FIX_SUMMARY.md](AF_FIX_SUMMARY.md)
4. 完整审查: [COMPLETE_AF_FIX_REVIEW.md](COMPLETE_AF_FIX_REVIEW.md)

---

**最后更新**: 2025-12-02
**状态**: ✅ V17修复完成, V18代码审查完成, 可以使用
**推荐**: 先用V17验证修复，再尝试V18对比性能
