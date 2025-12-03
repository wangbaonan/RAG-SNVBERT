# 快速开始指南

**最简单的使用方式 - 3步开始训练**

---

## 🚀 使用V17 (修复版，推荐先用)

### Step 1: 进入目录

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert
```

### Step 2: 确认修复已应用

```bash
grep "use_dynamic_mask=True" src/train_with_val_optimized.py
```

**预期输出**: 应该看到2行 (Line 122和Line 153)

### Step 3: 运行

```bash
bash run_v17_extreme_memory_fix.sh
```

**就这样！** V17会开始训练。

---

## 🚀 使用V18 (新版本，更快)

### Step 1: 进入目录

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert
```

### Step 2: 快速测试 (可选，推荐)

```bash
python test_embedding_rag.py
```

**预期输出**: `✓ All tests passed!`

### Step 3: 运行

```bash
bash run_v18_embedding_rag.sh
```

**就这样！** V18会开始训练。

---

## 📊 如何知道训练正常？

### V17正常日志

```bash
tail -f logs/v17_extreme_memfix/latest.log
```

**应该看到**:
```
Epoch 1 - TRAINING
EP_Train:0: 100%|| 5745/5745 [2:03:15<00:00,  1.29s/it]
Epoch 1 VAL Summary
Avg Loss:      110.27
Avg F1:        0.9505  ← 应该 > 0.93

Epoch 2 - TRAINING
...
Epoch 2 VAL Summary
Avg Loss:      ~105   ← 应该稳定或下降
Avg F1:        ~0.95  ← 应该稳定
```

### V18正常日志

```bash
tail -f logs/v18_embedding_rag/latest.log
```

**应该看到**:
```
▣ 构建Embedding-based RAG索引
✓ 预编码完成! 总耗时: 523s

Epoch 1 - TRAINING
...
Epoch 1 VAL Summary
Avg F1:        0.9500

▣ 刷新Reference Embeddings
✓ 刷新完成! 耗时: 495s

Epoch 2 - TRAINING
...
```

---

## ❌ 异常情况

### 如果V17看到这个 - 异常

```
Epoch 2 VAL Summary
Avg Loss:      355    ← 太高! (应该~110)
Avg F1:        0.86   ← 太低! (应该~0.95)
```

**原因**: dynamic mask修复未应用

**解决**: 检查 `src/train_with_val_optimized.py` Line 122是否有 `use_dynamic_mask=True`

### 如果V18报错 - 检查

```
ModuleNotFoundError: No module named 'af_embedding'
```

**原因**: 新文件不存在

**解决**:
```bash
ls src/model/embedding/af_embedding.py
# 应该存在
```

---

## 🛑 停止训练

```bash
# 找到进程
ps aux | grep python

# 停止
kill -9 <PID>
```

---

## 📈 监控训练

### 实时日志

```bash
# V17
tail -f logs/v17_extreme_memfix/latest.log

# V18
tail -f logs/v18_embedding_rag/latest.log
```

### GPU使用

```bash
watch -n 1 nvidia-smi
```

**正常**: GPU利用率 > 80%, 内存 15-20GB

---

## 🆘 遇到问题？

1. **V17崩溃** → 查看 [V17_REAL_ISSUE_FIXED.md](V17_REAL_ISSUE_FIXED.md)
2. **V18报错** → 查看 [AF_FIX_SUMMARY.md](AF_FIX_SUMMARY.md)
3. **完整指南** → 查看 [CURRENT_CODE_STATUS.md](CURRENT_CODE_STATUS.md)
4. **详细文档** → 查看 [HOW_TO_RUN.md](HOW_TO_RUN.md)

---

**就这么简单！**

选择一个版本，进入目录，运行脚本，训练开始！🚀
