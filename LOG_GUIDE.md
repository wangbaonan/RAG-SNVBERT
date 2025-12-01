# 训练日志使用指南

## 📝 快速开始

### 1. 运行带日志的训练

```bash
# 使用带日志保存的脚本
bash run_v12_split_val_with_log.sh
```

**日志保存位置**: `logs/baseline_gamma5_recon30/training_YYYYMMDD_HHMMSS.log`

**特点**:
- ✅ 同时输出到终端和文件
- ✅ 每次运行创建新的时间戳文件
- ✅ 创建 `latest.log` 符号链接指向最新日志
- ✅ 捕获所有stdout和stderr

---

## 📊 实时监控训练

### 方法1: 查看实时日志

```bash
# 查看最新日志
tail -f logs/baseline_gamma5_recon30/latest.log

# 只看epoch summary
tail -f logs/baseline_gamma5_recon30/latest.log | grep -A 10 'Summary'

# 只看validation F1
tail -f logs/baseline_gamma5_recon30/latest.log | grep 'VAL Summary' -A 3 | grep 'F1:'
```

### 方法2: 提取关键指标

```bash
# 提取所有epoch的validation F1
grep 'VAL Summary' -A 10 logs/baseline_gamma5_recon30/latest.log | grep 'F1:'

# 提取训练和验证F1 (对比过拟合)
grep -E '(TRAIN|VAL) Summary' -A 5 logs/baseline_gamma5_recon30/latest.log | grep 'F1:'

# 查看最佳验证性能
grep 'New best' logs/baseline_gamma5_recon30/latest.log
```

---

## 🔍 训练后分析

### 单次训练分析

```bash
# 分析最新训练
python scripts/analyze_training_log.py logs/baseline_gamma5_recon30/latest.log

# 生成图表
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/training_20250101_120000.log \
    --output logs/analysis/
```

**输出示例**:
```
============================================================
Analyzing: logs/baseline_gamma5_recon30/training_20250101_120000.log
============================================================

📊 Training Summary:
  Total epochs: 20
  Train samples: 20
  Val samples: 20

🏆 Best Validation Performance:
  Epoch: 8
  Val F1: 0.7345
  Val Precision: 0.7421
  Val Recall: 0.7271

📈 Training at Best Val Epoch:
  Train F1: 0.7823
  Overfitting Gap: 0.0478

📉 Final Epoch Performance:
  Epoch: 20
  Val F1: 0.7329
  ⚠️  Performance degraded from best by 0.0016

⏱️  Convergence Speed:
  Epochs to F1>0.6: 2
  Epochs to F1>0.7: 5

📊 Plot saved to: logs/analysis/training_20250101_120000_analysis.png
```

**生成的图表包含**:
- F1曲线 (Train vs Val)
- Loss曲线 (Train vs Val)
- Precision/Recall/F1对比
- Overfitting gap随epoch变化

---

## 📈 对比不同配置

### 对比baseline和优化版本

```bash
# 假设你运行了两个版本:
# 1. Baseline (gamma=5, recon=30%)
# 2. Optimized (gamma=2.5, no recon)

python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/training_20250101_120000.log \
    logs/optimized_gamma25_norecon/training_20250101_140000.log \
    --labels "Baseline (gamma=5)" "Optimized (gamma=2.5)" \
    --compare \
    --output logs/comparison/
```

**输出对比表**:
```
Run                            Best Val F1  @Epoch   Final Val F1 Overfitting
--------------------------------------------------------------------------------
Baseline (gamma=5)             0.7345       8        0.7329       0.0478
Optimized (gamma=2.5)          0.7856       5        0.7841       0.0312
```

**结论**:
- ✅ Optimized版本F1提升: 0.7345 → 0.7856 (+0.0511, +7%)
- ✅ 收敛更快: 8 epochs → 5 epochs
- ✅ 过拟合更少: 0.0478 → 0.0312

---

## 🗂️ 日志目录结构

```
logs/
├── baseline_gamma5_recon30/         # 当前配置 (gamma=5, recon=30%)
│   ├── training_20250101_120000.log
│   ├── training_20250101_140000.log
│   └── latest.log -> training_20250101_140000.log
│
├── optimized_gamma25_norecon/       # 优化配置 (gamma=2.5, no recon)
│   ├── training_20250101_150000.log
│   └── latest.log
│
├── optimized_gamma25_lowrecon/      # 另一个配置 (gamma=2.5, recon=5%)
│   └── ...
│
├── analysis/                        # 单次分析结果
│   ├── training_20250101_120000_analysis.png
│   └── ...
│
└── comparison/                      # 对比分析结果
    ├── comparison.png
    └── ...
```

---

## 🎯 推荐工作流

### 阶段1: Baseline训练 (当前)

```bash
# 1. 运行baseline (不修改代码)
bash run_v12_split_val_with_log.sh

# 2. 实时监控
tail -f logs/baseline_gamma5_recon30/latest.log | grep 'Summary' -A 10

# 3. 训练几个epoch后分析
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    --output logs/analysis/
```

**观察指标**:
- ✅ Val F1是否稳定增长
- ✅ Train F1 - Val F1的gap (过拟合程度)
- ✅ Loss曲线是否平滑 (训练稳定性)
- ✅ 是否触发early stopping

---

### 阶段2: 应用优化 (修改gamma和recon)

```bash
# 1. 创建优化版本的运行脚本
# (见下文 run_v13_optimized_with_log.sh)

# 2. 运行优化版本
bash run_v13_optimized_with_log.sh

# 3. 同时对比两个版本
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    logs/optimized_gamma25_norecon/latest.log \
    --labels "Baseline" "Optimized" \
    --compare \
    --output logs/comparison/
```

**判断标准**:
```
优化成功的信号:
✅ Best Val F1 提升 > 3%
✅ 收敛速度提升 (减少30%+ epochs)
✅ Loss曲线更平滑
✅ Overfitting gap < 0.05

优化失败的信号:
❌ Val F1下降
❌ 训练不稳定 (loss震荡)
❌ 过拟合加剧
```

---

## 🔧 快速命令参考

### 查看训练进度

```bash
# 当前epoch
grep 'Epoch.*Summary' logs/baseline_gamma5_recon30/latest.log | tail -2

# 最佳F1
grep 'New best f1:' logs/baseline_gamma5_recon30/latest.log | tail -1

# Early stopping触发
grep 'Early stopping' logs/baseline_gamma5_recon30/latest.log
```

### 提取数据到CSV

```bash
# 提取validation F1到CSV
grep 'VAL Summary' -A 10 logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | \
    awk '{print NR","$3}' > val_f1.csv

# 查看CSV
cat val_f1.csv
# 输出:
# 1,0.6823
# 2,0.6956
# 3,0.7123
# ...
```

### 对比两次训练的关键epoch

```bash
# Baseline的epoch 10
grep 'Epoch 10 VAL Summary' -A 15 logs/baseline_gamma5_recon30/latest.log

# Optimized的epoch 10
grep 'Epoch 10 VAL Summary' -A 15 logs/optimized_gamma25_norecon/latest.log
```

---

## 📋 日志文件格式示例

```
============================================================
Epoch 1 TRAIN Summary
============================================================
Avg Loss:      0.6234
Avg Accuracy:  0.7123

Haplotype Metrics:
  - F1:        0.7045
  - Precision: 0.7189
  - Recall:    0.6905

Genotype Metrics:
  - F1:        0.6823
  - Precision: 0.6945
  - Recall:    0.6705
============================================================

============================================================
Epoch 1 VAL Summary
============================================================
Avg Loss:      0.6512
Avg Accuracy:  0.6987

Haplotype Metrics:
  - F1:        0.6823
  - Precision: 0.6945
  - Recall:    0.6705

Genotype Metrics:
  - F1:        0.6512
  - Precision: 0.6634
  - Recall:    0.6395
============================================================

✓ New best f1: 0.6823
EP:1 Model Saved: .../rag_bert.model.best.pth
```

---

## ⚠️ 注意事项

1. **日志文件大小**
   - 每个epoch约50-100 KB
   - 20 epochs约1-2 MB
   - 定期清理旧日志

2. **磁盘空间**
   - 确保logs目录有足够空间
   - 建议保留10GB以上

3. **时间戳格式**
   - 格式: `YYYYMMDD_HHMMSS`
   - 示例: `20250101_120534`
   - 便于按时间排序

4. **符号链接**
   - `latest.log`始终指向最新日志
   - 如果不存在,检查是否支持符号链接

---

## 🚀 高级技巧

### 1. 自动邮件通知 (训练完成)

```bash
# 在run脚本末尾添加
bash run_v12_split_val_with_log.sh
# 训练完成后发送邮件
echo "Training finished. Best Val F1: $(grep 'New best f1:' logs/baseline_gamma5_recon30/latest.log | tail -1)" | \
    mail -s "Training Complete" your_email@example.com
```

### 2. 自动对比baseline

```bash
# 创建自动对比脚本
cat > scripts/auto_compare.sh << 'EOF'
#!/bin/bash
BASELINE_LOG="logs/baseline_gamma5_recon30/latest.log"
NEW_LOG="logs/optimized_gamma25_norecon/latest.log"

if [ -f "$BASELINE_LOG" ] && [ -f "$NEW_LOG" ]; then
    python scripts/analyze_training_log.py \
        $BASELINE_LOG $NEW_LOG \
        --labels "Baseline" "New" \
        --compare \
        --output logs/comparison/
    echo "✓ Comparison saved to logs/comparison/comparison.png"
else
    echo "❌ Logs not found"
fi
EOF

chmod +x scripts/auto_compare.sh
bash scripts/auto_compare.sh
```

### 3. 监控训练异常

```bash
# 检测NaN loss
watch -n 10 "grep 'NaN\|nan\|inf' logs/baseline_gamma5_recon30/latest.log | tail -5"

# 检测训练停止
watch -n 30 "tail -1 logs/baseline_gamma5_recon30/latest.log"
```

---

## 📞 常见问题

**Q: 日志文件没有创建?**
A: 检查logs目录权限: `mkdir -p logs/baseline_gamma5_recon30`

**Q: tee命令不识别?**
A: Windows用户可能需要Git Bash或WSL

**Q: 如何在Windows查看实时日志?**
A: 使用PowerShell: `Get-Content logs\baseline_gamma5_recon30\latest.log -Wait -Tail 50`

**Q: 符号链接不工作?**
A: Windows需要管理员权限,或直接使用时间戳文件

**Q: 如何快速找到最佳epoch?**
A: `grep 'New best f1:' logs/baseline_gamma5_recon30/latest.log`

---

现在你可以开始baseline训练并保留完整日志了！🚀
