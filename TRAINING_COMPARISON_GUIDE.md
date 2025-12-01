# 🎯 训练对比完整指南

## 📋 目录

1. [当前状态](#1-当前状态)
2. [Baseline训练](#2-baseline训练)
3. [后续优化方案](#3-后续优化方案)
4. [如何判断改进效果](#4-如何判断改进效果)
5. [文件清单](#5-文件清单)

---

## 1. 当前状态

### 已完成的工作

✅ **Validation支持**: 从训练集划分15%作为验证集
✅ **Early Stopping**: 5个epoch无改进自动停止
✅ **日志保存**: 完整训练日志保存到文件
✅ **分析脚本**: 自动提取指标和生成图表

### 当前配置 (Baseline)

```python
# 模型参数
dims=128, layers=8, heads=4
train_batch=64, val_batch=128

# Loss配置
Focal Loss: gamma=5  # ← 问题1: 过高
Recon Loss: 30% (0.15+0.15)  # ← 问题2: 权重过大

# RAG配置
K=1 (检索1个参考序列)

# 训练配置
epochs=20, patience=5
学习率: 1e-5, warmup=20k steps
```

### 已识别的问题

| 问题 | 严重程度 | 预期影响 |
|-----|---------|---------|
| **Focal gamma=5** | 🔴 HIGH | 数据利用率<3%, 收敛慢2-3x |
| **Recon loss过重** | 🟡 MEDIUM | 梯度冲突, 可能降低5-10% F1 |
| **动态loss切换** | 🟡 MEDIUM | 训练不稳定 |

---

## 2. Baseline训练

### 🎯 目标

在做任何优化前,先建立baseline性能基准:
1. 记录完整训练日志
2. 观察是否存在过拟合
3. 评估收敛速度
4. 作为后续对比标准

### 📝 运行Baseline

```bash
# 确保在项目根目录
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 运行baseline训练 (保留日志)
bash run_v12_split_val_with_log.sh
```

**预期运行时间**:
- 单个epoch: ~15-20分钟 (取决于GPU)
- 建议先跑5-10个epochs观察趋势

### 📊 实时监控

**终端1: 运行训练**
```bash
bash run_v12_split_val_with_log.sh
```

**终端2: 监控日志**
```bash
# 查看实时输出
tail -f logs/baseline_gamma5_recon30/latest.log

# 只看epoch summary
tail -f logs/baseline_gamma5_recon30/latest.log | grep 'Summary' -A 10

# 只看validation F1
tail -f logs/baseline_gamma5_recon30/latest.log | grep 'VAL Summary' -A 3 | grep 'F1:'
```

### 🔍 关键观察指标

训练5-10个epochs后,观察以下指标:

#### 1. Validation F1趋势

```bash
grep 'VAL Summary' -A 10 logs/baseline_gamma5_recon30/latest.log | grep 'F1:'
```

**健康信号**:
```
Epoch 1: F1: 0.6823
Epoch 2: F1: 0.6956  ← 稳定增长
Epoch 3: F1: 0.7123  ← 稳定增长
Epoch 4: F1: 0.7245
Epoch 5: F1: 0.7312
```

**问题信号**:
```
Epoch 1: F1: 0.6823
Epoch 2: F1: 0.6734  ← 下降
Epoch 3: F1: 0.6912  ← 震荡
Epoch 4: F1: 0.6856  ← 震荡
```

#### 2. 过拟合程度

```bash
# 对比train和val F1
grep -E '(TRAIN|VAL) Summary' -A 5 logs/baseline_gamma5_recon30/latest.log | grep 'F1:'
```

**健康信号**:
```
TRAIN F1: 0.7456
VAL F1:   0.7123  ← Gap = 0.0333 (可接受)
```

**过拟合信号**:
```
TRAIN F1: 0.8234
VAL F1:   0.6823  ← Gap = 0.1411 (严重过拟合!)
```

**判断标准**:
- Gap < 0.05: ✅ 轻微或无过拟合
- Gap 0.05-0.10: ⚠️ 中等过拟合
- Gap > 0.10: 🔴 严重过拟合

#### 3. Loss曲线稳定性

```bash
grep 'Avg Loss:' logs/baseline_gamma5_recon30/latest.log | head -10
```

**健康信号** (平滑下降):
```
Epoch 1 TRAIN: Loss: 0.6234
Epoch 1 VAL:   Loss: 0.6512
Epoch 2 TRAIN: Loss: 0.5987
Epoch 2 VAL:   Loss: 0.6245
Epoch 3 TRAIN: Loss: 0.5756
Epoch 3 VAL:   Loss: 0.6123
```

**问题信号** (震荡):
```
Epoch 1: Loss: 0.6234
Epoch 2: Loss: 0.8123  ← 突然增大!
Epoch 3: Loss: 0.5456  ← 剧烈波动
Epoch 4: Loss: 0.7234
```

#### 4. 收敛速度

```bash
# 查看到达F1>0.7需要多少epochs
grep 'VAL Summary' -A 10 logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | \
    awk '{print NR, $3}' | \
    awk '$2 > 0.7 {print "Epoch", $1, "F1:", $2; exit}'
```

**预期** (基于gamma=5的问题):
- Epochs to F1>0.7: 8-12 (慢)

**优化后预期**:
- Epochs to F1>0.7: 3-5 (快)

---

## 3. 后续优化方案

### 🔧 优化配置对比

| 配置项 | Baseline | 优化方案A | 优化方案B | 优化方案C |
|--------|----------|-----------|-----------|-----------|
| **Focal gamma** | 5 | **2.5** | **2.5** | **2.0** |
| **Recon loss** | 30% | **0%** (移除) | **5%** (降低) | **0%** |
| **预期收益** | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **风险** | - | 低 | 极低 | 极低 |

**推荐**: 优化方案A (最激进,最可能获得显著提升)

### 📝 运行优化版本

**等baseline训练完成后** (或观察5-10个epochs后),运行优化版本:

```bash
# 创建优化配置的运行脚本 (见下文)
bash run_v13_optimized_gamma25_norecon.sh
```

### 🔄 对比分析

训练完成后,对比baseline和优化版本:

```bash
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    logs/optimized_gamma25_norecon/latest.log \
    --labels "Baseline (γ=5, recon=30%)" "Optimized (γ=2.5, no recon)" \
    --compare \
    --output logs/comparison/
```

---

## 4. 如何判断改进效果

### ✅ 成功的优化

**指标1: Val F1提升**
```
Baseline:  Best Val F1 = 0.7123
Optimized: Best Val F1 = 0.7645  ← +0.0522 (+7.3%)

判断: ✅ 显著提升 (>5%)
```

**指标2: 收敛速度**
```
Baseline:  Epochs to F1>0.7: 10
Optimized: Epochs to F1>0.7: 4   ← 提升2.5x

判断: ✅ 收敛更快
```

**指标3: 训练稳定性**
```
Baseline:  Loss震荡, std(loss) = 0.12
Optimized: Loss平滑, std(loss) = 0.03

判断: ✅ 更稳定
```

**指标4: 过拟合程度**
```
Baseline:  Train-Val F1 gap = 0.0856
Optimized: Train-Val F1 gap = 0.0312

判断: ✅ 过拟合减少
```

### ❌ 失败的优化

**信号1**: Val F1下降或无提升 (<1%)
```
Baseline:  Best Val F1 = 0.7123
Optimized: Best Val F1 = 0.7089  ← 下降

判断: ❌ 优化失败
```

**信号2**: 训练不稳定 (loss爆炸/NaN)
```
Epoch 1: Loss = 0.623
Epoch 2: Loss = NaN  ← 训练崩溃

判断: ❌ 优化失败 (参数设置有问题)
```

**信号3**: 严重过拟合加剧
```
Baseline:  Gap = 0.0478
Optimized: Gap = 0.1523  ← 过拟合加剧

判断: ❌ 优化失败 (需要更强正则化)
```

### 📊 完整评估表

| 维度 | Baseline | Optimized | 提升 | 评级 |
|-----|----------|-----------|------|------|
| **Best Val F1** | 0.7123 | 0.7645 | +7.3% | ⭐⭐⭐⭐⭐ |
| **Epochs到最佳** | 10 | 4 | -60% | ⭐⭐⭐⭐⭐ |
| **Loss平滑度** | 0.12 | 0.03 | -75% | ⭐⭐⭐⭐⭐ |
| **过拟合Gap** | 0.0856 | 0.0312 | -64% | ⭐⭐⭐⭐⭐ |
| **总评** | - | - | - | **🏆 优化成功!** |

---

## 5. 文件清单

### 已创建的文件

```
VCF-Bert/
├── run_v12_split_val_with_log.sh      # Baseline训练脚本 (带日志)
├── scripts/
│   └── analyze_training_log.py        # 日志分析脚本
├── logs/                               # 日志目录
│   ├── baseline_gamma5_recon30/       # Baseline日志
│   │   ├── training_YYYYMMDD_HHMMSS.log
│   │   └── latest.log
│   ├── analysis/                      # 单次分析结果
│   └── comparison/                    # 对比结果
├── LOG_GUIDE.md                       # 日志使用详细指南
├── TRAINING_COMPARISON_GUIDE.md       # 本文档
├── MODEL_ARCHITECTURE_ANALYSIS.md     # 模型架构分析
├── FOCAL_LOSS_ANALYSIS.md             # Focal Loss深度分析
├── RECON_LOSS_ANALYSIS.md             # Recon Loss深度分析
└── PRECOMPUTE_ANALYSIS.md             # 预计算可行性分析
```

### 文档阅读顺序

1. **本文档** (TRAINING_COMPARISON_GUIDE.md) - 开始这里
2. **LOG_GUIDE.md** - 详细的日志使用方法
3. **MODEL_ARCHITECTURE_ANALYSIS.md** - 理解优化的理论基础
4. **FOCAL_LOSS_ANALYSIS.md** - 深入理解gamma问题
5. **RECON_LOSS_ANALYSIS.md** - 深入理解recon loss问题

---

## 🚀 下一步行动

### 立即执行 (现在)

```bash
# 1. 确保在正确目录
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 2. 运行baseline训练
bash run_v12_split_val_with_log.sh

# 3. 新开终端监控
tail -f logs/baseline_gamma5_recon30/latest.log | grep 'Summary' -A 10
```

### 训练中 (实时)

```bash
# 每隔1小时检查一次
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    --output logs/analysis/

# 查看生成的图表
ls -lh logs/analysis/*.png
```

### 训练后 (5-10 epochs后)

**决策点**: 根据baseline表现决定是否继续或提前优化

**选项A**: Baseline表现良好 (Val F1 > 0.7, 稳定增长)
→ 继续训练到early stopping

**选项B**: Baseline表现不佳 (Val F1 < 0.65, 震荡)
→ 提前停止,立即应用优化

**选项C**: Baseline表现中等
→ 再训练5个epochs观察

### 应用优化 (baseline完成后)

```bash
# 1. 创建优化脚本 (需要修改pretrain_with_val.py)
# 见下一节

# 2. 运行优化训练
bash run_v13_optimized_gamma25_norecon.sh

# 3. 对比分析
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    logs/optimized_gamma25_norecon/latest.log \
    --compare --output logs/comparison/

# 4. 查看对比结果
cat logs/comparison/comparison.txt
```

---

## 📞 如果遇到问题

### 问题1: 日志文件为空

**检查**:
```bash
ls -lh logs/baseline_gamma5_recon30/
```

**解决**:
```bash
# 确保目录存在
mkdir -p logs/baseline_gamma5_recon30

# 检查tee命令
which tee

# 手动重定向 (如果tee不可用)
python -m src.train_with_val ... > logs/baseline.log 2>&1
```

### 问题2: 训练崩溃/NaN

**检查日志**:
```bash
grep -i 'nan\|inf\|error' logs/baseline_gamma5_recon30/latest.log
```

**可能原因**:
- 学习率过大
- 梯度爆炸
- 数据问题

**应急方案**:
```bash
# 降低学习率重试
# 修改 src/train_with_val.py:
lr = 5e-6  # 从1e-5降低
```

### 问题3: 显存不足

**检查显存**:
```bash
nvidia-smi
```

**解决**:
```bash
# 降低batch size
--train_batch_size 32  # 从64降到32
--val_batch_size 64    # 从128降到64
```

---

## 🎯 预期时间线

```
Day 1 (今天):
├─ 00:00 - 启动baseline训练
├─ 00:15 - 第一个epoch完成,初步观察
├─ 01:30 - 5个epochs完成,分析趋势
└─ 03:00 - 10个epochs,决定是否继续

Day 2 (如果继续):
├─ 00:00 - Baseline训练完成 (假设20 epochs)
├─ 00:30 - 完整分析baseline结果
├─ 01:00 - 准备优化版本代码
└─ 01:30 - 启动优化训练

Day 3:
├─ 00:00 - 优化训练完成
└─ 00:30 - 对比分析,得出结论
```

---

## 📚 参考文档

- [LOG_GUIDE.md](LOG_GUIDE.md) - 日志使用详细指南
- [MODEL_ARCHITECTURE_ANALYSIS.md](MODEL_ARCHITECTURE_ANALYSIS.md) - 架构分析
- [FOCAL_LOSS_ANALYSIS.md](FOCAL_LOSS_ANALYSIS.md) - Gamma深度分析
- [RECON_LOSS_ANALYSIS.md](RECON_LOSS_ANALYSIS.md) - Recon Loss分析
- [FINAL_SOLUTION.md](FINAL_SOLUTION.md) - 之前的validation解决方案

---

**现在开始baseline训练,观察几个epochs后再决定下一步!** 🚀

记住: **不要急于优化,先建立可靠的baseline基准!**
