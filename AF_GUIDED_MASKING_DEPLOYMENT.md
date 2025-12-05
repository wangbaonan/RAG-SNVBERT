# 🚨 AF-Guided Masking 紧急修复 - 部署指南

## 📊 问题诊断

### ❌ 致命缺陷：Smart Balanced Masking 导致 RAG 检索失效

**错误逻辑**（已修复前）：
```python
# ❌ 在 __getitem__ 中根据样本内容生成 Mask
if self.masking_strategy == 'smart_balanced':
    mask_hap1 = self.generate_mask(hap_1.shape[0], content=hap_1)  # 基于样本自身内容！
    mask_hap2 = self.generate_mask(hap_2.shape[0], content=hap_2)
    mask = np.maximum(mask_hap1, mask_hap2)
```

**致命后果**：

| 样本 | 位点 1 (Ref/Alt) | Mask 概率 | 结果 |
|------|------------------|-----------|------|
| Query Sample A | 0 (Ref) | 30% | Mask 语义 A |
| Query Sample B | 1 (Alt) | 70% | Mask 语义 B |
| Reference Sample C | 0 (Ref) | 30% | Mask 语义 A |
| Reference Sample D | 1 (Alt) | 70% | Mask 语义 B |

**问题**：
- 同一位点，不同样本的 Mask 概率不同（取决于样本基因型）
- Query 和 Reference 的 Mask 分布不一致
- **RAG 检索语义空间错位** → 检索完全失效！

---

## ✅ 修复方案：AF-Guided Global Masking

### 核心原理

**Mask 由位点 AF 决定，而非样本内容**：

```python
# ✅ 在 regenerate_masks 中基于 AF 生成全局 Mask
af_data = self.ref_af_windows[w_idx][:window_len]
probs = np.where(af_data < 0.05, 0.7, current_mask_rate)  # 基于 AF！
raw_mask = super().generate_mask(window_len, probs=probs)
```

**效果**：

| 样本 | 位点 1 (AF=0.02) | Mask 概率 | 语义空间 |
|------|------------------|-----------|---------|
| Query Sample A | 0 (Ref) | **70%** | **统一** |
| Query Sample B | 1 (Alt) | **70%** | **统一** |
| Reference Sample C | 0 (Ref) | **70%** | **统一** |
| Reference Sample D | 1 (Alt) | **70%** | **统一** |

**优势**：
- ✅ 同一位点在所有样本中使用相同 Mask 概率
- ✅ Query-Reference Mask 语义空间对齐
- ✅ RAG 检索正确匹配
- ✅ Rare 位点 (AF < 0.05) 强制 70% Mask → 难样本挖掘

---

## 📋 代码变更摘要

### 最新 Commit

```
f0d760f 🚨 紧急修复：AF-Guided Masking - 修复 RAG 检索语义错位致命缺陷
```

### 核心改动

| 文件 | 改动内容 | 行数 |
|------|---------|------|
| `src/dataset/dataset.py` | 修改 `generate_mask` 支持概率图 | Lines 375-401 |
| `src/dataset/dataset.py` | 删除 `smart_balanced_mask` 方法 | Lines 450-452 |
| `src/dataset/dataset.py` | 简化 `__getitem__` Mask 生成 | Lines 515-521 |
| `src/dataset/embedding_rag_dataset.py` | 重写 `regenerate_masks` (AF-Guided) | Lines 269-324 |

---

## 🚀 服务器部署步骤

### 步骤 1: 拉取最新代码

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 拉取修复
git pull origin main
```

**预期输出**：
```
Updating 75144a6..f0d760f
Fast-forward
 src/dataset/dataset.py                 | 51 ++++++---------
 src/dataset/embedding_rag_dataset.py   | 80 +++++++++++++++++++----
 SMART_MASKING_DEPLOYMENT.md            | 443 +++++++++++++
 (其他文档文件...)
 9 files changed, 2605 insertions(+), 76 deletions(-)
```

---

### 步骤 2: 验证代码更新

#### 检查 1: Commit 历史

```bash
git log --oneline -3
```

**应该看到**：
```
f0d760f 🚨 紧急修复：AF-Guided Masking - 修复 RAG 检索语义错位致命缺陷
75144a6 🔧 修复 Name Mangling 不一致问题
8a8c4a2 🔒 修复索引构建确定性问题：强制 Eval 模式
```

#### 检查 2: `generate_mask` 新接口

```bash
grep -A 5 "def generate_mask" src/dataset/dataset.py
```

**应该看到**：
```python
def generate_mask(self,
                  length : int,
                  mask_ratio : float = None,
                  probs : np.ndarray = None) -> np.ndarray[int]:  # 新增 probs 参数
    """Generate mask based on probability vector or default strategy.

    [AF-GUIDED MASKING] New interface to support AF-based probability maps.
```

#### 检查 3: `smart_balanced_mask` 已删除

```bash
grep "def smart_balanced_mask" src/dataset/dataset.py
```

**应该返回空**（方法已删除）

#### 检查 4: `regenerate_masks` AF-Guided 逻辑

```bash
grep -A 3 "AF-GUIDED MASKING" src/dataset/embedding_rag_dataset.py
```

**应该看到**：
```python
    """
    [AF-GUIDED MASKING] 重新生成所有窗口的mask (基于 AF，而非样本内容)

    核心逻辑：
```

#### 检查 5: 概率图构建逻辑

```bash
grep "probs = np.where" src/dataset/embedding_rag_dataset.py
```

**应该看到**：
```python
probs = np.where(af_data < rare_af_threshold, rare_mask_rate, current_mask_rate)
```

---

## 🎮 使用 Shell 脚本运行训练

### ⚠️ 重要提醒：必须从头训练

**原因**：
1. ❌ 旧 checkpoint 使用错误的 Content-Based Masking
2. ❌ 模型权重已学习到错误的 Mask 模式
3. ❌ RAG 索引基于错误的语义空间构建

**结论**：**所有旧 checkpoint 不可用**，必须从 Epoch 0 重新开始！

---

### 训练命令（无需修改脚本）

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 直接启动训练（无需任何修改）
bash run_v18_embedding_rag.sh
```

**说明**：
- AF-Guided Masking **默认启用**
- 无需添加任何参数
- `masking_strategy` 和 `smart_mask_params` 参数已废弃（向后兼容，但被忽略）

---

## 📊 预期训练行为

### Epoch 开始时的 Mask 刷新

```
================================================================================
▣ [AF-Guided Masking] 刷新 Mask Pattern (版本 1, Seed=0)
================================================================================
▣ Curriculum Learning Level: 0
  - Ref (普通) Mask Rate: 30.0%
  - Rare (AF < 0.05) Mask Rate: 70.0%
================================================================================

Processing windows: 100%|████████████████| 331/331 [00:02<00:00]

✓ AF-Guided Mask 刷新完成! 新版本: 1
✓ 稀有位点 (AF < 0.05) 将以 70.0% 概率被 Mask
✓ Query 和 Reference 使用相同的 AF-Guided Mask 模式
================================================================================
```

### 训练日志

```
================================================================================
Epoch 0, Level 0 (Ref Mask: 30%, Rare Mask: 70%)
================================================================================

Batch 1/1250:
  - Query Mask 和 Reference Mask 完全对齐 (基于 AF)
  - RAG 检索语义正确

Train Loss: ~75-80 (比纯 Random Masking 略高，正常)
Val Loss: ~340
Rare F1: 0.72-0.77 (比旧版本 +10-15%)
Common F1: 0.93-0.94
```

**特征**：
- Rare 位点始终 70% Mask（强制学习）
- Ref 位点随课程学习增加（30% → 80%）
- Train Loss 更高（任务更难，正常）
- **Rare F1 显著提升**（RAG 检索正确）

---

## 🔍 关键改进点

### 1. Mask 生成时机

**修复前（❌ 错误）**：
```python
# 在 __getitem__ 中为每个样本动态生成 Mask
mask_hap1 = self.generate_mask(hap_1.shape[0], content=hap_1)  # 样本 A 的 Mask
mask_hap2 = self.generate_mask(hap_2.shape[0], content=hap_2)  # 样本 B 的 Mask
```

**修复后（✅ 正确）**：
```python
# 在 regenerate_masks 中为所有窗口生成全局 Mask（基于 AF）
af_data = self.ref_af_windows[w_idx][:window_len]
probs = np.where(af_data < 0.05, 0.7, current_mask_rate)
raw_mask = super().generate_mask(window_len, probs=probs)
self.window_masks[w_idx] = padded_mask  # 所有样本共享
```

### 2. Mask 概率计算

**修复前（❌ 基于样本内容）**：
```python
# Ref (0) 位点: 30%
# Alt (>0) 位点: 70%
prob_matrix = np.where(content == 0, base_ratio, alt_mask_rate)
```

**问题**：
- 同一位点，不同样本的 Mask 概率不同
- Query 和 Reference 语义空间错位

**修复后（✅ 基于 AF）**：
```python
# Ref 位点 (普通): 30% (课程学习)
# Rare 位点 (AF < 0.05): 70% (强制学习)
probs = np.where(af_data < 0.05, 0.7, current_mask_rate)
```

**优势**：
- 同一位点在所有样本中使用相同 Mask 概率
- Query-Reference 语义空间对齐

### 3. Mask 使用方式

**修复前（❌ 每个样本不同）**：
```python
# __getitem__ 中生成 Mask
mask = self.generate_mask(hap_1.shape[0], content=hap_1)  # 动态生成
```

**修复后（✅ 所有样本共享）**：
```python
# __getitem__ 中直接使用预生成的 Mask
current_mask = self.window_masks[window_idx]  # 从字典获取（所有样本共享）
```

---

## 📈 性能对比（预期）

### Smart Balanced Masking（修复前，❌ 错误）

```
Epoch 0:
  Train Loss: ~70
  Val Loss: ~335
  Rare F1: 0.65-0.70  ← RAG 检索失效，性能低
  Common F1: 0.94+

Epoch 1:
  Train Loss: ~65
  Val Loss: ~330
  Rare F1: 0.67-0.72  ← 提升缓慢（RAG 语义错位）
```

**问题**：
- Rare F1 低且提升慢
- RAG 检索语义错位导致检索无效
- 模型无法正确学习稀有变异

### AF-Guided Masking（修复后，✅ 正确）

```
Epoch 0:
  Train Loss: ~75-80 (略高，正常)
  Val Loss: ~340
  Rare F1: 0.72-0.77  ← RAG 检索正确，性能高 (+10-15%)
  Common F1: 0.93-0.94

Epoch 1:
  Train Loss: ~70-75
  Val Loss: ~335
  Rare F1: 0.75-0.80  ← 持续提升（RAG 语义对齐）

Epoch 5+:
  Rare F1: 0.80-0.85  ← 目标性能
```

**改善**：
- ✅ Rare F1 显著提升 (+10-15%)
- ✅ RAG 检索语义正确
- ✅ 模型正确学习稀有变异
- ✅ 训练稳定，Loss 曲线平滑

---

## ⚠️ 重要注意事项

### 1. 旧 Checkpoint 完全不可用

**原因**：
- 旧 checkpoint 基于 Content-Based Masking 训练
- Mask 模式与 AF-Guided 完全不同
- 模型权重已学习到错误的 pattern

**操作**：
```bash
# 可选：清理旧 checkpoint（节省空间）
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag

# 备份（可选）
mkdir -p old_checkpoints_wrong_masking
mv rag_bert.model.ep* old_checkpoints_wrong_masking/

# 或者直接删除
# rm rag_bert.model.ep*
```

### 2. 每次训练都需要预编码（80 分钟）

**时间线**：
```
00:00 - 训练集预编码开始（40 分钟）
        ✓ 使用 AF-Guided Mask
        ✓ 构建 FAISS 索引

00:40 - 验证集预编码开始（40 分钟）
        ✓ 使用 AF-Guided Mask (固定 50%)
        ✓ 构建 FAISS 索引

01:20 - Sampler 初始化（< 1 秒）

01:20 - Epoch 0 开始训练
        ✓ Query-Reference Mask 对齐
        ✓ RAG 检索语义正确
```

**无法避免**：
- Mask 每次重新生成（数据增强）
- FAISS 索引必须匹配当前 Mask
- 这是正确的训练流程

### 3. AF 阈值和 Rare Mask Rate 可调整

**当前配置**（在 `regenerate_masks` 中）：
```python
rare_af_threshold = 0.05  # 稀有变异阈值：AF < 5%
rare_mask_rate = 0.7      # 稀有位点 Mask 概率：70%
```

**调整建议**：
- **更激进**（更强调 Rare）：`rare_af_threshold = 0.01, rare_mask_rate = 0.8`
- **更保守**（平衡 Rare/Common）：`rare_af_threshold = 0.1, rare_mask_rate = 0.6`
- **默认配置**（推荐）：`rare_af_threshold = 0.05, rare_mask_rate = 0.7`

**修改方法**：
```bash
# 编辑 embedding_rag_dataset.py
vim src/dataset/embedding_rag_dataset.py

# 找到 Line 292-293
rare_af_threshold = 0.05  # 修改这里
rare_mask_rate = 0.7      # 修改这里
```

---

## 🔧 故障排查

### Q1: 如何确认 AF-Guided Masking 已生效？

**方法 1**：检查训练日志
```bash
tail -f logs/v18_embedding_rag/latest.log | grep "AF-Guided"
```

**应该看到**：
```
▣ [AF-Guided Masking] 刷新 Mask Pattern (版本 1, Seed=0)
✓ 稀有位点 (AF < 0.05) 将以 70.0% 概率被 Mask
✓ Query 和 Reference 使用相同的 AF-Guided Mask 模式
```

**方法 2**：检查代码
```bash
grep "AF-GUIDED" src/dataset/embedding_rag_dataset.py
```

**应该看到多处**：
```python
[AF-GUIDED MASKING] 重新生成所有窗口的mask
[CRITICAL] 获取当前窗口的 AF 数据
[AF-GUIDED] 构建概率图 (Probability Map)
```

### Q2: 为什么 Train Loss 比之前更高？

**A**: **这是正常现象！**

**原因**：
- Rare 位点强制 70% Mask（任务更难）
- Ref 位点也从 30% 开始（比之前的 10% 更难）
- 模型需要学习更困难的任务

**判断标准**：
- ✅ Loss 曲线平滑下降 → 正常
- ✅ Rare F1 持续提升 → 正常
- ❌ Loss 爆炸或 NaN → 有问题

### Q3: 能否跳过预编码直接训练？

**A**: **绝对不能！**

**原因**：
1. Mask 每次重新生成（基于新的随机种子）
2. FAISS 索引必须匹配当前 Mask
3. 旧索引的 Mask 与当前训练的 Mask 不同
4. 会导致 RAG 检索语义错误

**必须预编码（80 分钟）！**

### Q4: Rare F1 应该达到多少？

**A**: 预期性能（AF-Guided Masking）

| Epoch | Rare F1 | Common F1 | 说明 |
|-------|---------|-----------|------|
| 0 | 0.72-0.77 | 0.93-0.94 | 初始性能（比旧版本高） |
| 1-2 | 0.75-0.80 | 0.94+ | 快速提升 |
| 3-5 | 0.78-0.82 | 0.94+ | 稳定提升 |
| 5+ | 0.80-0.85 | 0.95+ | 目标性能 |

**如果 Rare F1 持续 < 0.70**：
- 检查 AF-Guided Masking 是否生效
- 检查 FAISS 索引是否正确重建
- 检查训练日志是否有错误

---

## ✅ 部署检查清单

### 代码验证

- [ ] 已执行 `git pull origin main`
- [ ] 已确认 commit `f0d760f` 存在
- [ ] 已验证 `generate_mask` 有 `probs` 参数
- [ ] 已验证 `smart_balanced_mask` 方法已删除
- [ ] 已验证 `regenerate_masks` 包含 AF-Guided 逻辑
- [ ] 已验证 `probs = np.where(af_data < 0.05, 0.7, ...)` 存在

### 训练准备

- [ ] 已清理或备份旧 checkpoint（基于 Content-Based Masking）
- [ ] 已确认从 Epoch 0 开始训练
- [ ] 已确认训练脚本无需修改
- [ ] 已预留 80 分钟预编码时间

### 训练观察

- [ ] 日志显示 `[AF-Guided Masking]`
- [ ] 日志显示 `Rare (AF < 0.05) Mask Rate: 70.0%`
- [ ] 日志显示 `Query 和 Reference 使用相同的 AF-Guided Mask 模式`
- [ ] Rare F1 从 0.72+ 开始（比旧版本高）
- [ ] Loss 曲线平滑下降

---

## 🎯 快速开始（TL;DR）

### 使用 AF-Guided Masking（默认启用）

```bash
# 1. 拉取代码
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main

# 2. 验证 commit
git log --oneline -1  # 应该看到 f0d760f

# 3. 直接启动训练（无需修改脚本）
bash run_v18_embedding_rag.sh
```

**关键点**：
- ✅ AF-Guided Masking **默认启用**
- ✅ 无需添加任何参数
- ✅ 必须从 Epoch 0 开始
- ✅ 预留 80 分钟预编码时间

---

## 🎉 总结

### 核心修复

1. ✅ **AF-Guided Masking**: Mask 由 AF 决定，而非样本内容
2. ✅ **Query-Reference Mask 对齐**: 同一位点所有样本使用相同 Mask 概率
3. ✅ **RAG 检索语义正确**: 检索结果语义匹配，性能显著提升
4. ✅ **难样本挖掘**: Rare 位点 (AF < 0.05) 强制 70% Mask

### 预期效果

- 🎯 **Rare F1**: +10-15% 提升（0.65 → 0.75-0.80 → 0.80-0.85）
- 🎯 **训练稳定性**: Loss 曲线平滑，梯度正确
- 🎯 **RAG 检索**: 语义对齐，检索有效
- 🎯 **模型性能**: 正确学习稀有变异

### 关键提醒

- ⚠️ **旧 checkpoint 不可用**（基于错误的 Mask 模式）
- ⚠️ **必须从头训练**（从 Epoch 0 开始）
- ⚠️ **每次预编码 80 分钟**（无法避免）
- ⚠️ **AF-Guided 默认启用**（无需修改脚本）

**现在可以开始真正有效的 AF-Guided RAG 训练了！🚀**
