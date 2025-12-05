# 架构优化部署指南 - 单一事实来源 + 验证集策略修正

## 🎯 本次修复内容

### 修复 1: 单一事实来源 (Robustness)
**问题**: `__getitem__` 硬编码了 `window_idx` 计算逻辑，与父类解耦性差
**解决**: 优先从父类输出获取 `window_idx`，仅在缺失时回退到计算

### 修复 2: 验证集策略修正 (Critical!)
**问题**: 验证集 Mask 每个 Epoch 都刷新，导致 Loss 不可比
**解决**: 禁用验证集 Mask 刷新，保持题目固定，仅更新索引（答案）

---

## 📋 服务器部署步骤

### 步骤 1: 拉取最新代码

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 拉取最新修复
git pull origin main
```

**应该看到**:
```
Updating 59174b1..2ad4dd5
Fast-forward
 src/dataset/embedding_rag_dataset.py | XX ++++++++++---
 src/train_embedding_rag.py          | YY +++--
 2 files changed, 23 insertions(+), 5 deletions(-)
```

### 步骤 2: 验证修复已生效

#### 验证修复 1: 单一事实来源

```bash
grep -A 5 "def __getitem__" src/dataset/embedding_rag_dataset.py | grep -A 3 "window_idx"
```

**应该看到**:
```python
if 'window_idx' in output:
    window_idx = int(output['window_idx'])
else:
    # 回退逻辑：保持与父类逻辑一致 (Sample-Major)
    window_idx = item % self.window_count
```

#### 验证修复 2: 验证集策略修正

```bash
grep -B 2 -A 2 "VALIDATION STRATEGY FIX" src/train_embedding_rag.py
```

**应该看到**:
```python
# [VALIDATION STRATEGY FIX] 验证集 Mask 必须固定，严禁刷新！
# if rag_val_loader:
#     rag_val_loader.regenerate_masks(seed=epoch)  # ← 已禁用！
print(f"✓ 验证集 Mask 保持固定（50%），确保评估基准一致")
```

同时检查验证集索引仍会重建:

```bash
grep -A 3 "验证集索引必须更新" src/train_embedding_rag.py
```

**应该看到**:
```python
if rag_val_loader:
    # 验证集索引必须更新（答案随 Embedding Layer 变化）
    rag_val_loader.rebuild_indexes(embedding_layer, device=device)
    print(f"✓ 验证集索引已重建（答案更新，题目不变）")
```

### 步骤 3: 检查 Commit 历史

```bash
git log --oneline -3
```

**应该看到**:
```
2ad4dd5 🔧 架构优化：单一事实来��� + 验证集策略修正
59174b1 🚨 修复三个致命 Bug：语义错位 + Batch 顺序 + Sampler 随机性
...
```

查看详细 Commit 信息:

```bash
git show 2ad4dd5 --stat
```

---

## 🚀 部署训练

### 重要决策: 是否需要从头训练？

#### ❌ 不需要从头训练！

**原因**:
1. **修复 1** (单一事实来源): 只是代码重构，逻辑未变
   - `window_idx` 计算结果完全相同
   - 不影响数据语义

2. **修复 2** (验证集策略): 只影响验证评估方式
   - 训练过程不受影响
   - 模型权重仍然有效

#### ✅ 可以继续训练！

**建议策略**:
- 如果当前训练正在进行: 继续跑完，观察验证 Loss 曲线是否更稳定
- 如果训练已停止: 从最新 checkpoint 恢复
- 如果想重新训练: 删除旧索引后重启（可选）

### 方案 A: 继续当前训练（推荐）

```bash
# 无需操作，继续运行即可
# 下次 Epoch refresh 时会自动应用新的验证策略
```

**观察点**:
- 日志应显示: `✓ 验证集 Mask 保持固定（50%），确保评估基准一致`
- 验证 Loss 应该可比（不会因为 Mask 变化而突变）

### 方案 B: 从头开始训练（可选）

```bash
# 1. 停止当前训练（如果正在运行）
ps aux | grep train_embedding_rag
kill -9 <PID>

# 2. 清理旧索引（可选，系统会自动重建）
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data
rm -rf faiss_indexes_train faiss_indexes_val

# 3. 启动训练
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
bash run_v18_embedding_rag.sh
```

### 方案 C: 从 Checkpoint 恢复（推荐用于中断训练）

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

# 启动训练（会自动检测最新 checkpoint）
bash run_v18_embedding_rag.sh
```

---

## 📊 预期效果对比

### 修复前（旧版本）

**验证集评估问题**:
```
Epoch 0: Val Loss=340  ← 50% mask
Epoch 1: Val Loss=335  ← 60% mask (不可比！)
Epoch 2: Val Loss=682  ← 70% mask (暴涨！)
Epoch 3: Val Loss=330  ← 80% mask (更不可比！)
```

**问题**:
- ❌ Loss 无法比较（题目难度每次都变）
- ❌ Early Stopping 失效（无法判断是否过拟合）
- ❌ 模型选择困难（不知道哪个 checkpoint 最好）

### 修复后（新版本）

**验证集评估改善**:
```
Epoch 0: Val Loss=340  ← 50% mask (固定题目)
Epoch 1: Val Loss=335  ← 50% mask (可比！Loss 下降)
Epoch 2: Val Loss=330  ← 50% mask (可比！持续下降)
Epoch 3: Val Loss=328  ← 50% mask (可比！收敛趋势明显)
```

**改善**:
- ✅ Loss 可比（题目固定）
- ✅ Early Stopping 生效（可准确判断过拟合）
- ✅ 模型选择容易（Loss 最低 = 最佳模型）

---

## 🔍 验证修复成功

### 1. 检查日志输出

启动训练后，观察日志中的关键信息：

```bash
tail -f logs/v18_embedding_rag/latest.log
```

**在每个 Epoch 的 Refresh 阶段应该看到**:

```
================================================================================
🔄 Epoch X Refresh: 刷新训练Mask + 重建索引（匹配最新Embedding）
================================================================================
✓ 训练集 Mask 已刷新（数据增强）
✓ 训练集索引已重建（匹配最新 Embedding）
✓ 验证集 Mask 保持固定（50%），确保评估基准一致  ← 关键！
✓ 验证集索引已重建（答案更新，题目不变）  ← 关键！
```

### 2. 观察验证 Loss 曲线

**正常现象** (修复后):
- Epoch 0-3: Loss 平滑下降或稳定
- 不应出现暴涨（如 133 → 682）

**异常现象** (如果仍出现):
- 检查代码是否正确拉取（git log 确认）
- 检查是否有旧的 Python 进程在运行

### 3. 对比 F1 指标

**预期**:
- Rare F1 应持续提升（因为之前三个致命 Bug 已修复）
- Common F1 保持高水平

---

## ⚠️ 注意事项

### 1. 训练集仍会刷新 Mask

**这是正确的行为！**
- 训练集每个 Epoch 都会随机生成新的 Mask
- 这是数据增强的一部分，有助于泛化
- 训练集 Loss 可能略有波动（正常）

### 2. 验证集 Mask 固定

**这也是正确的行为！**
- 验证集 Mask 在初始化后永久固定
- 确保每个 Epoch 评估的是相同的任务
- 验证集 Loss 应该单调下降（收敛）或稳定（过拟合）

### 3. 验证集索引仍会重建

**必须重建！**
- Embedding Layer 每个 Epoch 都在更新
- 索引中的 Embedding 必须与最新模型匹配
- 这是"答案更新"，题目（Mask）保持不变

---

## 📞 常见问题

### Q1: 为什么不需要从头训练？
**A**: 因为：
1. 修复 1 只是代码重构，逻辑未变
2. 修复 2 只影响验证方式，不影响训练过程
3. 已有的模型权重仍然有效

### Q2: 我应该信任哪个 Epoch 的 Loss？
**A**: 修复后的所有 Epoch Loss 都可信！因为：
- 验证集 Mask 固定（题目相同）
- Loss 可直接比较
- 最低 Loss = 最佳模型

### Q3: 修复前的 checkpoint 还能用吗？
**A**: 可以用！但：
- 之前的验证 Loss 不可比（Mask 每次都变）
- 建议以修复后的第一个 Epoch 为基准
- 后续 Loss 曲线将更可靠

### Q4: 如何确认修复真的生效了？
**A**: 看日志中的三个关键输出：
1. ✅ `✓ 训练集 Mask 已刷新（数据增强）`
2. ✅ `✓ 验证集 Mask 保持固定（50%），确保评估基准一致`
3. ✅ `✓ 验证集索引已重建（答案更新，题目不变）`

---

## ✅ 部署检查清单

部署前确认：

- [ ] 已拉取最新代码（commit 2ad4dd5）
- [ ] 已验证修复 1: `window_idx` 从 output 优先获取
- [ ] 已验证修复 2: 验证集 Mask 刷新被禁用
- [ ] 已验证修复 2: 验证集索引重建仍启用
- [ ] 已查看 git log 确认 commit 存在

部署后观察：

- [ ] 日志显示 `✓ 验证集 Mask 保持固定`
- [ ] 验证 Loss 曲线平滑（无暴涨）
- [ ] Rare F1 持续提升
- [ ] 训练稳定进行

---

## 🎯 总结

本次修复解决了两个重要问题：

1. **代码架构优化**: 遵循"单一事实来源"原则，提高可维护性
2. **验证策略修正**: 固定验证集题目，确保 Loss 可比性

**核心改进**:
- ✅ 验证 Loss 现在可以直接比较
- ✅ Early Stopping 可以正确工作
- ✅ 模型选择更加可靠
- ✅ 代码更加健壮和可维护

**建议**:
- 继续当前训练即可
- 观察验证 Loss 是否更稳定
- 如需从头训练，清理索引后重启

**现在可以放心训练了！🚀**
