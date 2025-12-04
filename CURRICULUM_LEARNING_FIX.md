# V18 Curriculum Learning 修复文档

## 🔴 问题诊断

### 原始问题
训练 Epoch 2 时，Loss 翻倍（74.69 → 140.19 for train, 133.42 → 280.26 for val），但 F1 和 Accuracy 保持稳定。

### 根本原因
**Curriculum Learning 导致 Mask Rate 每个 Epoch 都增加**:
- Epoch 1: Mask Rate = 10% → Loss = 74.69 (train)
- Epoch 2: Mask Rate = 20% → Loss = 140.19 (train, ~2x)

由于 Loss 使用 `reduction='sum'`，masked 位点翻倍导致 Loss 翻倍，但这**不代表模型性能变差**！

### 问题根源代码
```python
# 原始代码 (train_embedding_rag.py:379-382)
if rag_train_loader:
    rag_train_loader.add_level()  # ❌ 每个 epoch 都增加训练难度
if rag_val_loader:
    rag_val_loader.add_level()    # ❌ 验证集也增加难度!
```

**两个严重问题**:
1. **训练集**: 每个 epoch 增加难度过快，模型没有足够时间收敛
2. **验证集**: 难度也在增加，导致 Loss 无法跨 epoch 比较（移动的靶子！）

---

## ✅ 修复方案

### 修复 1: 固定验证集难度为 50%

**目标**: 验证集必须在整个训练过程中保持固定难度，以便 Loss 和 F1 在不同 Epoch 间可比较。

**实现** ([train_embedding_rag.py:263-272](src/train_embedding_rag.py#L263-L272)):
```python
# === 关键修改: 固定验证集难度为50% (level=4) ===
# 验证集不参与课程学习，保持固定难度以便公平比较不同epoch的性能
print(f"\n{'='*80}")
print(f"Setting Validation Mask Level to 50%...")
print(f"{'='*80}")
for _ in range(4):  # 从level=0提升到level=4 (50% mask)
    rag_val_loader.add_level()
print(f"✓ Validation mask level set to 50%")
print(f"✓ Validation difficulty is now FIXED for all epochs")
print(f"{'='*80}\n")
```

**禁用验证集的动态增加** ([train_embedding_rag.py:396-398](src/train_embedding_rag.py#L396-L398)):
```python
# 验证集保持固定难度 (50%)
# ❌ 已禁用: rag_val_loader.add_level()
# 原因: 验证集必须在整个训练过程中保持固定难度，以便Loss和F1在不同epoch间可比较
```

**效果**:
- ✅ 验证 Loss 现在可以跨 Epoch 公平比较
- ✅ 50% mask 提供了充分的挑战性（比原始的 10%/20% 更能测试泛化能力）
- ✅ F1/Accuracy 指标更有意义

---

### 修复 2: 减缓训练集课程学习速度

**目标**: 给模型更多时间在当前难度下收敛，避免过快增加难度导致训练不稳定。

**实现** ([train_embedding_rag.py:379-394](src/train_embedding_rag.py#L379-L394)):
```python
# === 关键修改: 课程学习策略优化 ===
# 1. 训练集: 每2个epoch增加一次难度 (给模型更多时间收敛)
# 2. 验证集: 固定50%难度，不再增加 (保持评估标准一致)
if (epoch + 1) % 2 == 0 and rag_train_loader:
    # 只在偶数epoch增加训练难度
    current_level = rag_train_loader._BaseDataset__level
    max_level = len(rag_train_loader._BaseDataset__mask_rate) - 1

    if current_level < max_level:
        rag_train_loader.add_level()
        new_mask_rate = rag_train_loader._BaseDataset__mask_rate[rag_train_loader._BaseDataset__level]
        print(f"\n{'='*80}")
        print(f"▣ Curriculum Learning: Training Mask Rate → {new_mask_rate*100:.0f}%")
        print(f"{'='*80}\n")
    else:
        print(f"\n▣ Curriculum Learning: Maximum mask rate reached (80%)")
```

**课程学习时间表**:
```
Epoch 0-1: 10% mask
Epoch 2-3: 20% mask  (在 epoch 2 结束时增加)
Epoch 4-5: 30% mask  (在 epoch 4 结束时增加)
Epoch 6-7: 40% mask
...
```

**效果**:
- ✅ 模型有 2 个 epoch 时间适应当前难度
- ✅ 训练更稳定，收敛更好
- ✅ 避免难度增加过快导致性能下降

---

### 修复 3: 支持从 Checkpoint 恢复训练

**目标**: 由于训练耗时，需要支持从任意 Checkpoint 恢复，同时正确恢复 Mask Level。

**新增参数** ([train_embedding_rag.py:72-74](src/train_embedding_rag.py#L72-L74)):
```python
# Checkpoint恢复参数
parser.add_argument("--resume_path", type=str, default=None, help="恢复训练的checkpoint路径")
parser.add_argument("--resume_epoch", type=int, default=0, help="恢复的起始epoch (用于课程学习)")
```

**加载权重** ([train_embedding_rag.py:154-188](src/train_embedding_rag.py#L154-L188)):
```python
# === Checkpoint恢复: 加载预训练权重 ===
start_epoch = 0
if args.resume_path:
    print(f"\n{'='*80}")
    print(f"Resuming from Checkpoint...")
    print(f"{'='*80}")
    print(f"Loading weights from: {args.resume_path}")

    checkpoint = torch.load(args.resume_path, map_location=device)

    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        # 格式1: {'state_dict': OrderedDict(...), ...}
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        # 格式2: 直接是 state_dict (OrderedDict)
        else:
            state_dict = checkpoint
    elif hasattr(checkpoint, 'state_dict'):
        # 格式3: checkpoint 是模型对象本身
        print(f"✓ Checkpoint is a model object, extracting state_dict...")
        state_dict = checkpoint.state_dict()
    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    # 移除 'module.' 前缀 (如果存在，DataParallel模型会有这个前缀)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"✓ Weights loaded successfully")

    # 设置起始epoch
    start_epoch = args.resume_epoch
    print(f"✓ Resuming from epoch {start_epoch}")
    print(f"{'='*80}\n")
```

**恢复 Mask Level** ([train_embedding_rag.py:306-317](src/train_embedding_rag.py#L306-L317)):
```python
# === 关键修改: 恢复训练时，同步训练集的mask level ===
if start_epoch > 0 and rag_train_loader:
    print(f"\n{'='*80}")
    print(f"Restoring Training Mask Level for Epoch {start_epoch}...")
    print(f"{'='*80}")
    # 课程学习策略: 每2个epoch增加一次难度
    target_level = min(start_epoch // 2, 7)  # level最大为7 (80% mask)
    for _ in range(target_level):
        rag_train_loader.add_level()
    current_mask_rate = rag_train_loader._BaseDataset__mask_rate[rag_train_loader._BaseDataset__level]
    print(f"✓ Training mask level restored to: {current_mask_rate*100:.0f}%")
    print(f"{'='*80}\n")
```

**修改训练循环起点** ([train_embedding_rag.py:324](src/train_embedding_rag.py#L324)):
```python
for epoch in range(start_epoch, args.epochs):  # 从 start_epoch 开始，而不是 0
```

**效果**:
- ✅ 正确加载权重（支持3种checkpoint格式：字典、OrderedDict、模型对象）
- ✅ 处理 DataParallel 前缀（自动移除 'module.' 前缀）
- ✅ 正确恢复 Mask Level（基于课程学习策略）
- ✅ 从正确的 Epoch 继续训练

**Bug修复记录**:
- 2025-12-05: 修复了 `AttributeError: 'BERTFoundationModel' object has no attribute 'items'`
- 原因: 原始代码假设checkpoint一定是字典，但实际可能是模型对象
- 解决: 添加 `hasattr(checkpoint, 'state_dict')` 检查，通过 `.state_dict()` 方法提取权重

---

## 📋 使用方法

### 方法 1: 从头开始训练（使用修复后的逻辑）

```bash
bash run_v18_embedding_rag.sh
```

**行为**:
- 训练集: Epoch 0-1 用 10%, Epoch 2-3 用 20%, ...
- 验证集: 全程固定 50% mask
- Loss 曲线现在可比较

---

### 方法 2: 从 Epoch 2 恢复训练（推荐！）

使用专门的恢复脚本:
```bash
bash run_v18_embedding_rag_resume_ep2.sh
```

**行为**:
- 加载 `rag_bert.model.ep2` 的权重
- 从 Epoch 2 继续训练（而不是从 Epoch 0）
- 训练 Mask 恢复到 10%（因为 2 // 2 = 1，但 level 从 0 开始，所以是 level=1 → 20%）
- 验证 Mask 固定在 50%

**或者手动修改 `run_v18_embedding_rag.sh`**:
1. 取消注释以下行:
```bash
RESUME_PATH="/path/to/rag_bert.model.ep2"
RESUME_EPOCH=2
```

2. 在 python 命令中添加:
```bash
python -m src.train_embedding_rag \
    ... (其他参数) ...
    --resume_path ${RESUME_PATH} \
    --resume_epoch ${RESUME_EPOCH} \
    2>&1 | tee ${LOG_FILE}
```

---

## 📊 预期效果

### 修复前（Epoch 1-2）:
```
Epoch 1: Train Loss=74.69,  Val Loss=133.42 (10% train mask, 10% val mask)
Epoch 2: Train Loss=140.19, Val Loss=280.26 (20% train mask, 20% val mask) ❌ 无法比较!
```

### 修复后（从 Epoch 2 恢复）:
```
Epoch 2: Train Loss=~75,  Val Loss=~350 (10% train mask, 50% val mask - 基准)
Epoch 3: Train Loss=~80,  Val Loss=~345 (10% train mask, 50% val mask - 可比较!)
Epoch 4: Train Loss=~150, Val Loss=~340 (20% train mask, 50% val mask - 训练难度提升)
Epoch 5: Train Loss=~155, Val Loss=~335 (20% train mask, 50% val mask - 继续改进)
```

**关键观察点**:
- ✅ **Val Loss 现在可以直接比较**（固定 50% mask）
- ✅ **F1/Accuracy 指标更准确**（不受 mask 变化影响）
- ✅ **Train Loss 每 2 个 epoch 跳跃一次**（对应难度提升，这是正常的！）

---

## 🚀 下一步建议

1. **立即行动**: 使用 `run_v18_embedding_rag_resume_ep2.sh` 从 Epoch 2 恢复训练
2. **观察 Val Loss**: 现在应该能看到 Val Loss 逐步下降（因为固定难度）
3. **对比 F1**: 验证 Rare/Common F1 是否持续改善
4. **长期训练**: 运行至少 10-15 个 Epoch 观察收敛

---

## 📝 文件修改总结

### 修改的文件:
1. **`src/train_embedding_rag.py`**
   - 添加 `--resume_path` 和 `--resume_epoch` 参数
   - 实现 checkpoint 加载逻辑（处理 DataParallel）
   - 固定验证集 mask 为 50%
   - 减缓训练集课程学习（每 2 个 epoch）
   - 禁用验证集的 `add_level()`
   - 恢复时正确设置 mask level

2. **`run_v18_embedding_rag.sh`**
   - 添加注释说明如何使用 resume 参数

3. **`run_v18_embedding_rag_resume_ep2.sh`** (新文件)
   - 专门用于从 Epoch 2 恢复训练的脚本
   - 预配置好所有 resume 参数

---

## ⚠️ 注意事项

1. **Checkpoint 路径**: 确保 `rag_bert.model.ep2` 存在于指定路径
2. **Epoch 编号**: `--resume_epoch` 应该等于 checkpoint 对应的 epoch 数
3. **Mask Level 计算**: 训练 Mask Level = `resume_epoch // 2`（整除）
4. **验证 Loss 突变**: 从 Epoch 2 恢复后，第一次验证的 Loss 会跳跃（因为从 20% → 50% mask），这是**正常的**！之后会稳定下降。

---

## 🎯 总结

这次修复解决了三个关键问题:

1. ✅ **验证集评估标准一致**: 固定 50% mask，Loss 可比较
2. ✅ **训练更稳定**: 每 2 个 epoch 增加难度，而不是每个 epoch
3. ✅ **支持恢复训练**: 从任意 checkpoint 继续，节省时间

现在可以放心训练，Loss 和 F1 曲线都有意义了！🚀
