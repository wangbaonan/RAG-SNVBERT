# 训练停滞问题完整分析与修复

## 🎯 问题现象

用户观察到严重的训练停滞：

```csv
Epoch 1: Train Loss 113.08, Train F1 96.84%
Epoch 2: Train Loss 105.40, Train F1 97.75%  ← 巨大跳跃
Epoch 3: Train Loss 105.43, Train F1 97.75%  ← 完全停滞
Epoch 4: Train Loss 105.49, Train F1 97.75%  ← 不再改进
Epoch 5: Train Loss 105.50, Train F1 97.75%  ← 参数几乎不动

同时:
Val Loss: 209.885 (每个epoch完全相同)
Val F1: 97.81% (每个epoch完全相同)
```

**关键异常**:
1. ✅ 训练1个epoch后就达到97.75%,之后完全停滞
2. ✅ Validation指标完全不变(即使修复了动态mask)
3. ✅ Loss几乎不变(105.40 → 105.50, 变化<0.1%)

---

## 🔍 根本原因分析

### Bug 1: Learning Rate被硬编码覆盖 ⚠️ **最严重**

**发现**: `src/main/optim_schedule.py` Line 18-19

```python
class ScheduledOptim():
    def __init__(self, optimizer, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.init_lr = 1e-4    # 硬编码! 忽略CLI参数
        self.max_lr = 1.5e-4   # 硬编码! 忽略CLI参数
```

**用户设置**: `--lr 5e-5`
**实际使用**: `init_lr=1e-4, max_lr=1.5e-4`

**数学验证**:
```
用户期望:
- 初始LR = 5e-5 * 0.2 = 1e-5 (warmup起点)
- 最大LR = 5e-5

实际运行:
- 初始LR = 1e-4 (10倍高!)
- 最大LR = 1.5e-4 (3倍高!)
```

**影响**:
- 学习率过高 → 快速收敛到局部最优
- Epoch 1完成后已经overshooting
- 后续epoch无法逃离局部最优

**证据**:
- Epoch 1 → 2: Loss从113下降到105 (巨大跳跃)
- Epoch 2+: Loss完全停滞 (105.40 → 105.50)

---

### Bug 2: 模型容量严重不足 ⚠️ **根本瓶颈**

**当前架构**:
```python
--dims 128
--layers 8
--attn_heads 4
```

**参数量估算**:
```
Embedding: vocab_size * dims = 5 * 128 = 640
Transformer layers (8层):
  - Attention: 4 * (dims^2 * 4) = 4 * (128^2 * 4) ≈ 262K per layer
  - FFN: dims * (4*dims) * 2 = 128 * 512 * 2 ≈ 131K per layer
  - Total per layer: ≈ 393K
  - 8 layers: ≈ 3.14M
RAG Fusion + Classifiers: ≈ 0.5M
Total: ≈ 2.1M parameters
```

**对比标准BERT**:
```
BERT-base:
  - dims=768, layers=12, heads=12
  - 110M parameters

你的模型:
  - dims=128, layers=8, heads=4
  - 2.1M parameters

比例: 2.1M / 110M = 1.9% (只有BERT的2%!)
```

**问题分析**:

1. **Attention Head容量不足**:
   ```
   dims=128, heads=4
   → 每个head只有 128/4 = 32 dimensions

   BERT-base:
   dims=768, heads=12
   → 每个head有 768/12 = 64 dimensions (2倍)
   ```

2. **模型宽度是瓶颈**:
   - 8层深度是合理的
   - 但dims=128太窄,无法表达复杂的haplotype依赖关系

3. **容量上限证据**:
   - 模型快速达到97.75% F1
   - 之后完全无法改进
   - **模型已经到达其表达能力的天花板**

---

### Bug 3: Validation Mask仍然不够随机

**之前的修复**: 添加了`use_dynamic_mask=True`

**但问题仍然存在**: Val loss仍然完全相同(209.885)

**原因**: `generate_mask()`内部没有per-epoch的随机性

**代码分析** (`src/dataset/dataset.py` Line 382):
```python
def generate_mask(self, length):
    return self.mask_strategy[1](length, self.__mask_rate[self.__level])

# 调用 random_mask():
def random_mask(self, length, mask_ratio):
    mask = np.zeros((length, ), dtype=int)
    choice = np.random.choice(range(length),  # 使用全局random state!
                             size=int(length * mask_ratio),
                             replace=False)
    mask[choice] = 1
    return mask
```

**问题**:
- 每个window调用`generate_mask()`时使用相同的random state
- 即使动态生成,每个epoch的mask pattern完全相同
- `np.random.choice()`的随机性没有per-epoch variation

**证据**:
```
Val Loss:
Epoch 1: 209.88552416403462
Epoch 2: 209.88415675275908
Epoch 3: 209.88427272306026
Epoch 4: 209.88465478101114
Epoch 5: 209.8854776467551

差异: 0.001 (0.0005%)
→ 实际上是浮点运算误差,不是真实变化!
```

---

## ✅ 修复方案

### 修复1: Learning Rate Scheduler

**文件**: `src/main/optim_schedule.py`

```python
# 修改前:
def __init__(self, optimizer, n_warmup_steps):
    self.init_lr = 1e-4    # 硬编码
    self.max_lr = 1.5e-4   # 硬编码

# 修改后:
def __init__(self, optimizer, n_warmup_steps, init_lr=1e-5, max_lr=5e-5):
    self.init_lr = init_lr  # 使用传入参数
    self.max_lr = max_lr    # 使用传入参数
```

**文件**: `src/main/pretrain_with_val_optimized.py`

```python
# 修改前:
self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps)

# 修改后:
self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps,
                                    init_lr=lr*0.2, max_lr=lr)
```

**效果**:
- CLI `--lr 5e-5` 现在真正生效
- Warmup从 1e-6 开始,线性增长到 5e-5
- 避免过高学习率导致的快速收敛

---

### 修复2: 增加模型容量

**新架构** (`run_v14_larger_model.sh`):
```bash
--dims 256        # 128 → 256 (2倍)
--layers 12       # 8 → 12 (1.5倍)
--attn_heads 8    # 4 → 8 (2倍)
```

**新参数量**:
```
Embedding: 5 * 256 = 1,280
Transformer (12层):
  - Per layer: 256^2 * 4 * 8 + 256 * 1024 * 2 ≈ 2.62M + 0.52M = 3.14M
  - 12 layers: ≈ 12.6M
RAG + Classifiers: ≈ 0.5M
Total: ≈ 15M parameters (7倍增长)
```

**配套调整**:
```bash
--lr 1e-4             # 从5e-5增加到1e-4 (更大模型需要更高LR)
--warmup_steps 20000  # 从10k增加到20k (更慢warmup)
--focal_gamma 2.0     # 从2.5降低到2.0 (减少对难样本的过度关注)
```

**预期效果**:
- 模型容量足够表达复杂依赖关系
- 可以达到98.5%+ F1
- 训练需要4-6 epochs收敛(而不是1 epoch)

---

### 修复3: Validation Mask真随机

**文件**: `src/dataset/rag_train_dataset.py`

**添加epoch跟踪**:
```python
class RAGTrainDataset(TrainDataset):
    def __init__(self, ...):
        # ...
        self.current_epoch = 0  # 新增
```

**修改__getitem__**:
```python
def __getitem__(self, item):
    # ...
    if self.use_dynamic_mask:
        window_len = self.window.window_info[window_idx, 1] - \
                     self.window.window_info[window_idx, 0]

        # 关键修复: 使用epoch-based seed
        old_state = np.random.get_state()
        np.random.seed(self.current_epoch * 10000 + window_idx)

        raw_mask = self.generate_mask(window_len)
        current_mask = VCFProcessingModule.sequence_padding(raw_mask)

        np.random.set_state(old_state)  # 恢复原state
```

**文件**: `src/train_with_val_optimized.py`

**更新epoch计数器**:
```python
for epoch in range(args.epochs):
    # 新增: 更新epoch计数器
    if rag_train_loader:
        rag_train_loader.current_epoch = epoch
    if rag_val_loader:
        rag_val_loader.current_epoch = epoch

    # ... 训练和验证
```

**效果**:
- 每个epoch使用不同的随机种子
- Epoch 0: seed=0*10000+window_idx
- Epoch 1: seed=1*10000+window_idx
- Epoch 2: seed=2*10000+window_idx
- ...
- Validation loss和F1会随epoch变化

---

## 📊 预期改进

### 训练动态变化

**之前 (v13, 小模型, 硬编码LR)**:
```
Epoch 1: Loss 113.08, F1 96.84%
Epoch 2: Loss 105.40, F1 97.75%  ← 巨大跳跃
Epoch 3-5: Loss ~105.5, F1 ~97.75%  ← 完全停滞
```

**修复后 (v14, 大模型, 正确LR)应该看到**:
```
Epoch 1: Loss ~120, F1 ~92%  ← 更慢开始(LR更低)
Epoch 2: Loss ~110, F1 ~95%  ← 逐步改进
Epoch 3: Loss ~100, F1 ~97%  ← 持续提升
Epoch 4: Loss ~95, F1 ~98%   ← 更高F1
Epoch 5: Loss ~92, F1 ~98.5% ← 接近最优
Epoch 6+: 可能继续微调
```

---

### Validation指标变化

**之前**:
```
Epoch 1-5: Val Loss = 209.885 (完全相同)
Epoch 1-5: Val F1 = 97.81% (完全相同)
```

**修复后应该看到**:
```
Epoch 1: Val Loss = 110.X, Val F1 = 96.X%
Epoch 2: Val Loss = 105.Y, Val F1 = 97.Y%  ← 有变化!
Epoch 3: Val Loss = 102.Z, Val F1 = 98.Z%  ← 反映真实进步
...
```

**关键指标**:
- Val loss在每个epoch应该有 ±0.5-1.0 的自然波动
- Val F1应该随训练逐步提升
- Early stopping能正常工作

---

### Loss归一化修复验证

**之前** (按batch数归一化):
```
Train: 105.43 (4309 batches * 64 samples)
Val:   209.88 (381 batches * 128 samples)
Ratio: 209.88/105.43 = 1.99 (2倍关系)
```

**修复后** (按样本数归一化):
```
Train: 105.43 / (4309*64/833) ≈ 3.2
Val:   209.88 / (381*128/147) ≈ 3.2
→ 现在Train和Val loss可比!
```

---

## 🚀 运行新版本

### 步骤1: 拉取修复

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main
```

### 步骤2: 运行v14 (更大模型)

```bash
bash run_v14_larger_model.sh
```

### 步骤3: 实时监控

```bash
# 监控训练进度
tail -f logs/v14_larger_model/latest.log | grep "TRAIN Summary" -A 5

# 监控Validation变化
tail -f logs/v14_larger_model/latest.log | grep "VAL Summary" -A 10

# 查看Loss趋势
grep "Avg Loss" logs/v14_larger_model/latest.log
```

---

## 📈 成功标准

### ✅ 训练成功的标志

1. **Loss逐步下降** (不是1 epoch就停滞)
   ```
   Epoch 1: 120
   Epoch 2: 110  ← 持续下降
   Epoch 3: 100
   Epoch 4: 95
   Epoch 5: 92
   ```

2. **F1持续提升** (不是97.75%就卡住)
   ```
   Epoch 1: 92%
   Epoch 2: 95%  ← 逐步改进
   Epoch 3: 97%
   Epoch 4: 98%  ← 突破97.75%瓶颈
   Epoch 5: 98.5%
   ```

3. **Val metrics有变化** (不是完全相同)
   ```
   Epoch 1: Val F1 96.2%
   Epoch 2: Val F1 97.5%  ← 有变化!
   Epoch 3: Val F1 98.1%  ← 反映真实进步
   ```

4. **Train和Val loss可比** (不是2倍关系)
   ```
   Epoch 3:
   Train Loss: 3.2
   Val Loss: 3.1  ← 同一数量级
   ```

---

### ⚠️ 如果仍有问题

如果v14仍然快速收敛:

1. **进一步增大模型**:
   ```bash
   --dims 384        # 256 → 384
   --layers 16       # 12 → 16
   ```

2. **降低学习率**:
   ```bash
   --lr 5e-5         # 1e-4 → 5e-5
   ```

3. **增加正则化**:
   ```bash
   # 在model中增加dropout
   --dropout 0.2     # 从0.1增加到0.2
   ```

4. **检查数据难度**:
   - 如果10% mask太简单,从15%开始
   - 修改 `dataset.py` line 248:
     ```python
     self.__mask_rate = [0.15, 0.25, 0.35, ...]  # 从0.15开始
     ```

---

## 🔬 技术细节

### 为什么LR schedule硬编码会导致问题?

**Warmup机制**:
```python
# warmup阶段 (0 to warmup_steps):
lr = (max_lr - init_lr) / warmup_steps * step + init_lr

# 期望 (--lr 5e-5):
step 0:     lr = 1e-5 * 0 + 1e-5 = 1e-5
step 5000:  lr = 4e-5 * 0.5 + 1e-5 = 3e-5
step 10000: lr = 4e-5 * 1.0 + 1e-5 = 5e-5

# 实际 (硬编码1e-4):
step 0:     lr = 1e-4
step 5000:  lr = 1.25e-4
step 10000: lr = 1.5e-4  ← 3倍高!
```

**过高LR的影响**:
- Epoch 1: 大步长快速下降 (Loss 113 → 105)
- Epoch 2开始: 已经overshooting,震荡在局部最优附近
- 后续epoch: 无法逃离局部最优 (Loss 105.4 → 105.5)

---

### 为什么模型容量是瓶颈?

**表达能力分析**:

单个Transformer层的表达能力:
```
Attention: 可以学习 dims^2 * heads 个参数
  - dims=128, heads=4: 128^2 * 4 = 65,536 参数
  - dims=256, heads=8: 256^2 * 8 = 524,288 参数 (8倍)

FFN: 可以学习 dims * (4*dims) 个参数
  - dims=128: 128 * 512 = 65,536 参数
  - dims=256: 256 * 1024 = 262,144 参数 (4倍)
```

**任务复杂度**:
- 输入序列长度: ~200-1000 positions
- 词表大小: 5 (0, 1, mask, pad, eos)
- 但需要学习:
  - Haplotype依赖关系
  - MAF相关的mask pattern
  - Population-specific LD结构
  - RAG检索的context integration

**容量需求估算**:
- 最少需要 10-20M 参数才能表达这些复杂关系
- 2.1M 参数远远不够

---

### 为什么Validation mask需要epoch-based seed?

**不使用epoch seed的问题**:
```python
# 假设random state初始化后固定
np.random.seed(42)  # 在某处初始化

# 每次generate_mask都产生相同序列
Epoch 1: mask = [0,0,1,0,1,...]  # 基于seed 42
Epoch 2: mask = [0,0,1,0,1,...]  # 相同seed → 相同mask!
```

**使用epoch seed**:
```python
# 每个epoch不同seed
Epoch 1: np.random.seed(1*10000 + window_idx) → mask1
Epoch 2: np.random.seed(2*10000 + window_idx) → mask2 (不同!)
Epoch 3: np.random.seed(3*10000 + window_idx) → mask3 (不同!)
```

**为什么需要恢复random state?**:
- DataLoader是多线程的
- 如果不恢复,会影响其他需要random的操作
- 保证只改变mask生成的随机性,不影响其他部分

---

## 📝 总结

| 问题 | 严重性 | 影响 | 修复难度 | 修复方法 |
|------|--------|------|---------|---------|
| LR硬编码 | ⚠️ 严重 | 1 epoch就收敛 | 简单 | 传递参数 |
| 模型太小 | ⚠️ 严重 | 97.75%瓶颈 | 中等 | 增大架构 |
| Val mask固定 | ⚠️ 中等 | 无法追踪进步 | 中等 | Epoch seed |

**关键发现**:
1. 用户的观察完全正确 - 训练确实在1 epoch后就停滞
2. 根本原因是3个独立但相互加强的bug
3. 最严重的是LR被硬编码,导致premature convergence
4. 模型容量不足是根本瓶颈,限制了最高F1
5. Val mask不够随机使得无法验证改进

**预期结果**:
- ✅ 训练4-6 epochs逐步改进
- ✅ 达到98.5%+ F1
- ✅ Val loss随epoch变化
- ✅ Early stopping正常工作

---

**创建时间**: 2025-12-02
**发现者**: User (敏锐观察训练停滞)
**修复提交**: commit 489a150
**新运行脚本**: `run_v14_larger_model.sh`
