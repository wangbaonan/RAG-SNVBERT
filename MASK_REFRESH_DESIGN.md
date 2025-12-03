# Dynamic Mask with Periodic Refresh - 设计方案

**目标**: 在保持语义对齐的前提下，支持mask的数据增强

---

## 🎯 设计原则

### 用户的正确思路

1. **检索阶段**：Query和Reference都用相同的mask
   - 目的：语义对齐，检索基于相同的"已知位点"

2. **使用阶段**：Retrieved sequences是完整的（无mask）
   - 目的：为模型提供完整的参考信息

3. **数据增强**：Mask pattern定期变化
   - 目的：防止过拟合特定mask pattern
   - 方法：每N个batch或每M个epoch重新生成mask

---

## 📋 完整流程

### 初始化阶段

```python
# 1. 生成初始mask (seed=0)
for w_idx in range(window_count):
    np.random.seed(w_idx)  # 每个窗口不同seed
    raw_mask = self.generate_mask(window_len)
    self.window_masks[w_idx] = raw_mask

# 2. 用初始mask编码Reference (用于检索)
for w_idx in range(window_count):
    mask = self.window_masks[w_idx]
    ref_tokenized_masked = self.tokenize(raw_ref, mask)  # ← 用mask!
    ref_emb_masked = embedding_layer(ref_tokenized_masked, af, pos=True)

    # 构建FAISS索引
    index.add(ref_emb_masked.flatten())
    self.embedding_indexes[w_idx] = index

# 3. 同时保存完整的Reference (用于后续返回)
for w_idx in range(window_count):
    ref_tokenized_complete = self.tokenize(raw_ref, zero_mask)  # ← 无mask!
    ref_emb_complete = embedding_layer(ref_tokenized_complete, af, pos=True)
    self.ref_embeddings_complete[w_idx] = ref_emb_complete  # 保存完整版本
```

### 训练阶段

```python
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # === 步骤1: 检查是否需要刷新mask ===
        if should_refresh_mask(epoch, batch_idx):
            # 重新生成所有窗口的mask
            dataset.regenerate_masks(seed=get_current_seed(epoch, batch_idx))

            # 重建FAISS索引 (用新mask编码Reference)
            dataset.rebuild_indexes(embedding_layer)
            # 注意: ref_embeddings_complete不需要重建，永远是完整的

        # === 步骤2: 检索 (Query和Reference都用当前mask) ===
        # Query用当前mask
        query_masked = tokenize(query, current_mask)  # ← mask
        query_emb = embedding_layer(query_masked, af, pos=True)

        # 检索 (索引中的Reference也是用相同mask编码的)
        topk_indices = faiss_index.search(query_emb.flatten(), k=16)

        # === 步骤3: 返回完整的Reference ===
        retrieved_complete = []
        for idx in topk_indices:
            # 从完整版本中获取
            retrieved_complete.append(ref_embeddings_complete[idx])  # ← 无mask!

        # === 步骤4: 模型使用完整Reference ===
        output = model(query_emb, retrieved_complete)
        loss = criterion(output, target)
        loss.backward()
```

---

## 🔧 关键实现细节

### 1. 数据结构

```python
class EmbeddingRAGDataset:
    def __init__(self, ...):
        # Mask相关
        self.window_masks = []  # 当前mask pattern [window_count]
        self.mask_version = 0   # 当前mask版本号

        # Reference的两个版本
        self.ref_tokens_masked = []    # 用于构建索引 (masked)
        self.ref_tokens_complete = []  # 用于返回 (complete)

        self.ref_embeddings_complete = []  # 完整embedding [window_count][num_haps, L, D]

        # FAISS索引 (基于masked embeddings)
        self.embedding_indexes = []  # [window_count]
```

### 2. Mask刷新函数

```python
def regenerate_masks(self, seed: int):
    """重新生成所有窗口的mask"""
    self.mask_version += 1
    print(f"\n{'='*80}")
    print(f"▣ 刷新Mask (Version {self.mask_version}, Seed={seed})")
    print(f"{'='*80}")

    for w_idx in range(self.window_count):
        window_len = self.window.window_info[w_idx, 1] - \
                     self.window.window_info[w_idx, 0]

        # 生成新mask
        np.random.seed(seed * 10000 + w_idx)
        raw_mask = self.generate_mask(window_len)
        padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

        self.window_masks[w_idx] = padded_mask

def rebuild_indexes(self, embedding_layer, device='cuda'):
    """用当前mask重建FAISS索引"""
    print(f"  → 重建FAISS索引 (基于新mask)")
    start_time = time.time()

    with torch.no_grad():
        for w_idx in tqdm(range(self.window_count), desc="重建索引"):
            # 获取完整的reference tokens
            ref_tokens_complete = self.ref_tokens_complete[w_idx]  # [num_haps, L]
            ref_af = self.ref_af_windows[w_idx]  # [L]

            # 应用当前mask
            current_mask = self.window_masks[w_idx]  # [L]
            ref_tokens_masked = self.apply_mask(ref_tokens_complete, current_mask)

            # 用masked版本编码 (用于检索)
            ref_tokens_tensor = torch.LongTensor(ref_tokens_masked).to(device)
            ref_af_tensor = torch.FloatTensor(ref_af).to(device)
            ref_emb_masked = embedding_layer(ref_tokens_tensor, af=ref_af_tensor, pos=True)

            # 重建索引
            ref_emb_flat = ref_emb_masked.reshape(num_haps, -1).cpu().numpy().astype(np.float32)
            self.embedding_indexes[w_idx].reset()
            self.embedding_indexes[w_idx].add(ref_emb_flat)

    print(f"  ✓ 重建完成! 耗时: {time.time() - start_time:.2f}s")

def apply_mask(self, tokens, mask):
    """应用mask到token序列"""
    masked_tokens = tokens.copy()
    mask_token_id = 4  # [MASK] token
    masked_tokens[mask == 1] = mask_token_id
    return masked_tokens
```

### 3. Collate函数修改

```python
def embedding_rag_collate_fn(batch_list, dataset, embedding_layer, k_retrieve, device='cuda'):
    """
    关键修改:
    1. Query用当前mask编码
    2. 检索 (索引中Reference也是用相同mask)
    3. 返回完整的Reference embeddings
    """
    final_batch = defaultdict(list)

    # 按窗口分组
    window_groups = defaultdict(list)
    for sample in batch_list:
        win_idx = int(sample['window_idx'])
        window_groups[win_idx].append(sample)

    with torch.no_grad():
        for win_idx, group in window_groups.items():
            index = dataset.embedding_indexes[win_idx]  # Masked index
            ref_emb_complete = dataset.ref_embeddings_complete[win_idx]  # Complete embeddings
            current_mask = dataset.window_masks[win_idx]  # 当前mask

            # === Query编码 (用当前mask) ===
            # 注意: sample['hap_1'] 已经在__getitem__中用相同mask tokenized了
            h1_tokens = torch.stack([s['hap_1'] for s in group]).to(device)
            h2_tokens = torch.stack([s['hap_2'] for s in group]).to(device)
            af_batch = torch.stack([s['af'] for s in group]).to(device)

            h1_emb = embedding_layer(h1_tokens, af=af_batch, pos=True)
            h2_emb = embedding_layer(h2_tokens, af=af_batch, pos=True)

            # === 检索 (在masked space) ===
            h1_flat = h1_emb.reshape(B, -1).cpu().numpy().astype(np.float32)
            h2_flat = h2_emb.reshape(B, -1).cpu().numpy().astype(np.float32)

            D1, I1 = index.search(h1_flat, k=k_retrieve)
            D2, I2 = index.search(h2_flat, k=k_retrieve)

            # === 返回完整Reference ===
            for i, sample in enumerate(group):
                topk_h1 = []
                for k in range(k_retrieve):
                    ref_idx = I1[i, k]
                    # 返回完整版本! ← 关键!
                    topk_h1.append(ref_emb_complete[ref_idx])
                sample['rag_emb_h1'] = torch.stack(topk_h1)

                topk_h2 = []
                for k in range(k_retrieve):
                    ref_idx = I2[i, k]
                    topk_h2.append(ref_emb_complete[ref_idx])
                sample['rag_emb_h2'] = torch.stack(topk_h2)

            # 收集数据
            for sample in group:
                for key in sample:
                    final_batch[key].append(sample[key])

    # Stack
    for key in final_batch:
        if key not in ["window_idx", "hap1_nomask", "hap2_nomask"]:
            final_batch[key] = torch.stack(final_batch[key])

    return dict(final_batch)
```

### 4. 训练脚本修改

```python
# train_embedding_rag.py

# 配置
REFRESH_MASK_EVERY_N_EPOCHS = 1  # 每个epoch刷新一次
# 或
# REFRESH_MASK_EVERY_N_BATCHES = 500  # 每500个batch刷新一次

for epoch in range(args.epochs):
    # === Epoch开始时刷新mask ===
    if epoch > 0 and epoch % REFRESH_MASK_EVERY_N_EPOCHS == 0:
        print(f"\n{'='*80}")
        print(f"▣ Epoch {epoch}: 刷新Mask和索引")
        print(f"{'='*80}")

        # 重新生成mask
        rag_train_loader.regenerate_masks(seed=epoch)
        rag_val_loader.regenerate_masks(seed=epoch)  # 验证集用相同mask

        # 重建索引 (用新mask)
        rag_train_loader.rebuild_indexes(model.embedding, device=device)
        rag_val_loader.rebuild_indexes(model.embedding, device=device)

        print(f"✓ 刷新完成!\n")

    # 更新dataset的epoch (用于dynamic mask seed)
    rag_train_loader.current_epoch = epoch
    rag_val_loader.current_epoch = epoch

    # === 训练 ===
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        # (可选) 每N个batch刷新
        # if batch_idx > 0 and batch_idx % REFRESH_MASK_EVERY_N_BATCHES == 0:
        #     rag_train_loader.regenerate_masks(seed=epoch * 10000 + batch_idx)
        #     rag_train_loader.rebuild_indexes(model.embedding, device)

        loss = train_step(batch)

    # === 验证 ===
    validate(...)

    # === Epoch结束时刷新模型embeddings (保持原有逻辑) ===
    # 这里刷新的是ref_embeddings_complete (完整版本)
    print(f"\n{'='*80}")
    print(f"▣ Epoch {epoch+1}: 刷新Reference Embeddings (完整版本)")
    print(f"{'='*80}")
    rag_train_loader.refresh_complete_embeddings(model.embedding, device)
    print(f"✓ 完成!\n")
```

---

## ⚖️ 刷新频率的Trade-off

### Option A: 每个Epoch刷新一次 (推荐)

```python
REFRESH_MASK_EVERY_N_EPOCHS = 1
```

**优点**:
- 计算开销可控 (每个epoch ~8分钟额外开销)
- 足够的数据增强 (20 epochs = 20个不同mask)
- 实现简单

**缺点**:
- 每个epoch内mask固定
- 数据增强效果有限

**适用**: 大多数情况，平衡性能和开销

---

### Option B: 每N个Batch刷新一次

```python
REFRESH_MASK_EVERY_N_BATCHES = 500  # ~每2小时刷新一次
```

**优点**:
- 更频繁的数据增强
- 防止overfitting到特定mask

**缺点**:
- 计算开销更高
- 训练可能更慢

**适用**: 如果发现过拟合mask pattern

---

### Option C: 自适应刷新

```python
# 根据验证集性能决定是否刷新
if val_f1_plateau_for_N_epochs:
    # 性能停滞 → 刷新mask尝试escape
    refresh_mask()
```

**优点**:
- 智能，只在需要时刷新
- 最小化不必要的计算

**缺点**:
- 实现复杂
- 可能不稳定

---

## 📊 性能预估

### 初始化 (首次)

```
原来: 15分钟 (构建masked索引 + 构建complete embeddings)
现在: 18分钟 (多一次complete embeddings的编码)
增加: +20%
```

### 每个Epoch

```
训练: 1.3小时 (不变)
刷新mask + 重建索引: ~8分钟 (与原来的refresh_embeddings相同)
刷新complete embeddings: ~8分钟 (原有逻辑)
总计: ~1.5小时/epoch
```

### 每500 Batch刷新 (如果选择Option B)

```
训练500 batch: ~20分钟
刷新mask + 重建索引: ~8分钟
总计: ~28分钟/500batch
开销: +40%
```

---

## 🎯 推荐配置

### 初期训练 (探索阶段)

```python
# 每个epoch刷新一次
REFRESH_MASK_EVERY_N_EPOCHS = 1
REFRESH_EMBEDDINGS_EVERY_N_EPOCHS = 1  # 原有逻辑

# 预期:
# - 每个epoch: ~1.5小时
# - 20 epochs: ~30小时
# - 20个不同的mask pattern
```

### 如果过拟合

```python
# 更频繁刷新
REFRESH_MASK_EVERY_N_BATCHES = 500
```

---

## 🔍 与V17对比

| 特性 | V17 (修复后) | V18 (此方案) |
|------|-------------|-------------|
| **Mask对齐** | ✅ Reference=Query | ✅ Reference=Query (检索阶段) |
| **完整信息** | ❌ Reference永远masked | ✅ 返回完整Reference |
| **Dynamic mask** | ❌ 不支持 | ✅ 定期刷新 |
| **数据增强** | ❌ | ✅ 20个mask (if 每epoch刷新) |
| **计算开销** | 4.2h/epoch | 1.5h/epoch (仍快3x) |

---

## 📝 总结

### 关键设计

1. **两套Reference Embeddings**:
   - Masked版本：用于构建FAISS索引和检索
   - Complete版本：用于返回给模型

2. **定期刷新Mask**:
   - 每N个epoch或每M个batch
   - 重新生成mask → 重建索引 (masked版本)
   - Complete版本不需要重建

3. **保持语义对齐**:
   - Query和Reference都用相同的mask编码 (检索时)
   - 检索后返回完整Reference (使用时)

### 实现复杂度

- **代码改动**: 中等 (~200行)
- **计算开销**: +20% (如果每epoch刷新)
- **收益**: 数据增强 + 语义对齐 + 完整信息

---

**建议**: 先实现Option A (每epoch刷新)，运行实验，根据结果决定是否需要更频繁刷新。
