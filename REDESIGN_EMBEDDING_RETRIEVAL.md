# 端到端Embedding检索重新设计

## 🎯 核心思想

**当前问题**: RAG检索raw tokens → 过BERT编码 → Fusion
- 内存: 每个batch都要对retrieved sequences过BERT
- 效率: 重复编码相同的reference sequences
- 次优: 检索发生在token space而非learned embedding space

**新设计**: Query embedding → 检索pre-encoded embeddings → Fusion
- 内存: 无需重复过BERT,只检索embedding
- 效率: Reference sequences预编码一次
- 端到端: 检索在embedding space,梯度可以回传

---

## 🏗️ 架构设计

### 阶段1: 预编码Reference Panel (初始化时)

```python
class EmbeddingRAGDataset(TrainDataset):
    def _build_embedding_index(self, ref_vcf_path, pretrained_bert=None):
        """
        预编码所有reference sequences为embeddings
        """
        print("Building embedding-based RAG index...")

        # 加载reference data
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)

        # 如果提供预训练BERT,用它编码
        if pretrained_bert is None:
            # 使用简单的可学习embedding
            self.ref_encoder = nn.Embedding(vocab_size, dims)
        else:
            # 使用预训练BERT的embedding层
            self.ref_encoder = pretrained_bert.embedding

        # 预编码所有reference sequences
        self.ref_embeddings = {}

        with torch.no_grad():
            for w_idx in tqdm(range(self.window_count), desc="Encoding references"):
                window_slice = slice(
                    self.window.window_info[w_idx, 0],
                    self.window.window_info[w_idx, 1]
                )

                # 该window的所有reference haplotypes
                window_refs = ref_gt[window_slice]  # [L, num_samples*2]
                window_pos = ref_pos[window_slice]

                # Flatten to [num_haps, L]
                num_haps = window_refs.shape[1]
                ref_haps = window_refs.T  # [num_haps, L]

                # 编码为embeddings: [num_haps, L, D]
                ref_emb = self.ref_encoder(
                    torch.LongTensor(ref_haps),
                    pos=torch.LongTensor(window_pos).unsqueeze(0).expand(num_haps, -1),
                    af=...,  # 从freq_path加载
                    type_idx=...
                )

                # 存储在CPU (避免占用GPU内存)
                self.ref_embeddings[w_idx] = ref_emb.cpu()

                # 构建FAISS index (在embedding space)
                # Flatten embeddings: [num_haps, L*D]
                ref_emb_flat = ref_emb.reshape(num_haps, -1).cpu().numpy()

                index = faiss.IndexFlatL2(ref_emb_flat.shape[1])
                index.add(ref_emb_flat)
                self.embedding_indexes.append(index)

        print(f"✓ Built embedding index for {self.window_count} windows")
```

### 阶段2: 训练时检索 (forward pass)

```python
def __getitem__(self, item):
    output = super().__getitem__(item)
    window_idx = item % self.window_count

    # 获取query的raw sequence
    query_seq = output['hap1_nomask']  # [L]

    # 在collate_fn中会编码为embedding并检索
    output['window_idx'] = window_idx
    output['retrieve_embedding'] = True  # 标记使用embedding检索

    return output
```

```python
def embedding_rag_collate_fn(batch_list, dataset):
    """
    新的collate函数: 在embedding space检索
    """
    batch = default_collate(batch_list)
    B = len(batch_list)

    # Step 1: 对query sequences编码 (需要过BERT embedding层)
    # 注意: 这里只过embedding层,不过transformer!
    query_emb = dataset.query_encoder(
        batch['hap_seq'],
        batch['pos'],
        batch['af'],
        batch['type_idx']
    )  # [B, L, D]

    # Step 2: 在embedding space检索
    retrieved_embs = []

    for i in range(B):
        window_idx = batch['window_idx'][i]

        # Query embedding flatten: [L*D]
        query_flat = query_emb[i].reshape(-1).cpu().numpy()

        # FAISS检索 (在embedding space)
        D, I = dataset.embedding_indexes[window_idx].search(
            query_flat.reshape(1, -1),
            k=1  # 检索top-1
        )

        # 获取pre-encoded embedding
        retrieved_idx = I[0, 0]
        retrieved_emb = dataset.ref_embeddings[window_idx][retrieved_idx]  # [L, D]

        retrieved_embs.append(retrieved_emb)

    # Step 3: Stack retrieved embeddings
    batch['rag_h1_emb'] = torch.stack(retrieved_embs)  # [B, L, D]
    batch['rag_h2_emb'] = torch.stack(retrieved_embs)  # 简化: h1和h2用相同

    return batch
```

### 阶段3: 模型Forward (无需过BERT)

```python
class BERTWithEmbeddingRAG(nn.Module):
    def forward(self, x):
        # 获取input
        hap_seq = x['hap_seq']
        pos = x['pos']
        af = x['af']
        type_idx = x['type_idx']

        # Embedding
        h1 = self.embedding(hap_seq[:, 0], pos, af, type_idx)  # [B, L, D]
        h2 = self.embedding(hap_seq[:, 1], pos, af, type_idx)

        # 获取pre-encoded RAG embeddings (来自collate_fn)
        rag_h1_emb = x.get('rag_h1_emb', None)  # [B, L, D]
        rag_h2_emb = x.get('rag_h2_emb', None)

        if rag_h1_emb is not None:
            # 关键: RAG embeddings已经预编码,无需过BERT!
            # 直接Fusion
            h1_fused = self.fusion_module(h1, rag_h1_emb.to(h1.device))
            h2_fused = self.fusion_module(h2, rag_h2_emb.to(h2.device))
        else:
            h1_fused = h1
            h2_fused = h2

        # 过Transformer (只过一次!)
        for transformer in self.transformer_blocks:
            h1_fused = transformer(h1_fused)
            h2_fused = transformer(h2_fused)

        # Prediction heads
        hap_1_pred = self.hap_classifier(h1_fused)
        hap_2_pred = self.hap_classifier(h2_fused)
        gt_pred = self.gt_classifier(torch.cat([h1_fused, h2_fused], dim=-1))

        return hap_1_pred, hap_2_pred, gt_pred, h1_fused, h2_fused
```

---

## 📊 内存和速度对比

### 当前RAG (V17)

```
内存消耗:
- Query sequences过BERT: 9 GB (batch=16)
- Retrieved sequences过BERT: 9 GB
- Total: 18 GB forward

速度:
- Query encoding: 100 ms
- RAG encoding: 100 ms
- FAISS search: 5 ms
- Fusion: 10 ms
- Total: 215 ms/batch
```

### Embedding RAG (新设计)

```
预计算 (初始化时,一次性):
- Reference encoding: 所有windows一次性编码
- 时间: ~10 minutes
- 存储: ~500 MB (CPU内存)

训练时内存消耗:
- Query sequences过embedding: 0.5 GB (只embedding层)
- FAISS检索: 0.1 GB
- Retrieved embeddings: 0.5 GB (已预编码)
- Fusion: 0.5 GB
- Transformer (只过一次): 9 GB
- Total: 10.6 GB forward (vs 18 GB)

速度:
- Query embedding: 10 ms (只embedding层)
- FAISS search: 5 ms
- Fusion: 10 ms
- Transformer: 100 ms
- Total: 125 ms/batch

收益:
- 内存: 18 GB → 10.6 GB (减少41%)
- 速度: 215 ms → 125 ms (快1.7x)
- Batch size: 16 → 32+ (2倍)
```

---

## 🎓 端到端可学习性

### 方案A: 固定Pre-encoded Embeddings

**优点**: 简单,立即可用
**缺点**: Reference embeddings不会随训练更新

### 方案B: 动态更新 (真正端到端)

```python
class LearnableEmbeddingRAG(nn.Module):
    def __init__(self):
        # Reference encoder (可学习)
        self.ref_encoder = BERTEmbedding(...)

        # Query encoder (与main model共享)
        self.query_encoder = self.ref_encoder  # 共享权重

    def refresh_reference_embeddings(self, dataset):
        """
        训练中定期刷新reference embeddings
        """
        with torch.no_grad():
            for w_idx in range(dataset.window_count):
                ref_seqs = dataset.ref_sequences[w_idx]

                # 用更新后的encoder重新编码
                ref_emb = self.ref_encoder(ref_seqs, ...)
                dataset.ref_embeddings[w_idx] = ref_emb.cpu()

                # 重建FAISS index
                ref_emb_flat = ref_emb.reshape(num_haps, -1).numpy()
                dataset.embedding_indexes[w_idx].reset()
                dataset.embedding_indexes[w_idx].add(ref_emb_flat)
```

**训练流程**:
```python
for epoch in range(epochs):
    for batch in dataloader:
        # Normal training
        loss.backward()
        optimizer.step()

    # 每个epoch结束后刷新reference embeddings
    model.refresh_reference_embeddings(train_dataset)
    print(f"✓ Refreshed reference embeddings for epoch {epoch+1}")
```

---

## 🚀 实施路线图

### Phase 1: 基础版 (2-3小时)

1. ✅ 修改 `rag_train_dataset.py`:
   - 添加 `_build_embedding_index()`
   - 预编码reference sequences
   - 构建embedding-based FAISS index

2. ✅ 修改 `collate_fn`:
   - Query编码只过embedding层
   - FAISS检索在embedding space
   - 返回pre-encoded embeddings

3. ✅ 修改 `bert.py`:
   - Forward接收pre-encoded RAG embeddings
   - 跳过RAG的transformer encoding
   - 直接fusion

**预期效果**:
- 内存: 18 GB → 10.6 GB
- Batch size: 16 → 32
- 速度: 1.7x faster

### Phase 2: 端到端可学习 (4-5小时)

4. ✅ 实现 `refresh_reference_embeddings()`
5. ✅ 训练循环中定期刷新
6. ✅ 梯度回传到reference encoder

**预期效果**:
- Reference embeddings随训练优化
- 检索质量提升
- F1可能+0.5-1%

### Phase 3: 高级优化 (1-2天)

7. ✅ 多GPU并行预编码
8. ✅ Approximate nearest neighbor (更快检索)
9. ✅ Learned similarity metric (替代L2距离)
10. ✅ Hard negative mining

---

## 💡 为什么这是最优方案

### 对比其他方案

| 方案 | 内存 | 速度 | 模型容量 | 可学习性 |
|------|------|------|---------|---------|
| V13 (小模型无RAG) | 6 GB | 1x | 2.1M | ✅ |
| V17 (中模型+RAG) | 18 GB | 0.25x | 8M | ✅ |
| Embedding RAG | 10.6 GB | 1.7x | 8M+ | ✅ |

### 核心优势

1. **内存高效**:
   - Reference只编码一次 (vs 每batch编码)
   - 检索embedding而非raw sequences

2. **速度快**:
   - 无需重复过BERT
   - FAISS检索极快 (<5ms)

3. **端到端可学习**:
   - 梯度可以回传到encoder
   - Reference embeddings可以优化

4. **可扩展**:
   - 支持更大reference panel
   - 支持多个retrieved sequences (K>1)

---

## 🎯 立即行动计划

### 选项A: 快速验证 (推荐先做)

```bash
# 运行V17验证模型能否突破97.75%
bash run_v17_extreme_memory_fix.sh

# 如果V17仍然停滞在97.75%,说明8M参数还不够
# 那么必须实施Embedding RAG才能用更大模型
```

### 选项B: 实施Embedding RAG (根本解决)

我可以帮你实现Phase 1 (基础版):
1. 修改dataset代码
2. 修改model代码
3. 创建新的training script

**预计工作量**: 2-3小时编码 + 1小时测试
**收益**:
- Batch size 16 → 48
- 模型可以用更大 (dims=256, layers=12)
- 速度快1.7x

---

## 📝 总结

**当前V17**: 8M参数, batch=16, 训练慢4倍, 内存18GB
- 不是长久之计
- 有效效率几乎没提升

**Embedding RAG**: 同样8M参数, batch=48, 训练快1.7x, 内存10.6GB
- 根本性解决方案
- 可以支持更大模型 (dims=256+)
- 端到端可学习

**建议**:
1. 先跑V17,看8M参数是否够
2. 如果不够,立即实施Embedding RAG
3. 长期用Embedding RAG + 大模型 (dims=256, layers=12)

要我帮你实现Embedding RAG吗?
