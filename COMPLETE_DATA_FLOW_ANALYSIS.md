# V18 Embedding RAG - 完整数据流分析

## 审查日期
2025-12-03

## 目标
完整审阅所有代码，追踪数据从Dataset到模型的完整流动过程，确认无逻辑错误。

---

## 第一部分: 过滤机制详解

### 什么是过滤？

**定义**: 移除在reference panel中不存在的训练数据位点（SNP位置）

**代码位置**: [embedding_rag_dataset.py:117-133](embedding_rag_dataset.py#L117-L133)

**触发条件**:
```python
# 获取训练窗口的位点
train_pos = self.pos[current_slice]  # 训练数据的SNP位置

# 获取reference panel的位点
ref_pos = ...  # reference VCF的SNP位置

# 检查每个训练位点是否在reference中存在
for idx, p in enumerate(train_pos):
    matches = np.where(ref_pos == p)[0]
    if len(matches) > 0:  # ✅ 找到了
        valid_pos_mask.append(idx)
    # else: ❌ 没找到，被过滤
```

### 具体例子

**场景**: 训练数据和参考面板来自不同测序平台

```
训练数据 (Illumina chip):
  窗口chr21:10000000-10001000
  位点: [10000100, 10000200, 10000300, 10000400, 10000500]

参考面板 (1000 Genomes Project):
  包含位点: [10000100, 10000200, 10000400, 10000500]
  不包含:   [10000300]  ← 这个位点在1KGP中没有

过滤过程:
  位点10000100: ✅ matches = [456] (在ref_pos的第456个位置)
  位点10000200: ✅ matches = [457]
  位点10000300: ❌ matches = []    ← 过滤!
  位点10000400: ✅ matches = [458]
  位点10000500: ✅ matches = [459]

过滤后:
  train_pos: [10000100, 10000200, 10000400, 10000500]
  window_len: 5 → 4
  valid_pos_mask: [0, 1, 3, 4]
```

### 为什么需要过滤？

1. **数据来源不同**: 训练数据和参考面板可能来自不同测序平台
2. **位点覆盖度**: reference panel未必包含所有训练数据的位点
3. **无法获取参考**: 如果位点不在reference中，就无法从reference获取该位点的基因型
4. **必须移除**: 保留会导致无法检索，或者检索到错误的位点

### 过滤的影响

```
原始窗口:
  window.window_info[w_idx] = [start=100, stop=105]
  window_len = 5
  train_pos = [100, 101, 102, 103, 104]

过滤后 (假设位点102不在ref中):
  train_pos = [100, 101, 103, 104]
  window_len = 4  ← 长度变化!
  current_slice = [0, 1, 3, 4]  ← 索引变化!

关键:
  - 所有后续操作必须基于过滤后的长度
  - mask生成: 4个位点
  - AF计算: 4个值
  - tokenize: 4个位点
```

---

## 第二部分: 完整数据流分析

### 阶段0: 数据加载

#### 0.1 训练数据加载

**入口**: `EmbeddingRAGDataset.from_file()` [embedding_rag_dataset.py:406-444](embedding_rag_dataset.py#L406-L444)

```python
# 1. 加载base dataset (训练数据)
base_dataset = TrainDataset.from_file(
    vocab=vocab,
    vcfpath=vcfpath,      # 训练VCF (h5格式)
    panelpath=panelpath,  # 人群panel
    freqpath=freqpath,    # 频率数据
    windowpath=windowpath # 窗口定义
)

# 包含:
# - self.vcf: [N_samples, N_snps, 2] 训练数据基因型
# - self.pos: [N_snps] 训练数据SNP位置
# - self.panel: [N_samples] 人群标签
# - self.freq: [5][N_pops][N_snps] 频率信息
#   - freq[0] = REF频率
#   - freq[1] = HET频率
#   - freq[2] = HOM频率
#   - freq[3] = AF (Allele Frequency)
#   - freq[4] = 未使用
# - self.window: 窗口信息
#   - window.window_info: [N_windows, 2] (start, stop)索引
```

**数据示例**:
```
训练数据:
  vcf.shape: (1000 samples, 50000 SNPs, 2 haplotypes)
  pos.shape: (50000,)  # SNP物理位置
  window_count: 331
  window_info[0]: [0, 1031]  # 第一个窗口: 0-1031个SNP

频率数据:
  freq.shape: (5, N_pops, N_snps)
  freq[3][5][1000] = 0.35  # SNP 1000的全局AF = 0.35
```

#### 0.2 参考面板加载

**代码**: `_load_ref_data()` [embedding_rag_dataset.py:352-373](embedding_rag_dataset.py#L352-L373)

```python
ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)

# ref_gt: [N_ref_snps, N_ref_samples, 2] reference基因型
# ref_pos: [N_ref_snps] reference SNP位置

# 示例:
# ref_gt.shape: (150508 SNPs, 1004 samples, 2 haplotypes)
# ref_pos.shape: (150508,)
```

**关键**: reference数据独立于训练数据
- 不同的样本集
- 可能不同的SNP集合
- **必须通过位置匹配**

---

### 阶段1: 预编码 (Initialization)

**入口**: `_build_embedding_indexes()` [embedding_rag_dataset.py:70-228](embedding_rag_dataset.py#L70-L228)

#### 1.1 窗口循环

```python
for w_idx in range(self.window_count):  # 331个窗口
    # 获取当前窗口范围
    current_slice = slice(
        self.window.window_info[w_idx, 0],  # start
        self.window.window_info[w_idx, 1]   # stop
    )
    # 例如: slice(0, 1031) - 第一个窗口包含1031个SNP
```

#### 1.2 位点过滤 (关键步骤!)

```python
# 步骤1.2.1: 获取训练位点
train_pos = self.pos[current_slice]
# train_pos: [1031] 训练数据的SNP物理位置
# 例如: [10000100, 10000200, ..., 10031000]

# 步骤1.2.2: 在reference中查找匹配
ref_indices = []      # reference中的索引
valid_pos_mask = []   # 有效位点的mask

for idx, p in enumerate(train_pos):
    # p: 训练位点的物理位置 (例如: 10000100)
    matches = np.where(ref_pos == p)[0]
    # 在ref_pos中查找物理位置 = p 的索引

    if len(matches) > 0:
        ref_indices.append(matches[0])  # ref中的索引
        valid_pos_mask.append(idx)      # train中的索引

# 示例结果:
# train_pos长度: 1031
# matches找到:   1030个
# 被过滤:        1个 (位点10000300不在reference中)
```

#### 1.3 同步更新 (关键修复!)

```python
# 步骤1.3.1: 检查是否有过滤
if len(ref_indices) < len(train_pos):
    # 有位点被过滤

    if len(valid_pos_mask) == 0:
        # 所有位点都被过滤 → 跳过这个窗口
        print(f"⚠ 跳过窗口 {w_idx}: 没有可用位点")
        continue

    # 步骤1.3.2: 同步更新三个变量 (关键!)
    valid_indices = current_slice.start + np.array(valid_pos_mask)
    # 例如: 0 + [0,1,3,4,...,1029] = [0,1,3,4,...,1029]

    current_slice = valid_indices
    # 从slice对象变为索引数组

    train_pos = train_pos[valid_pos_mask]
    # 只保留有效位点

    window_len = len(train_pos)
    # 更新为过滤后的长度: 1030

# 步骤1.3.3: 保存实际长度 (关键修复!)
self.window_actual_lens.append(window_len)
# 1030 (过滤后) 而不是 1031 (原始)
```

**数据流示例**:
```
原始:
  current_slice: slice(0, 1031)
  train_pos: [位点1, 位点2, 位点3, ..., 位点1031]
  window_len: 1031

过滤后:
  current_slice: [0, 1, 3, 4, ..., 1030]  ← 跳过索引2
  train_pos: [位点1, 位点2, 位点4, ..., 位点1031]  ← 移除位点3
  window_len: 1030  ← 减少1
```

#### 1.4 生成Mask

```python
# 步骤1.4.1: 生成masked版本的mask
raw_mask = self.generate_mask(window_len)
# window_len = 1030 (过滤后的长度!)
# raw_mask: [1030] 其中约10%为1 (被mask)

padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')
# padding到MAX_SEQ_LEN=1030
# padded_mask: [1030]

# 步骤1.4.2: 保存mask
self.raw_window_masks.append(raw_mask)     # [1030]
self.window_masks.append(padded_mask)      # [1030]

# 步骤1.4.3: 生成complete版本的mask (全0)
raw_mask_complete = np.zeros_like(raw_mask)  # [1030] 全0
padded_mask_complete = VCFProcessingModule.sequence_padding(
    raw_mask_complete, dtype='int'
)
```

**关键**: mask长度 = window_len (过滤后) = 1030

#### 1.5 提取Reference数据

```python
# 步骤1.5.1: 从reference中提取对应位点
raw_ref = ref_gt[current_slice, :, :]
# current_slice: [0,1,3,4,...,1030] (过滤后的索引)
# raw_ref: [1030, 1004, 2]
#   - 1030个位点 (过滤后)
#   - 1004个样本
#   - 2条单倍型

# 步骤1.5.2: reshape
raw_ref = raw_ref.reshape(raw_ref.shape[0], -1)
# [1030, 2008] (1004*2 = 2008条单倍型)

raw_ref = raw_ref.T
# [2008, 1030] (每行是一条单倍型序列)
```

**数据示例**:
```
raw_ref[0]: [0, 1, 0, 1, 1, ..., 0]  # 第1条单倍型，1030个位点
raw_ref[1]: [0, 0, 1, 1, 0, ..., 1]  # 第2条单倍型
...
raw_ref[2007]: [1, 0, 0, 1, 1, ..., 0]  # 第2008条单倍型
```

#### 1.6 Tokenize

```python
# 步骤1.6.1: Tokenize masked版本
ref_tokens_masked = self.tokenize(raw_ref, padded_mask)
# raw_ref: [2008, 1030] 基因型 (0/1)
# padded_mask: [1030] mask pattern
#
# tokenize过程:
#   for each position i in [0, 1030):
#       if padded_mask[i] == 1:
#           token[i] = 4  # [MASK]
#       else:
#           token[i] = raw_ref[:, i]  # 0或1
#
# ref_tokens_masked: [2008, 1030]

# 步骤1.6.2: Tokenize complete版本
ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)
# padded_mask_complete全为0，不mask任何位点
# ref_tokens_complete: [2008, 1030] (无mask)

# 步骤1.6.3: 保存
self.ref_tokens_masked.append(ref_tokens_masked)
self.ref_tokens_complete.append(ref_tokens_complete)
```

**Tokenize示例**:
```
raw_ref[0]:           [0, 1, 0, 1, 1, 0, ...]
padded_mask:          [0, 1, 0, 0, 1, 0, ...]
ref_tokens_masked[0]: [0, 4, 0, 1, 4, 0, ...]  ← 位置1,4被mask
ref_tokens_complete[0]:[0, 1, 0, 1, 1, 0, ...]  ← 完全一样
```

#### 1.7 计算AF

```python
# 步骤1.7.1: 从频率数据获取AF
AF_IDX = 3       # AF在freq的第3维
GLOBAL_IDX = 5   # Global population索引

ref_af = np.array([
    self.freq[AF_IDX][GLOBAL_IDX][self.pos_to_idx[p]]
    if p in self.pos_to_idx else 0.0
    for p in train_pos  # 遍历过滤后的位点!
], dtype=np.float32)
# ref_af: [1030] 每个位点的AF值

# 步骤1.7.2: Padding
ref_af = VCFProcessingModule.sequence_padding(ref_af, dtype='float')
# ref_af: [1030] (已padding)

# 步骤1.7.3: 保存
self.ref_af_windows.append(ref_af)
```

**AF示例**:
```
train_pos: [10000100, 10000200, 10000400, ...]
ref_af:    [0.35,     0.12,     0.67,     ...]
           ↑          ↑         ↑
           位点100    位点200   位点400的AF值
```

#### 1.8 编码Embeddings

```python
# 步骤1.8.1: 扩展AF到所有单倍型
num_haps = ref_tokens_masked.shape[0]  # 2008
ref_af_expanded = np.tile(ref_af, (num_haps, 1))
# [2008, 1030] 每条单倍型共享相同的AF

# 步骤1.8.2: 转换为tensor并移到GPU
ref_tokens_masked_tensor = torch.LongTensor(ref_tokens_masked).to(device)
# [2008, 1030]
ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
# [2008, 1030]

# 步骤1.8.3: 编码masked版本
ref_emb_masked = embedding_layer(
    ref_tokens_masked_tensor,  # [2008, 1030]
    af=ref_af_tensor,          # [2008, 1030]
    pos=True
)
# ref_emb_masked: [2008, 1030, 192]
#   - 2008条单倍型
#   - 1030个位点
#   - 192维embedding

# 步骤1.8.4: 编码complete版本
ref_tokens_complete_tensor = torch.LongTensor(ref_tokens_complete).to(device)
ref_emb_complete = embedding_layer(
    ref_tokens_complete_tensor,
    af=ref_af_tensor,
    pos=True
)
# ref_emb_complete: [2008, 1030, 192]

# 步骤1.8.5: 移到CPU保存
self.ref_embeddings_masked.append(ref_emb_masked.cpu())
self.ref_embeddings_complete.append(ref_emb_complete.cpu())
```

**Embedding维度追踪**:
```
Input:
  tokens: [2008, 1030] LongTensor
  af:     [2008, 1030] FloatTensor

Embedding Layer:
  token_emb: [2008, 1030, 192]
  af_emb:    [2008, 1030, 192]
  pos_emb:   [2008, 1030, 192]
  combined:  [2008, 1030, 192]

Output:
  [2008, 1030, 192]
```

#### 1.9 构建FAISS索引

```python
# 步骤1.9.1: Flatten embeddings
num_haps, L, D = ref_emb_masked.shape  # 2008, 1030, 192
ref_emb_flat = ref_emb_masked.reshape(num_haps, L * D)
# [2008, 197760] (1030 * 192 = 197760)

# 步骤1.9.2: 转为numpy
ref_emb_flat_np = ref_emb_flat.cpu().numpy().astype(np.float32)

# 步骤1.9.3: 创建FAISS索引
index = faiss.IndexFlatL2(L * D)  # 维度 = 197760
index.add(ref_emb_flat_np)        # 添加2008条向量

# 步骤1.9.4: 保存
self.embedding_indexes.append(index)
```

**FAISS索引内容**:
```
索引维度: 197760
向量数量: 2008
每个向量代表: 一条reference单倍型的masked embedding
```

#### 1.10 预编码完成

```python
# 总结:
print(f"✓ 预编码完成!")
print(f"  - 窗口数量: {self.window_count}")  # 331
print(f"  - 每窗口单倍型数: ~2008")
print(f"  - Mask版本号: {self.mask_version}")  # 0
print(f"  - 存储大小: {storage_mb:.1f} MB")  # 1486.4 MB
```

**存储内容**:
```
self.ref_tokens_complete[w_idx]:     [2008, 1030] - 完整tokens
self.ref_tokens_masked[w_idx]:       [2008, 1030] - masked tokens
self.ref_embeddings_complete[w_idx]: [2008, 1030, 192] - 完整embeddings (CPU)
self.ref_embeddings_masked[w_idx]:   [2008, 1030, 192] - masked embeddings (CPU)
self.embedding_indexes[w_idx]:       FAISS index - 2008个向量
self.window_masks[w_idx]:            [1030] - mask pattern
self.ref_af_windows[w_idx]:          [1030] - AF值
self.window_actual_lens[w_idx]:      1030 - 实际长度
```

---

### 阶段2: 训练数据获取 (__getitem__)

**入口**: `EmbeddingRAGDataset.__getitem__(item)` [embedding_rag_dataset.py:375-403](embedding_rag_dataset.py#L375-L403)

#### 2.1 获取Base数据

```python
# 步骤2.1.1: 调用父类获取基础数据
output = super().__getitem__(item)
# 从TrainDataset继承，返回:
# {
#   'hap1_nomask': [L] 第一条单倍型 (无mask)
#   'hap2_nomask': [L] 第二条单倍型 (无mask)
#   'af':   [L] Allele frequency
#   'af_p': [L] Population-specific AF
#   'ref':  [L] REF genotype frequency
#   'het':  [L] HET genotype frequency
#   'hom':  [L] HOM genotype frequency
#   'pos':  [L] 位置编码
#   'type': [L] SNP type
#   'window_idx': scalar
# }
```

**Base Dataset示例**:
```python
# item = 15
output = {
    'hap1_nomask': [0, 1, 0, 1, 1, ..., 0],  # 样本8的单倍型1
    'hap2_nomask': [0, 0, 1, 1, 0, ..., 1],  # 样本8的单倍型2
    'af':   [0.35, 0.12, 0.67, ..., 0.23],   # Global AF
    'pos':  [0.001, 0.002, 0.003, ..., 0.1], # 归一化位置
    'window_idx': 0  # 第0个窗口
}
```

#### 2.2 生成Mask

```python
window_idx = item % self.window_count  # item 15 → window 0

# 步骤2.2.1: 选择mask策略
if self.use_dynamic_mask:
    # 动态mask (每次调用都不同)
    window_len = self.window_actual_lens[window_idx]  # 1030

    old_state = np.random.get_state()
    np.random.seed(self.current_epoch * 10000 + window_idx)

    raw_mask = self.generate_mask(window_len)
    current_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

    np.random.set_state(old_state)
else:
    # 静态mask (使用预编码的mask)
    current_mask = self.window_masks[window_idx]  # [1030]

output['mask'] = current_mask
```

**Mask示例**:
```
current_mask: [0, 1, 0, 0, 1, 0, 1, 0, ..., 0]
              ↑  ↑        ↑     ↑
              保留 mask    保留  mask保留
```

#### 2.3 Tokenize Query

```python
# 步骤2.3.1: 应用mask到单倍型
output['hap_1'] = self.tokenize(output['hap1_nomask'], current_mask)
output['hap_2'] = self.tokenize(output['hap2_nomask'], current_mask)

# tokenize过程:
# for i in range(len(hap1_nomask)):
#     if current_mask[i] == 1:
#         hap_1[i] = 4  # [MASK]
#     else:
#         hap_1[i] = hap1_nomask[i]
```

**Tokenize示例**:
```
hap1_nomask: [0, 1, 0, 1, 1, 0, 1, ...]
current_mask:[0, 1, 0, 0, 1, 0, 1, ...]
hap_1:       [0, 4, 0, 1, 4, 0, 4, ...]
             ↑  ↑     ↑  ↑     ↑
             保留mask 保留保留  mask mask
```

#### 2.4 转换为Tensor

```python
for key in self.long_fields:
    output[key] = torch.LongTensor(output[key])
for key in self.float_fields:
    output[key] = torch.FloatTensor(output[key])

return output
```

**最终output**:
```python
{
    'hap_1': LongTensor([0, 4, 0, 1, 4, ...]),  # masked
    'hap_2': LongTensor([0, 0, 4, 4, 0, ...]),  # masked
    'af':    FloatTensor([0.35, 0.12, 0.67, ...]),
    'pos':   FloatTensor([0.001, 0.002, ...]),
    'mask':  LongTensor([0, 1, 0, 0, 1, ...]),
    'window_idx': 0,
    'hap1_nomask': [0, 1, 0, 1, 1, ...],  # 保留用于其他用途
    'hap2_nomask': [0, 0, 1, 1, 0, ...]
}
```

---

### 阶段3: Batch Collate (RAG检索)

**入口**: `embedding_rag_collate_fn()` [embedding_rag_dataset.py:451-529](embedding_rag_dataset.py#L451-L529)

#### 3.1 按窗口分组

```python
batch_list = [sample1, sample2, ..., sample32]  # batch_size=32

# 步骤3.1.1: 按window_idx分组
window_groups = defaultdict(list)
for sample in batch_list:
    win_idx = int(sample['window_idx'])
    window_groups[win_idx].append(sample)

# 结果:
# window_groups = {
#     0: [sample1, sample3, sample5, ...],  # 15个样本
#     1: [sample2, sample4, ...],           # 10个样本
#     2: [sample6, ...],                    # 7个样本
# }
```

#### 3.2 编码Query Embeddings

```python
for win_idx, group in window_groups.items():
    # 步骤3.2.1: 获取FAISS索引和reference embeddings
    index = dataset.embedding_indexes[win_idx]
    # FAISS索引 (基于masked embeddings)

    ref_emb_complete = dataset.ref_embeddings_complete[win_idx]
    # [2008, 1030, 192] (CPU)

    # 步骤3.2.2: Batch化query tokens
    h1_tokens = torch.stack([s['hap_1'] for s in group]).to(device)
    # [B, 1030] 例如B=15
    h2_tokens = torch.stack([s['hap_2'] for s in group]).to(device)
    # [B, 1030]
    af_batch = torch.stack([s['af'] for s in group]).to(device)
    # [B, 1030]

    # 步骤3.2.3: 编码query (masked版本!)
    h1_emb = embedding_layer(h1_tokens, af=af_batch, pos=True)
    # [B, 1030, 192]
    h2_emb = embedding_layer(h2_tokens, af=af_batch, pos=True)
    # [B, 1030, 192]
```

**Query Embedding示例**:
```
group中有15个样本:
h1_tokens: [15, 1030] - masked tokens
h1_emb:    [15, 1030, 192] - masked embeddings

h1_emb[0]:  第1个样本的单倍型1的embedding [1030, 192]
h1_emb[1]:  第2个样本的单倍型1的embedding [1030, 192]
...
```

#### 3.3 FAISS检索

```python
    # 步骤3.3.1: Flatten embeddings
    B, L, D = h1_emb.shape  # 15, 1030, 192
    h1_emb_flat = h1_emb.reshape(B, L * D)
    # [15, 197760]
    h2_emb_flat = h2_emb.reshape(B, L * D)
    # [15, 197760]

    # 步骤3.3.2: 转为numpy
    h1_emb_flat_np = h1_emb_flat.cpu().numpy().astype(np.float32)
    h2_emb_flat_np = h2_emb_flat.cpu().numpy().astype(np.float32)

    # 步骤3.3.3: 在FAISS中检索
    D1, I1 = index.search(h1_emb_flat_np, k=k_retrieve)
    # D1: [B, k] 距离
    # I1: [B, k] 索引 (reference单倍型的索引)
    # 例如: k=1
    # I1 = [[234], [567], [890], ...]
    #       ↑      ↑      ↑
    #       样本1  样本2  样本3最近的ref单倍型索引

    D2, I2 = index.search(h2_emb_flat_np, k=k_retrieve)
    # I2: [B, k]
```

**检索过程详解**:
```
Query (样本1的h1):
  h1_emb_flat[0]: [197760维向量]

FAISS索引包含:
  ref[0]: [197760维]  距离=15.3
  ref[1]: [197760维]  距离=8.2
  ...
  ref[234]: [197760维]  距离=2.1  ← 最近!
  ...
  ref[2007]: [197760维]  距离=12.5

检索结果:
  I1[0] = [234]  # 第234条reference单倍型最近
  D1[0] = [2.1]  # L2距离
```

#### 3.4 获取Retrieved Embeddings (关键!)

```python
    # 步骤3.4.1: 收集retrieved embeddings
    for i, sample in enumerate(group):
        # 对于每个query样本

        # h1的top-k
        topk_h1 = []
        for k in range(k_retrieve):
            ref_idx = I1[i, k]  # 例如: 234
            # 关键: 返回COMPLETE embeddings!
            topk_h1.append(ref_emb_complete[ref_idx])
            # ref_emb_complete[234]: [1030, 192] - 完整!

        sample['rag_emb_h1'] = torch.stack(topk_h1)
        # [k, 1030, 192] 例如k=1: [1, 1030, 192]

        # h2的top-k
        topk_h2 = []
        for k in range(k_retrieve):
            ref_idx = I2[i, k]
            topk_h2.append(ref_emb_complete[ref_idx])

        sample['rag_emb_h2'] = torch.stack(topk_h2)
        # [k, 1030, 192]
```

**关键设计**:
- **检索**: 在masked space进行 (Query和Reference都是masked)
- **返回**: Complete embeddings (提供完整信息)

**示例**:
```
样本1:
  Query h1 (masked): [1030, 192] 有10%位点被mask
  检索到: ref[234] (也是masked，相同mask pattern)
  距离最近 (语义对齐!)

  返回给模型: ref_complete[234] (无mask，完整信息)
```

#### 3.5 组装Final Batch

```python
    # 步骤3.5.1: 收集所有样本
    for sample in group:
        for key in sample:
            final_batch[key].append(sample[key])

# 步骤3.5.2: Stack为tensor
for key in final_batch:
    if key not in ["window_idx", "hap1_nomask", "hap2_nomask"]:
        final_batch[key] = torch.stack(final_batch[key])

return dict(final_batch)
```

**Final Batch结构**:
```python
{
    'hap_1': [32, 1030],         # Query tokens (masked)
    'hap_2': [32, 1030],
    'af':    [32, 1030],
    'pos':   [32, 1030],
    'mask':  [32, 1030],
    'rag_emb_h1': [32, 1, 1030, 192],  # Retrieved complete embeddings
    'rag_emb_h2': [32, 1, 1030, 192],
    ...
}
```

---

### 阶段4: 模型Forward Pass

**入口**: `BERTWithEmbeddingRAG.forward()` [src/model/bert.py]

#### 4.1 编码Query

```python
# 假设模型代码:
def forward(self, batch):
    # 步骤4.1.1: 获取inputs
    h1_tokens = batch['hap_1']  # [B, L]
    h2_tokens = batch['hap_2']  # [B, L]
    af = batch['af']            # [B, L]

    # 步骤4.1.2: Embedding
    h1_emb = self.embedding(h1_tokens, af=af, pos=True)
    # [B, L, D] 例如[32, 1030, 192]
    h2_emb = self.embedding(h2_tokens, af=af, pos=True)
    # [B, L, D]

    # 步骤4.1.3: 获取retrieved embeddings
    rag_h1 = batch['rag_emb_h1']  # [B, k, L, D]
    rag_h2 = batch['rag_emb_h2']  # [B, k, L, D]
```

#### 4.2 融合RAG信息

```python
    # 步骤4.2.1: 移到正确设备
    rag_h1 = rag_h1.to(h1_emb.device)
    rag_h2 = rag_h2.to(h2_emb.device)

    # 步骤4.2.2: Squeeze k维度 (如果k=1)
    if rag_h1.shape[1] == 1:
        rag_h1 = rag_h1.squeeze(1)  # [B, L, D]
        rag_h2 = rag_h2.squeeze(1)  # [B, L, D]

    # 步骤4.2.3: 融合策略 (多种可能)

    # 选项1: 直接拼接
    h1_combined = torch.cat([h1_emb, rag_h1], dim=-1)
    # [B, L, 2D] [32, 1030, 384]

    # 选项2: 加权求和
    h1_combined = h1_emb + 0.5 * rag_h1
    # [B, L, D]

    # 选项3: Attention融合
    h1_combined = self.rag_attention(
        query=h1_emb,
        key=rag_h1,
        value=rag_h1
    )
```

**融合示例** (拼接方式):
```
Query embedding:
  h1_emb[0,0,:]:    [192维] 第1个样本第1个位点的query embedding

Retrieved embedding:
  rag_h1[0,0,:]:    [192维] 检索到的reference embedding

Combined:
  h1_combined[0,0,:]: [384维] = [query 192维 | reference 192维]
```

#### 4.3 Transformer处理

```python
    # 步骤4.3.1: 通过Transformer layers
    h1_output = self.transformer(h1_combined)
    # [B, L, D] 或 [B, L, 2D] 取决于融合方式
    h2_output = self.transformer(h2_combined)

    # 步骤4.3.2: Pooling (如果需要)
    h1_pooled = h1_output.mean(dim=1)  # [B, D]
    h2_pooled = h2_output.mean(dim=1)  # [B, D]

    # 步骤4.3.3: 预测
    logits = self.classifier(torch.cat([h1_pooled, h2_pooled], dim=-1))
    # [B, num_classes]

    return logits
```

---

### 阶段5: Epoch刷新机制

#### 5.1 Epoch开始: 刷新Mask和索引

**代码**: [train_embedding_rag.py:263-278](train_embedding_rag.py#L263-L278)

```python
if epoch > 0:
    # 步骤5.1.1: 重新生成mask pattern
    rag_train_loader.regenerate_masks(seed=epoch)
```

**regenerate_masks详解**:
```python
def regenerate_masks(self, seed: int):
    self.mask_version += 1  # 0 → 1

    for w_idx in range(self.window_count):
        # 使用实际长度!
        window_len = self.window_actual_lens[w_idx]  # 1030

        # 生成新mask (不同随机种子)
        np.random.seed(seed * 10000 + w_idx)
        raw_mask = self.generate_mask(window_len)
        # 新的mask pattern，不同于epoch 0

        padded_mask = VCFProcessingModule.sequence_padding(raw_mask)

        # 更新
        self.window_masks[w_idx] = padded_mask
```

**Mask变化示例**:
```
Epoch 0 mask: [0, 1, 0, 0, 1, 0, 1, ...]
Epoch 1 mask: [1, 0, 0, 1, 0, 1, 0, ...]  ← 不同!
              ↑           ↑
              变化        变化
```

#### 5.2 重建FAISS索引

```python
    # 步骤5.2.1: 用新mask重建索引
    rag_train_loader.rebuild_indexes(embedding_layer, device=device)
```

**rebuild_indexes详解**:
```python
def rebuild_indexes(self, embedding_layer, device='cuda'):
    with torch.no_grad():
        for w_idx in range(self.window_count):
            # 步骤1: 获取完整tokens
            ref_tokens_complete = self.ref_tokens_complete[w_idx]
            # [2008, 1030] 无mask

            # 步骤2: 应用新mask
            current_mask = self.window_masks[w_idx]  # 新mask!
            ref_tokens_masked = self._apply_mask_to_tokens(
                ref_tokens_complete, current_mask
            )
            # [2008, 1030] 新mask pattern

            # 步骤3: 用最新模型重新编码
            ref_emb_masked = embedding_layer(
                torch.LongTensor(ref_tokens_masked).to(device),
                af=...,
                pos=True
            )
            # [2008, 1030, 192] 新的embeddings!

            # 步骤4: 重建FAISS索引
            self.embedding_indexes[w_idx].reset()
            self.embedding_indexes[w_idx].add(ref_emb_masked.flatten())

            # 步骤5: 保存
            self.ref_embeddings_masked[w_idx] = ref_emb_masked.cpu()
```

**关键点**:
- 用新的mask pattern
- 用最新训练的embedding_layer (参数已更新!)
- 索引包含最新的learned representations

#### 5.3 Epoch结束: 刷新Complete Embeddings

**代码**: [train_embedding_rag.py:295-298](train_embedding_rag.py#L295-L298)

```python
# Epoch训练和验证完成后
rag_train_loader.refresh_complete_embeddings(embedding_layer, device)
```

**refresh_complete_embeddings详解**:
```python
def refresh_complete_embeddings(self, embedding_layer, device='cuda'):
    with torch.no_grad():
        for w_idx in range(self.window_count):
            # 步骤1: 获取完整tokens (不变)
            ref_tokens_complete = self.ref_tokens_complete[w_idx]
            # [2008, 1030]

            # 步骤2: 用最新模型重新编码
            ref_emb_complete = embedding_layer(
                torch.LongTensor(ref_tokens_complete).to(device),
                af=...,
                pos=True
            )
            # [2008, 1030, 192] 用最新embedding_layer!

            # 步骤3: 更新
            self.ref_embeddings_complete[w_idx] = ref_emb_complete.cpu()
```

**为什么要刷新**:
- Epoch训练后，embedding_layer参数已更新
- Complete embeddings需要反映最新的learned representations
- 确保返回给模型的embeddings是最新的

---

## 第三部分: 完整数据维度追踪

### 预编码阶段
```
窗口w_idx = 0:

1. 位点过滤:
   train_pos原始: [1031]
   过滤后:        [1030]
   window_actual_lens[0] = 1030  ✅

2. Reference提取:
   raw_ref: [1030, 1004, 2] → [2008, 1030]

3. Tokenize:
   ref_tokens_masked:   [2008, 1030]
   ref_tokens_complete: [2008, 1030]

4. AF:
   ref_af: [1030]
   ref_af_expanded: [2008, 1030]

5. Embeddings:
   ref_emb_masked:   [2008, 1030, 192]
   ref_emb_complete: [2008, 1030, 192]

6. FAISS:
   index维度: 197760 (1030 * 192)
   向量数:    2008
```

### 训练阶段
```
__getitem__(item=15):
  window_idx = 0
  window_len = window_actual_lens[0] = 1030  ✅

  mask: [1030]
  hap_1: [1030]  (masked)
  hap_2: [1030]  (masked)
  af: [1030]
```

### Collate阶段
```
batch_size = 32, 假设都是window 0:

  Query编码:
    h1_tokens: [32, 1030]
    h1_emb:    [32, 1030, 192]

  FAISS检索:
    h1_emb_flat: [32, 197760]
    I1: [32, 1] (k=1)

  Retrieved:
    rag_emb_h1: [32, 1, 1030, 192]
```

### 模型阶段
```
Forward:
  h1_emb: [32, 1030, 192]  (query)
  rag_h1: [32, 1, 1030, 192] → squeeze → [32, 1030, 192]

  Combine:
    concat: [32, 1030, 384]
    或 add: [32, 1030, 192]

  Transformer:
    output: [32, 1030, D]

  Predict:
    logits: [32, num_classes]
```

### 刷新阶段
```
Epoch 1结束 → Epoch 2开始:

regenerate_masks:
  window_len = window_actual_lens[0] = 1030  ✅
  new_mask: [1030]  (不是1031!)

rebuild_indexes:
  ref_tokens_masked: [2008, 1030]
  ref_emb_masked: [2008, 1030, 192]
  index维度: 197760  ✅ 不变!

refresh_complete:
  ref_emb_complete: [2008, 1030, 192]
```

---

## 第四部分: 潜在问题检查

### ✅ 问题1: 维度一致性
**状态**: 已修复
- 所有阶段都使用`window_actual_lens`
- 维度始终一致: 1030

### ⚠️ 问题2: 窗口跳过

**代码**: [embedding_rag_dataset.py:125-127](embedding_rag_dataset.py#L125-L127)
```python
if len(valid_pos_mask) == 0:
    print(f"⚠ 跳过窗口 {w_idx}: 没有可用位点")
    continue
```

**问题**:
```
假设窗口5被跳过:
  预编码时:
    w_idx=0: append到列表 → index 0
    w_idx=1: append到列表 → index 1
    ...
    w_idx=5: continue (不append) → 没有index 5!
    w_idx=6: append到列表 → index 5 (not 6!)

  训练时:
    sample的window_idx=6
    访问: embedding_indexes[6]
    实际: 访问到的是w_idx=7的数据! ❌
```

**解决方案**:
```python
# 选项1: 不允许跳过
if len(valid_pos_mask) == 0:
    raise ValueError(f"窗口 {w_idx} 没有可用位点!")

# 选项2: 填充空窗口
if len(valid_pos_mask) == 0:
    # 添加空数据占位
    self.ref_tokens_complete.append(None)
    self.embedding_indexes.append(None)
    ...
    continue

# 在collate_fn中检查
if dataset.embedding_indexes[win_idx] is None:
    raise ValueError(f"窗口 {win_idx} 无效")

# 选项3: 重新映射索引
self.valid_window_mapping = {}  # {original: actual}
actual_idx = 0
for w_idx in range(window_count):
    if has_valid_data:
        self.valid_window_mapping[w_idx] = actual_idx
        actual_idx += 1

# 使用时:
actual_idx = self.valid_window_mapping[win_idx]
index = self.embedding_indexes[actual_idx]
```

**当前建议**: 监控是否有跳过，如果有则需要修复

### ✅ 问题3: AF访问
**状态**: 已修复
- 使用整数索引
- 有保护: `if p in self.pos_to_idx`

### ✅ 问题4: window_len一致性
**状态**: 已修复
- 所有地方都使用`window_actual_lens`

---

## 总结

### 核心数据流
```
数据加载 → 过滤位点 → Tokenize → 编码 → FAISS索引
    ↓
训练采样 → 应用Mask → 编码Query → FAISS检索 → 获取Complete
    ↓
模型Forward → 融合RAG → Transformer → 预测
    ↓
Epoch刷新 → 新Mask → 重建索引 → 刷新Complete
```

### 关键设计
1. **过滤机制**: 移除不在reference中的位点
2. **Mask对齐**: 检索时Query和Reference使用相同mask
3. **Complete返回**: 返回无mask的完整embeddings
4. **端到端学习**: 每个epoch刷新embeddings
5. **维度一致性**: 使用window_actual_lens保证一致

### 已修复问题
1. ✅ 维度对齐 (train_pos, current_slice同步)
2. ✅ window_len一致性
3. ✅ AF计算正确

### 待监控问题
1. ⚠️ 窗口跳过机制 (如果发生需要修复)

### 运行建议
- 继续当前训练
- 监控是否有"跳过窗口"警告
- 检查Epoch 2的mask刷新是否正常
