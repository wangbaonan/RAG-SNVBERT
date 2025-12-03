# embedding_rag_dataset.py
# 端到端可学习的Embedding RAG Dataset
# 关键特性:
# 1. 检索在embedding space进行 (端到端可学习)
# 2. 每个epoch刷新reference embeddings (保持最新)
# 3. 内存优化: reference embeddings存在CPU

import time
import os
import h5py
from collections import defaultdict
import faiss
import allel
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional

from .dataset import TrainDataset
from .utils import timer, PanelProcessingModule, VCFProcessingModule

# DEFAULT MAXIMUM SEQUENCE LENGTH
INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030


class EmbeddingRAGDataset(TrainDataset):
    """
    Embedding RAG Dataset with end-to-end learnable retrieval

    核心改进:
    - FAISS检索在embedding space (learned representation)
    - Reference embeddings在每个epoch后刷新 (端到端可学习)
    - 内存优化: embeddings存在CPU, 仅在需要时移到GPU
    """

    def __init__(self, vocab, vcf, pos, panel, freq, window,
                 type_to_idx, pop_to_idx, pos_to_idx,
                 ref_vcf_path=None,
                 embedding_layer=None,
                 build_ref_data=True,
                 n_gpu=1,
                 maf_mask_percentage=10,
                 use_dynamic_mask=False):
        super().__init__(vocab, vcf, pos, panel, freq, window,
                        type_to_idx, pop_to_idx, pos_to_idx)

        self.maf_mask_percentage = maf_mask_percentage
        self.use_dynamic_mask = use_dynamic_mask
        self.current_epoch = 0

        # Embedding RAG特有的数据结构
        self.ref_tokens_complete = []     # 完整tokens (无mask) [window_idx][num_haps, L]
        self.raw_window_masks = []        # 原始mask
        self.window_masks = []            # 填充后的mask
        self.mask_version = 0             # 当前mask版本号
        self.ref_af_windows = []          # AF信息 [window_idx][L]
        self.window_actual_lens = []      # 每个窗口过滤后的实际长度 [window_idx]

        # FAISS索引路径 (保存到磁盘，不占用内存)
        self.index_dir = os.path.join(os.path.dirname(ref_vcf_path) if ref_vcf_path else ".", "faiss_indexes")
        self.index_paths = []             # 索引文件路径

        # 存储embedding layer引用 (用于按需编码)
        self.embedding_layer = embedding_layer
        self.embed_dim = embedding_layer.embed_size if embedding_layer else None

        if build_ref_data and ref_vcf_path and embedding_layer:
            self._build_embedding_indexes(ref_vcf_path, embedding_layer)

    def _build_embedding_indexes(self, ref_vcf_path: str, embedding_layer):
        """
        构建Embedding-based FAISS索引 (内存优化版)

        核心改进:
        1. 只在内存保存tokens和AF (小数据)
        2. FAISS索引保存到磁盘 (不占内存)
        3. Complete embeddings按需编码 (不预存储)

        内存消耗: ~11GB (vs 1.6TB原设计)
        """
        print("=" * 80)
        print("▣ 构建Embedding-based RAG索引 (内存优化版)")
        print("=" * 80)
        start_time = time.time()

        # 创建FAISS索引目录
        os.makedirs(self.index_dir, exist_ok=True)
        print(f"✓ FAISS索引目录: {self.index_dir}")

        # 加载参考数据
        load_start = time.time()
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        print(f"✓ 加载参考数据: 样本数={ref_gt.shape[1]} | 位点数={ref_gt.shape[0]} | "
              f"耗时={time.time()-load_start:.2f}s")

        # 获取设备
        device = next(embedding_layer.parameters()).device
        print(f"✓ Embedding层设备: {device}")
        print(f"✓ Embedding维度: {self.embed_dim}")
        print()

        # 对每个window进行预编码
        with torch.no_grad():  # 关键: 预编码时不计算梯度
            for w_idx in tqdm(range(self.window_count), desc="预编码窗口"):
                current_slice = slice(
                    self.window.window_info[w_idx, 0],
                    self.window.window_info[w_idx, 1]
                )
                window_len = current_slice.stop - current_slice.start

                # === 步骤1: 先处理参考数据以确定有效位点 ===
                train_pos = self.pos[current_slice]
                ref_indices = []
                valid_pos_mask = []

                for idx, p in enumerate(train_pos):
                    matches = np.where(ref_pos == p)[0]
                    if len(matches) > 0:
                        ref_indices.append(matches[0])
                        valid_pos_mask.append(idx)

                # 如果有位点被过滤，更新current_slice和train_pos
                if len(ref_indices) < len(train_pos):
                    if len(valid_pos_mask) == 0:
                        print(f"  ⚠ 跳过窗口 {w_idx}: 没有可用位点")
                        continue
                    # 关键修复: 同时更新current_slice和train_pos
                    valid_indices = current_slice.start + np.array(valid_pos_mask)
                    current_slice = valid_indices
                    train_pos = train_pos[valid_pos_mask]
                    # 更新window_len为过滤后的长度
                    window_len = len(train_pos)

                # 保存每个窗口的实际长度 (用于regenerate_masks)
                self.window_actual_lens.append(window_len)

                # === 步骤2: 生成mask (用于语义对齐) ===
                # 现在基于过滤后的window_len生成mask
                raw_mask = self.generate_mask(window_len)
                padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

                # 保存mask用于后续刷新
                self.raw_window_masks.append(raw_mask)
                self.window_masks.append(padded_mask)

                # 创建完整版本的mask (全0, 用于complete版本)
                raw_mask_complete = np.zeros_like(raw_mask)
                padded_mask_complete = VCFProcessingModule.sequence_padding(
                    raw_mask_complete, dtype='int'
                )

                # 获取reference sequences
                raw_ref = ref_gt[current_slice, :, :]  # [L, num_samples, 2]
                raw_ref = raw_ref.reshape(raw_ref.shape[0], -1)  # [L, num_haps]
                raw_ref = raw_ref.T  # [num_haps, L]

                # Tokenize两个版本
                # 1. Masked版本: 用于检索 (与Query语义对齐)
                ref_tokens_masked = self.tokenize(raw_ref, padded_mask)  # [num_haps, MAX_SEQ_LEN]

                # 2. Complete版本: 用于按需编码 (提供完整信息)
                ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)  # [num_haps, MAX_SEQ_LEN]
                self.ref_tokens_complete.append(ref_tokens_complete)  # 只保存complete tokens

                # === 步骤3: 计算AF (Reference的真实AF) ===
                # 从reference panel计算每个位点的AF
                # 现在train_pos和raw_ref维度已对齐，可以安全计算
                # AF=3, GLOBAL=5 (constants from dataset.py)
                AF_IDX = 3
                GLOBAL_IDX = 5
                # 使用列表推导式，类似base dataset的实现
                ref_af = np.array([
                    self.freq[AF_IDX][GLOBAL_IDX][self.pos_to_idx[p]]
                    if p in self.pos_to_idx else 0.0
                    for p in train_pos
                ], dtype=np.float32)

                # Padding到MAX_SEQ_LEN
                ref_af = VCFProcessingModule.sequence_padding(ref_af, dtype='float')  # [MAX_SEQ_LEN]

                # 保存AF信息用于后续刷新
                self.ref_af_windows.append(ref_af)  # 只保存一份，后续会扩展

                # 扩展到所有haplotypes (每个haplotype共享相同的AF)
                num_haps_in_window = ref_tokens_masked.shape[0]
                ref_af_expanded = np.tile(ref_af, (num_haps_in_window, 1))  # [num_haps, MAX_SEQ_LEN]

                # === 步骤4: 编码masked版本并构建FAISS索引 ===
                # 只编码masked版本 (用于构建索引)
                ref_tokens_masked_tensor = torch.LongTensor(ref_tokens_masked).to(device)
                ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                ref_emb_masked = embedding_layer(ref_tokens_masked_tensor, af=ref_af_tensor, pos=True)  # [num_haps, L, D]

                # === 步骤5: 用Masked embeddings构建FAISS索引 ===
                num_haps, L, D = ref_emb_masked.shape
                ref_emb_masked_flat = ref_emb_masked.reshape(num_haps, L * D)  # [num_haps, L*D]
                ref_emb_masked_flat_np = ref_emb_masked_flat.cpu().numpy().astype(np.float32)

                # 构建FAISS索引 (基于masked embeddings)
                index = faiss.IndexFlatL2(L * D)  # L*D维空间
                index.add(ref_emb_masked_flat_np)

                # === 步骤6: 保存FAISS索引到磁盘 (不保存到内存!) ===
                index_path = os.path.join(self.index_dir, f"index_{w_idx}.faiss")
                faiss.write_index(index, index_path)
                self.index_paths.append(index_path)

                # 清理GPU内存
                del ref_tokens_masked_tensor, ref_af_tensor, ref_emb_masked, ref_emb_masked_flat, ref_emb_masked_flat_np, index
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        # 计算存储大小 (只计算内存中的tokens和AF)
        total_haps = sum(tokens.shape[0] for tokens in self.ref_tokens_complete)
        tokens_mb = (total_haps * MAX_SEQ_LEN * 8) / (1024 ** 2)  # int64, 8 bytes
        af_mb = (self.window_count * MAX_SEQ_LEN * 4) / (1024 ** 2)  # float32, 4 bytes
        memory_mb = tokens_mb + af_mb

        # 计算磁盘占用 (FAISS索引)
        disk_gb = (total_haps * MAX_SEQ_LEN * self.embed_dim * 4) / (1024 ** 3)  # float32

        print()
        print("=" * 80)
        print(f"✓ 预编码完成! (内存优化版)")
        print(f"  - 窗口数: {self.window_count}")
        print(f"  - 总单体型数: {total_haps}")
        print(f"  - Embedding维度: {self.embed_dim}")
        print(f"  - FAISS索引维度: {MAX_SEQ_LEN * self.embed_dim}")
        print(f"  - Mask版本号: {self.mask_version}")
        print(f"  - 内存占用: {memory_mb:.1f} MB (tokens + AF) ✅")
        print(f"  - 磁盘占用: {disk_gb:.1f} GB (FAISS索引)")
        print(f"  - 总耗时: {time.time() - start_time:.2f}s")
        print("=" * 80)

    def regenerate_masks(self, seed: int):
        """
        重新生成所有窗口的mask (用于数据增强)

        Args:
            seed: 随机种子 (通常是epoch number)
        """
        self.mask_version += 1
        print(f"\n{'='*80}")
        print(f"▣ 刷新Mask Pattern (版本 {self.mask_version}, Seed={seed})")
        print(f"{'='*80}")

        for w_idx in range(self.window_count):
            # 使用过滤后的实际长度 (不是window.window_info的原始长度!)
            window_len = self.window_actual_lens[w_idx]

            # 生成新mask
            np.random.seed(seed * 10000 + w_idx)
            raw_mask = self.generate_mask(window_len)
            padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

            # 更新保存的mask
            self.raw_window_masks[w_idx] = raw_mask
            self.window_masks[w_idx] = padded_mask

        print(f"✓ Mask刷新完成! 新版本: {self.mask_version}")
        print(f"{'='*80}\n")

    def load_index(self, w_idx):
        """从磁盘加载FAISS索引"""
        return faiss.read_index(self.index_paths[w_idx])

    def encode_complete_embeddings(self, w_idx, device='cuda', grad_enabled=False):
        """
        按需编码complete embeddings (支持梯度回传)

        ⚠️ DEPRECATED: 此方法会对整个参考面板编码，导致OOM
        推荐使用 process_batch_retrieval 的 Lazy Encoding 策略

        仅在 rebuild_indexes 中使用（无梯度模式）

        Args:
            w_idx: 窗口索引
            device: GPU设备
            grad_enabled: 是否启用梯度计算 (训练时True, 索引重建时False)

        Returns:
            ref_emb_complete: [num_haps, L, D] GPU tensor
        """
        ref_tokens = self.ref_tokens_complete[w_idx]
        ref_af = self.ref_af_windows[w_idx]

        # 扩展AF
        num_haps = ref_tokens.shape[0]
        ref_af_expanded = np.tile(ref_af, (num_haps, 1))

        # 编码 (根据grad_enabled决定是否启用梯度)
        if grad_enabled:
            # 训练模式：启用梯度，让Reference编码参与端到端学习
            ref_emb = self.embedding_layer(
                torch.LongTensor(ref_tokens).to(device),
                af=torch.FloatTensor(ref_af_expanded).to(device),
                pos=True
            )
        else:
            # 索引重建模式：不需要梯度
            with torch.no_grad():
                ref_emb = self.embedding_layer(
                    torch.LongTensor(ref_tokens).to(device),
                    af=torch.FloatTensor(ref_af_expanded).to(device),
                    pos=True
                )

        return ref_emb  # [num_haps, L, D] 在GPU上

    def process_batch_retrieval(self, batch, embedding_layer, device, k_retrieve=1):
        """
        在主进程中执行RAG检索 (Lazy Encoding 优化版 - 解决OOM)

        关键优化:
        1. **Lazy Encoding**: 先检索，只编码检索到的少量 haplotypes
        2. 避免对整个参考面板（数千个haplotypes）全量编码
        3. Query编码和Reference编码都在计算图中（grad_enabled=True）
        4. 显存占用: O(B*k) 而非 O(num_total_haps)

        核心流程:
        1. 编码Query (masked版本, 带梯度)
        2. FAISS检索获取Top-K索引 (不可微)
        3. **按需提取**: 只从CPU内存提取检索到的 haplotypes 的 tokens
        4. **按需编码**: 只编码这少量 retrieved tokens (带梯度!)
        5. 返回带梯度的RAG embeddings

        Args:
            batch: CPU batch from DataLoader (dict)
            embedding_layer: Embedding层 (用于编码)
            device: GPU设备
            k_retrieve: 检索K个最近邻

        Returns:
            batch: 增强后的batch，包含 'rag_emb_h1' 和 'rag_emb_h2'
        """
        # 1. 将batch数据移到GPU
        h1_tokens = batch['hap_1'].to(device)  # [B, L] masked tokens
        h2_tokens = batch['hap_2'].to(device)
        af_batch = batch['af'].to(device)
        window_idx_list = batch['window_idx']  # list

        # 2. 按窗口分组
        window_groups = defaultdict(list)
        for i, win_idx in enumerate(window_idx_list):
            win_idx = int(win_idx)
            window_groups[win_idx].append(i)

        # 初始化RAG embedding存储
        B = h1_tokens.size(0)
        L = h1_tokens.size(1)
        D = self.embed_dim

        # 使用list存储，最后stack
        rag_emb_h1_list = []
        rag_emb_h2_list = []

        # 3. 处理每个窗口（Lazy Encoding）
        for win_idx, indices in window_groups.items():
            # 获取当前窗口的样本
            h1_win = h1_tokens[indices]  # [B_win, L]
            h2_win = h2_tokens[indices]
            af_win = af_batch[indices]

            # === 步骤1: 编码Query (带梯度!) ===
            h1_emb = embedding_layer(h1_win, af=af_win, pos=True)  # [B_win, L, D]
            h2_emb = embedding_layer(h2_win, af=af_win, pos=True)

            B_win = h1_emb.size(0)

            # === 步骤2: FAISS检索获取Top-K索引 ===
            with torch.no_grad():
                # 加载FAISS索引
                index = self.load_index(win_idx)

                # Flatten并检索
                h1_emb_flat = h1_emb.reshape(B_win, L * D).cpu().numpy().astype(np.float32)
                h2_emb_flat = h2_emb.reshape(B_win, L * D).cpu().numpy().astype(np.float32)

                D1, I1 = index.search(h1_emb_flat, k=k_retrieve)  # [B_win, k]
                D2, I2 = index.search(h2_emb_flat, k=k_retrieve)

            # === 步骤3: 按需提取 - 只取检索到的 haplotypes ===
            # 获取所有需要的唯一索引
            unique_indices_h1 = np.unique(I1.flatten())
            unique_indices_h2 = np.unique(I2.flatten())
            unique_indices = np.unique(np.concatenate([unique_indices_h1, unique_indices_h2]))

            # 从CPU内存提取对应的 tokens 和 AF
            ref_tokens_complete = self.ref_tokens_complete[win_idx]  # [num_haps, L]
            ref_af = self.ref_af_windows[win_idx]  # [L]

            # 只提取需要的 haplotypes
            retrieved_tokens = ref_tokens_complete[unique_indices]  # [num_retrieved, L]
            num_retrieved = retrieved_tokens.shape[0]

            # 扩展AF（每个haplotype共享相同的AF）
            retrieved_af = np.tile(ref_af, (num_retrieved, 1))  # [num_retrieved, L]

            # === 步骤4: 按需编码 - 只编码检索到的 (带梯度!) ===
            # 转为GPU tensor并编码
            retrieved_tokens_tensor = torch.LongTensor(retrieved_tokens).to(device)
            retrieved_af_tensor = torch.FloatTensor(retrieved_af).to(device)

            # 关键: 不使用 torch.no_grad()，保持梯度
            retrieved_emb = embedding_layer(
                retrieved_tokens_tensor,
                af=retrieved_af_tensor,
                pos=True
            )  # [num_retrieved, L, D] 带梯度!

            # 创建索引映射: old_index -> new_index
            index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

            # === 步骤5: 收集Retrieved Embeddings (保持梯度) ===
            for i in range(B_win):
                # h1的top-k
                topk_h1 = []
                for k in range(k_retrieve):
                    old_ref_idx = I1[i, k]
                    new_ref_idx = index_map[old_ref_idx]
                    topk_h1.append(retrieved_emb[new_ref_idx])  # [L, D] 带梯度!
                rag_emb_h1_list.append(torch.stack(topk_h1))  # [k, L, D]

                # h2的top-k
                topk_h2 = []
                for k in range(k_retrieve):
                    old_ref_idx = I2[i, k]
                    new_ref_idx = index_map[old_ref_idx]
                    topk_h2.append(retrieved_emb[new_ref_idx])  # [L, D] 带梯度!
                rag_emb_h2_list.append(torch.stack(topk_h2))  # [k, L, D]

        # 4. 堆叠为batch (保持梯度)
        batch['rag_emb_h1'] = torch.stack(rag_emb_h1_list)  # [B, k, L, D]
        batch['rag_emb_h2'] = torch.stack(rag_emb_h2_list)

        return batch

    def rebuild_indexes(self, embedding_layer, device='cuda'):
        """
        用当前mask重建FAISS索引 (在regenerate_masks后调用)

        核心流程:
        1. 用当前mask重新tokenize reference sequences
        2. 用最新embedding layer编码masked版本
        3. 重建FAISS索引
        """
        print(f"▣ 重建FAISS索引 (基于新Mask)")
        start_time = time.time()

        with torch.no_grad():
            for w_idx in tqdm(range(self.window_count), desc="重建索引"):
                # 获取完整tokens和AF
                ref_tokens_complete = self.ref_tokens_complete[w_idx]  # [num_haps, L]
                ref_af = self.ref_af_windows[w_idx]  # [L]
                current_mask = self.window_masks[w_idx]  # [L]

                # 应用当前mask到完整tokens
                ref_tokens_masked = self._apply_mask_to_tokens(ref_tokens_complete, current_mask)

                # 扩展AF
                num_haps = ref_tokens_masked.shape[0]
                ref_af_expanded = np.tile(ref_af, (num_haps, 1))

                # 用masked tokens编码
                ref_tokens_tensor = torch.LongTensor(ref_tokens_masked).to(device)
                ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                ref_emb_masked = embedding_layer(ref_tokens_tensor, af=ref_af_tensor, pos=True)

                # 重建FAISS索引
                num_haps, L, D = ref_emb_masked.shape
                ref_emb_flat = ref_emb_masked.reshape(num_haps, L * D)
                ref_emb_flat_np = ref_emb_flat.cpu().numpy().astype(np.float32)

                # 构建新索引
                index = faiss.IndexFlatL2(L * D)
                index.add(ref_emb_flat_np)

                # 保存到磁盘 (覆盖旧索引)
                index_path = self.index_paths[w_idx]
                faiss.write_index(index, index_path)

                # 清理GPU内存
                del ref_tokens_tensor, ref_af_tensor, ref_emb_masked, ref_emb_flat, ref_emb_flat_np, index
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        print(f"✓ 索引重建完成! 耗时: {time.time() - start_time:.2f}s")
        print(f"{'='*80}\n")

    # refresh_complete_embeddings方法已删除
    # 现在使用encode_complete_embeddings()按需编码，不预存储

    def _apply_mask_to_tokens(self, tokens, mask):
        """
        应用mask到token序列

        Args:
            tokens: [num_haps, L] 完整的tokens
            mask: [L] mask pattern (0=keep, 1=mask)

        Returns:
            masked_tokens: [num_haps, L] 应用mask后的tokens
        """
        masked_tokens = tokens.copy()
        mask_token_id = 4  # [MASK] token ID
        mask_positions = (mask == 1)
        masked_tokens[:, mask_positions] = mask_token_id
        return masked_tokens

    def _load_ref_data(self, ref_vcf_path: str) -> tuple:
        """加载并预处理参考VCF数据"""
        h5_path = os.path.splitext(ref_vcf_path)[0] + ".h5"
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                return f['calldata/GT'][:], f['variants/POS'][:]

        # 从VCF读取
        callset = allel.read_vcf(
            ref_vcf_path,
            fields=['variants/POS', 'calldata/GT'],
            alt_number=1
        )
        ref_gt = callset['calldata/GT'][:]
        ref_pos = callset['variants/POS']

        # 保存为HDF5
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('calldata/GT', data=ref_gt, compression='gzip')
            h5f.create_dataset('variants/POS', data=ref_pos, compression='gzip')

        return ref_gt, ref_pos

    def __getitem__(self, item) -> dict:
        output = super().__getitem__(item)
        window_idx = item % self.window_count

        # 根据配置选择静态或动态mask
        if self.use_dynamic_mask:
            # 使用过滤后的实际长度
            window_len = self.window_actual_lens[window_idx]

            old_state = np.random.get_state()
            np.random.seed(self.current_epoch * 10000 + window_idx)

            raw_mask = self.generate_mask(window_len)
            current_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

            np.random.set_state(old_state)
        else:
            current_mask = self.window_masks[window_idx]

        output['mask'] = current_mask
        output['hap_1'] = self.tokenize(output['hap1_nomask'], current_mask)
        output['hap_2'] = self.tokenize(output['hap2_nomask'], current_mask)

        for key in self.long_fields:
            output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            output[key] = torch.FloatTensor(output[key])

        return output

    @classmethod
    def from_file(cls,
                  vocab,
                  vcfpath,
                  panelpath,
                  freqpath,
                  windowpath,
                  typepath,
                  poppath,
                  pospath,
                  ref_vcf_path=None,
                  embedding_layer=None,
                  build_ref_data=True,
                  n_gpu=1,
                  use_dynamic_mask=False):
        """
        从文件创建EmbeddingRAGDataset

        重要: 必须提供embedding_layer用于预编码!
        """
        # 调用父类创建基础dataset
        base_dataset = TrainDataset.from_file(
            vocab, vcfpath, panelpath, freqpath, windowpath,
            typepath, poppath, pospath
        )

        # 创建EmbeddingRAGDataset
        rag_dataset = cls(
            vocab=base_dataset.vocab,
            vcf=base_dataset.vcf,
            pos=base_dataset.pos,
            panel=base_dataset.panel,
            freq=base_dataset.freq,
            window=base_dataset.window,
            type_to_idx=base_dataset.type_to_idx,
            pop_to_idx=base_dataset.pop_to_idx,
            pos_to_idx=base_dataset.pos_to_idx,
            ref_vcf_path=ref_vcf_path,
            embedding_layer=embedding_layer,
            build_ref_data=build_ref_data,
            n_gpu=n_gpu,
            use_dynamic_mask=use_dynamic_mask
        )
        return rag_dataset


def embedding_rag_collate_fn(batch_list, dataset=None, embedding_layer=None, k_retrieve=1):
    """
    纯CPU Collate函数 (重构版 - 解耦加载与计算)

    关键改进:
    - 只在CPU上堆叠基础数据 (tokens, AF, pos, window_idx)
    - 不做任何GPU操作，不调用embedding_layer
    - 不执行检索逻辑
    - 支持 num_workers > 0 (无CUDA fork风险)

    检索和编码逻辑移至 dataset.process_batch_retrieval() 在主进程执行

    Args:
        batch_list: list of samples from __getitem__
        dataset: 保留参数兼容性，但本函数不使用
        embedding_layer: 保留参数兼容性，但本函数不使用
        k_retrieve: 保留参数兼容性，但本函数不使用
    """
    final_batch = defaultdict(list)

    # 简单堆叠所有样本的数据
    for sample in batch_list:
        for key in sample:
            final_batch[key].append(sample[key])

    # 转换为张量（只在CPU上操作）
    for key in final_batch:
        if key in ["window_idx", "hap1_nomask", "hap2_nomask"]:
            # window_idx保持为list，其他保持原样
            continue
        try:
            final_batch[key] = torch.stack(final_batch[key])
        except (RuntimeError, TypeError) as e:
            # 如果无法stack，保持为list
            pass

    return dict(final_batch)
