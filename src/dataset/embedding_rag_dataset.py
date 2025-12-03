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
        self.ref_tokens_masked = []       # Masked tokens (用于检索) [window_idx][num_haps, L]
        self.ref_embeddings_complete = [] # 完整embeddings (返回给模型) [window_idx][num_haps, L, D] (CPU)
        self.ref_embeddings_masked = []   # Masked embeddings (用于FAISS索引) [window_idx][num_haps, L, D] (CPU)
        self.embedding_indexes = []       # FAISS索引 (基于masked embeddings) [window_idx]
        self.raw_window_masks = []        # 原始mask
        self.window_masks = []            # 填充后的mask
        self.mask_version = 0             # 当前mask版本号
        self.ref_af_windows = []          # AF信息 [window_idx][L]
        self.window_actual_lens = []      # 每个窗口过滤后的实际长度 [window_idx]

        # 存储embedding layer引用 (用于刷新)
        self.embedding_layer = embedding_layer
        self.embed_dim = embedding_layer.embed_size if embedding_layer else None

        if build_ref_data and ref_vcf_path and embedding_layer:
            self._build_embedding_indexes(ref_vcf_path, embedding_layer)

    def _build_embedding_indexes(self, ref_vcf_path: str, embedding_layer):
        """
        构建Embedding-based FAISS索引 (修改版 - 支持mask对齐)

        核心流程:
        1. 加载reference data
        2. 对每个window:
           - 生成mask (用于检索阶段的语义对齐)
           - Tokenize两个版本:
             a) Masked版本: 用于构建FAISS索引 (与Query mask一致)
             b) Complete版本: 用于返回给模型 (提供完整信息)
           - 编码两个版本的embeddings
           - 用masked embeddings构建FAISS索引
           - 存储complete embeddings用于返回
        """
        print("=" * 80)
        print("▣ 构建Embedding-based RAG索引")
        print("=" * 80)
        start_time = time.time()

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
                self.ref_tokens_masked.append(ref_tokens_masked)

                # 2. Complete版本: 用于返回给模型 (提供完整信息)
                ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)  # [num_haps, MAX_SEQ_LEN]
                self.ref_tokens_complete.append(ref_tokens_complete)

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

                # === 步骤4: 编码两个版本的embeddings (传入AF!) ===
                # 4a. Masked版本 (用于检索)
                ref_tokens_masked_tensor = torch.LongTensor(ref_tokens_masked).to(device)
                ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                ref_emb_masked = embedding_layer(ref_tokens_masked_tensor, af=ref_af_tensor, pos=True)  # [num_haps, L, D]

                # 4b. Complete版本 (用于返回)
                ref_tokens_complete_tensor = torch.LongTensor(ref_tokens_complete).to(device)
                ref_emb_complete = embedding_layer(ref_tokens_complete_tensor, af=ref_af_tensor, pos=True)  # [num_haps, L, D]

                # === 步骤5: 用Masked embeddings构建FAISS索引 ===
                num_haps, L, D = ref_emb_masked.shape
                ref_emb_masked_flat = ref_emb_masked.reshape(num_haps, L * D)  # [num_haps, L*D]
                ref_emb_masked_flat_np = ref_emb_masked_flat.cpu().numpy().astype(np.float32)

                # 构建FAISS索引 (基于masked embeddings)
                index = faiss.IndexFlatL2(L * D)  # L*D维空间
                index.add(ref_emb_masked_flat_np)
                self.embedding_indexes.append(index)

                # === 步骤6: 存储embeddings到CPU ===
                self.ref_embeddings_masked.append(ref_emb_masked.cpu())      # 用于重建索引
                self.ref_embeddings_complete.append(ref_emb_complete.cpu())  # 用于返回给模型

        # 计算存储大小 (两套embeddings)
        total_haps = sum(emb.shape[0] for emb in self.ref_embeddings_complete)
        storage_mb = (total_haps * MAX_SEQ_LEN * self.embed_dim * 4 * 2) / (1024 ** 2)  # float32 * 2 (masked + complete)

        print()
        print("=" * 80)
        print(f"✓ 预编码完成!")
        print(f"  - 窗口数: {self.window_count}")
        print(f"  - 总单体型数: {total_haps}")
        print(f"  - Embedding维度: {self.embed_dim}")
        print(f"  - FAISS索引维度: {MAX_SEQ_LEN * self.embed_dim}")
        print(f"  - Mask版本号: {self.mask_version}")
        print(f"  - 存储大小: {storage_mb:.1f} MB (两套embeddings, CPU RAM)")
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

                # 更新masked embeddings和tokens
                self.ref_tokens_masked[w_idx] = ref_tokens_masked
                self.ref_embeddings_masked[w_idx] = ref_emb_masked.cpu()

                # 重建FAISS索引
                num_haps, L, D = ref_emb_masked.shape
                ref_emb_flat = ref_emb_masked.reshape(num_haps, L * D)
                ref_emb_flat_np = ref_emb_flat.cpu().numpy().astype(np.float32)

                self.embedding_indexes[w_idx].reset()
                self.embedding_indexes[w_idx].add(ref_emb_flat_np)

        print(f"✓ 索引重建完成! 耗时: {time.time() - start_time:.2f}s")
        print(f"{'='*80}\n")

    def refresh_complete_embeddings(self, embedding_layer, device='cuda'):
        """
        刷新完整版本的reference embeddings (每个epoch结束后调用)

        关键: 用最新的embedding layer重新编码完整版本
        这确保返回给模型的embeddings是最新的learned representations
        """
        print(f"▣ 刷新Complete Embeddings")
        start_time = time.time()

        with torch.no_grad():
            for w_idx in tqdm(range(self.window_count), desc="刷新Complete"):
                # 获取完整tokens和AF
                ref_tokens_complete = self.ref_tokens_complete[w_idx]
                ref_af = self.ref_af_windows[w_idx]

                # 扩展AF
                num_haps = ref_tokens_complete.shape[0]
                ref_af_expanded = np.tile(ref_af, (num_haps, 1))

                # 重新编码完整版本
                ref_tokens_tensor = torch.LongTensor(ref_tokens_complete).to(device)
                ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                ref_emb_complete = embedding_layer(ref_tokens_tensor, af=ref_af_tensor, pos=True)

                # 更新完整embeddings
                self.ref_embeddings_complete[w_idx] = ref_emb_complete.cpu()

        print(f"✓ Complete刷新完成! 耗时: {time.time() - start_time:.2f}s")
        print(f"{'='*80}\n")

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


def embedding_rag_collate_fn(batch_list, dataset, embedding_layer, k_retrieve=1):
    """
    Embedding RAG的collate函数 (修改版 - 支持mask对齐)

    核心流程:
    1. Query用当前mask编码 (已在__getitem__中tokenized)
    2. 在FAISS中检索 (索引基于masked embeddings, 语义对齐)
    3. 返回COMPLETE embeddings (提供完整信息给模型)

    关键设计:
    - 检索阶段: Query和Reference都是masked (语义对齐)
    - 使用阶段: 返回complete embeddings (提供完整信息)
    """
    final_batch = defaultdict(list)
    device = next(embedding_layer.parameters()).device

    # 按窗口分组
    window_groups = defaultdict(list)
    for sample in batch_list:
        win_idx = int(sample['window_idx'])
        window_groups[win_idx].append(sample)

    # 处理每个窗口
    with torch.no_grad():  # 检索操作不需要梯度
        for win_idx, group in window_groups.items():
            # 获取FAISS索引 (基于masked embeddings)
            index = dataset.embedding_indexes[win_idx]
            # 获取完整embeddings (用于返回给模型)
            ref_emb_complete = dataset.ref_embeddings_complete[win_idx]  # [num_haps, L, D] (CPU)

            # 批量编码queries (用当前mask, 已在__getitem__中tokenized)
            h1_tokens = torch.stack([s['hap_1'] for s in group]).to(device)  # [B, L] (已masked)
            h2_tokens = torch.stack([s['hap_2'] for s in group]).to(device)  # [B, L] (已masked)
            af_batch = torch.stack([s['af'] for s in group]).to(device)  # [B, L]

            # 编码query (masked版本, 与索引语义对齐)
            h1_emb = embedding_layer(h1_tokens, af=af_batch, pos=True)  # [B, L, D]
            h2_emb = embedding_layer(h2_tokens, af=af_batch, pos=True)

            B, L, D = h1_emb.shape

            # Flatten并检索 (在masked space检索)
            h1_emb_flat = h1_emb.reshape(B, L * D).cpu().numpy().astype(np.float32)
            h2_emb_flat = h2_emb.reshape(B, L * D).cpu().numpy().astype(np.float32)

            D1, I1 = index.search(h1_emb_flat, k=k_retrieve)  # [B, k]
            D2, I2 = index.search(h2_emb_flat, k=k_retrieve)

            # 收集retrieved embeddings (返回COMPLETE版本!)
            for i, sample in enumerate(group):
                # h1的top-k (完整embeddings)
                topk_h1 = []
                for k in range(k_retrieve):
                    ref_idx = I1[i, k]
                    topk_h1.append(ref_emb_complete[ref_idx])  # [L, D] - 完整!
                sample['rag_emb_h1'] = torch.stack(topk_h1)  # [k, L, D]

                # h2的top-k (完整embeddings)
                topk_h2 = []
                for k in range(k_retrieve):
                    ref_idx = I2[i, k]
                    topk_h2.append(ref_emb_complete[ref_idx])  # [L, D] - 完整!
                sample['rag_emb_h2'] = torch.stack(topk_h2)  # [k, L, D]

            # 收集数据
            for sample in group:
                for key in sample:
                    final_batch[key].append(sample[key])

    # 转换为张量
    for key in final_batch:
        if key in ["window_idx", "hap1_nomask", "hap2_nomask"]:
            continue
        try:
            final_batch[key] = torch.stack(final_batch[key])
        except RuntimeError as e:
            print(f"Warning: Failed to stack {key}: {e}")

    return dict(final_batch)
