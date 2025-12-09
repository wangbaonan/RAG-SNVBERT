# embedding_rag_infer_dataset.py
# V18 Embedding RAG 专用推理数据集
# 关键特性:
# 1. Imputation Masking: Mask 位置由数据缺失情况决定
# 2. 对称 Masking: Query 和 Reference 在相同位置 Mask
# 3. Lazy Encoding: 检索后按需编码 Complete Reference

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

from .dataset import InferDataset
from .utils import timer, PanelProcessingModule, VCFProcessingModule

# DEFAULT MAXIMUM SEQUENCE LENGTH
INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030


class EmbeddingRAGInferDataset(InferDataset):
    """
    V18 Embedding RAG Inference Dataset

    核心逻辑:
    1. Imputation Masking: Mask = All_Ref_Positions - Target_Known_Positions
    2. Symmetric Masking: Query 和 Reference 在相同位置 Mask
    3. Key (索引): Masked Reference Embeddings
    4. Value (生成): Complete Reference Tokens
    """

    def __init__(self, vocab, vcf, pos, panel, freq, type_to_idx, pop_to_idx, pos_to_idx,
                 ref_vcf_path=None,
                 embedding_layer=None,
                 build_ref_data=True,
                 n_gpu=1,
                 build_index: bool = True,
                 name='infer'):
        super().__init__(vocab, vcf, pos, panel, freq, type_to_idx, pop_to_idx, pos_to_idx)

        # Embedding layer (用于构建索引)
        self.embedding_layer = embedding_layer
        self.embed_dim = embedding_layer.embed_size if embedding_layer else 384

        # RAG 相关属性
        self.ref_tokens_complete = []    # Complete Reference Tokens (Value)
        self.ref_af_windows = []         # Reference AF
        self.infer_masks = []            # Imputation Masks
        self.raw_window_masks = []       # Raw Masks (未 Padding)
        self.index_paths = []            # FAISS 索引路径

        # GPU 缓存 (单槽位)
        self.gpu_res = faiss.StandardGpuResources()
        self.cached_index = None
        self.cached_window_idx = -1

        # FAISS 索引目录
        self.index_dir = f"maf_data/faiss_indexes_{name}"

        # 构建索引
        self.ref_vcf_path = ref_vcf_path
        if not build_index:
            print("First loading, no index building.")
        if build_ref_data and ref_vcf_path and build_index and embedding_layer:
            print("Building Embedding RAG Indexes for Inference...")
            self._build_embedding_indexes(ref_vcf_path)

    def _load_ref_data(self, ref_vcf_path: str):
        """加载参考数据"""
        if ref_vcf_path.endswith('.h5'):
            with h5py.File(ref_vcf_path, 'r') as f:
                ref_gt = f['gt'][:]
                ref_pos = f['variants/POS'][:]
        else:
            vcf_data = allel.read_vcf(
                ref_vcf_path,
                fields=['variants/POS', 'calldata/GT'],
                samples=None
            )
            ref_gt = vcf_data['calldata/GT']
            ref_pos = vcf_data['variants/POS']
        return ref_gt, ref_pos

    def _build_embedding_indexes(self, ref_vcf_path: str):
        """
        构建 Embedding RAG 索引 (推理版本)

        关键逻辑:
        1. 计算 Imputation Mask: Ref_Positions - Target_Known_Positions
        2. 对称 Masking: Query 和 Reference 在相同位置 Mask
        3. Key (索引): Masked Reference Embeddings
        4. Value (生成): Complete Reference Tokens
        """
        print("=" * 80)
        print("▣ 构建 Embedding RAG 推理索引")
        print("=" * 80)
        start_time = time.time()

        # 创建 FAISS 索引目录
        os.makedirs(self.index_dir, exist_ok=True)
        print(f"✓ FAISS 索引目录: {self.index_dir}")

        # 加载参考数据
        load_start = time.time()
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        print(f"✓ 加载参考数据: 样本数={ref_gt.shape[1]} | 位点数={ref_gt.shape[0]} | "
              f"耗时={time.time()-load_start:.2f}s")

        # 获取设备
        device = next(self.embedding_layer.parameters()).device
        print(f"✓ Embedding 层设备: {device}")
        print(f"✓ Embedding 维度: {self.embed_dim}")
        print()

        # 保存原始状态并切换到 eval 模式
        was_training = self.embedding_layer.training
        self.embedding_layer.eval()

        # 预构建位置字典
        pos_to_index = {}
        for idx, p in enumerate(ref_pos):
            if p not in pos_to_index:
                pos_to_index[p] = idx

        try:
            with torch.no_grad():
                for w_idx in tqdm(range(self.window_count), desc="预编码推理窗口"):
                    start_idx = INFER_WINDOW_LEN * w_idx
                    end_idx = min(start_idx + INFER_WINDOW_LEN, self.ori_pos.shape[0])

                    # === 步骤1: 计算 Imputation Mask ===
                    # Mask_Positions = All_Reference_Positions - Target_Known_Positions
                    # Mask=1: 需要填补的位置
                    # Mask=0: 已知位置（作为 Context）
                    mask = np.array([
                        1 if self.position_needed[i] else 0
                        for i in range(start_idx, end_idx)
                    ], dtype=int)

                    window_len = len(mask)

                    # Padding
                    if window_len < INFER_WINDOW_LEN:
                        pad_len = INFER_WINDOW_LEN - window_len
                        mask = np.pad(mask, (0, pad_len), mode='constant')

                    # 保存 Mask
                    self.raw_window_masks.append(mask[:window_len].copy())
                    padded_mask = VCFProcessingModule.sequence_padding(mask, dtype='int')
                    self.infer_masks.append(padded_mask)

                    # === 步骤2: 获取 Reference Sequences ===
                    current_pos = self.ori_pos[start_idx:end_idx]
                    ref_indices = []
                    valid_pos_mask = []

                    for idx, p in enumerate(current_pos):
                        ref_idx = pos_to_index.get(p, -1)
                        if ref_idx != -1:
                            ref_indices.append(ref_idx)
                            valid_pos_mask.append(idx)

                    # 如果有位点被过滤，更新 mask 和 window_len
                    if len(ref_indices) < len(current_pos):
                        if len(valid_pos_mask) == 0:
                            print(f"  ⚠ 跳过窗口 {w_idx}: 没有可用位点")
                            continue
                        # 更新 mask 和长度
                        mask = mask[valid_pos_mask]
                        window_len = len(mask)
                        current_pos = current_pos[valid_pos_mask]

                    # 获取 Reference Sequences
                    raw_ref = ref_gt[ref_indices, :, :]  # [L, num_samples, 2]
                    raw_ref = raw_ref.reshape(raw_ref.shape[0], -1)  # [L, num_haps]
                    raw_ref = raw_ref.T  # [num_haps, L]

                    # === 步骤3: 计算 AF (Reference 的真实 AF) ===
                    AF_IDX = 3
                    GLOBAL_IDX = 5
                    ref_af = np.array([
                        self.freq[AF_IDX][GLOBAL_IDX][self.pos_to_idx[p]]
                        if p in self.pos_to_idx else 0.0
                        for p in current_pos
                    ], dtype=np.float32)

                    # Padding AF
                    ref_af = VCFProcessingModule.sequence_padding(ref_af, dtype='float')
                    self.ref_af_windows.append(ref_af)

                    # === 步骤4: Tokenize (Masked 和 Complete 版本) ===
                    # Masked 版本: 用于构建索引 (Key)
                    # 关键: 对称 Masking - Reference 在相同位置 Mask
                    padded_mask_for_ref = VCFProcessingModule.sequence_padding(
                        mask[:window_len], dtype='int'
                    )
                    ref_tokens_masked = self.tokenize(raw_ref, padded_mask_for_ref)  # [num_haps, MAX_SEQ_LEN]

                    # Complete 版本: 用于按需编码 (Value)
                    padded_mask_complete = np.zeros_like(padded_mask_for_ref)
                    ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)
                    self.ref_tokens_complete.append(ref_tokens_complete)

                    # === 步骤5: 编码 Masked Reference 并构建 FAISS 索引 ===
                    num_haps = ref_tokens_masked.shape[0]
                    ref_af_expanded = np.tile(ref_af, (num_haps, 1))

                    # 编码 Masked 版本
                    ref_tokens_masked_tensor = torch.LongTensor(ref_tokens_masked).to(device)
                    ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                    ref_emb_masked = self.embedding_layer(
                        ref_tokens_masked_tensor,
                        af=ref_af_tensor,
                        pos=True
                    )  # [num_haps, L, D]

                    # === 步骤6: 构建 FAISS 索引 ===
                    num_haps, L, D = ref_emb_masked.shape
                    ref_emb_masked_flat = ref_emb_masked.reshape(num_haps, L * D)
                    ref_emb_masked_flat_np = ref_emb_masked_flat.cpu().numpy().astype(np.float32)

                    # 构建 FAISS 索引
                    index = faiss.IndexFlatL2(L * D)
                    index.add(ref_emb_masked_flat_np)

                    # 保存到磁盘
                    index_path = os.path.join(self.index_dir, f"index_{w_idx}.faiss")
                    faiss.write_index(index, index_path)
                    self.index_paths.append(index_path)

                    # 清理 GPU 和 CPU 内存
                    del ref_tokens_masked_tensor, ref_af_tensor, ref_emb_masked, ref_emb_masked_flat, ref_emb_masked_flat_np, index
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        finally:
            # 恢复原始训练状态
            self.embedding_layer.train(was_training)

        # 计算存储大小
        total_haps = sum(tokens.shape[0] for tokens in self.ref_tokens_complete)
        tokens_mb = (total_haps * MAX_SEQ_LEN * 8) / (1024 ** 2)
        af_mb = (self.window_count * MAX_SEQ_LEN * 4) / (1024 ** 2)
        memory_mb = tokens_mb + af_mb

        disk_gb = (total_haps * MAX_SEQ_LEN * self.embed_dim * 4) / (1024 ** 3)

        print()
        print("=" * 80)
        print(f"✓ 推理索引构建完成!")
        print(f"  - 窗口数: {self.window_count}")
        print(f"  - 总单体型数: {total_haps}")
        print(f"  - Embedding 维度: {self.embed_dim}")
        print(f"  - FAISS 索引维度: {MAX_SEQ_LEN * self.embed_dim}")
        print(f"  - 内存占用: {memory_mb:.1f} MB (tokens + AF)")
        print(f"  - 磁盘占用: {disk_gb:.1f} GB (FAISS 索引)")
        print(f"  - 总耗时: {time.time() - start_time:.2f}s")
        print("=" * 80)

    def load_index(self, w_idx):
        """
        加载 FAISS 索引 (GPU 加速 + 单槽位缓存)
        """
        # 单槽位缓存命中
        if w_idx == self.cached_window_idx and self.cached_index is not None:
            return self.cached_index

        # 缓存未命中: 从磁盘加载并转为 GPU 索引
        if self.cached_index is not None:
            del self.cached_index
            self.cached_index = None

        # 从磁盘读取 CPU 索引
        cpu_index = faiss.read_index(self.index_paths[w_idx])

        # 转换为 GPU 索引
        gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)

        # 缓存 GPU 索引
        self.cached_index = gpu_index
        self.cached_window_idx = w_idx

        # 释放 CPU 索引
        del cpu_index

        return self.cached_index

    def __getitem__(self, item) -> dict:
        """
        返回推理数据

        关键: Imputation Mask 应用到 Target Sample
        """
        output = super().__getitem__(item)

        # 获取窗口索引
        if 'window_idx' in output:
            window_idx = int(output['window_idx'])
        else:
            window_idx = item % self.window_count

        # 使用 Imputation Mask
        current_mask = self.infer_masks[window_idx]
        output['mask'] = current_mask

        # Tokenize (应用 Imputation Mask)
        output['hap_1'] = self.tokenize(output['hap1_nomask'], current_mask)
        output['hap_2'] = self.tokenize(output['hap2_nomask'], current_mask)

        # Float fields
        for key in self.long_fields:
            output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            output[key] = torch.FloatTensor(output[key])

        return output

    def process_batch_retrieval(self, batch, embedding_layer, device, k_retrieve=1):
        """
        在主进程中执行 RAG 检索 (Lazy Encoding 优化版)

        核心流程:
        1. 编码 Query (Masked 版本, 带梯度)
        2. FAISS 检索获取 Top-K 索引 (不可微)
        3. 按需提取: 只从 CPU 内存提取检索到的 Complete Reference Tokens
        4. 按需编码: 只编码检索到的 Complete Reference (带梯度)
        5. 返回带梯度的 RAG Embeddings
        """
        window_idx = int(batch['window_idx'][0].item())

        # 1. 编码 Query (Masked 版本)
        hap_1_tokens = batch['hap_1'].to(device)
        hap_2_tokens = batch['hap_2'].to(device)
        af = batch['af'].to(device)

        # Query Embeddings (Masked)
        query_h1_emb = embedding_layer(hap_1_tokens, af=af, pos=True)  # [B, L, D]
        query_h2_emb = embedding_layer(hap_2_tokens, af=af, pos=True)

        # 2. FAISS 检索
        B, L, D = query_h1_emb.shape
        query_h1_flat = query_h1_emb.reshape(B, L * D).detach().cpu().numpy().astype(np.float32)
        query_h2_flat = query_h2_emb.reshape(B, L * D).detach().cpu().numpy().astype(np.float32)

        # 加载 FAISS 索引
        index = self.load_index(window_idx)

        # 检索 Top-K
        _, I1 = index.search(query_h1_flat, k_retrieve)  # [B, K]
        _, I2 = index.search(query_h2_flat, k_retrieve)

        # 3. 按需提取 Complete Reference Tokens
        ref_tokens_complete = self.ref_tokens_complete[window_idx]  # [num_haps, MAX_SEQ_LEN]
        ref_af = self.ref_af_windows[window_idx]  # [MAX_SEQ_LEN]

        # 提取检索到的 Complete Tokens
        all_indices = np.unique(np.concatenate([I1.flatten(), I2.flatten()]))
        retrieved_tokens = ref_tokens_complete[all_indices]  # [num_retrieved, MAX_SEQ_LEN]

        # 4. 按需编码 Complete Reference (带梯度)
        num_retrieved = retrieved_tokens.shape[0]
        ref_af_expanded = np.tile(ref_af, (num_retrieved, 1))

        retrieved_tokens_tensor = torch.LongTensor(retrieved_tokens).to(device)
        ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)

        # 编码 Complete Reference (带梯度!)
        retrieved_emb = embedding_layer(
            retrieved_tokens_tensor,
            af=ref_af_tensor,
            pos=True
        )  # [num_retrieved, L, D] 带梯度

        # 5. 重映射索引
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_indices)}

        # 6. 构建输出 (保持梯度)
        rag_emb_h1_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=retrieved_emb.dtype)
        rag_emb_h2_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=retrieved_emb.dtype)

        for i in range(B):
            # h1 的 top-k
            for k in range(k_retrieve):
                old_ref_idx = I1[i, k]
                new_ref_idx = index_map[old_ref_idx]
                rag_emb_h1_final[i, k] = retrieved_emb[new_ref_idx]  # [L, D] 带梯度!

            # h2 的 top-k
            for k in range(k_retrieve):
                old_ref_idx = I2[i, k]
                new_ref_idx = index_map[old_ref_idx]
                rag_emb_h2_final[i, k] = retrieved_emb[new_ref_idx]  # [L, D] 带梯度!

        # 7. 赋值到 batch (保持梯度和正确顺序)
        batch['rag_emb_h1'] = rag_emb_h1_final.contiguous()  # [B, k, L, D]
        batch['rag_emb_h2'] = rag_emb_h2_final.contiguous()

        return batch

    @classmethod
    def from_file(cls, vocab, vcfpath, panelpath, freqpath, typepath, poppath, pospath,
                  ref_vcf_path=None, embedding_layer=None, build_ref_data=True, n_gpu=1, name='infer'):
        """从文件加载推理数据集"""
        # 调用父类的 from_file
        dataset = super(EmbeddingRAGInferDataset, cls).from_file(
            vocab, vcfpath, panelpath, freqpath, typepath, poppath, pospath
        )

        # 创建 EmbeddingRAGInferDataset 实例
        return cls(
            vocab=vocab,
            vcf=dataset.vcf,
            pos=dataset.pos,
            panel=dataset.panel,
            freq=dataset.freq,
            type_to_idx=dataset.type_to_idx,
            pop_to_idx=dataset.pop_to_idx,
            pos_to_idx=dataset.pos_to_idx,
            ref_vcf_path=ref_vcf_path,
            embedding_layer=embedding_layer,
            build_ref_data=build_ref_data,
            n_gpu=n_gpu,
            build_index=True,
            name=name
        )
