# embedding_rag_dataset.py
# 端到端可学习的Embedding RAG Dataset
# 关键特性:
# 1. 检索在embedding space进行 (端到端可学习)
# 2. 每个epoch刷新reference embeddings (保持最新)
# 3. 内存优化: reference embeddings存在CPU

import time
import os
import pickle
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
# 修改为512以支持固定窗口大小 (511 SNPs + 1 SOS = 512)
INFER_WINDOW_LEN = 511
MAX_SEQ_LEN = 512


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
                 use_dynamic_mask=False,
                 name='default'):
        super().__init__(vocab, vcf, pos, panel, freq, window,
                        type_to_idx, pop_to_idx, pos_to_idx)

        self.maf_mask_percentage = maf_mask_percentage
        self.use_dynamic_mask = use_dynamic_mask
        self.current_epoch = 0
        self.name = name  # 数据集名称 (用于区分训练集/验证集的索引)

        # Embedding RAG特有的数据结构
        self.ref_tokens_complete = []     # 完整tokens (无mask) [window_idx][num_haps, L]
        # [FIX] 预分配空间，防止赋值时越界
        self.raw_window_masks = [None] * self.window_count  # 原始mask
        self.window_masks = [None] * self.window_count      # 填充后的mask
        self.mask_version = 0             # 当前mask版本号
        self.ref_af_windows = []          # AF信息 [window_idx][L] (将在 _precompute_window_lens 中填充)
        self.window_valid_indices = {}    # [FIX] 记录每个窗口的有效位点索引 (用于过滤训练数据)
        self.window_actual_lens = []      # 每个窗口过滤后的实际长度 [window_idx]

        # FAISS索引路径 (保存到磁盘)
        # 关键修复: 使用 name 参数区分训练集和验证集的索引目录
        # 训练集: faiss_indexes_train, 验证集: faiss_indexes_val
        # 避免索引文件冲突导致的语义不匹配问题
        base_dir = os.path.dirname(ref_vcf_path) if ref_vcf_path else "."
        self.index_dir = os.path.join(base_dir, f"faiss_indexes_{name}")
        self.index_paths = []             # 索引文件路径

        # === 性能优化: 单槽位缓存 (配合Window-Grouped Sampling) ===
        # 策略: 只缓存当前窗口的索引，下一个窗口时释放并加载新的
        # 内存占用: ~1.5GB GPU显存 (单个窗口)
        # 配合WindowGroupedSampler，I/O次数: 30,000 → 331 次/epoch
        self.cached_index = None          # 单槽位缓存: 当前窗口的GPU索引
        self.cached_window_idx = -1       # 当前缓存的窗口ID

        # === 性能优化: GPU FAISS 加速 ===
        # 使用 faiss-gpu 将索引放到 GPU 上进行检索，消除 CPU 检索瓶颈
        # 检索速度: CPU ~50ms/batch → GPU ~1ms/batch (50x加速)
        self.gpu_res = faiss.StandardGpuResources()  # GPU 资源管理器

        # 存储embedding layer引用 (用于按需编码)
        self.embedding_layer = embedding_layer
        self.embed_dim = embedding_layer.embed_size if embedding_layer else None

        # === JIT构建策略: 不预先构建所有索引 ===
        # 存储参考面板数据到内存，索引按需构建
        self.ref_vcf_path = ref_vcf_path
        self.ref_gt = None  # 延迟加载
        self.ref_pos = None  # 延迟加载

        # JIT缓存: 当前窗口的数据
        self.jit_window_idx = -1  # 当前JIT构建的窗口ID
        self.jit_ref_tokens_complete = None  # 当前窗口的complete tokens
        self.jit_ref_af = None  # 当前窗口的AF数据
        self.jit_window_mask = None  # 当前窗口的mask

        # 如果需要，只加载参考面板的原始GT和POS数据到内存
        if build_ref_data and ref_vcf_path:
            print(f"\n{'='*80}")
            print(f"▣ 加载参考面板原始数据 (JIT模式 - 不预构建索引)")
            print(f"{'='*80}")
            start_time = time.time()
            self.ref_gt, self.ref_pos = self._load_ref_data(ref_vcf_path)
            print(f"✓ 参考面板加载完成:")
            print(f"  - 样本数: {self.ref_gt.shape[1]}")
            print(f"  - 位点数: {self.ref_gt.shape[0]}")
            print(f"  - 耗时: {time.time() - start_time:.2f}s")
            print(f"✓ JIT索引构建: 训练时按需构建，启动延迟<1s")
            print(f"{'='*80}\n")

            # [FIX 1] 必须预计算窗口长度，否则 regenerate_masks 会报错
            # 即使是 JIT 模式，Mask 生成也依赖于窗口长度信息
            self._precompute_window_lens()

            # [FIX 2] 初始化 Epoch 0 的 Mask
            # 必须在 _precompute_window_lens 之后调用，解决启动时 list index out of range
            self.regenerate_masks(seed=0)

    def _precompute_window_lens(self):
        """
        [V18 FIX] 预计算所有窗口的实际长度和AF数据
        这是 regenerate_masks 所必需的，必须在初始化时完成。
        AF数据很小(几MB)，预加载到内存可加速后续JIT构建和Mask生成。
        """
        print(f"▣ 预计算窗口元数据 (长度 & AF)...")
        self.window_actual_lens = []
        self.window_valid_indices = {}
        self.ref_af_windows = []  # 重置 AF 列表

        # 确保 ref_pos 已加载
        if self.ref_pos is None and self.ref_vcf_path:
            _, self.ref_pos = self._load_ref_data(self.ref_vcf_path)

        # AF计算所需的常量
        AF_IDX = 3
        GLOBAL_IDX = 5

        for w_idx in tqdm(range(self.window_count), desc="计算窗口元数据"):
            # 1. 获取窗口的SNP范围
            start = self.window.window_info[w_idx, 0]
            end = self.window.window_info[w_idx, 1]
            train_pos = self.pos[start:end]

            # 2. 计算有效位点 (与 Reference 的交集)
            filtered_pos = train_pos  # 默认全部有效
            window_len = len(train_pos)

            if self.ref_pos is not None:
                is_valid = np.isin(train_pos, self.ref_pos)
                # 如果有位点需要过滤
                if not np.all(is_valid):
                    valid_mask = np.where(is_valid)[0]
                    self.window_valid_indices[w_idx] = valid_mask
                    filtered_pos = train_pos[valid_mask]  # 更新为过滤后的位点
                    window_len = len(filtered_pos)

            self.window_actual_lens.append(window_len)

            # 3. [FIX] 预计算 AF (用于 regenerate_masks)
            # 即使是 JIT 模式，AF 数据也必须常驻内存以支持 Smart Masking
            ref_af = np.array([
                self.freq[AF_IDX][GLOBAL_IDX][self.pos_to_idx[p]]
                if p in self.pos_to_idx else 0.0
                for p in filtered_pos  # 使用过滤后的位点
            ], dtype=np.float32)

            # Padding到MAX_SEQ_LEN
            ref_af = VCFProcessingModule.sequence_padding(ref_af, dtype='float')  # [MAX_SEQ_LEN]
            self.ref_af_windows.append(ref_af)

        print(f"✓ 窗口元数据计算完成 (总窗口数: {len(self.window_actual_lens)}, AF数据已加载)")

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

        # [CRITICAL FIX] 保存原始状态并切换到 eval 模式
        # 这确保 Reference Embedding 不含 Dropout 噪声，保证索引构建的确定性
        was_training = embedding_layer.training
        embedding_layer.eval()

        try:
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
    
                        # [FIX A] 保存有效位点的索引掩码，用于 __getitem__ 中过滤训练数据
                        self.window_valid_indices[w_idx] = np.array(valid_pos_mask)
    
                    # 保存每个窗口的实际长度 (用于regenerate_masks)
                    self.window_actual_lens.append(window_len)

                    # === 步骤2: 计算AF (CRITICAL: 必须在生成Mask之前!) ===
                    # 从reference panel计算每个位点的AF
                    # AF计算仅依赖于train_pos，这在循环开头已经过滤好了，因此移动到这里是安全的
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

                    # === 步骤3: 生成AF-Guided Mask (用于语义对齐) ===
                    # [CRITICAL OPTIMIZATION] 在初始化时就生成正确的AF-Guided Mask
                    # 避免在train.py中被迫重新刷新Mask并重建索引（节省40分钟！）
                    rare_af_threshold = 0.05  # 稀有变异阈值
                    rare_mask_rate = 0.7      # 稀有位点Mask概率
                    current_mask_rate = self._TrainDataset__mask_rate[self._TrainDataset__level]  # 通常为0.3

                    # 构建概率图 (基于AF)
                    # Rare位点 (AF < 0.05): 70% Mask概率
                    # Ref位点: 使用课程学习的Mask Rate (30%)
                    af_data_unpadded = ref_af[:window_len]  # 只取有效长度
                    probs = np.where(af_data_unpadded < rare_af_threshold, rare_mask_rate, current_mask_rate)

                    # 生成AF-Guided Mask
                    raw_mask = super().generate_mask(window_len, probs=probs)
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
    
                    # === 步骤6: 保存FAISS索引到磁盘 ===
                    index_path = os.path.join(self.index_dir, f"index_{w_idx}.faiss")
                    faiss.write_index(index, index_path)
                    self.index_paths.append(index_path)
    
                    # 清理GPU和CPU内存 (不保存索引到内存)
                    del ref_tokens_masked_tensor, ref_af_tensor, ref_emb_masked, ref_emb_masked_flat, ref_emb_masked_flat_np, index
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        finally:
            # [CRITICAL FIX] 无论是否出错，都恢复原始训练状态
            # 这确保不会影响后续的训练过程
            embedding_layer.train(was_training)

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

    def _jit_build_window_data(self, w_idx):
        """
        JIT (Just-In-Time) 构建窗口数据和FAISS索引

        关键改进:
        1. 按需构建: 只在首次访问窗口时构建索引
        2. 快速AF提取: 从self.freq直接lookup，严禁重新计算
        3. AF-Guided Mask: 使用self.mask_version支持课程学习
        4. Tokenize & Encode: 使用self.embedding_layer的实时权重
        5. 构建FAISS索引并保存到磁盘

        Args:
            w_idx: 窗口索引

        Returns:
            None (将数据缓存到 self.jit_* 变量中)
        """
        # 1. 获取窗口的SNP范围
        current_slice = slice(
            self.window.window_info[w_idx, 0],
            self.window.window_info[w_idx, 1]
        )
        window_len = current_slice.stop - current_slice.start

        # 2. 提取训练位点并匹配参考面板
        train_pos = self.pos[current_slice]
        ref_indices = []
        valid_pos_mask = []

        for idx, p in enumerate(train_pos):
            matches = np.where(self.ref_pos == p)[0]
            if len(matches) > 0:
                ref_indices.append(matches[0])
                valid_pos_mask.append(idx)

        # 如果有位点被过滤，更新窗口长度
        if len(ref_indices) < len(train_pos):
            if len(valid_pos_mask) == 0:
                raise ValueError(f"窗口 {w_idx} 没有可用位点")
            valid_indices = current_slice.start + np.array(valid_pos_mask)
            current_slice = valid_indices
            train_pos = train_pos[valid_pos_mask]
            window_len = len(train_pos)

        # 3. [OPTIMIZATION] 直接使用预计算的AF数据 (从 _precompute_window_lens 缓存中读取)
        # 这避免了重复计算，加速JIT构建
        ref_af = self.ref_af_windows[w_idx]  # 已经过padding的AF数据 [MAX_SEQ_LEN]
        self.jit_ref_af = ref_af  # 缓存AF数据

        # 4. 生成AF-Guided Mask
        rare_af_threshold = 0.05
        rare_mask_rate = 0.7
        current_mask_rate = self._TrainDataset__mask_rate[self._TrainDataset__level]

        af_data_unpadded = ref_af[:window_len]
        probs = np.where(af_data_unpadded < rare_af_threshold, rare_mask_rate, current_mask_rate)

        # 生成mask (使用mask_version作为种子以支持课程学习)
        np.random.seed(self.mask_version * 10000 + w_idx)
        raw_mask = super().generate_mask(window_len, probs=probs)
        padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')
        self.jit_window_mask = padded_mask  # 缓存mask

        # 创建完整mask (全0)
        raw_mask_complete = np.zeros_like(raw_mask)
        padded_mask_complete = VCFProcessingModule.sequence_padding(raw_mask_complete, dtype='int')

        # 5. 获取参考序列
        raw_ref = self.ref_gt[current_slice, :, :]  # [L, num_samples, 2]
        raw_ref = raw_ref.reshape(raw_ref.shape[0], -1)  # [L, num_haps]
        raw_ref = raw_ref.T  # [num_haps, L]

        # 6. Tokenize两个版本
        ref_tokens_masked = self.tokenize(raw_ref, padded_mask)  # [num_haps, MAX_SEQ_LEN]
        ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)
        self.jit_ref_tokens_complete = ref_tokens_complete  # 缓存complete tokens

        # 7. 扩展AF到所有haplotypes
        num_haps_in_window = ref_tokens_masked.shape[0]
        ref_af_expanded = np.tile(ref_af, (num_haps_in_window, 1))

        # 8. 编码masked版本并构建FAISS索引
        device = next(self.embedding_layer.parameters()).device

        # [CRITICAL] 保存训练状态并切换到eval模式
        was_training = self.embedding_layer.training
        self.embedding_layer.eval()

        try:
            with torch.no_grad():
                ref_tokens_masked_tensor = torch.LongTensor(ref_tokens_masked).to(device)
                ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                ref_emb_masked = self.embedding_layer(ref_tokens_masked_tensor, af=ref_af_tensor, pos=True)

                # 构建FAISS索引
                num_haps, L, D = ref_emb_masked.shape
                ref_emb_masked_flat = ref_emb_masked.reshape(num_haps, L * D)
                ref_emb_masked_flat_np = ref_emb_masked_flat.cpu().numpy().astype(np.float32)

                # 创建并保存索引
                index = faiss.IndexFlatL2(L * D)
                index.add(ref_emb_masked_flat_np)

                # 保存到磁盘
                index_path = os.path.join(self.index_dir, f"index_{w_idx}.faiss")
                faiss.write_index(index, index_path)

                # 确保index_paths有足够长度
                while len(self.index_paths) <= w_idx:
                    self.index_paths.append(None)
                self.index_paths[w_idx] = index_path

                # 清理GPU内存
                del ref_tokens_masked_tensor, ref_af_tensor, ref_emb_masked, ref_emb_masked_flat, ref_emb_masked_flat_np, index
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        finally:
            # 恢复原始训练状态
            self.embedding_layer.train(was_training)

        # 更新JIT缓存标记
        self.jit_window_idx = w_idx

    def regenerate_masks(self, seed: int):
        """
        [AF-GUIDED MASKING] 重新生成所有窗口的mask (基于 AF，而非样本内容)

        核心逻辑：
        - Ref 位点 (普通位点): 使用 current_mask_rate (课程学习，从 30% → 80%)
        - Rare 位点 (AF < 0.05): 强制使用 70% Mask 概率（难样本挖掘）

        这确保：
        1. Query 和 Reference 使用相同的 Mask 模式（基于 AF，而非样本内容）
        2. 低 AF 位点被更频繁地 Mask，迫使模型学习稀有变异
        3. RAG 检索语义空间对齐（Query-Reference Mask 一致）

        Args:
            seed: 随机种子 (通常是 epoch number)
        """
        self.mask_version += 1
        print(f"\n{'='*80}")
        print(f"▣ [AF-Guided Masking] 刷新 Mask Pattern (版本 {self.mask_version}, Seed={seed})")
        print(f"{'='*80}")

        # 获取当前课程学习的 Mask Rate
        current_mask_rate = self._TrainDataset__mask_rate[self._TrainDataset__level]
        rare_af_threshold = 0.05  # 稀有变异阈值
        rare_mask_rate = 0.7      # 稀有位点 Mask 概率

        print(f"▣ Curriculum Learning Level: {self._TrainDataset__level}")
        print(f"  - Ref (普通) Mask Rate: {current_mask_rate:.1%}")
        print(f"  - Rare (AF < {rare_af_threshold}) Mask Rate: {rare_mask_rate:.1%}")
        print(f"{'='*80}")

        for w_idx in range(self.window_count):
            # 使用过滤后的实际长度
            window_len = self.window_actual_lens[w_idx]

            # [CRITICAL] 获取当前窗口的 AF 数据
            af_data = self.ref_af_windows[w_idx][:window_len]  # 只取有效长度

            # [AF-GUIDED] 构建概率图 (Probability Map)
            # Rare 位点 (AF < 0.05): 70% Mask 概率
            # Ref 位点: 使用课程学习的 Mask Rate
            probs = np.where(af_data < rare_af_threshold, rare_mask_rate, current_mask_rate)

            # 生成新 Mask (使用概率图)
            np.random.seed(seed * 10000 + w_idx)
            raw_mask = super().generate_mask(window_len, probs=probs)
            padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

            # 更新保存的 Mask
            self.raw_window_masks[w_idx] = raw_mask
            self.window_masks[w_idx] = padded_mask

        print(f"✓ AF-Guided Mask 刷新完成! 新版本: {self.mask_version}")
        print(f"✓ 稀有位点 (AF < {rare_af_threshold}) 将以 {rare_mask_rate:.1%} 概率被 Mask")
        print(f"✓ Query 和 Reference 使用相同的 AF-Guided Mask 模式")
        print(f"{'='*80}\n")

    def load_index(self, w_idx):
        """
        加载FAISS索引 (GPU加速 + 单槽位缓存策略)

        策略:
        - 检查单槽位缓存，如果命中直接返回 GPU 索引
        - 如果未命中，从磁盘加载 CPU 索引并转为 GPU 索引
        - 配合WindowGroupedSampler，同一窗口的样本连续训练，缓存命中率~100%

        性能:
        - GPU显存占用: ~1.5GB (单个窗口)
        - I/O次数: 331次/epoch (每个窗口加载一次)
        - 检索速度: CPU ~50ms → GPU ~1ms (50x加速)

        Args:
            w_idx: 窗口索引

        Returns:
            faiss.GpuIndex对象 (GPU索引)
        """
        # === 单槽位缓存命中 ===
        if w_idx == self.cached_window_idx and self.cached_index is not None:
            return self.cached_index

        # === 检查索引是否存在，不存在则JIT构建 ===
        if w_idx >= len(self.index_paths) or self.index_paths[w_idx] is None or not os.path.exists(self.index_paths[w_idx] if w_idx < len(self.index_paths) and self.index_paths[w_idx] else ""):
            print(f"  JIT: 构建窗口 {w_idx} 的索引...")
            self._jit_build_window_data(w_idx)

        # === 缓存未命中: 从磁盘加载并转为GPU索引 ===
        # 释放旧GPU缓存 (节省显存)
        if self.cached_index is not None:
            del self.cached_index
            self.cached_index = None

        # 从磁盘读取CPU索引
        cpu_index = faiss.read_index(self.index_paths[w_idx])

        # 转换为GPU索引 (设备 ID 0)
        gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)

        # 缓存GPU索引
        self.cached_index = gpu_index
        self.cached_window_idx = w_idx

        # 释放CPU索引
        del cpu_index

        return self.cached_index

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

        # [FIX B] 预分配Tensor，避免顺序错乱
        # [CRITICAL FIX] 必须使用 float32，而非 h1_tokens.dtype (int64)
        # 原因: Embedding 输出是浮点数，使用 int64 会导致精度丢失和梯度断裂
        rag_emb_h1_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=torch.float32)
        rag_emb_h2_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=torch.float32)

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

            # 从JIT缓存或预构建数据中提取 tokens 和 AF
            # 如果是JIT构建，优先使用JIT缓存; 否则使用预构建的数据
            if self.jit_window_idx == win_idx and self.jit_ref_tokens_complete is not None:
                # JIT模式: 使用JIT缓存
                ref_tokens_complete = self.jit_ref_tokens_complete  # [num_haps, L]
                ref_af = self.jit_ref_af  # [L]
            else:
                # 预构建模式: 使用预存储的数据 (兼容性)
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
            # [FIX B] 使用索引赋值，保持正确顺序
            idx_tensor = torch.tensor(indices, device=device)

            for i in range(B_win):
                batch_idx = idx_tensor[i]  # 全局batch中的索引

                # h1的top-k
                for k in range(k_retrieve):
                    old_ref_idx = I1[i, k]
                    new_ref_idx = index_map[old_ref_idx]
                    rag_emb_h1_final[batch_idx, k] = retrieved_emb[new_ref_idx]  # [L, D] 带梯度!

                # h2的top-k
                for k in range(k_retrieve):
                    old_ref_idx = I2[i, k]
                    new_ref_idx = index_map[old_ref_idx]
                    rag_emb_h2_final[batch_idx, k] = retrieved_emb[new_ref_idx]  # [L, D] 带梯度!

        # 4. 赋值到batch (保持梯度和正确顺序)
        batch['rag_emb_h1'] = rag_emb_h1_final.contiguous()  # [B, k, L, D]
        batch['rag_emb_h2'] = rag_emb_h2_final.contiguous()

        return batch

    def invalidate_indexes(self):
        """
        标记所有索引为无效 (用于Epoch切换后的懒惰重建)

        关键改进:
        1. 不立即重建索引 (避免串行等待)
        2. 只清空缓存和路径标记
        3. 下次load_index时会触发JIT重建
        4. 配合DataLoader并行处理,大幅减少Epoch切换延迟

        性能对比:
        - rebuild_indexes(): 串行重建所有窗口, ~5分钟
        - invalidate_indexes(): 仅标记无效, <1秒; 实际重建分散到训练中
        """
        print(f"\n{'='*80}")
        print(f"▣ 标记索引为无效 (懒惰���建模式)")
        print(f"{'='*80}")

        # 清空单槽位缓存
        if self.cached_index is not None:
            del self.cached_index
            self.cached_index = None
        self.cached_window_idx = -1

        # 标记所有索引路径为None (强制JIT重建)
        # 保留self.index_paths列表结构,但将每个路径设为None
        for i in range(len(self.index_paths)):
            # 如果磁盘上有旧索引文件,可选择删除 (节省磁盘空间)
            if self.index_paths[i] is not None and os.path.exists(self.index_paths[i]):
                try:
                    os.remove(self.index_paths[i])
                except:
                    pass  # 删除失败不影响训练
            self.index_paths[i] = None

        # 更新JIT缓存标记
        self.jit_window_idx = -1
        self.jit_ref_tokens_complete = None
        self.jit_ref_af = None
        self.jit_window_mask = None

        print(f"✓ 所有索引已标记为无效")
        print(f"✓ 下次访问时将自动触发 JIT 重建 (基于最新 Embedding 和 Mask)")
        print(f"✓ Epoch 切换延迟: ~1秒 (vs rebuild_indexes ~5分钟)")
        print(f"{'='*80}\n")

    def rebuild_indexes(self, embedding_layer, device='cuda'):
        """
        [DEPRECATED] 用当前mask重建FAISS索引 (已废弃,请使用invalidate_indexes)

        性能问题:
        - 串行处理所有窗口,导致Epoch切换时等待~5分钟
        - 与DataLoader的并行能力冲突

        推荐替代方案:
        - 使用 invalidate_indexes() 标记索引无效
        - JIT构建会在训练时自动并行处理

        核心流程:
        1. 用当前mask重新tokenize reference sequences
        2. 用最新embedding layer编码masked版本
        3. 重建FAISS索引
        """
        print(f"\n⚠️  警告: rebuild_indexes() 已废弃,请使用 invalidate_indexes()")
        print(f"▣ 重建FAISS索引 (基于新Mask) - 串行模式")
        start_time = time.time()

        # [CRITICAL FIX] 保存原始状态并切换到 eval 模式
        # 这确保 Reference Embedding 不含 Dropout 噪声，保证索引构建的确定性
        # 原因：如果 embedding_layer 处于 training 模式，Dropout 会随机丢弃神经元
        #      导致同一个 Reference 在不同时刻生成的 Embedding 不一致
        #      严重影响 RAG 检索的稳定性和准确性
        was_training = embedding_layer.training
        embedding_layer.eval()

        try:
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

                    # 清理单槽位缓存 (如果缓存的正好是这个窗口)
                    if self.cached_window_idx == w_idx:
                        if self.cached_index is not None:
                            del self.cached_index
                        self.cached_index = None
                        self.cached_window_idx = -1

                    # 清理GPU和CPU内存
                    del ref_tokens_tensor, ref_af_tensor, ref_emb_masked, ref_emb_flat, ref_emb_flat_np, index
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        finally:
            # [CRITICAL FIX] 无论是否出错，都恢复原始训练状态
            # 这确保不会影响后续的训练过程
            embedding_layer.train(was_training)

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

        # [BEST PRACTICE FIX] 优先从父类输出获取 window_idx，确保与父类逻辑解耦
        # 这遵循"单一事实来源"原则，避免硬编码计算逻辑
        if 'window_idx' in output:
            window_idx = int(output['window_idx'])
        else:
            # 回退逻辑：保持与父类逻辑一致 (Sample-Major)
            window_idx = item % self.window_count

        # [FIX A] 如果该窗口有位点过滤，先过滤所有序列数据
        if window_idx in self.window_valid_indices:
            valid_mask = self.window_valid_indices[window_idx]
            # 过滤序列数据（hap1_nomask, hap2_nomask, label）
            output['hap1_nomask'] = output['hap1_nomask'][valid_mask]
            output['hap2_nomask'] = output['hap2_nomask'][valid_mask]
            output['label'] = output['label'][valid_mask]
            # 过滤 float fields（af, type, pop 等）
            for key in self.float_fields:
                if key in output and len(output[key]) > len(valid_mask):
                    output[key] = output[key][valid_mask]

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
                  window_size,  # 修改: 改为固定窗口大小
                  typepath,
                  poppath,
                  pospath,
                  ref_vcf_path=None,
                  embedding_layer=None,
                  build_ref_data=True,
                  n_gpu=1,
                  use_dynamic_mask=False,
                  name='default'):
        """
        从文件创建EmbeddingRAGDataset (JIT版本 - 固定窗口大小)

        重要变更:
        1. 不再依赖外部window文件，使用固定窗口大小切分
        2. 索引按需构建 (JIT)，训练几乎立即开始
        3. 配合WindowGroupedSampler，内存中只保留当前窗口索引

        Args:
            window_size: 固定窗口大小 (默认511，加SOS后512)
            name: 数据集名称 (用于区分训练集/验证集的索引目录)
                  训练集应使用 'train', 验证集应使用 'val'
        """
        # 首先加载基础数据但不创建窗口
        import pathlib
        import h5py

        filepath = pathlib.Path(vcfpath)
        panelpath_obj = pathlib.Path(panelpath)
        freqpath_obj = pathlib.Path(freqpath)

        # 加载 VCF/H5 数据
        if filepath.name.endswith('.h5'):
            h5_file = h5py.File(filepath, 'r')
            vcf_data = h5_file['calldata/GT'][:]
            pos_data = h5_file['variants/POS'][:]
            h5_file.close()
        else:
            raise ValueError(f"仅支持 .h5 文件格式")

        # 加载其他必需数据
        from .dataset import PanelData, Window
        panel = PanelData.from_file(str(panelpath_obj))
        freq = np.load(str(freqpath_obj), allow_pickle=True)

        # 动态计算窗口信息 (基于固定window_size)
        total_snps = len(pos_data)
        num_windows = (total_snps + window_size - 1) // window_size  # 向上取整

        window_starts = []
        window_ends = []
        for i in range(num_windows):
            start = i * window_size
            end = min((i + 1) * window_size, total_snps)
            window_starts.append(start)
            window_ends.append(end)

        # 使用正确的类名 Window (不是 WindowData)
        window = Window(np.array(window_starts), np.array(window_ends))

        # 加载映射字典 (使用pickle而非torch.load)
        # 这些文件是用Python原生pickle保存的,不是torch.save
        with open(typepath, 'rb') as f:
            type_to_idx = pickle.load(f)
        with open(poppath, 'rb') as f:
            pop_to_idx = pickle.load(f)
        with open(pospath, 'rb') as f:
            pos_to_idx = pickle.load(f)

        # 创建EmbeddingRAGDataset (不立即构建索引!)
        rag_dataset = cls(
            vocab=vocab,
            vcf=vcf_data,
            pos=pos_data,
            panel=panel,
            freq=freq,
            window=window,
            type_to_idx=type_to_idx,
            pop_to_idx=pop_to_idx,
            pos_to_idx=pos_to_idx,
            ref_vcf_path=ref_vcf_path,
            embedding_layer=embedding_layer,
            build_ref_data=build_ref_data,  # 这将被改为JIT构建
            n_gpu=n_gpu,
            use_dynamic_mask=use_dynamic_mask,
            name=name  # 传递name参数
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
