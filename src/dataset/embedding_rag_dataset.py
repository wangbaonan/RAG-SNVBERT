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
    Embedding RAG Dataset with JIT (Just-in-Time) GPU Indexing

    核心改进 (V18 GPU-JIT):
    - 移除磁盘 FAISS 索引，改为 GPU 显存即时索引
    - Query 和 Reference 参数完全同步的 End-to-End 训练
    - 使用 L2 距离进行 SNP Imputation 检索
    - Reference 数据保存在 CPU，按需上传到 GPU 构建索引
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
        self.name = name

        # Embedding RAG 数据结构 (CPU)
        self.ref_tokens_complete = []     # 完整tokens (无mask) [window_idx][num_haps, L]
        self.raw_window_masks = []        # 原始mask
        self.window_masks = []            # 填充后的mask
        self.mask_version = 0             # 当前mask版本号
        self.ref_af_windows = []          # AF信息 [window_idx][L]
        self.window_valid_indices = {}    # 记录每个窗口的有效位点索引
        self.window_actual_lens = []      # 每个窗口过滤后的实际长度

        # === JIT GPU 索引缓存 (替代磁盘 FAISS) ===
        # 策略: 当前窗口的 Reference Embeddings 保存在 GPU 显存
        # 切换窗口时自动释放旧缓存并构建新索引
        self.jit_cache_win_idx = -1       # 当前缓存的窗口ID (-1 表示无缓存)
        self.jit_ref_emb_search = None    # [Num_Ref, L, D] Masked Reference Embeddings (GPU)
        self.jit_ref_tokens_raw = None    # [Num_Ref, L] Complete Reference Tokens (GPU)
        self.jit_ref_af_raw = None        # [Num_Ref, L] Reference AF (GPU)

        # 存储embedding layer引用 (用于按需编码)
        self.embedding_layer = embedding_layer
        self.embed_dim = embedding_layer.embed_size if embedding_layer else None

        # 加载 Reference 数据到 CPU 内存
        if build_ref_data and ref_vcf_path and embedding_layer:
            self._load_ref_data_to_memory(ref_vcf_path, embedding_layer)

    def _load_ref_data_to_memory(self, ref_vcf_path: str, embedding_layer):
        """
        加载 Reference 数据到 CPU 内存 (JIT GPU-Indexing 模式)

        核心改进:
        1. 只加载 Complete Tokens 和 AF 到 CPU 内存
        2. 移除所有 FAISS 索引构建和磁盘 I/O
        3. GPU 索引将在训练时 Just-in-Time 构建

        内存消耗: ~11GB (仅 CPU 内存)
        """
        print("=" * 80)
        print("▣ 加载 Reference 数据到内存 (JIT GPU-Indexing 模式)")
        print("=" * 80)
        start_time = time.time()

        # 加载参考数据
        load_start = time.time()
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        print(f"✓ 加载参考数据: 样本数={ref_gt.shape[1]} | 位点数={ref_gt.shape[0]} | "
              f"耗时={time.time()-load_start:.2f}s")

        print(f"✓ Embedding维度: {self.embed_dim}")
        print()

        # 对每个window加载tokens和AF数据
        for w_idx in tqdm(range(self.window_count), desc="加载窗口数据"):
            current_slice = slice(
                self.window.window_info[w_idx, 0],
                self.window.window_info[w_idx, 1]
            )
            window_len = current_slice.stop - current_slice.start

            # === 步骤1: 向量化位点匹配 (NumPy Vectorized) ===
            train_pos = self.pos[current_slice]

            # [VECTORIZED] 使用 searchsorted 批量查找 (假设 ref_pos 已排序)
            # 时间复杂度: O(n log m) vs 原 O(n * m)
            found_indices = np.searchsorted(ref_pos, train_pos)

            # [SAFETY] 边界处理: Clamp 索引到有效范围
            found_indices = np.clip(found_indices, 0, len(ref_pos) - 1)

            # [CRITICAL] 严格验证匹配: 只有位置值完全相等才计入
            # 防止 searchsorted 的近似匹配导致数据错位
            is_match = (ref_pos[found_indices] == train_pos)

            # 生成有效位点的索引
            valid_pos_mask = np.where(is_match)[0]  # Training Panel 中的索引
            ref_indices = found_indices[is_match]   # Reference VCF 中的索引

            # 如果有位点被过滤
            if len(ref_indices) < len(train_pos):
                if len(valid_pos_mask) == 0:
                    print(f"  ⚠ 跳过窗口 {w_idx}: 没有可用位点")
                    continue
                # 更新数据
                train_pos = train_pos[valid_pos_mask]
                window_len = len(train_pos)
                self.window_valid_indices[w_idx] = valid_pos_mask

            # 保存每个窗口的实际长度
            self.window_actual_lens.append(window_len)

            # === 步骤2: 计算AF ===
            AF_IDX = 3
            GLOBAL_IDX = 5
            ref_af = np.array([
                self.freq[AF_IDX][GLOBAL_IDX][self.pos_to_idx[p]]
                if p in self.pos_to_idx else 0.0
                for p in train_pos
            ], dtype=np.float32)

            # Padding到MAX_SEQ_LEN
            ref_af = VCFProcessingModule.sequence_padding(ref_af, dtype='float')
            self.ref_af_windows.append(ref_af)

            # === 步骤3: 生成AF-Guided Mask ===
            rare_af_threshold = 0.05
            rare_mask_rate = 0.7
            current_mask_rate = self._TrainDataset__mask_rate[self._TrainDataset__level]

            af_data_unpadded = ref_af[:window_len]
            probs = np.where(af_data_unpadded < rare_af_threshold, rare_mask_rate, current_mask_rate)

            raw_mask = super().generate_mask(window_len, probs=probs)
            padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')

            self.raw_window_masks.append(raw_mask)
            self.window_masks.append(padded_mask)

            # === 步骤4: Tokenize Complete版本 (保存到CPU) ===
            raw_mask_complete = np.zeros_like(raw_mask)
            padded_mask_complete = VCFProcessingModule.sequence_padding(
                raw_mask_complete, dtype='int'
            )

            # [CRITICAL FIX] 使用精确的 ref_indices 提取数据，而非 current_slice
            # 原代码: raw_ref = ref_gt[current_slice, :, :]  ← RISKY! 假设行号一一对应
            # 新代码: raw_ref = ref_gt[ref_indices, :, :]  ← SAFE! 使用精确匹配的索引
            raw_ref = ref_gt[ref_indices, :, :]  # [L, num_samples, 2]

            # [SAFETY ASSERT] 开发模式下验证数据对齐
            assert raw_ref.shape[0] == len(ref_indices), \
                f"Data alignment error: raw_ref length {raw_ref.shape[0]} != ref_indices length {len(ref_indices)}"

            raw_ref = raw_ref.reshape(raw_ref.shape[0], -1)  # [L, num_haps]
            raw_ref = raw_ref.T  # [num_haps, L]

            # Complete版本: 用于JIT时Re-encoding
            ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)
            self.ref_tokens_complete.append(ref_tokens_complete)

        # 计算内存占用
        total_haps = sum(tokens.shape[0] for tokens in self.ref_tokens_complete)
        tokens_mb = (total_haps * MAX_SEQ_LEN * 8) / (1024 ** 2)  # int64, 8 bytes
        af_mb = (self.window_count * MAX_SEQ_LEN * 4) / (1024 ** 2)  # float32, 4 bytes
        memory_mb = tokens_mb + af_mb

        print()
        print("=" * 80)
        print(f"✓ Reference 数据加载完成! (JIT GPU-Indexing 模式)")
        print(f"  - 窗口数: {self.window_count}")
        print(f"  - 总单体型数: {total_haps}")
        print(f"  - Embedding维度: {self.embed_dim}")
        print(f"  - Mask版本号: {self.mask_version}")
        print(f"  - CPU内存占用: {memory_mb:.1f} MB (tokens + AF) ✅")
        print(f"  - GPU索引: Just-in-Time 构建 (训练时按需)")
        print(f"  - 总耗时: {time.time() - start_time:.2f}s")
        print("=" * 80)

    def clear_jit_cache(self):
        """
        清空 JIT GPU 缓存并释放显存

        使用场景:
        - Epoch 结束后
        - 验证结束后
        - 需要彻底释放 GPU 显存时
        """
        self.jit_cache_win_idx = -1
        self.jit_ref_emb_search = None
        self.jit_ref_tokens_raw = None
        self.jit_ref_af_raw = None

        # 强制释放 GPU 显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def process_batch_retrieval(self, batch, embedding_layer, device, k_retrieve=1):
        """
        JIT GPU索引检索 (End-to-End Learnable, 零磁盘I/O)

        核心改进 (V18 GPU-JIT):
        1. **JIT GPU Indexing**: 窗口切换时在GPU即时构建索引
        2. **L2距离检索**: 使用 torch.cdist + topk(largest=False) for SNP Imputation
        3. **Masked Search, Complete Re-encode**: 检索用Masked, 最终用Complete重编码
        4. **参数同步**: Query和Reference使用完全相同的模型参数

        核心流程:
        1. JIT构建索引 (如果窗口变化): Masked Reference → GPU Embeddings
        2. 编码Query (Masked版本, 带梯度)
        3. L2距离检索获取Top-K索引 (使用torch.cdist)
        4. 提取Complete Reference Tokens并Re-encode (带梯度!)
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

        rag_emb_h1_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=torch.float32)
        rag_emb_h2_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=torch.float32)

        # 3. 处理每个窗口（JIT GPU Indexing）
        for win_idx, indices in window_groups.items():
            # === JIT构建GPU索引 (如果窗口变化) ===
            if self.jit_cache_win_idx != win_idx:
                # 清空旧缓存
                self.jit_cache_win_idx = -1
                self.jit_ref_emb_search = None
                self.jit_ref_tokens_raw = None
                self.jit_ref_af_raw = None

                # 从CPU加载当前窗口的Complete Reference Tokens和AF
                ref_tokens_complete = self.ref_tokens_complete[win_idx]  # [num_haps, L] numpy
                ref_af = self.ref_af_windows[win_idx]  # [L] numpy
                current_mask = self.window_masks[win_idx]  # [L] numpy

                # 上传到GPU
                ref_tokens_gpu = torch.LongTensor(ref_tokens_complete).to(device)
                ref_af_gpu = torch.FloatTensor(ref_af).to(device)
                mask_gpu = torch.LongTensor(current_mask).to(device)

                # 应用Mask生成Masked Reference Tokens (在GPU上)
                # Masked tokens用于检索（与Query语义对齐）
                ref_tokens_masked = self._apply_mask_to_tokens_gpu(ref_tokens_gpu, mask_gpu)

                num_haps = ref_tokens_masked.size(0)
                ref_af_expanded = ref_af_gpu.unsqueeze(0).expand(num_haps, -1)  # [num_haps, L]

                # **关键**: 使用eval()模式编码，防止Dropout导致索引不稳定
                was_training = embedding_layer.training
                embedding_layer.eval()

                with torch.no_grad():
                    # 编码Masked Reference用于检索
                    ref_emb_search = embedding_layer(
                        ref_tokens_masked,
                        af=ref_af_expanded,
                        pos=True
                    )  # [num_haps, L, D]

                # 恢复训练状态
                embedding_layer.train(was_training)

                # 缓存到GPU显存
                self.jit_ref_emb_search = ref_emb_search  # [num_haps, L, D]
                self.jit_ref_tokens_raw = ref_tokens_gpu  # Complete tokens [num_haps, L]
                self.jit_ref_af_raw = ref_af_gpu  # [L]
                self.jit_cache_win_idx = win_idx

            # 获取当前窗口的样本
            h1_win = h1_tokens[indices]  # [B_win, L]
            h2_win = h2_tokens[indices]
            af_win = af_batch[indices]

            # === 步骤1: 编码Query (带梯度!) ===
            h1_emb = embedding_layer(h1_win, af=af_win, pos=True)  # [B_win, L, D]
            h2_emb = embedding_layer(h2_win, af=af_win, pos=True)

            B_win = h1_emb.size(0)

            # === 步骤2: L2距离检索 (CRITICAL: 使用largest=False!) ===
            # Flatten embeddings for distance calculation
            h1_emb_flat = h1_emb.reshape(B_win, L * D)  # [B_win, L*D]
            h2_emb_flat = h2_emb.reshape(B_win, L * D)
            ref_emb_flat = self.jit_ref_emb_search.reshape(-1, L * D)  # [num_haps, L*D]

            # 计算L2距离
            dists_h1 = torch.cdist(h1_emb_flat, ref_emb_flat, p=2)  # [B_win, num_haps]
            dists_h2 = torch.cdist(h2_emb_flat, ref_emb_flat, p=2)

            # **CRITICAL**: 使用largest=False获取最小距离的索引 (L2距离越小越好)
            _, I1 = dists_h1.topk(k_retrieve, largest=False, dim=1)  # [B_win, k]
            _, I2 = dists_h2.topk(k_retrieve, largest=False, dim=1)

            # === 步骤3: 提取并Re-encode Complete Reference (带梯度!) ===
            # 合并所有需要的索引
            all_indices = torch.cat([I1.flatten(), I2.flatten()]).unique()  # [num_unique]

            # 从GPU缓存中提取Complete Reference Tokens
            retrieved_tokens = self.jit_ref_tokens_raw[all_indices]  # [num_unique, L]
            retrieved_af = self.jit_ref_af_raw.unsqueeze(0).expand(retrieved_tokens.size(0), -1)

            # **Re-encode** Complete Reference (带梯度!)
            retrieved_emb = embedding_layer(
                retrieved_tokens,
                af=retrieved_af,
                pos=True
            )  # [num_unique, L, D] 带梯度!

            # 创建索引映射
            index_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(all_indices)}

            # === 步骤4: 收集Retrieved Embeddings ===
            idx_tensor = torch.tensor(indices, device=device)

            for i in range(B_win):
                batch_idx = idx_tensor[i]

                # h1的top-k
                for k in range(k_retrieve):
                    old_ref_idx = int(I1[i, k])
                    new_ref_idx = index_map[old_ref_idx]
                    rag_emb_h1_final[batch_idx, k] = retrieved_emb[new_ref_idx]

                # h2的top-k
                for k in range(k_retrieve):
                    old_ref_idx = int(I2[i, k])
                    new_ref_idx = index_map[old_ref_idx]
                    rag_emb_h2_final[batch_idx, k] = retrieved_emb[new_ref_idx]

        # 4. 赋值到batch (保持梯度和正确顺序)
        batch['rag_emb_h1'] = rag_emb_h1_final.contiguous()  # [B, k, L, D]
        batch['rag_emb_h2'] = rag_emb_h2_final.contiguous()

        return batch

    def _apply_mask_to_tokens_gpu(self, tokens, mask):
        """
        在GPU上应用mask到tokens

        Args:
            tokens: [num_haps, L] LongTensor (GPU)
            mask: [L] LongTensor (GPU), 1表示需要mask的位置

        Returns:
            masked_tokens: [num_haps, L] LongTensor (GPU)
        """
        masked_tokens = tokens.clone()
        mask_token_id = self.vocab.mask_index
        # 将mask位置替换为[MASK] token
        masked_tokens[:, mask == 1] = mask_token_id
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
                  windowpath,
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
        从文件创建EmbeddingRAGDataset

        重要: 必须提供embedding_layer用于预编码!

        Args:
            name: 数据集名称 (用于区分训练集/验证集的索引目录)
                  训练集应使用 'train', 验证集应使用 'val'
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
