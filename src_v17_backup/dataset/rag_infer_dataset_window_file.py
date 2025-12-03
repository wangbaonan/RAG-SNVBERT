import time
import os
import h5py
from collections import defaultdict
import faiss
import allel
import numpy as np
import torch
import random
from tqdm import tqdm
from typing import Dict, List, Optional
# 引入你原先的 TrainDataset
from .dataset import InferDataset
from .utils import timer, PanelProcessingModule, VCFProcessingModule

# DEFAULT MAXIMUM SEQUENCE LENGTH
INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030

class RAGInferDataset(InferDataset):
    def __init__(self, vocab, vcf, pos, panel, freq, window, type_to_idx, pop_to_idx, pos_to_idx,
                 ref_vcf_path=None, build_ref_data=True, n_gpu=1, build_index: bool = False):
        super().__init__(vocab, vcf, pos, panel, freq, window, type_to_idx, pop_to_idx, pos_to_idx)
        
        # RAG相关属性
        self.ref_data_windows = []    # [num_windows, num_positions, sample * hap]
        self.raw_ref_data_windows = []
        self.window_indexes = []
        self.infer_masks = []
        self.raw_window_masks = []
        self.ref_vcf_path = ref_vcf_path
        if not build_index:
            print("First loading no index building.")
        if build_ref_data and ref_vcf_path and build_index:
            print("Building.")
            self._build_faiss_indexes(ref_vcf_path)

    def _build_faiss_indexes(self, ref_vcf_path: str):
        """优化版（保持完全兼容）"""
        print("▣ 开始构建推理FAISS索引2.0")
        start_time = time.time()
    
        # 加载参考数据
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        print(f"▨ 加载参考数据完成 | 样本数={ref_gt.shape[1]} 位点数={ref_gt.shape[0]}")

        # === 关键优化1：预构建位置字典 ===
        # 保持原始行为：当存在重复位置时，取第一个出现的索引
        self.pos_to_index = {}
        for idx, pos in enumerate(ref_pos):
            if pos not in self.pos_to_index:  # 确保第一个出现的索引
                self.pos_to_index[pos] = idx

        # === 关键优化2：预分配内存 ===
        total_windows = self.window_count
        self.raw_window_masks = [None] * total_windows
        self.infer_masks = [None] * total_windows
        self.raw_ref_data_windows = [None] * total_windows
        self.ref_data_windows = [None] * total_windows
        self.window_indexes = [None] * total_windows

        # === 关键优化3：向量化处理 ===
        position_needed = np.zeros_like(self.position_needed)  # 确保长度一致
        if len(self.position_needed) < len(self.ori_pos):
            pad_len = len(self.ori_pos) - len(self.position_needed)

            #修改
            #position_needed = np.pad(self.position_needed, (0, pad_len), mode='constant')

        # 主处理循环
        for w_idx in tqdm(range(total_windows), desc="处理推理窗口"):
            start_idx = self.window.window_info[w_idx, 0]
            end_idx = self.window.window_info[w_idx, 1]
        
            # === 优化4：向量化掩码生成 ===
            window_slice = slice(start_idx, end_idx)
            #修改
            #mask = position_needed[window_slice].astype(int)
            mask = np.array([
                1 if self.position_needed[i] else 0 
                for i in range(start_idx, end_idx)
            ], dtype=int)
            if len(mask) < INFER_WINDOW_LEN:
                pad_len = INFER_WINDOW_LEN - len(mask)
                mask = np.pad(mask, (0, pad_len), mode='constant')
            self.raw_window_masks[w_idx] = mask.copy()  # 保持原始数据存储
        
            padded_mask = VCFProcessingModule.sequence_padding(mask, dtype='int')
            self.infer_masks[w_idx] = padded_mask

            # === 优化5：高效索引查询 ===
            current_pos = self.ori_pos[window_slice]
            ref_indices = []
            for p in current_pos:
                idx = self.pos_to_index.get(p, -1)
                if idx != -1:  # 保持原始跳过不存在位置的行为
                    ref_indices.append(idx)
        
            # === 优化6：内存视图减少拷贝 ===
            raw_ref = ref_gt[ref_indices, :, :]
            self.raw_ref_data_windows[w_idx] = raw_ref  # 保持原始形状存储

            # 保持原始数据处理流程
            reshaped_ref = raw_ref.reshape(raw_ref.shape[0], -1).T
            ref_tokenized = self.tokenize(reshaped_ref, padded_mask)
            self.ref_data_windows[w_idx] = ref_tokenized

            # === 优化7：FAISS配置优化 ===
            index_data = ref_tokenized.astype(np.float32, copy=False)  # 避免重复拷贝
            index = faiss.IndexFlatL2(index_data.shape[1])
            index.add(index_data)
            self.window_indexes[w_idx] = index

        print(f"\n✔ 推理索引构建完成 | 总窗口数={total_windows} 总耗时{time.time()-start_time:.2f}s")

    def _generate_infer_mask(self, start_idx: int, end_idx: int) -> np.ndarray:
        """根据needed_pos生成推理掩码"""
        mask = np.zeros((end_idx - start_idx,), dtype=int)
        for i in range(start_idx, end_idx):
            if i >= len(self.position_needed):
                break
            mask[i - start_idx] = int(self.position_needed[i])
        return mask

    def __getitem__(self, item) -> dict:
        output = super().__getitem__(item)
        window_idx = item % self.window_count
        
        # 添加RAG所需字段
        output['window_idx'] = window_idx
        output['mask'] = self.infer_masks[window_idx]
        
        # 重新处理单体型数据（确保使用正确的mask）
        output['hap_1'] = self.tokenize(output['hap1_nomask'], output['mask'])
        output['hap_2'] = self.tokenize(output['hap2_nomask'], output['mask'])
        for key in self.long_fields:
            output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            output[key] = torch.FloatTensor(output[key])
        return output

    """
    def __getitem__(self, item) -> dict:
        output = super().__getitem__(item)
        window_idx = item % self.window_count
        
        # ================== 类型安全调试函数 ==================
        def _safe_convert(tensor_data):
            ### 将可能的Tensor转换为numpy并进行设备处理
            if torch.is_tensor(tensor_data):
                return tensor_data.cpu().numpy()
            return tensor_data

        def _safe_unique(data, name):
            ### 安全获取唯一值
            arr = _safe_convert(data)
            unique_values = np.unique(arr)
            print(f"{name} unique: {unique_values}")
            return unique_values

        def _safe_sum(data, condition):
            ### 安全计算满足条件的元素数量
            arr = _safe_convert(data)
            return np.sum(arr == condition)

        # ================== 调试信息输出 ==================
        print("\n=== 父类数据检查 ===")
        # 检查hap数据（兼容Tensor和numpy）
        h1_arr = _safe_convert(output['hap_1'])
        h2_arr = _safe_convert(output['hap_2'])
        _safe_unique(h1_arr, "[父类] hap_1")
        _safe_unique(h2_arr, "[父类] hap_2")
        
        # 检查mask分布（显式类型转换）
        mask_arr = _safe_convert(output['mask'])
        print(f"[父类] mask值分布: 0={_safe_sum(mask_arr, 0)}, 1={_safe_sum(mask_arr, 1)}")
        
        print("\n=== 原始数据特征 ===")
        # 检查未mask数据（确保访问正确字段）
        hap1_nomask = _safe_convert(output.get('hap1_nomask', h1_arr))  # 防止字段不存在
        hap2_nomask = _safe_convert(output.get('hap2_nomask', h2_arr))
        _safe_unique(hap1_nomask, "hap1_nomask")
        _safe_unique(hap2_nomask, "hap2_nomask")
        
        # ================== 核心处理逻辑 ==================
        # 添加RAG所需字段（保持原始逻辑）
        output['window_idx'] = window_idx
        output['mask'] = self.infer_masks[window_idx]  # 假设这是numpy数组
        
        print("\n=== 子类mask检查 ===")
        current_mask = _safe_convert(output['mask'])
        print(f"当前window_idx: {window_idx}")
        print(f"当前mask形状: {current_mask.shape}")
        print(f"当前mask值分布: 0={_safe_sum(current_mask, 0)}, 1={_safe_sum(current_mask, 1)}")
        
        # ================== 关键调试点：tokenize处理 ==================
        print("\n=== 子类处理流程 ===")
        h1_before = hap1_nomask.copy()
        h2_before = hap2_nomask.copy()
        
        # 确保tokenize输入类型正确
        output['hap_1'] = self.tokenize(
            hap1_nomask.astype(np.int64), 
            current_mask.astype(np.int64)
        )
        output['hap_2'] = self.tokenize(
            hap2_nomask.astype(np.int64),
            current_mask.astype(np.int64)
        )
        
        # 打印处理对比（限制长度为10）
        print(f"[处理前] hap1_nomask[:10]: {h1_before[:10].tolist()}")
        print(f"[处理后] hap_1[:10]: {_safe_convert(output['hap_1'])[:10].tolist()}")
        print(f"[处理前] hap2_nomask[:10]: {h2_before[:10].tolist()}")
        print(f"[处理后] hap_2[:10]: {_safe_convert(output['hap_2'])[:10].tolist()}")
        
        # ================== 最终检查 ==================
        print("\n=== 子类最终数据 ===")
        _safe_unique(output['hap_1'], "最终hap_1")
        _safe_unique(output['hap_2'], "最终hap_2")
        
        # ================== 类型转换 ==================
        # 保持原有类型转换逻辑
        for key in self.long_fields:
            if isinstance(output[key], np.ndarray):
                output[key] = torch.LongTensor(output[key])
            elif torch.is_tensor(output[key]) and output[key].dtype != torch.long:
                output[key] = output[key].long()
                
        for key in self.float_fields:
            if isinstance(output[key], np.ndarray):
                output[key] = torch.FloatTensor(output[key])
            elif torch.is_tensor(output[key]) and output[key].dtype != torch.float:
                output[key] = output[key].float()
        
        return output

    def __getitem__(self, item) -> dict:
        output = super().__getitem__(item)
        window_idx = item % self.window_count
        
        # ================== 增强调试函数 ==================
        def _get_token_distribution(data, name):
            ### 获取详细token分布
            arr = _safe_convert(data)
            total = len(arr)
            print(f"\n▨ {name}统计 (长度={total})")
            
            # 统计所有可能token（0-6）
            counts = np.bincount(arr.astype(int), minlength=7)
            for token in range(7):
                count = counts[token]
                ratio = count / total if total > 0 else 0
                print(f"  ▸ Token {token}: {count} ({ratio:.2%})", end='')
                if token == 0:
                    print(" ← padding token" if count > 0 else "")
                else:
                    print()

        def _compare_length(before, after, name):
            ### 比较处理前后长度差异
            len_before = len(_safe_convert(before))
            len_after = len(_safe_convert(after))
            print(f"  ▸ {name}长度变化: {len_before} → {len_after}")
            if len_before != len_after:
                print(f"  !!! 警告：长度不一致 (差={abs(len_after - len_before)})")

        def _safe_convert(tensor_data):
            if torch.is_tensor(tensor_data):
                return tensor_data.cpu().numpy()
            return tensor_data

        # ================== 父类数据深度分析 ==================
        print("\n=== 父类数据检查 ===")
        h1_parent = _safe_convert(output['hap_1'])
        h2_parent = _safe_convert(output['hap_2'])
        
        # 原始长度和分布
        _get_token_distribution(h1_parent, "[父类] hap_1 token分布")
        _get_token_distribution(h2_parent, "[父类] hap_2 token分布")
        
        # 检查mask有效性
        mask_parent = _safe_convert(output['mask'])
        print(f"\n[父类] mask有效性检查 (长度={len(mask_parent)})")
        print(f"  ▸ 有mask位置数: {np.sum(mask_parent == 1)}")
        print(f"  ▸ 对应h1实际mask数: {np.sum(h1_parent[mask_parent == 1] == 4)}")
        print(f"  ▸ 对应h2实际mask数: {np.sum(h2_parent[mask_parent == 1] == 4)}")

        # ================== 子类处理流程 ================== 
        output['window_idx'] = window_idx
        current_mask = _safe_convert(self.infer_masks[window_idx])
        output['mask'] = current_mask
        
        print("\n=== 子类处理检查 ===")
        # 原始未mask数据统计
        h1_raw = _safe_convert(output.get('hap1_nomask', h1_parent))
        h2_raw = _safe_convert(output.get('hap2_nomask', h2_parent))
        _get_token_distribution(h1_raw, "[子类输入] hap1_nomask")
        _get_token_distribution(h2_raw, "[子类输入] hap2_nomask")
        
        # 应用mask处理
        output['hap_1'] = self.tokenize(h1_raw.astype(np.int64), current_mask.astype(np.int64))
        output['hap_2'] = self.tokenize(h2_raw.astype(np.int64), current_mask.astype(np.int64))
        
        # ================== 处理结果验证 ==================
        print("\n=== 处理结果验证 ===")
        # 长度一致性检查
        h1_processed = _safe_convert(output['hap_1'])
        h2_processed = _safe_convert(output['hap_2'])
        _compare_length(h1_raw, h1_processed, "hap1")
        _compare_length(h2_raw, h2_processed, "hap2")
        
        # 最终分布统计
        _get_token_distribution(h1_processed, "[子类输出] hap_1")
        _get_token_distribution(h2_processed, "[子类输出] hap_2")
        
        # 检查mask应用情况
        print(f"\n▨ mask应用验证 (当前mask含{np.sum(current_mask)}个1)")
        masked_positions = np.where(current_mask == 1)[0]
        if len(masked_positions) > 0:
            first_pos = masked_positions[0]
            print(f"  首个mask位置: {first_pos}")
            print(f"  hap1原始值: {h1_raw[first_pos]} → 处理后: {h1_processed[first_pos]}")
            print(f"  hap2原始值: {h2_raw[first_pos]} → 处理后: {h2_processed[first_pos]}")
        
        # ================== 类型转换 ==================
        for key in self.long_fields:
            if isinstance(output[key], np.ndarray):
                output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            if isinstance(output[key], np.ndarray):
                output[key] = torch.FloatTensor(output[key])
        
        return output
    """


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
                 build_ref_data=True,
                 n_gpu=1):
        # 创建基础InferDataset
        base_dataset = super().from_file(vocab, vcfpath, panelpath, freqpath, windowpath,
                                       typepath, poppath, pospath)
        
        # 创建RAGInferDataset实例
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
            build_ref_data=build_ref_data,
            n_gpu=n_gpu
        )
        return rag_dataset

    def _load_ref_data(self, ref_vcf_path: str) -> tuple:
        """加载参考数据（复用训练方法）"""
        h5_path = os.path.splitext(ref_vcf_path)[0] + ".h5"
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                return f['calldata/GT'][:], f['variants/POS'][:]
        
        # 从VCF读取
        callset = allel.read_vcf(ref_vcf_path, 
                               fields=['variants/POS', 'calldata/GT'],
                               alt_number=1)
        ref_gt = callset['calldata/GT'][:]
        ref_pos = callset['variants/POS']
        
        # 保存为HDF5
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('calldata/GT', data=ref_gt, compression='gzip')
            h5f.create_dataset('variants/POS', data=ref_pos, compression='gzip')
            
        return ref_gt, ref_pos

