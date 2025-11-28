# rag_train_dataset.py

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
from .dataset import TrainDataset
from .utils import timer, PanelProcessingModule, VCFProcessingModule

# DEFAULT MAXIMUM SEQUENCE LENGTH
INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030

class RAGTrainDataset(TrainDataset):
    def __init__(self, vocab, vcf, pos, panel, freq, window, 
                 type_to_idx, pop_to_idx, pos_to_idx, 
                 ref_vcf_path=None, build_ref_data=True, n_gpu=1,maf_mask_percentage=10):
        super().__init__(vocab, vcf, pos, panel, freq, window, 
                        type_to_idx, pop_to_idx, pos_to_idx)
        self.maf_mask_percentage = maf_mask_percentage
        self.ref_data_windows = []    # [num_windows, num_positions, sample * hap]  所有窗口上的ref_data
        self.raw_ref_data_windows = []
        self.window_indexes = [] # 每个窗口的FAISS索引
        self.raw_window_masks = []    # 原始mask（未填充）
        self.window_masks = []        # 填充后的mask（980长度）

        if build_ref_data and ref_vcf_path:
            self._build_faiss_indexes(ref_vcf_path)

    def _build_faiss_indexes(self, ref_vcf_path: str):
        """构建FAISS索引（完整EOS处理）"""
        print("▣ 开始构建FAISS索引")
        start_time = time.time()
        
        # 加载参考数据
        load_start = time.time()
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        print(f"▨ 加载参考数据完成 | 样本数={ref_gt.shape[1]} 位点数={ref_gt.shape[0]} 耗时{time.time()-load_start:.2f}s")

        # 主处理循环
        for w_idx in tqdm(range(self.window_count), desc="处理窗口"):
            current_slice = slice(self.window.window_info[w_idx, 0], self.window.window_info[w_idx, 1])
            window_len = current_slice.stop - current_slice.start
            #print(f"\n▩ 窗口{w_idx} [{current_slice.start}-{current_slice.stop}] 开始处理")
            
            # === 步骤1: 生成mask ===
            mask_start = time.time()
            current_pos = self.pos[current_slice]
            #raw_mask = self.maf_mask(current_pos, self.maf_mask_percentage) 
            raw_mask = self.generate_mask(window_len)
            assert len(raw_mask) == len(current_pos), "Mask长度与输入位点数量不一致"

            # 检查2: mask率有效性
            mask_rate = raw_mask.sum() / len(raw_mask)
            #print(f"全局mask比例: {mask_rate:.2%} (预期约{self.maf_mask_percentage}%)")

            # 检查3: 是否产生有效mask
            assert raw_mask.sum() > 0, "未生成任何mask位点！"

            # raw_mask = self.generate_mask(window_len)
            padded_mask = VCFProcessingModule.sequence_padding(raw_mask, dtype='int')
            self.raw_window_masks.append(raw_mask)
            self.window_masks.append(padded_mask)
            #print(f"生成Mask完成 | 原始长度={len(raw_mask)} 填充长度={len(padded_mask)} 耗时{time.time()-mask_start:.2f}s")

            # =========================
            # === 步骤2: 处理参考数据 ==
            # =========================

            ref_start = time.time()
            # 获取参考数据索引 
            # train_pos是训练数据中从vcf提取的窗口内的物理位置变量
            train_pos = self.pos[current_slice]
            # ref_indices 列表，存储 train_pos 中每个元素在 ref_pos 中的索引
            ref_indices = [np.where(ref_pos == p)[0][0] for p in train_pos]
            #print("ref_indices")
            #print(ref_indices)
            #print("ref_gt.shape:")
            #print(ref_gt.shape)
            raw_ref = ref_gt[current_slice, :, :]
            #print("raw_ref")
            #print(raw_ref.shape)
            self.raw_ref_data_windows.append(raw_ref) # 存储三维的原始refdata
            raw_ref = raw_ref.reshape(raw_ref.shape[0], -1)
            #print("raw_ref")
            #print(raw_ref.shape)
            raw_ref = raw_ref.T # 样本维度和特征（位点维度之前混淆了）
            #print("raw_ref.T")
            #print(raw_ref.shape)
            # raw_ref = raw_ref.reshape(raw_ref.shape[0], -1) # 合并hap维度
            ref_tokenized = self.tokenize(raw_ref, padded_mask) # MAX_SEQ_LEN
            # ref_tokenized.reshape(raw_ref.shape[0], -1)
            
            # self.raw_ref_data_windows.append(raw_ref) # 窗口内的数据维度 [pos, sample * hap(2)]
            self.ref_data_windows.append(ref_tokenized)
            #print(f"  ↳ 参考数据处理完成 | 样本数={ref_tokenized.shape[0]} 耗时{time.time()-ref_start:.2f}s")

            # === 步骤3: 构建索引 ===
            index_start = time.time()
            # index_data = ref_tokenized.T.astype(np.float32) # (2 * samples, -1) 
            index_data = ref_tokenized.astype(np.float32) # (2 * samples, -1)  第一个维度应是样本维度 第二个是特征维度
            index = faiss.IndexFlatL2(index_data.shape[1])  # 向量维度 = n_sites
            index.add(index_data) # (2 * samples, -1) 

            self.window_indexes.append(index)
            #print(f"  ↳ FAISS索引构建完成 | 向量维度={index.d} 数据量={index.ntotal} 耗时{time.time()-index_start:.2f}s")

        print(f"\n✔ 所有窗口处理完成 | 总窗口数={self.window_count} 总耗时{time.time()-start_time:.2f}s")

    def _load_ref_data(self, ref_vcf_path: str) -> tuple:
        """加载并预处理参考VCF数据"""
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

    def __getitem__(self, item) -> dict:
        output = super().__getitem__(item)
        window_idx = item % self.window_count
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
                  build_ref_data=True,
                  n_gpu=1):
        # 调用父类的 from_file 方法创建基础的 TrainDataset 实例
        base_dataset = super().from_file(vocab, vcfpath, panelpath, freqpath, windowpath, typepath, poppath, pospath)

        # 使用基础数据集的信息创建 RAGTrainDataset 实例
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



def rag_collate_fn_with_dataset(batch_list, dataset, k_retrieve):
    """修正后的批处理函数（完整EOS处理）"""
    total_start = time.time()
    final_batch = defaultdict(list)
    # print(f"\n▣ 开始处理批次 | 样本数={len(batch_list)}")
    
    # 按窗口分组
    window_groups = defaultdict(list)
    for sample in batch_list:
        win_idx = int(sample['window_idx'])
        window_groups[win_idx].append(sample)
    
    # 处理每个窗口
    for win_idx, group in window_groups.items():
        win_start = time.time()
        #print(f"\n▩ 处理窗口{win_idx} | 包含样本数={len(group)}")

        # 获取索引和参考数据
        index = dataset.window_indexes[win_idx] # √
        # 原始未tokenize的refdata 还需要使用空的mask进行tokenize后再提供给
        raw_ref_window = dataset.raw_ref_data_windows[win_idx]
        raw_mask = dataset.raw_window_masks[win_idx]
        raw_mask_unmasked = np.zeros_like(raw_mask)
        padded_unmasked_mask = VCFProcessingModule.sequence_padding(raw_mask_unmasked, dtype='int')

        # 已tokenize的refdata
        ref_data = dataset.ref_data_windows[win_idx]
        
        
        # 构建批量查询
        queries = []
        for sample in group:
            # 提取有效特征（与索引构建时相同）
            h1 = sample['hap_1'].numpy().flatten().astype(np.float32)  # (980,) → (1,980)
            h2 = sample['hap_2'].numpy().flatten().astype(np.float32)
            #print("--------shape-----------")
            #print("h1")
            #print(h1.shape)
            #print("h2")
            #print(h2.shape)
            #print("ref_data")
            #print(ref_data.shape)
            queries.extend([h1, h2])

        queries = np.vstack(queries) if queries else np.array([])

        # 批量检索
        if queries.size > 0:
            search_start = time.time()
            D, I = index.search(np.array(queries), k=k_retrieve)
            #print(f"  ↳ 检索完成 | 查询数={len(queries)} 耗时{time.time()-search_start:.4f}s")
            #print(f"    示例结果: Q0→{I[0][0]} (距离={D[0][0]:.2f}) Q1→{I[1][0]} (距离={D[1][0]:.2f})")
            
            
            
            for query_idx in range(len(queries)):
                # 根据query_idx定位对应的样本和单体型
                sample_idx = query_idx // 2  # 每个样本生成2个查询（h1和h2）
                hap_type = query_idx % 2     # 0表示h1，1表示h2
                key = f'rag_seg_h{1 if hap_type == 0 else 2}'
                sample = group[sample_idx]

                topk_seqs = []
                for k in range(k_retrieve):
                    # 获取对应的参考序列
                    ref_idx = I[query_idx][k]    # 检索结果索引
                    ref_sample_idx = ref_idx // 2  # 假设参考数据每个样本有2个单体型
                    ref_hap_idx = ref_idx % 2
                
                    # 将结果写入原始样本
                    rag_seq = raw_ref_window[:, ref_sample_idx, ref_hap_idx] 
                    orig_ref_seq = dataset.tokenize(rag_seq.reshape(1, -1), padded_unmasked_mask)
                    orig_ref_seq = torch.from_numpy(orig_ref_seq).squeeze(0)
                    topk_seqs.append(orig_ref_seq)
                
                sample[key] = torch.stack(topk_seqs, dim = 0)


                if query_idx < 2:  # 仅打印前两个查询的示例（避免过多输出）
                    continue
                    #print(f"\n[Debug] 窗口{win_idx} 样本{sample_idx} {key} 前25位点:")
                    #print(sample[key][:].tolist())  # 将张量转换为列表打印
        
        # 收集数据
        for sample in group:
            for key in sample:
                final_batch[key].append(sample[key])
        
        #print(f"◈ 窗口{win_idx}处理完成 | 耗时{time.time()-win_start:.2f}s")
    
    # 转换为张量
    #for key in final_batch:
    #    if key == "window_idx" or key == "":
    #        continue
    #    final_batch[key] = torch.stack(final_batch[key])
    
    #print("\n▣ 转换前的字段维度检查:")
    for key in final_batch:
        if key in ['rag_seg_h1', 'rag_seg_h2']:
            lengths = [tensor.shape[0] for tensor in final_batch[key]]
            #print(f"{key}: 长度分布 {set(lengths)}")

    #print("\n▣ 字段类型检查:")
    for key in final_batch:
        first_element = final_batch[key][0]
        type_info = type(first_element)
        if isinstance(first_element, (torch.Tensor, np.ndarray)):
            type_info = f"{type_info} (dtype={first_element.dtype}, shape={first_element.shape})"
        #print(f"字段 '{key}': {type_info}")

    # 转换为张量
    for key in final_batch:
        if key == "window_idx" or key == "hap1_nomask" or key == "hap2_nomask":
            continue
        try:
            final_batch[key] = torch.stack(final_batch[key])
            #print(f"{key}: 成功堆叠为张量，形状 {final_batch[key].shape}")
        except RuntimeError as e:
            #print(f"❌ {key}: 堆叠失败，错误信息: {e}")
            # 打印前3个样本的具体维度
            for i, tensor in enumerate(final_batch[key][:3]):
                continue
                #print(f"  样本{i} 维度: {tensor.shape}")

    total_time = time.time() - total_start
    #print(f"\n✔ 批次处理完成 | 总耗时{total_time:.2f}s 平均{total_time/len(batch_list):.4f}s/样本")
    return dict(final_batch)
