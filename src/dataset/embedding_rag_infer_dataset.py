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

# [MODIFIED] 同步修改为 512，与训练保持一致
INFER_WINDOW_LEN = 510
MAX_SEQ_LEN = 1030


class EmbeddingRAGInferDataset(InferDataset):
    """
    V18 Embedding RAG Inference Dataset
    """

    def __init__(self, vocab, vcf, pos, panel, freq, type_to_idx, pop_to_idx, pos_to_idx,
                 ref_vcf_path=None,
                 embedding_layer=None,
                 build_ref_data=True,
                 n_gpu=1,
                 build_index: bool = True,
                 name='infer'):
        super().__init__(vocab, vcf, pos, panel, freq, type_to_idx, pop_to_idx, pos_to_idx)

        self.embedding_layer = embedding_layer
        self.embed_dim = embedding_layer.embed_size if embedding_layer else 384

        self.ref_tokens_complete = []    
        self.ref_af_windows = []         
        self.infer_masks = []            
        self.raw_window_masks = []       
        self.index_paths = []            

        self.gpu_res = faiss.StandardGpuResources()
        self.cached_index = None
        self.cached_window_idx = -1

        self.index_dir = f"maf_data/faiss_indexes_{name}"

        self.ref_vcf_path = ref_vcf_path
        if not build_index:
            print("First loading, no index building.")
        if build_ref_data and ref_vcf_path and build_index and embedding_layer:
            print("Building Embedding RAG Indexes for Inference...")
            self._build_embedding_indexes(ref_vcf_path)

    def _load_ref_data(self, ref_vcf_path: str):
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
        print("=" * 80)
        print("▣ 构建 Embedding RAG 推理索引")
        print("=" * 80)
        start_time = time.time()

        os.makedirs(self.index_dir, exist_ok=True)
        print(f"✓ FAISS 索引目录: {self.index_dir}")

        load_start = time.time()
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        print(f"✓ 加载参考数据: 样本数={ref_gt.shape[1]} | 位点数={ref_gt.shape[0]} | "
              f"耗时={time.time()-load_start:.2f}s")

        device = next(self.embedding_layer.parameters()).device
        print(f"✓ Embedding 层设备: {device}")
        print(f"✓ Embedding 维度: {self.embed_dim}")
        print()

        was_training = self.embedding_layer.training
        self.embedding_layer.eval()

        pos_to_index = {}
        for idx, p in enumerate(ref_pos):
            if p not in pos_to_index:
                pos_to_index[p] = idx

        try:
            with torch.no_grad():
                for w_idx in tqdm(range(self.window_count), desc="预编码推理窗口"):
                    start_idx = INFER_WINDOW_LEN * w_idx
                    end_idx = min(start_idx + INFER_WINDOW_LEN, self.ori_pos.shape[0])

                    mask = np.array([
                        1 if self.position_needed[i] else 0
                        for i in range(start_idx, end_idx)
                    ], dtype=int)

                    window_len = len(mask)

                    if window_len < INFER_WINDOW_LEN:
                        pad_len = INFER_WINDOW_LEN - window_len
                        mask = np.pad(mask, (0, pad_len), mode='constant')

                    self.raw_window_masks.append(mask[:window_len].copy())
                    padded_mask = VCFProcessingModule.sequence_padding(mask, dtype='int')
                    self.infer_masks.append(padded_mask)

                    current_pos = self.ori_pos[start_idx:end_idx]
                    ref_indices = []
                    valid_pos_mask = []

                    for idx, p in enumerate(current_pos):
                        ref_idx = pos_to_index.get(p, -1)
                        if ref_idx != -1:
                            ref_indices.append(ref_idx)
                            valid_pos_mask.append(idx)

                    if len(ref_indices) < len(current_pos):
                        if len(valid_pos_mask) == 0:
                            print(f"  ⚠ 跳过窗口 {w_idx}: 没有可用位点")
                            continue
                        mask = mask[valid_pos_mask]
                        window_len = len(mask)
                        current_pos = current_pos[valid_pos_mask]

                    raw_ref = ref_gt[ref_indices, :, :]  
                    raw_ref = raw_ref.reshape(raw_ref.shape[0], -1) 
                    raw_ref = raw_ref.T  

                    AF_IDX = 3
                    GLOBAL_IDX = 5
                    ref_af = np.array([
                        self.freq[AF_IDX][GLOBAL_IDX][self.pos_to_idx[p]]
                        if p in self.pos_to_idx else 0.0
                        for p in current_pos
                    ], dtype=np.float32)

                    ref_af = VCFProcessingModule.sequence_padding(ref_af, dtype='float')
                    self.ref_af_windows.append(ref_af)

                    padded_mask_for_ref = VCFProcessingModule.sequence_padding(
                        mask[:window_len], dtype='int'
                    )
                    ref_tokens_masked = self.tokenize(raw_ref, padded_mask_for_ref)

                    padded_mask_complete = np.zeros_like(padded_mask_for_ref)
                    ref_tokens_complete = self.tokenize(raw_ref, padded_mask_complete)
                    self.ref_tokens_complete.append(ref_tokens_complete)

                    num_haps = ref_tokens_masked.shape[0]
                    ref_af_expanded = np.tile(ref_af, (num_haps, 1))

                    ref_tokens_masked_tensor = torch.LongTensor(ref_tokens_masked).to(device)
                    ref_af_tensor = torch.FloatTensor(ref_af_expanded).to(device)
                    ref_emb_masked = self.embedding_layer(
                        ref_tokens_masked_tensor,
                        af=ref_af_tensor,
                        pos=True
                    ) 

                    num_haps, L, D = ref_emb_masked.shape
                    ref_emb_masked_flat = ref_emb_masked.reshape(num_haps, L * D)
                    ref_emb_masked_flat_np = ref_emb_masked_flat.cpu().numpy().astype(np.float32)

                    index = faiss.IndexFlatL2(L * D)
                    index.add(ref_emb_masked_flat_np)

                    index_path = os.path.join(self.index_dir, f"index_{w_idx}.faiss")
                    faiss.write_index(index, index_path)
                    self.index_paths.append(index_path)

                    del ref_tokens_masked_tensor, ref_af_tensor, ref_emb_masked, ref_emb_masked_flat, ref_emb_masked_flat_np, index
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

        finally:
            self.embedding_layer.train(was_training)

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
        if w_idx == self.cached_window_idx and self.cached_index is not None:
            return self.cached_index

        if self.cached_index is not None:
            del self.cached_index
            self.cached_index = None

        cpu_index = faiss.read_index(self.index_paths[w_idx])
        gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)

        self.cached_index = gpu_index
        self.cached_window_idx = w_idx
        del cpu_index

        return self.cached_index

    def __getitem__(self, item) -> dict:
        output = super().__getitem__(item)

        if 'window_idx' in output:
            window_idx = int(output['window_idx'])
        else:
            window_idx = item % self.window_count

        current_mask = self.infer_masks[window_idx]
        output['mask'] = torch.from_numpy(current_mask).long() if isinstance(current_mask, np.ndarray) else torch.LongTensor(current_mask)
        output['window_idx'] = torch.tensor(window_idx, dtype=torch.long)

        output['hap_1'] = self.tokenize(output['hap1_nomask'], current_mask)
        output['hap_2'] = self.tokenize(output['hap2_nomask'], current_mask)

        for key in self.long_fields:
            if not isinstance(output[key], torch.Tensor):
                output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            if not isinstance(output[key], torch.Tensor):
                output[key] = torch.FloatTensor(output[key])

        return output

    def process_batch_retrieval(self, batch, embedding_layer, device, k_retrieve=1):
        hap_1_tokens = batch['hap_1'].to(device)
        hap_2_tokens = batch['hap_2'].to(device)
        af = batch['af'].to(device)
        window_idx_list = batch['window_idx']

        window_groups = defaultdict(list)
        for i, win_idx in enumerate(window_idx_list):
            win_idx = int(win_idx.item())
            window_groups[win_idx].append(i)

        B = hap_1_tokens.size(0)
        L = hap_1_tokens.size(1)
        D = self.embed_dim

        rag_emb_h1_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=torch.float32)
        rag_emb_h2_final = torch.zeros(B, k_retrieve, L, D, device=device, dtype=torch.float32)

        with torch.no_grad():
            for win_idx, indices in window_groups.items():
                h1_win = hap_1_tokens[indices]
                h2_win = hap_2_tokens[indices]
                af_win = af[indices]

                h1_emb = embedding_layer(h1_win, af=af_win, pos=True)
                h2_emb = embedding_layer(h2_win, af=af_win, pos=True)

                B_win = h1_emb.size(0)

                index = self.load_index(win_idx)

                h1_emb_flat = h1_emb.reshape(B_win, L * D).cpu().numpy().astype(np.float32)
                h2_emb_flat = h2_emb.reshape(B_win, L * D).cpu().numpy().astype(np.float32)

                _, I1 = index.search(h1_emb_flat, k=k_retrieve)
                _, I2 = index.search(h2_emb_flat, k=k_retrieve)

                unique_indices = np.unique(np.concatenate([I1.flatten(), I2.flatten()]))

                ref_tokens_complete = self.ref_tokens_complete[win_idx]
                ref_af = self.ref_af_windows[win_idx]

                retrieved_tokens = ref_tokens_complete[unique_indices]
                num_retrieved = retrieved_tokens.shape[0]

                retrieved_af = np.tile(ref_af, (num_retrieved, 1))

                retrieved_tokens_tensor = torch.LongTensor(retrieved_tokens).to(device)
                ref_af_tensor = torch.FloatTensor(retrieved_af).to(device)

                retrieved_emb = embedding_layer(
                    retrieved_tokens_tensor,
                    af=ref_af_tensor,
                    pos=True
                )

                index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

                for i in range(B_win):
                    batch_idx = indices[i]

                    for k in range(k_retrieve):
                        old_ref_idx = I1[i, k]
                        new_ref_idx = index_map[old_ref_idx]
                        rag_emb_h1_final[batch_idx, k] = retrieved_emb[new_ref_idx]

                    for k in range(k_retrieve):
                        old_ref_idx = I2[i, k]
                        new_ref_idx = index_map[old_ref_idx]
                        rag_emb_h2_final[batch_idx, k] = retrieved_emb[new_ref_idx]

        batch['rag_emb_h1'] = rag_emb_h1_final.contiguous()
        batch['rag_emb_h2'] = rag_emb_h2_final.contiguous()

        return batch

    @classmethod
    def from_file(cls, vocab, vcfpath, panelpath, freqpath, typepath, poppath, pospath,
                  ref_vcf_path=None, embedding_layer=None, build_ref_data=True, n_gpu=1, name='infer'):
        dataset = super(EmbeddingRAGInferDataset, cls).from_file(
            vocab, vcfpath, panelpath, freqpath, typepath, poppath, pospath
        )

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