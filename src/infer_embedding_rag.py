"""
V18 Embedding RAG Inference Script (Complete & Corrected)

修复清单:
1. [Speed] WindowMajorSampler: 索引公式修正为 `s * num_windows + w`。
2. [Geometry] VCF Shape: 修正维度变换为 [W*L, S]。
3. [Alignment] Token Slicing: 切除 [SOS] 和 Padding，防止移码。
4. [Metadata] Real REF/ALT/CHROM: 从源文件读取，不再输出硬编码值。
5. [Output] Output All Sites: 输出全量 VCF (包括原始位点和填补位点)。
"""

import argparse
import os
import time
import re
import torch
import numpy as np
import allel
from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler

from .model import BERTWithEmbeddingRAG
from .dataset import PanelData, WordVocab
from .dataset.embedding_rag_infer_dataset import EmbeddingRAGInferDataset
from .dataset.embedding_rag_dataset import embedding_rag_collate_fn
from .dataset.utils import VCFProcessingModule

# 必须与 utils.py 中的定义保持一致
INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030

class WindowMajorSampler(Sampler):
    """Window-Major Sampler for efficient FAISS caching."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.total_items = len(dataset)
        self.num_windows = dataset.window_count
        self.num_samples = self.total_items // self.num_windows
        print(f"\n▣ WindowMajorSampler Initialized")
        print(f"  - Total items: {self.total_items}")
        print(f"  - Num Windows: {self.num_windows}")
        print(f"  - Real Samples: {self.num_samples}")

    def __iter__(self):
        for w in range(self.num_windows):
            for s in range(self.num_samples):
                # 跨步索引公式: s * num_windows + w
                yield s * self.num_windows + w

    def __len__(self):
        return self.total_items

def infer():
    parser = argparse.ArgumentParser(description="V18 Embedding RAG Inference")
    parser.add_argument("--ref_panel", type=str, required=True)
    parser.add_argument("--infer_dataset", type=str, required=True)
    parser.add_argument("--infer_panel", type=str, required=True)
    parser.add_argument("-f", "--freq_path", type=str, required=True)
    parser.add_argument("--type_path", type=str, required=True)
    parser.add_argument("--pop_path", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("-c", "--check_point", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("--chrom", type=str, default=None, help="Chromosome ID (e.g. '21' or 'chr21')")
    parser.add_argument("-d", "--dims", type=int, default=384)
    parser.add_argument("-l", "--layers", type=int, default=6)
    parser.add_argument("-a", "--attn_heads", type=int, default=8)
    parser.add_argument("-b", "--infer_batch_size", type=int, default=32)
    parser.add_argument("-n", "--num_workers", type=int, default=4)
    parser.add_argument("--k_retrieve", type=int, default=1)
    parser.add_argument("--with_cuda", type=bool, default=True)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None)

    args = parser.parse_args()

    if args.with_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_devices[0]}" if args.cuda_devices else "cuda")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("▣ V18 Inference: Full Production Version")
    print("=" * 80)
    
    os.makedirs(args.output_path, exist_ok=True)

    # 1. Load Vocab
    print("\n▣ Step 1: Loading Vocabulary")
    panel = PanelData.from_file(args.infer_panel)
    vocab = WordVocab(list(panel.pop_class_dict.keys()))

    # 2. Load Model
    print("\n▣ Step 2: Loading Model")
    model = BERTWithEmbeddingRAG(len(vocab), args.dims, args.layers, args.attn_heads, dropout=0.1)
    checkpoint = torch.load(args.check_point, map_location=device)
    if isinstance(checkpoint, dict):
        if any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint
    model.to(device)
    model.eval()
    
    if hasattr(model, 'bert'): embedding_layer = model.bert.embedding
    elif hasattr(model, 'embedding'): embedding_layer = model.embedding
    else: raise AttributeError("Cannot find embedding layer")

    # 3. Create Dataset
    print("\n▣ Step 3: Creating Dataset")
    infer_dataset = EmbeddingRAGInferDataset.from_file(
        vocab, args.infer_dataset, args.infer_panel, args.freq_path, 
        args.type_path, args.pop_path, args.pos_path, 
        ref_vcf_path=args.ref_panel, embedding_layer=embedding_layer,
        build_ref_data=True, n_gpu=1, name='infer'
    )

    # 4. DataLoader
    print("\n▣ Step 4: Creating DataLoader")
    window_sampler = WindowMajorSampler(infer_dataset)
    infer_data_loader = DataLoader(
        infer_dataset, batch_size=args.infer_batch_size, 
        sampler=window_sampler, num_workers=args.num_workers,
        collate_fn=embedding_rag_collate_fn
    )

    # 5. Inference
    print("\n▣ Step 5: Inference Loop")
    all_hap1, all_hap2, all_gt, all_mask = [], [], [], []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(infer_data_loader, desc="Imputing"):
            for k in batch:
                if isinstance(batch[k], torch.Tensor): batch[k] = batch[k].to(device)
            
            # Retrieval
            batch = infer_dataset.process_batch_retrieval(batch, embedding_layer, device, args.k_retrieve)
            
            # Forward
            if hasattr(model, 'bert'): outputs = model(batch); h1_out, h2_out = outputs[0], outputs[1]
            else: h1_out, h2_out, _, _ = model(batch)
            
            # Probabilities
            h1_prob = torch.softmax(h1_out, dim=-1)[:, :, 1]
            h2_prob = torch.softmax(h2_out, dim=-1)[:, :, 1]
            
            p00 = (1-h1_prob)*(1-h2_prob)
            p01 = (1-h1_prob)*h2_prob
            p10 = h1_prob*(1-h2_prob)
            p11 = h1_prob*h2_prob
            gt_prob = torch.stack([p00, p01, p10, p11], dim=-1)
            
            all_hap1.append(h1_prob.cpu().numpy())
            all_hap2.append(h2_prob.cpu().numpy())
            all_gt.append(gt_prob.cpu().numpy())
            all_mask.append(batch['mask'].cpu().numpy())

    inference_time = time.time() - start_time
    print(f"✓ Inference Time: {inference_time:.2f}s")

    # 6. VCF Reconstruction
    print("\n▣ Step 6: VCF Generation & Reconstruction")
    
    # --- Concatenate & Slice (Fix Alignment) ---
    hap1 = np.concatenate(all_hap1, axis=0)
    hap2 = np.concatenate(all_hap2, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    mask = np.concatenate(all_mask, axis=0)
    
    # 关键：切除两端多余 Token ([SOS] + Data + [Pad])
    valid_start = 1
    valid_end = 1 + INFER_WINDOW_LEN
    hap1 = hap1[:, valid_start:valid_end]
    hap2 = hap2[:, valid_start:valid_end]
    gt = gt[:, valid_start:valid_end, :]
    mask = mask[:, valid_start:valid_end]

    # --- Geometry Transform ---
    n_wins = infer_dataset.window_count
    n_samples = hap1.shape[0] // n_wins
    L = hap1.shape[1] # 1020

    print(f"  - Transforming Geometry: [W, S, L] -> [W*L, S]")

    # Reshape [W, S, L] -> Transpose [W, L, S] -> Flatten [W*L, S]
    hap1 = hap1.reshape(n_wins, n_samples, L).transpose(0, 2, 1).reshape(-1, n_samples)
    hap2 = hap2.reshape(n_wins, n_samples, L).transpose(0, 2, 1).reshape(-1, n_samples)
    gt = gt.reshape(n_wins, n_samples, L, 4).transpose(0, 2, 1, 3).reshape(-1, n_samples, 4)
    mask = mask.reshape(n_wins, n_samples, L).transpose(0, 2, 1).reshape(-1, n_samples)
    
    # --- Align to Original Positions ---
    n_variants = len(infer_dataset.ori_pos)
    
    if hap1.shape[0] >= n_variants:
        hap1, hap2, gt, mask = hap1[:n_variants], hap2[:n_variants], gt[:n_variants], mask[:n_variants]
    else:
        print(f"⚠ Warning: Short prediction. Padding with 0.")
        pad_len = n_variants - hap1.shape[0]
        hap1 = np.pad(hap1, ((0, pad_len), (0, 0)))
        hap2 = np.pad(hap2, ((0, pad_len), (0, 0)))
        gt = np.pad(gt, ((0, pad_len), (0, 0), (0, 0)))
        mask = np.pad(mask, ((0, pad_len), (0, 0)))
    
    # --- [关键修复] 获取真实的元数据 (CHROM, REF, ALT) ---
    print("\n▣ Loading Metadata from source VCF...")
    final_chrom = args.chrom if args.chrom else None
    
    try:
        read_fields = ['variants/REF', 'variants/ALT']
        if final_chrom is None: read_fields.append('variants/CHROM')
            
        vcf_chunk = allel.read_vcf(args.infer_dataset, fields=read_fields)
        
        if vcf_chunk and 'variants/REF' in vcf_chunk:
            arr_ref = vcf_chunk['variants/REF'][:n_variants]
            arr_alt = vcf_chunk['variants/ALT'][:n_variants]
            print(f"    ✓ Loaded {len(arr_ref)} REF/ALT records")
        else:
            print("    ⚠ Warning: REF/ALT not found in VCF. Output will contain '.'")
            arr_ref, arr_alt = None, None

        if final_chrom is None:
            if vcf_chunk and 'variants/CHROM' in vcf_chunk:
                unique_chroms = np.unique(vcf_chunk['variants/CHROM'])
                if len(unique_chroms) > 0:
                    final_chrom = str(unique_chroms[0])
                    print(f"    ✓ Auto-detected CHROM: {final_chrom}")
                else: final_chrom = "21"
            else: final_chrom = "21"
    except Exception as e:
        print(f"    ⚠ Error reading VCF metadata: {e}")
        arr_ref, arr_alt = None, None
        final_chrom = final_chrom if final_chrom else "21"

    # Generate VCF (Output ALL Sites)
    pos_flag = mask[:, 0].astype(bool)
    output_vcf = os.path.join(args.output_path, "imputed.vcf")
    
    try:
        VCFProcessingModule.generate_vcf_efficient_optimized(
            chr_id=final_chrom, 
            file_path=args.infer_dataset, 
            output_path=output_vcf,
            arr_hap1=hap1, arr_hap2=hap2, arr_gt=gt,
            arr_pos=infer_dataset.ori_pos, arr_pos_flag=pos_flag,
            arr_ref=arr_ref, arr_alt=arr_alt,
            output_all=True,  # [关键] 强制输出所有位点
            chunk_size=100000
        )
        print(f"\n✓ VCF Generation Successful: {output_vcf}")
    except Exception as e:
        print(f"⚠ VCF Gen Failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    infer()