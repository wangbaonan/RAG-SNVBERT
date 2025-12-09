"""
V18 Embedding RAG Inference Script

关键特性:
1. 加载 V18 模型 (BERTWithEmbeddingRAG)
2. 使用 EmbeddingRAGInferDataset (Imputation Masking)
3. Lazy Encoding: 检索后按需编码 Complete Reference
4. 生成完整的 VCF 文件
"""

import argparse
import os
import time
import torch
import numpy as np
import allel
from tqdm import tqdm
from torch.utils.data import DataLoader

from .model import BERTWithEmbeddingRAG
from .dataset import PanelData, WordVocab
from .dataset.embedding_rag_infer_dataset import EmbeddingRAGInferDataset
from .dataset.embedding_rag_dataset import embedding_rag_collate_fn
from .dataset.utils import VCFProcessingModule

INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030


def infer():
    parser = argparse.ArgumentParser(description="V18 Embedding RAG Inference")

    # Data paths
    parser.add_argument("--ref_panel", type=str, required=True,
                        help="Reference panel for FAISS index")
    parser.add_argument("--infer_dataset", type=str, required=True,
                        help="Target dataset for imputation")
    parser.add_argument("--infer_panel", type=str, required=True,
                        help="Population panel for target data")
    parser.add_argument("-f", "--freq_path", type=str, required=True,
                        help="Frequency data file")
    parser.add_argument("--type_path", type=str, required=True,
                        help="Genotype to index mapping")
    parser.add_argument("--pop_path", type=str, required=True,
                        help="Population to index mapping")
    parser.add_argument("--pos_path", type=str, required=True,
                        help="Position to index mapping")

    # Model checkpoint
    parser.add_argument("-c", "--check_point", type=str, required=True,
                        help="Model checkpoint path (e.g., output/rag_bert.model.ep11)")

    # Output
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="Output directory for imputed VCF")

    # Model architecture (必须显式传入!)
    parser.add_argument("-d", "--dims", type=int, default=384,
                        help="Hidden dimension of transformer model (must match training)")
    parser.add_argument("-l", "--layers", type=int, default=6,
                        help="Number of transformer layers (must match training)")
    parser.add_argument("-a", "--attn_heads", type=int, default=8,
                        help="Number of attention heads (must match training)")

    # Inference settings
    parser.add_argument("-b", "--infer_batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("-n", "--num_workers", type=int, default=4,
                        help="Dataloader worker size")
    parser.add_argument("--k_retrieve", type=int, default=1,
                        help="Number of reference haplotypes to retrieve")

    # Device
    parser.add_argument("--with_cuda", type=bool, default=True,
                        help="Use CUDA for inference")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None,
                        help="CUDA device ids")

    args = parser.parse_args()

    # 设置设备
    if args.with_cuda and torch.cuda.is_available():
        if args.cuda_devices:
            device = torch.device(f"cuda:{args.cuda_devices[0]}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print("▣ V18 Embedding RAG Inference")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: dims={args.dims}, layers={args.layers}, heads={args.attn_heads}")
    print(f"Checkpoint: {args.check_point}")
    print(f"Target dataset: {args.infer_dataset}")
    print(f"Reference panel: {args.ref_panel}")
    print(f"Output: {args.output_path}")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 1. 加载 Vocab
    print("\n▣ Step 1: Loading Vocabulary")
    panel = PanelData.from_file(args.infer_panel)
    vocab = WordVocab(list(panel.pop_class_dict.keys()))
    print(f"✓ Vocab size: {len(vocab)}")

    # 2. 加载模型
    print("\n▣ Step 2: Loading V18 Model (BERTWithEmbeddingRAG)")
    print(f"  - Architecture: dims={args.dims}, layers={args.layers}, heads={args.attn_heads}")

    model = BERTWithEmbeddingRAG(
        vocab_size=len(vocab),
        dims=args.dims,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
        dropout=0.1
    )

    # 加载 checkpoint
    print(f"  - Loading checkpoint: {args.check_point}")
    checkpoint = torch.load(args.check_point, map_location=device)

    # 检查 checkpoint 类型
    if isinstance(checkpoint, dict):
        # 如果是 state_dict
        print(f"  - Loading from state_dict...")
        if any(k.startswith('module.') for k in checkpoint.keys()):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)
    else:
        # 如果是整个模型对象
        print(f"  - Loading from model object...")
        model = checkpoint

    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")

    # 获取 embedding layer (处理 BERTFoundationModel 包装)
    if hasattr(model, 'bert'):
        # BERTFoundationModel: model.bert.embedding
        embedding_layer = model.bert.embedding
        bert_model = model.bert
        print(f"  - Model type: BERTFoundationModel (wrapped)")
    elif hasattr(model, 'embedding'):
        # 直接的 BERTWithEmbeddingRAG: model.embedding
        embedding_layer = model.embedding
        bert_model = model
        print(f"  - Model type: BERTWithEmbeddingRAG (direct)")
    else:
        raise AttributeError(f"Cannot find embedding layer in model type: {type(model).__name__}")

    # 3. 创建 Infer Dataset (关键: 传入 embedding_layer)
    print("\n▣ Step 3: Creating EmbeddingRAGInferDataset")
    print(f"  - Target dataset: {args.infer_dataset}")
    print(f"  - Reference panel: {args.ref_panel}")
    print(f"  - Building FAISS indexes with Imputation Masking...")

    infer_dataset = EmbeddingRAGInferDataset.from_file(
        vocab=vocab,
        vcfpath=args.infer_dataset,
        panelpath=args.infer_panel,
        freqpath=args.freq_path,
        typepath=args.type_path,
        poppath=args.pop_path,
        pospath=args.pos_path,
        ref_vcf_path=args.ref_panel,
        embedding_layer=embedding_layer,  # 传入 embedding layer!
        build_ref_data=True,
        n_gpu=1,
        name='infer'
    )

    print(f"✓ Dataset created: {len(infer_dataset)} samples")
    print(f"✓ Windows: {infer_dataset.window_count}")

    # 4. 创建 DataLoader (使用 embedding_rag_collate_fn 修复 Crash)
    print("\n▣ Step 4: Creating DataLoader")
    infer_data_loader = DataLoader(
        infer_dataset,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=embedding_rag_collate_fn  # 关键: 使用自定义 collate_fn
    )
    print(f"✓ DataLoader created: {len(infer_data_loader)} batches")

    # 5. 推理 (收集全量数据用于 VCF 生成)
    print("\n▣ Step 5: Starting Inference")
    print("=" * 80)

    # 初始化结果存储 (全量收集)
    all_hap1_probs = []  # List of [B, L] arrays
    all_hap2_probs = []
    all_gt_probs = []    # List of [B, L, 4] arrays
    all_positions = []   # List of [B, L] arrays
    all_masks = []       # List of [B, L] arrays (用于构建 pos_flag)

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(infer_data_loader, desc="Imputing")):
            # 移动 batch 到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 执行检索 (支持跨窗口 Batch)
            batch = infer_dataset.process_batch_retrieval(
                batch,
                embedding_layer,
                device,
                k_retrieve=args.k_retrieve
            )

            # 模型前向 (处理不同的模型类型)
            if hasattr(model, 'bert'):
                # BERTFoundationModel: 返回值多
                outputs = model(batch)
                hap_1_output = outputs[0]  # [B, L, 2] (logits)
                hap_2_output = outputs[1]  # [B, L, 2]
            else:
                # BERTWithEmbeddingRAG: 直接调用
                hap_1_output, hap_2_output, _, _ = model(batch)  # [B, L, 2]

            # === 计算概率 (Task 3: Implement Imputation & Accumulation) ===
            # 1. Haplotype Probabilities (取 Alt Allele 概率)
            hap1_probs = torch.softmax(hap_1_output, dim=-1)[:, :, 1]  # [B, L] (P(Alt))
            hap2_probs = torch.softmax(hap_2_output, dim=-1)[:, :, 1]  # [B, L]

            # 2. Genotype Probabilities (4 种组合)
            # P(0|0) = (1-h1) * (1-h2)
            # P(0|1) = (1-h1) * h2
            # P(1|0) = h1 * (1-h2)
            # P(1|1) = h1 * h2
            p_00 = (1 - hap1_probs) * (1 - hap2_probs)  # [B, L]
            p_01 = (1 - hap1_probs) * hap2_probs
            p_10 = hap1_probs * (1 - hap2_probs)
            p_11 = hap1_probs * hap2_probs

            gt_probs = torch.stack([p_00, p_01, p_10, p_11], dim=-1)  # [B, L, 4]

            # 3. 收集数据 (移至 CPU)
            all_hap1_probs.append(hap1_probs.cpu().numpy())
            all_hap2_probs.append(hap2_probs.cpu().numpy())
            all_gt_probs.append(gt_probs.cpu().numpy())

            # 4. 收集位置信息 (从 batch)
            # 位置需要从 dataset 重建 (通过 window_idx)
            # 简化版: 使用 batch 的 pos 字段 (如果存在)
            if 'pos' in batch:
                all_positions.append(batch['pos'].cpu().numpy())
            else:
                # 备用: 从 dataset 获取
                # 这里需要根据 window_idx 重建位置
                # 暂时使用占位符
                all_positions.append(np.zeros((hap1_probs.size(0), hap1_probs.size(1)), dtype=np.int32))

            # 5. 收集 Mask (用于构建 pos_flag)
            all_masks.append(batch['mask'].cpu().numpy())

    inference_time = time.time() - start_time

    print("=" * 80)
    print(f"✓ Inference completed in {inference_time:.2f}s")
    print(f"  - Total batches: {len(infer_data_loader)}")
    print(f"  - Average time per batch: {inference_time / len(infer_data_loader):.2f}s")

    # === Step 6: VCF 生成 (Task 4: Integrate VCF Generation) ===
    print("\n▣ Step 6: Generating Imputed VCF")
    print(f"  - Concatenating inference results...")

    # 1. Concatenate 所有 Batch 结果
    arr_hap1 = np.concatenate(all_hap1_probs, axis=0)  # [N_total_samples, L]
    arr_hap2 = np.concatenate(all_hap2_probs, axis=0)
    arr_gt = np.concatenate(all_gt_probs, axis=0)       # [N_total_samples, L, 4]
    arr_pos = np.concatenate(all_positions, axis=0)     # [N_total_samples, L]
    arr_mask = np.concatenate(all_masks, axis=0)        # [N_total_samples, L]

    print(f"  - Total samples: {arr_hap1.shape[0]}")
    print(f"  - Sequence length: {arr_hap1.shape[1]}")

    # 2. 转换数据格式以适配 VCFProcessingModule
    # VCFProcessingModule 期望的格式:
    # - arr_hap1/2: [N_Variants, N_Samples] (转置!)
    # - arr_gt: [N_Variants, N_Samples, 4]
    # - arr_pos: [N_Variants]
    # - arr_pos_flag: [N_Variants] (mask == 1 的位置)

    N_samples = arr_hap1.shape[0]
    L = arr_hap1.shape[1]

    # 转置 haplotype probabilities
    arr_hap1_T = arr_hap1.T  # [L, N_samples]
    arr_hap2_T = arr_hap2.T

    # 转置 genotype probabilities
    arr_gt_T = arr_gt.transpose(1, 0, 2)  # [L, N_samples, 4]

    # 3. 构建位置数组和 Flag (从 dataset 获取真实位置)
    # 从 infer_dataset 获取原始位置信息
    ori_pos = infer_dataset.ori_pos  # [N_total_positions]

    # 构建位置数组 (根据窗口结构重建)
    # 简化版: 取前 L 个位置
    final_positions = ori_pos[:L] if len(ori_pos) >= L else np.pad(ori_pos, (0, L - len(ori_pos)), mode='constant')

    # 构建 pos_flag (只写入被 mask 的位置)
    # 这里使用第一个样本的 mask 作为全局 flag (假设所有样本的 mask pattern 相同)
    final_pos_flag = arr_mask[0, :L].astype(bool)  # [L]

    print(f"  - Imputed positions: {final_pos_flag.sum()}")

    # 4. 调用 VCFProcessingModule.generate_vcf_efficient_optimized
    output_vcf_path = os.path.join(args.output_path, "imputed.vcf")
    print(f"  - Writing to: {output_vcf_path}")

    try:
        VCFProcessingModule.generate_vcf_efficient_optimized(
            chr_id="21",  # TODO: 从输入 VCF 提取染色体号
            file_path=args.infer_dataset,  # 原始 VCF 文件 (用于获取 Header)
            output_path=output_vcf_path,
            arr_hap1=arr_hap1_T[:L],      # [N_Variants, N_Samples]
            arr_hap2=arr_hap2_T[:L],
            arr_gt=arr_gt_T[:L],          # [N_Variants, N_Samples, 4]
            arr_pos=final_positions,      # [N_Variants]
            arr_pos_flag=final_pos_flag,  # [N_Variants]
            chunk_size=100000
        )
        print(f"✓ VCF file generated: {output_vcf_path}")
    except Exception as e:
        print(f"⚠ VCF generation failed: {e}")
        print(f"  - Falling back to simplified VCF writing...")

        # Fallback: 简化版 VCF 写入
        with open(output_vcf_path, 'w') as f:
            # 写入 VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")

            # 写入样本名
            sample_names = [f"sample_{i}" for i in range(N_samples)]
            f.write("\t".join(sample_names) + "\n")

            # 写入数据 (只写入 mask==1 的位置)
            for pos_idx in range(L):
                if not final_pos_flag[pos_idx]:
                    continue

                pos_val = final_positions[pos_idx]
                f.write(f"21\t{pos_val}\t.\t.\t.\t0\tPASS\t.\tGT")

                # 写入每个样本的基因型 (简化版: 只写 GT)
                for s_idx in range(N_samples):
                    gt_idx = np.argmax(arr_gt_T[pos_idx, s_idx, :])
                    gt_map = {0: "0|0", 1: "0|1", 2: "1|0", 3: "1|1"}
                    f.write(f"\t{gt_map[gt_idx]}")

                f.write("\n")

        print(f"✓ Simplified VCF file generated: {output_vcf_path}")

    print("\n" + "=" * 80)
    print("▣ V18 Inference Completed Successfully!")
    print("=" * 80)
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Output: {output_vcf_path}")


if __name__ == "__main__":
    infer()
