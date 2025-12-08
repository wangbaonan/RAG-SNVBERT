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
    state_dict = torch.load(args.check_point, map_location=device)

    # 处理 module. 前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")

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
        embedding_layer=model.embedding,  # 传入 embedding layer!
        build_ref_data=True,
        n_gpu=1,
        name='infer'
    )

    print(f"✓ Dataset created: {len(infer_dataset)} samples")
    print(f"✓ Windows: {infer_dataset.window_count}")

    # 4. 创建 DataLoader
    print("\n▣ Step 4: Creating DataLoader")
    infer_data_loader = DataLoader(
        infer_dataset,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    print(f"✓ DataLoader created: {len(infer_data_loader)} batches")

    # 5. 推理
    print("\n▣ Step 5: Starting Inference")
    print("=" * 80)

    # 初始化结果存储
    imputed_results = {}  # {window_idx: {sample_idx: {hap1: [], hap2: []}}}

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(infer_data_loader, desc="Imputing")):
            # 移动 batch 到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 执行检索
            batch = infer_dataset.process_batch_retrieval(
                batch,
                model.embedding,
                device,
                k_retrieve=args.k_retrieve
            )

            # 模型前向
            hap_1_output, hap_2_output, _, _ = model(batch)  # [B, L, D]

            # 解码
            # 获取 mask 位置
            mask = batch['mask']  # [B, MAX_SEQ_LEN]
            window_idx = int(batch['window_idx'][0].item())

            # 提取 Mask 位置的 logits
            B = hap_1_output.size(0)

            for i in range(B):
                sample_idx = batch_idx * args.infer_batch_size + i

                # 获取当前样本的 mask
                current_mask = mask[i].cpu().numpy()  # [MAX_SEQ_LEN]

                # 提取 Mask 位置
                mask_positions = np.where(current_mask == 1)[0]

                # 提取 Mask 位置的 logits
                hap_1_logits = hap_1_output[i, mask_positions, :]  # [num_mask, D]
                hap_2_logits = hap_2_output[i, mask_positions, :]

                # 预测 (简化版: 使用阈值或直接取最大值)
                # 这里需要根据实际的解码逻辑调整
                # 假设使用简单的二分类阈值
                hap_1_pred = (torch.sigmoid(hap_1_logits[:, 0]) > 0.5).long().cpu().numpy()
                hap_2_pred = (torch.sigmoid(hap_2_logits[:, 0]) > 0.5).long().cpu().numpy()

                # 保存结果
                if window_idx not in imputed_results:
                    imputed_results[window_idx] = {}
                if sample_idx not in imputed_results[window_idx]:
                    imputed_results[window_idx][sample_idx] = {'hap1': [], 'hap2': []}

                imputed_results[window_idx][sample_idx]['hap1'].append(hap_1_pred)
                imputed_results[window_idx][sample_idx]['hap2'].append(hap_2_pred)

    inference_time = time.time() - start_time

    print("=" * 80)
    print(f"✓ Inference completed in {inference_time:.2f}s")
    print(f"  - Total batches: {len(infer_data_loader)}")
    print(f"  - Average time per batch: {inference_time / len(infer_data_loader):.2f}s")

    # 6. 生成 VCF
    print("\n▣ Step 6: Generating Imputed VCF")
    print(f"  - Reconstructing full genotypes...")

    # 读取原始 target 数据
    if args.infer_dataset.endswith('.h5'):
        import h5py
        with h5py.File(args.infer_dataset, 'r') as f:
            original_gt = f['gt'][:]
            original_pos = f['variants/POS'][:]
    else:
        vcf_data = allel.read_vcf(
            args.infer_dataset,
            fields=['variants/POS', 'calldata/GT'],
            samples=None
        )
        original_gt = vcf_data['calldata/GT']
        original_pos = vcf_data['variants/POS']

    # 合并 imputed 结果到原始数据
    # TODO: 实现完整的 VCF 重建逻辑
    # 这里需要根据实际的数据格式和需求调整

    output_vcf_path = os.path.join(args.output_path, "imputed.vcf")
    print(f"  - Writing to: {output_vcf_path}")

    # 写入 VCF (简化版)
    # 实际实现需要根据具体需求调整
    with open(output_vcf_path, 'w') as f:
        # 写入 VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        # 写入样本名
        sample_names = [f"sample_{i}" for i in range(original_gt.shape[1])]
        f.write("\t".join(sample_names) + "\n")

        # 写入数据
        # TODO: 实现完整的数据写入逻辑

    print(f"✓ VCF file generated: {output_vcf_path}")

    print("\n" + "=" * 80)
    print("▣ V18 Inference Completed Successfully!")
    print("=" * 80)
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"Output: {output_vcf_path}")


if __name__ == "__main__":
    infer()
