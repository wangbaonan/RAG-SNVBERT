#!/bin/bash

# ==========================================
# RAG-SNVBERT训练脚本 - 无Validation版本
# Version: v11-no-val
# 临时方案：先不使用validation，确保训练能运行
# ==========================================

python -m src.train_with_val \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Train.maf01.vcf.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin \
    \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_no_val/rag_bert.model \
    \
    --dims 128 \
    --layers 8 \
    --attn_heads 4 \
    --train_batch_size 64 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 1

# 注意：这个版本没有validation，但代码已经改进：
# - RAG K从3降到1（节省显存）
# - 参数外部化（不再硬编码）
# - 训练指标会在每个epoch输出
