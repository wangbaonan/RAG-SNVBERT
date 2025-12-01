#!/bin/bash

# ==========================================
# RAG-SNVBERT训练脚本 - 使用划分的验证集
# Version: v12 (Split Validation)
# 解决方案：从训练集划分validation，避免SNP不匹配问题
# ==========================================

# 注意：首次运行前需要划分数据
# python scripts/split_data.py \
#     --input_h5 /cpfs01/.../maf_data/KGP.chr21.Train.maf01.vcf.h5 \
#     --input_panel /cpfs01/.../train.980.sample.panel \
#     --output_dir data/train_val_split \
#     --val_ratio 0.15

python -m src.train_with_val \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_split.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_panel.txt \
    \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_split.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_panel.txt \
    \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin \
    \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_split_val/rag_bert.model \
    \
    --dims 128 \
    --layers 8 \
    --attn_heads 4 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 1 \
    \
    --patience 5 \
    --val_metric f1 \
    --min_delta 0.001

# ==========================================
# 说明：
# 这个版本使用从训练集划分的validation
# 优点：
#   - SNP 100%匹配（都来自同一个文件）
#   - 不会有"找不到SNP"的问题
#   - 立即可用
# 缺点：
#   - 损失15%训练数据（但有validation可见性）
# ==========================================
