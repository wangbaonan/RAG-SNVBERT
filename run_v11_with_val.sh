#!/bin/bash

# ==========================================
# RAG-SNVBERT训练脚本 - 带Validation支持
# Version: v11 (2025-04-XX)
# Author: wangbaonan
# ==========================================

# 注意：运行前需要先准备验证数据
# python scripts/prepare_val_data.py \
#     --test_dir /cpfs01/.../New_VCF/Test \
#     --output_dir data/validation \
#     --mask_ratios 30

python -m src.train_with_val \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Train.maf01.vcf.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/validation/val_mask30.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin \
    \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_with_val/rag_bert.model \
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
# 参数说明
# ==========================================
#
# 数据参数：
#   --train_dataset: 训练H5文件
#   --val_dataset:   验证H5文件（新增）
#   --train_panel:   训练panel
#   --val_panel:     验证panel（新增）
#
# 模型参数：
#   --dims 128       你之前成功的配置
#   --layers 8
#   --attn_heads 4
#
# 训练参数：
#   --train_batch_size 64   训练batch size
#   --val_batch_size 128    验证batch size（可以更大）
#   --epochs 20             最大训练轮数
#
# 显存优化：
#   --rag_k 1              从3降到1，节省60-70%显存 ⭐
#   --grad_accum_steps 1   如果显存还不够，可以设为2或4
#
# Validation & Early Stopping：
#   --patience 5           5个epoch不改进就停止
#   --val_metric f1        监控F1分数（也可以用accuracy或loss）
#   --min_delta 0.001      最小改进阈值
#
# ==========================================
