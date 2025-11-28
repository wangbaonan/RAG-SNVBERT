#!/bin/bash

# 基础配置
MODEL_PATH="output_rag/bert.model.ep9.pth"
BASE_INPUT_DIR="data/New_VCF/Test/TestData/Test1"
BASE_OUTPUT_PREFIX="infer_output_rag_mask"
REF_PANEL="data/New_VCF/Panel/KGP.chr21.Panel.vcf.h5"
INFER_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/34_SNV_KGP_BioData/VCF/KGP_Imputation_info_520_42.txt"
max_jobs=1

process_mask() {
    local mask="$1"
    
    # 构建路径
    input_vcf="${BASE_INPUT_DIR}/KGP.chr21.Test1.Mask${mask}.vcf.gz"
    output_dir="${BASE_OUTPUT_PREFIX}${mask}/"
    imputed_file="${output_dir}/KGP.chr21.Test1.Mask${mask}_imputed.vcf.gz"

    mkdir -p "$output_dir"
    
    echo "Processing Mask${mask}..."
    
    # 执行推理
    python infer.py \
        -c "$MODEL_PATH" \
        --infer_dataset "$input_vcf" \
        --ref_panel "$REF_PANEL" \
        --infer_panel "$INFER_PANEL" \
        --freq_path data/Freq.npy \
        --type_path data/type_to_idx.bin \
        --pop_path data/pop_to_idx.bin \
        --pos_path data/pos_to_idx.bin \
        --output_path "$output_dir" \
        --dims 16 \
        --layers 4 \
        --attn_heads 2 \
        --infer_batch_size 128 \
        --cuda_devices 0

    # 生成VCF文件
    python generate_vcf.py \
        --hap1 "${output_dir}/HAP1.npy" \
        --hap2 "${output_dir}/HAP2.npy" \
        --gt "${output_dir}/GT.npy" \
        --pos "${output_dir}/POS.npy" \
        --pos_flag "${output_dir}/POS_Flag.npy" \
        --file_path "$input_vcf" \
        --output_path "$imputed_file" \
        --chr_id "chr21"

    echo "Mask${mask} processing completed. Output saved to: $imputed_file"
    echo "--------------------------------------------------"
}

# 完全兼容的并行处理函数
run_parallel() {
    local pids="" 
    local count=0
    
    for mask in 10 20 30 40 50 60 70 80 90; do
        # 启动任务
        process_mask "$mask" &
        pids="$pids $!"
        count=$((count + 1))
        
        # 控制并发数
        while [ "$count" -ge "$max_jobs" ]; do
            # 等待任意一个任务完成
            for pid in $pids; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    # 移除已完成的PID
                    pids=$(echo "$pids" | sed "s/\b$pid\b//g")
                    count=$((count - 1))
                    break
                fi
            done
            sleep 1
        done
    done
    
    # 等待剩余任务
    for pid in $pids; do
        wait "$pid"
    done
}

# 执行主程序
run_parallel
echo "All mask levels processed."
