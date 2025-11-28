#!/bin/sh  # 显式使用sh解释器

# 基础配置
MODEL_PATH="output_rag/bert.model.ep9.pth"
REF_PANEL="data/New_VCF/Panel/KGP.chr21.Panel.vcf.h5"
INFER_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/34_SNV_KGP_BioData/VCF/KGP_Imputation_info_520_42.txt"

# 处理函数
run_inference() {
    input_vcf=$1
    output_dir=$2
    
    mkdir -p "$output_dir"
    
    echo "Processing $input_vcf..."
    
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

    echo "Completed: $input_vcf"
    echo "Output -> $output_dir"
    echo "--------------------------------------------------"
}

# 处理Test4（使用空格分隔列表代替数组）
test4_subs="ASA Core GDA GSA Omni258 OmniExpress OmniZhongHua"
for sub in $test4_subs; do
    input_vcf="data/New_VCF/Test/TestData/Test4/KGP.chr21.Test4.${sub}.vcf.gz"
    output_dir="infer_output_rag_Test4_${sub}/"
    run_inference "$input_vcf" "$output_dir"
done

# 处理Test5
test5_subs="PGGHan1"
for sub in $test5_subs; do
    input_vcf="data/New_VCF/Test/TestData/Test5/KGP.chr21.Test5.${sub}.vcf.gz"
    output_dir="infer_output_rag_Test5_${sub}/"
    run_inference "$input_vcf" "$output_dir"
done

echo "All Test4 and Test5 processing completed!"
