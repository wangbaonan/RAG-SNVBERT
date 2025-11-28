import numpy as np
import argparse
from src.dataset import VCFProcessingModule

if __name__ == "__main__":
    # 设置 argparse
    parser = argparse.ArgumentParser(description="Process VCF files and generate imputed outputs.")

    # 添加命令行参数
    parser.add_argument("--hap1", type=str, required=True, help="Path to the HAP1 numpy file.")
    parser.add_argument("--hap2", type=str, required=True, help="Path to the HAP2 numpy file.")
    parser.add_argument("--gt", type=str, required=True, help="Path to the GT numpy file.")
    parser.add_argument("--pos", type=str, required=True, help="Path to the POS numpy file.")
    parser.add_argument("--pos_flag", type=str, required=True, help="Path to the POS_Flag numpy file.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input VCF file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output VCF file.")
    parser.add_argument("--chr_id", type=str, required=True, help="Chromosome ID to be processed.")

    args = parser.parse_args()

    # 加载数组
    arr_hap1 = np.load(args.hap1, mmap_mode='r')
    print("HAP1 loaded.")

    arr_hap2 = np.load(args.hap2, mmap_mode='r')
    print("HAP2 loaded.")

    arr_gt = np.load(args.gt, mmap_mode='r')
    print("GT loaded.")

    arr_pos = np.load(args.pos, mmap_mode='r')
    print("POS loaded.")

    arr_pos_flag = np.load(args.pos_flag, mmap_mode='r')
    print("POS_FLAG loaded.")

    # 调用 VCFProcessingModule 处理文件
    VCFProcessingModule.generate_vcf_efficient_optimized(
        chr_id=args.chr_id,
        file_path=args.file_path,
        output_path=args.output_path,
        arr_hap1=arr_hap1,
        arr_hap2=arr_hap2,
        arr_gt=arr_gt,
        arr_pos=arr_pos,
        arr_pos_flag=arr_pos_flag
    )