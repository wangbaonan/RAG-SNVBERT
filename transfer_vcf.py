import os
import argparse
import numpy as np


from src.dataset import VCFProcessingModule


ARR_HAP1 = "/home/user8/VCF-Bert/infer_output/HAP1.npy"
ARR_HAP2 = "/home/user8/VCF-Bert/infer_output/HAP2.npy"
ARR_GT = "/home/user8/VCF-Bert/infer_output/GT.npy"
ARR_POS = "/home/user8/VCF-Bert/infer_output/POS.npy"
ARR_POS_FLAG = "/home/user8/VCF-Bert/infer_output/POS_Flag.npy"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--check_point", type=str, required=True)
    args = parser.parse_args()

    arr_hap1 = np.load(ARR_HAP1, mmap_mode='r')
    print("HAP1 loaded.")

    arr_hap2 = np.load(ARR_HAP2, mmap_mode='r')
    print("HAP2 loaded.")

    arr_gt = np.load(ARR_GT, mmap_mode='r')
    print("GT loaded.")

    arr_pos = np.load(ARR_POS, mmap_mode='r')
    print("POS loaded.")

    arr_pos_flag = np.load(ARR_POS_FLAG, mmap_mode='r')
    print("POS_FLAG loaded.")

    VCFProcessingModule.generate_vcf(chr_id = "chr21",
                                     file_path = "/home/user8/VCF-Bert/data/Imputation_520_42/mask_vcf/KGP_chr21_masked_10.vcf.gz",
                                    #  file_path = "/home/user8/VCF-Bert/KGP_chr21_masked_10.vcf",
                                     output_path = "/home/user8/VCF-Bert/infer_output/KGP_chr21_masked_10_IMPUTED.vcf.gz",
                                     arr_hap1 = arr_hap1,
                                     arr_hap2 = arr_hap2,
                                     arr_gt = arr_gt,
                                     arr_pos = arr_pos,
                                     arr_pos_flag = arr_pos_flag)
    
    os.makedirs("infer_db", exist_ok=True)
    new_path = f"infer_db/{os.path.basename(args.check_point)}"
    os.makedirs(new_path)
    os.system(f"mv infer_output/* {new_path}")