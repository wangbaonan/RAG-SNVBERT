#!/usr/bin/env python
"""
准备验证数据集
将VCF格式的测试数据转换为H5格式用于验证
"""

import argparse
import os
import sys
import h5py
import allel
import numpy as np
from pathlib import Path

def vcf_to_h5(vcf_path, output_h5_path, compress=True):
    """
    将VCF文件转换为H5格式

    Args:
        vcf_path: 输入VCF文件路径
        output_h5_path: 输出H5文件路径
        compress: 是否压缩
    """
    print(f"Reading VCF: {vcf_path}")

    # 读取VCF数据
    callset = allel.read_vcf(vcf_path, fields=['calldata/GT', 'variants/CHROM',
                                                'variants/POS', 'variants/ID',
                                                'variants/REF', 'variants/ALT'])

    if callset is None:
        raise ValueError(f"Failed to read VCF file: {vcf_path}")

    gt_data = callset['calldata/GT']  # (n_variants, n_samples, 2)
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    ref = callset['variants/REF']
    alt = callset['variants/ALT']

    print(f"VCF Data shape: {gt_data.shape}")
    print(f"  Variants: {gt_data.shape[0]}")
    print(f"  Samples: {gt_data.shape[1]}")

    # 创建H5文件
    print(f"Writing H5: {output_h5_path}")
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    with h5py.File(output_h5_path, 'w') as f:
        # 基因型数据
        if compress:
            f.create_dataset('calldata/GT', data=gt_data, compression='gzip', compression_opts=9)
        else:
            f.create_dataset('calldata/GT', data=gt_data)

        # 变异位点信息
        f.create_dataset('variants/CHROM', data=chrom.astype('S'))
        f.create_dataset('variants/POS', data=pos)
        f.create_dataset('variants/REF', data=ref.astype('S'))
        f.create_dataset('variants/ALT', data=alt.astype('S'))

    print(f"✓ H5 file saved: {output_h5_path}")
    return output_h5_path

def prepare_validation_data(test_dir, output_dir, mask_ratios=['30'],
                           truth_vcf=None, compress=True):
    """
    准备验证数据集

    Args:
        test_dir: 测试数据目录（包含Masked_VCFs/）
        output_dir: 输出目录
        mask_ratios: mask比例列表，如 ['10', '30', '50']
        truth_vcf: 真实标签VCF路径
        compress: 是否压缩
    """
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 转换Masked VCF
    masked_dir = test_dir / "Masked_VCFs"
    if not masked_dir.exists():
        raise FileNotFoundError(f"Masked_VCFs directory not found: {masked_dir}")

    converted_files = {}

    for mask_ratio in mask_ratios:
        vcf_file = masked_dir / f"KGP.chr21.TestMask{mask_ratio}.vcf.gz"

        if not vcf_file.exists():
            print(f"⚠ Warning: VCF not found: {vcf_file}")
            continue

        h5_file = output_dir / f"val_mask{mask_ratio}.h5"
        vcf_to_h5(str(vcf_file), str(h5_file), compress=compress)
        converted_files[f'mask{mask_ratio}'] = str(h5_file)

    # 2. 转换Truth VCF（如果提供）
    if truth_vcf:
        truth_path = Path(truth_vcf)
        if not truth_path.exists():
            truth_path = test_dir / "Truth" / "KGP.chr21.TestTruth.vcf.gz"

        if truth_path.exists():
            truth_h5 = output_dir / "val_truth.h5"
            vcf_to_h5(str(truth_path), str(truth_h5), compress=compress)
            converted_files['truth'] = str(truth_h5)
        else:
            print(f"⚠ Warning: Truth VCF not found: {truth_path}")

    # 3. 生成配置文件
    config_file = output_dir / "val_config.txt"
    with open(config_file, 'w') as f:
        f.write("# Validation Data Configuration\n")
        f.write(f"# Generated from: {test_dir}\n\n")
        for key, path in converted_files.items():
            f.write(f"{key}={path}\n")

    print(f"\n{'='*60}")
    print(f"✓ Validation data prepared!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Converted files:")
    for key, path in converted_files.items():
        print(f"  - {key}: {path}")
    print(f"Config file: {config_file}")

    return converted_files

def main():
    parser = argparse.ArgumentParser(description='准备验证数据集')

    parser.add_argument('--test_dir', type=str, required=True,
                       help='测试数据目录路径（包含Masked_VCFs/和Truth/）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出H5文件的目录')
    parser.add_argument('--mask_ratios', type=str, nargs='+', default=['30'],
                       help='要转换的mask比例列表（如 10 30 50）')
    parser.add_argument('--truth_vcf', type=str, default=None,
                       help='真实标签VCF路径（可选，默认从Truth/目录查找）')
    parser.add_argument('--no_compress', action='store_true',
                       help='不压缩H5文件（加快读取速度但占用更多空间）')

    args = parser.parse_args()

    try:
        prepare_validation_data(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            mask_ratios=args.mask_ratios,
            truth_vcf=args.truth_vcf,
            compress=not args.no_compress
        )
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
