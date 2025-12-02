#!/usr/bin/env python
"""
修复数据分割 - 确保训练集和验证集无重叠
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def split_h5_data(input_h5, output_train, output_val, val_ratio=0.2, seed=42):
    """
    将H5文件分割为训练集和验证集（无重叠）

    Args:
        input_h5: 输入H5文件路径
        output_train: 训练集输出路径
        output_val: 验证集输出路径
        val_ratio: 验证集比例（默认0.2 = 20%）
        seed: 随机种子
    """

    print(f"\n{'='*70}")
    print(f"Data Splitting (No Overlap Guaranteed)")
    print(f"{'='*70}\n")

    print(f"Input: {input_h5}")
    print(f"Val ratio: {val_ratio * 100}%")
    print(f"Random seed: {seed}\n")

    # 设置随机种子
    np.random.seed(seed)

    # 读取所有样本key
    with h5py.File(input_h5, 'r') as f:
        all_keys = list(f.keys())
        total_samples = len(all_keys)

        print(f"Total samples: {total_samples}")

        # 随机打乱
        np.random.shuffle(all_keys)

        # 计算分割点
        val_size = int(total_samples * val_ratio)
        train_size = total_samples - val_size

        # 分割
        train_keys = all_keys[:train_size]
        val_keys = all_keys[train_size:]

        print(f"Train samples: {len(train_keys)}")
        print(f"Val samples: {len(val_keys)}")

        # 验证无重叠
        overlap = set(train_keys) & set(val_keys)
        assert len(overlap) == 0, f"ERROR: Found {len(overlap)} overlapping samples!"
        print("✓ No overlap verified\n")

        # 创建训练集
        print(f"Creating train set: {output_train}")
        with h5py.File(output_train, 'w') as train_f:
            for key in train_keys:
                f.copy(key, train_f)
        print("✓ Train set created")

        # 创建验证集
        print(f"Creating val set: {output_val}")
        with h5py.File(output_val, 'w') as val_f:
            for key in val_keys:
                f.copy(key, val_f)
        print("✓ Val set created")

    # 验证结果
    print(f"\n{'='*70}")
    print("Verification")
    print(f"{'='*70}\n")

    with h5py.File(output_train, 'r') as f:
        train_keys_verify = set(f.keys())
        print(f"Train H5: {len(train_keys_verify)} samples")

    with h5py.File(output_val, 'r') as f:
        val_keys_verify = set(f.keys())
        print(f"Val H5: {len(val_keys_verify)} samples")

    overlap_verify = train_keys_verify & val_keys_verify
    print(f"Overlap: {len(overlap_verify)} samples")

    if len(overlap_verify) == 0:
        print("\n✓ SUCCESS: Data split correctly with no overlap!")
    else:
        print(f"\n✗ ERROR: Found {len(overlap_verify)} overlapping samples!")
        return False

    print(f"\n{'='*70}\n")
    return True


def split_panel_file(input_panel, output_train_panel, output_val_panel,
                     train_keys, val_keys):
    """
    根据H5的key分割panel文件

    Args:
        input_panel: 输入panel文件
        output_train_panel: 训练panel输出
        output_val_panel: 验证panel输出
        train_keys: 训练集样本key列表
        val_keys: 验证集样本key列表
    """

    print(f"\nSplitting panel file: {input_panel}")

    # 读取panel文件
    with open(input_panel, 'r') as f:
        lines = f.readlines()

    if not lines:
        print("✗ Panel file is empty!")
        return False

    # 假设panel文件每行是一个样本，格式可能是sample_id开头
    # 需要根据实际格式调整
    train_lines = []
    val_lines = []

    for line in lines:
        # 假设每行以sample_id开头（需要根据实际情况调整）
        # 例如: HG00096_1_chr21_...
        sample_id = line.split()[0] if line.strip() else None

        if sample_id:
            if sample_id in train_keys:
                train_lines.append(line)
            elif sample_id in val_keys:
                val_lines.append(line)

    # 写入训练panel
    with open(output_train_panel, 'w') as f:
        f.writelines(train_lines)
    print(f"✓ Train panel: {len(train_lines)} lines")

    # 写入验证panel
    with open(output_val_panel, 'w') as f:
        f.writelines(val_lines)
    print(f"✓ Val panel: {len(val_lines)} lines")

    return True


def main():
    parser = argparse.ArgumentParser(description='修复数据分割，确保无重叠')
    parser.add_argument('--input_h5', required=True, help='输入H5文件')
    parser.add_argument('--input_panel', required=True, help='输入panel文件')
    parser.add_argument('--output_dir', default='data/train_val_split_fixed',
                       help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例（默认0.2）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_train_h5 = output_dir / 'train_split.h5'
    output_val_h5 = output_dir / 'val_split.h5'
    output_train_panel = output_dir / 'train_panel.txt'
    output_val_panel = output_dir / 'val_panel.txt'

    # 分割H5文件
    success = split_h5_data(
        args.input_h5,
        str(output_train_h5),
        str(output_val_h5),
        args.val_ratio,
        args.seed
    )

    if not success:
        print("✗ Failed to split H5 data!")
        return 1

    # 获取分割后的keys
    with h5py.File(str(output_train_h5), 'r') as f:
        train_keys = set(f.keys())

    with h5py.File(str(output_val_h5), 'r') as f:
        val_keys = set(f.keys())

    # 分割panel文件
    split_panel_file(
        args.input_panel,
        str(output_train_panel),
        str(output_val_panel),
        train_keys,
        val_keys
    )

    print(f"\n{'='*70}")
    print("All Done!")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Train H5: {output_train_h5}")
    print(f"  Val H5: {output_val_h5}")
    print(f"  Train panel: {output_train_panel}")
    print(f"  Val panel: {output_val_panel}")
    print(f"\n{'='*70}\n")

    return 0


if __name__ == '__main__':
    exit(main())
