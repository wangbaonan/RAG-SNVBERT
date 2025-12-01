#!/usr/bin/env python
"""
从训练数据中划分训练集和验证集
Split training data into train and validation sets by samples
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import sys


def split_h5_by_samples(input_h5, output_dir, val_ratio=0.15, random_seed=42):
    """
    按样本划分H5数据集为训练集和验证集

    Args:
        input_h5: 输入H5文件路径
        output_dir: 输出目录
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    print(f"=" * 60)
    print(f"Split Dataset by Samples")
    print(f"=" * 60)
    print(f"Input: {input_h5}")
    print(f"Output: {output_dir}")
    print(f"Val ratio: {val_ratio}")
    print(f"Random seed: {random_seed}")
    print()

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    print("Loading data...")
    with h5py.File(input_h5, 'r') as f:
        gt_data = f['calldata/GT'][:]  # (n_variants, n_samples, 2)

        # 读取variants组（包含POS等）
        metadata = {}
        if 'variants' in f:
            for key in f['variants'].keys():
                try:
                    metadata[f'variants/{key}'] = f[f'variants/{key}'][:]
                    print(f"  - Read variants/{key}: shape={f[f'variants/{key}'].shape}")
                except Exception as e:
                    print(f"  - Warning: Could not read variants/{key}: {e}")

        # 读取其他顶层组
        for key in f.keys():
            if key not in ['calldata', 'variants']:
                try:
                    metadata[key] = f[key][:]
                    print(f"  - Read {key}")
                except Exception as e:
                    print(f"  - Warning: Could not read {key}: {e}")

        # 读取calldata下的其他数据
        if 'calldata' in f:
            for key in f['calldata'].keys():
                if key != 'GT':
                    try:
                        metadata[f'calldata/{key}'] = f[f'calldata/{key}'][:]
                        print(f"  - Read calldata/{key}")
                    except Exception as e:
                        print(f"  - Warning: Could not read calldata/{key}: {e}")

    n_variants, n_samples, n_ploidy = gt_data.shape
    print(f"✓ Data loaded:")
    print(f"  - Variants: {n_variants}")
    print(f"  - Samples: {n_samples}")
    print(f"  - Ploidy: {n_ploidy}")

    # 划分样本
    print(f"\nSplitting samples...")
    np.random.seed(random_seed)
    sample_indices = np.arange(n_samples)
    np.random.shuffle(sample_indices)

    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val

    train_indices = sample_indices[:n_train]
    val_indices = sample_indices[n_train:]

    print(f"✓ Split completed:")
    print(f"  - Train samples: {n_train} ({(1-val_ratio)*100:.1f}%)")
    print(f"  - Val samples: {n_val} ({val_ratio*100:.1f}%)")

    # 保存训练集
    train_h5 = output_dir / "train_split.h5"
    print(f"\nSaving train set: {train_h5}")
    with h5py.File(train_h5, 'w') as f_train:
        # 保存GT数据
        f_train.create_dataset('calldata/GT',
                              data=gt_data[:, train_indices, :],
                              compression='gzip',
                              compression_opts=9)

        # 保存其他元数据
        for key, value in metadata.items():
            if 'calldata' in key and key != 'calldata/GT':
                # calldata下的其他数据也需要按样本切片
                if len(value.shape) > 0 and value.shape[0] == n_variants:
                    if len(value.shape) > 1 and value.shape[1] == n_samples:
                        f_train.create_dataset(key, data=value[:, train_indices])
                    else:
                        f_train.create_dataset(key, data=value)
            else:
                f_train.create_dataset(key, data=value)

    print(f"✓ Train set saved")

    # 保存验证集
    val_h5 = output_dir / "val_split.h5"
    print(f"\nSaving val set: {val_h5}")
    with h5py.File(val_h5, 'w') as f_val:
        # 保存GT数据
        f_val.create_dataset('calldata/GT',
                            data=gt_data[:, val_indices, :],
                            compression='gzip',
                            compression_opts=9)

        # 保存其他元数据
        for key, value in metadata.items():
            if 'calldata' in key and key != 'calldata/GT':
                if len(value.shape) > 0 and value.shape[0] == n_variants:
                    if len(value.shape) > 1 and value.shape[1] == n_samples:
                        f_val.create_dataset(key, data=value[:, val_indices])
                    else:
                        f_val.create_dataset(key, data=value)
            else:
                f_val.create_dataset(key, data=value)

    print(f"✓ Val set saved")

    return train_h5, val_h5, train_indices, val_indices


def split_panel(input_panel, output_dir, train_indices, val_indices):
    """
    划分panel文件

    Args:
        input_panel: 输入panel文件路径
        output_dir: 输出目录
        train_indices: 训练集样本索引
        val_indices: 验证集样本索引
    """
    print(f"\nSplitting panel file...")

    # 读取panel
    with open(input_panel, 'r') as f:
        lines = f.readlines()

    # 第一行可能是header
    has_header = not lines[0].split()[0].startswith('sample') and not lines[0][0].isdigit()

    if has_header:
        header = lines[0]
        samples = lines[1:]
    else:
        header = None
        samples = lines

    print(f"  - Total samples in panel: {len(samples)}")

    # 划分
    train_samples = [samples[i] for i in train_indices if i < len(samples)]
    val_samples = [samples[i] for i in val_indices if i < len(samples)]

    # 保存训练集panel
    train_panel = Path(output_dir) / "train_panel.txt"
    with open(train_panel, 'w') as f:
        if header:
            f.write(header)
        for line in train_samples:
            f.write(line)

    print(f"✓ Train panel saved: {train_panel} ({len(train_samples)} samples)")

    # 保存验证集panel
    val_panel = Path(output_dir) / "val_panel.txt"
    with open(val_panel, 'w') as f:
        if header:
            f.write(header)
        for line in val_samples:
            f.write(line)

    print(f"✓ Val panel saved: {val_panel} ({len(val_samples)} samples)")

    return train_panel, val_panel


def main():
    parser = argparse.ArgumentParser(description='从训练数据划分训练集和验证集')

    parser.add_argument('--input_h5', type=str, required=True,
                       help='输入H5文件路径')
    parser.add_argument('--input_panel', type=str, required=True,
                       help='输入panel文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例（默认0.15）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子（默认42）')

    args = parser.parse_args()

    try:
        # 划分H5数据
        train_h5, val_h5, train_idx, val_idx = split_h5_by_samples(
            args.input_h5,
            args.output_dir,
            args.val_ratio,
            args.random_seed
        )

        # 划分panel
        train_panel, val_panel = split_panel(
            args.input_panel,
            args.output_dir,
            train_idx,
            val_idx
        )

        print(f"\n{'=' * 60}")
        print(f"✓ Split completed successfully!")
        print(f"{'=' * 60}")
        print(f"\nOutput files:")
        print(f"  Train H5:    {train_h5}")
        print(f"  Train panel: {train_panel}")
        print(f"  Val H5:      {val_h5}")
        print(f"  Val panel:   {val_panel}")
        print(f"\nNext steps:")
        print(f"  Use these files in your training script:")
        print(f"    --train_dataset {train_h5}")
        print(f"    --train_panel {train_panel}")
        print(f"    --val_dataset {val_h5}")
        print(f"    --val_panel {val_panel}")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
