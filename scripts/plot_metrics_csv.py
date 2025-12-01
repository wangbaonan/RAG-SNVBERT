#!/usr/bin/env python
"""
从CSV文件绘制训练指标图表
包含Rare vs Common F1分解
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics_from_csv(csv_file, output_dir=None):
    """从CSV绘制完整的训练指标图表"""

    # 读取CSV
    df = pd.read_csv(csv_file)

    # 分离train和val
    train_df = df[df['mode'] == 'train']
    val_df = df[df['mode'] == 'val']

    print(f"\n{'='*60}")
    print(f"Plotting metrics from: {csv_file}")
    print(f"{'='*60}")
    print(f"Total epochs: {train_df['epoch'].max()}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    # 创建大图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')

    # 1. Overall F1
    if not train_df.empty:
        axes[0, 0].plot(train_df['epoch'], train_df['overall_f1'], 'b-o', label='Train', markersize=4)
    if not val_df.empty:
        axes[0, 0].plot(val_df['epoch'], val_df['overall_f1'], 'r-o', label='Val', markersize=4)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('Overall Haplotype F1')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Rare vs Common F1 (Val only)
    if not val_df.empty:
        axes[0, 1].plot(val_df['epoch'], val_df['overall_f1'], 'purple', marker='o',
                       label='Overall', markersize=4, linewidth=2)
        axes[0, 1].plot(val_df['epoch'], val_df['rare_f1'], 'red', marker='s',
                       label='Rare (MAF<0.05)', markersize=4, linestyle='--')
        axes[0, 1].plot(val_df['epoch'], val_df['common_f1'], 'green', marker='^',
                       label='Common (MAF>=0.05)', markersize=4, linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1: Rare vs Common')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Loss
    if not train_df.empty:
        axes[0, 2].plot(train_df['epoch'], train_df['loss'], 'b-o', label='Train', markersize=4)
    if not val_df.empty:
        axes[0, 2].plot(val_df['epoch'], val_df['loss'], 'r-o', label='Val', markersize=4)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Training Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Precision & Recall (Val)
    if not val_df.empty:
        axes[1, 0].plot(val_df['epoch'], val_df['overall_precision'], 'g-o',
                       label='Precision', markersize=4)
        axes[1, 0].plot(val_df['epoch'], val_df['overall_recall'], 'b-o',
                       label='Recall', markersize=4)
        axes[1, 0].plot(val_df['epoch'], val_df['overall_f1'], 'r-o',
                       label='F1', markersize=4)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation: Precision/Recall/F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Rare F1 breakdown
    if not val_df.empty:
        axes[1, 1].plot(val_df['epoch'], val_df['rare_precision'], 'g-o',
                       label='Precision', markersize=4, linestyle='--')
        axes[1, 1].plot(val_df['epoch'], val_df['rare_recall'], 'b-o',
                       label='Recall', markersize=4, linestyle='--')
        axes[1, 1].plot(val_df['epoch'], val_df['rare_f1'], 'r-o',
                       label='F1', markersize=4, linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Rare Variants (MAF<0.05)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Common F1 breakdown
    if not val_df.empty:
        axes[1, 2].plot(val_df['epoch'], val_df['common_precision'], 'g-o',
                       label='Precision', markersize=4, linestyle='--')
        axes[1, 2].plot(val_df['epoch'], val_df['common_recall'], 'b-o',
                       label='Recall', markersize=4, linestyle='--')
        axes[1, 2].plot(val_df['epoch'], val_df['common_f1'], 'r-o',
                       label='F1', markersize=4, linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Common Variants (MAF>=0.05)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / f"{Path(csv_file).stem}_plots.png"
    else:
        plot_file = Path(csv_file).parent / f"{Path(csv_file).stem}_plots.png"

    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_file}")
    plt.close()

    # 打印统计
    if not val_df.empty:
        print(f"\n{'='*60}")
        print(f"Summary Statistics")
        print(f"{'='*60}")

        best_overall = val_df.loc[val_df['overall_f1'].idxmax()]
        best_rare = val_df.loc[val_df['rare_f1'].idxmax()]
        best_common = val_df.loc[val_df['common_f1'].idxmax()]

        print(f"\nBest Overall F1:")
        print(f"  Epoch: {best_overall['epoch']:.0f}")
        print(f"  F1: {best_overall['overall_f1']:.4f}")
        print(f"  Rare F1: {best_overall['rare_f1']:.4f}")
        print(f"  Common F1: {best_overall['common_f1']:.4f}")

        print(f"\nBest Rare F1:")
        print(f"  Epoch: {best_rare['epoch']:.0f}")
        print(f"  Rare F1: {best_rare['rare_f1']:.4f}")
        print(f"  (Overall: {best_rare['overall_f1']:.4f})")

        print(f"\nBest Common F1:")
        print(f"  Epoch: {best_common['epoch']:.0f}")
        print(f"  Common F1: {best_common['common_f1']:.4f}")
        print(f"  (Overall: {best_common['overall_f1']:.4f})")

        # 计算rare vs common gap
        print(f"\n{'='*60}")
        print(f"Rare vs Common Performance Gap")
        print(f"{'='*60}")
        for idx, row in val_df.iterrows():
            gap = row['common_f1'] - row['rare_f1']
            print(f"Epoch {row['epoch']:.0f}: Common-Rare = {gap:+.4f} "
                  f"(Rare: {row['rare_f1']:.4f}, Common: {row['common_f1']:.4f})")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='绘制训练指标图表')
    parser.add_argument('csv_file', type=str, help='CSV指标文件')
    parser.add_argument('--output', '-o', type=str, help='输出目录')

    args = parser.parse_args()

    plot_metrics_from_csv(args.csv_file, output_dir=args.output)


if __name__ == '__main__':
    main()
