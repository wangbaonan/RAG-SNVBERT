#!/usr/bin/env python
"""
训练日志分析脚本
用于提取和对比不同训练run的关键指标
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_epoch_summary(log_file):
    """
    从日志文件中提取每个epoch的summary信息

    返回: DataFrame with columns:
        - epoch
        - mode (TRAIN/VAL)
        - avg_loss
        - avg_accuracy
        - hap_f1
        - hap_precision
        - hap_recall
        - gt_f1
        - gt_precision
        - gt_recall
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则匹配epoch summary块
    pattern = r"Epoch (\d+) (TRAIN|VAL) Summary.*?" \
              r"Avg Loss:\s+([\d.]+).*?" \
              r"Avg Accuracy:\s+([\d.]+).*?" \
              r"Haplotype Metrics:.*?" \
              r"F1:\s+([\d.]+).*?" \
              r"Precision:\s+([\d.]+).*?" \
              r"Recall:\s+([\d.]+).*?" \
              r"Genotype Metrics:.*?" \
              r"F1:\s+([\d.]+).*?" \
              r"Precision:\s+([\d.]+).*?" \
              r"Recall:\s+([\d.]+)"

    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        print(f"⚠️ Warning: No epoch summaries found in {log_file}")
        return pd.DataFrame()

    data = []
    for match in matches:
        epoch, mode, avg_loss, avg_acc, hap_f1, hap_p, hap_r, gt_f1, gt_p, gt_r = match
        data.append({
            'epoch': int(epoch),
            'mode': mode,
            'avg_loss': float(avg_loss),
            'avg_accuracy': float(avg_acc),
            'hap_f1': float(hap_f1),
            'hap_precision': float(hap_p),
            'hap_recall': float(hap_r),
            'gt_f1': float(gt_f1),
            'gt_precision': float(gt_p),
            'gt_recall': float(gt_r)
        })

    return pd.DataFrame(data)


def analyze_single_run(log_file, output_dir=None):
    """分析单次训练run"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {log_file}")
    print(f"{'='*60}\n")

    df = parse_epoch_summary(log_file)

    if df.empty:
        print("❌ No data found")
        return None

    # 分离train和val
    train_df = df[df['mode'] == 'TRAIN']
    val_df = df[df['mode'] == 'VAL']

    print(f"📊 Training Summary:")
    print(f"  Total epochs: {train_df['epoch'].max()}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")

    # 最佳性能
    if not val_df.empty:
        best_val_epoch = val_df.loc[val_df['hap_f1'].idxmax()]
        print(f"\n🏆 Best Validation Performance:")
        print(f"  Epoch: {best_val_epoch['epoch']}")
        print(f"  Val F1: {best_val_epoch['hap_f1']:.4f}")
        print(f"  Val Precision: {best_val_epoch['hap_precision']:.4f}")
        print(f"  Val Recall: {best_val_epoch['hap_recall']:.4f}")

        # 对应的train性能
        train_at_best = train_df[train_df['epoch'] == best_val_epoch['epoch']].iloc[0]
        print(f"\n📈 Training at Best Val Epoch:")
        print(f"  Train F1: {train_at_best['hap_f1']:.4f}")
        print(f"  Overfitting Gap: {train_at_best['hap_f1'] - best_val_epoch['hap_f1']:.4f}")

        if train_at_best['hap_f1'] - best_val_epoch['hap_f1'] > 0.1:
            print("  ⚠️  Significant overfitting detected!")

    # 最终性能
    if not val_df.empty:
        final_val = val_df.iloc[-1]
        print(f"\n📉 Final Epoch Performance:")
        print(f"  Epoch: {final_val['epoch']}")
        print(f"  Val F1: {final_val['hap_f1']:.4f}")

        if final_val['hap_f1'] < best_val_epoch['hap_f1'] - 0.02:
            print(f"  ⚠️  Performance degraded from best by {best_val_epoch['hap_f1'] - final_val['hap_f1']:.4f}")

    # 收敛速度
    if not val_df.empty and len(val_df) >= 3:
        # F1 > 0.6的第一个epoch
        epochs_to_06 = val_df[val_df['hap_f1'] > 0.6]
        if not epochs_to_06.empty:
            print(f"\n⏱️  Convergence Speed:")
            print(f"  Epochs to F1>0.6: {epochs_to_06.iloc[0]['epoch']}")

        epochs_to_07 = val_df[val_df['hap_f1'] > 0.7]
        if not epochs_to_07.empty:
            print(f"  Epochs to F1>0.7: {epochs_to_07.iloc[0]['epoch']}")

    # 生成图表
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # F1曲线
        if not train_df.empty:
            axes[0, 0].plot(train_df['epoch'], train_df['hap_f1'], 'b-o', label='Train F1')
        if not val_df.empty:
            axes[0, 0].plot(val_df['epoch'], val_df['hap_f1'], 'r-o', label='Val F1')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Haplotype F1')
        axes[0, 0].set_title('F1 Score over Epochs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss曲线
        if not train_df.empty:
            axes[0, 1].plot(train_df['epoch'], train_df['avg_loss'], 'b-o', label='Train Loss')
        if not val_df.empty:
            axes[0, 1].plot(val_df['epoch'], val_df['avg_loss'], 'r-o', label='Val Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision & Recall
        if not val_df.empty:
            axes[1, 0].plot(val_df['epoch'], val_df['hap_precision'], 'g-o', label='Precision')
            axes[1, 0].plot(val_df['epoch'], val_df['hap_recall'], 'b-o', label='Recall')
            axes[1, 0].plot(val_df['epoch'], val_df['hap_f1'], 'r-o', label='F1')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation Metrics (Haplotype)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Overfitting gap
        if not train_df.empty and not val_df.empty:
            merged = pd.merge(train_df[['epoch', 'hap_f1']],
                            val_df[['epoch', 'hap_f1']],
                            on='epoch',
                            suffixes=('_train', '_val'))
            merged['gap'] = merged['hap_f1_train'] - merged['hap_f1_val']
            axes[1, 1].plot(merged['epoch'], merged['gap'], 'purple', marker='o')
            axes[1, 1].axhline(y=0.1, color='red', linestyle='--', label='Overfitting threshold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train F1 - Val F1')
            axes[1, 1].set_title('Overfitting Gap')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_file = output_dir / f"{Path(log_file).stem}_analysis.png"
        plt.savefig(plot_file, dpi=150)
        print(f"\n📊 Plot saved to: {plot_file}")
        plt.close()

    return df


def compare_runs(log_files, labels=None, output_dir=None):
    """对比多个训练run"""
    print(f"\n{'='*60}")
    print(f"Comparing {len(log_files)} training runs")
    print(f"{'='*60}\n")

    all_dfs = []
    for i, log_file in enumerate(log_files):
        df = parse_epoch_summary(log_file)
        if df.empty:
            print(f"⚠️ Skipping {log_file} (no data)")
            continue

        label = labels[i] if labels and i < len(labels) else Path(log_file).stem
        df['run'] = label
        all_dfs.append(df)

    if not all_dfs:
        print("❌ No valid data to compare")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # 对比表格
    print(f"{'Run':<30} {'Best Val F1':<12} {'@Epoch':<8} {'Final Val F1':<12} {'Overfitting':<12}")
    print("-" * 80)

    for run in combined['run'].unique():
        run_df = combined[combined['run'] == run]
        val_df = run_df[run_df['mode'] == 'VAL']
        train_df = run_df[run_df['mode'] == 'TRAIN']

        if val_df.empty:
            continue

        best_val = val_df.loc[val_df['hap_f1'].idxmax()]
        final_val = val_df.iloc[-1]

        # 计算overfitting gap
        train_at_best = train_df[train_df['epoch'] == best_val['epoch']]
        if not train_at_best.empty:
            gap = train_at_best.iloc[0]['hap_f1'] - best_val['hap_f1']
        else:
            gap = 0.0

        print(f"{run:<30} {best_val['hap_f1']:<12.4f} {best_val['epoch']:<8.0f} "
              f"{final_val['hap_f1']:<12.4f} {gap:<12.4f}")

    # 生成对比图
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Val F1对比
        for run in combined['run'].unique():
            run_df = combined[(combined['run'] == run) & (combined['mode'] == 'VAL')]
            if not run_df.empty:
                axes[0].plot(run_df['epoch'], run_df['hap_f1'], marker='o', label=run)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation F1')
        axes[0].set_title('Validation F1 Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss对比
        for run in combined['run'].unique():
            run_df = combined[(combined['run'] == run) & (combined['mode'] == 'VAL')]
            if not run_df.empty:
                axes[1].plot(run_df['epoch'], run_df['avg_loss'], marker='o', label=run)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Validation Loss Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        comparison_file = output_dir / "comparison.png"
        plt.savefig(comparison_file, dpi=150)
        print(f"\n📊 Comparison plot saved to: {comparison_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='分析训练日志')
    parser.add_argument('log_files', nargs='+', help='日志文件路径')
    parser.add_argument('--labels', nargs='*', help='每个run的标签 (可选)')
    parser.add_argument('--output', '-o', type=str, help='输出图表目录')
    parser.add_argument('--compare', action='store_true', help='对比多个runs')

    args = parser.parse_args()

    if len(args.log_files) == 1 and not args.compare:
        # 单个run分析
        analyze_single_run(args.log_files[0], output_dir=args.output)
    else:
        # 多个runs对比
        compare_runs(args.log_files, labels=args.labels, output_dir=args.output)


if __name__ == '__main__':
    main()
