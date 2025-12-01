"""
增强版训练入口 - 添加Rare/Common F1输出和CSV日志
"""

import argparse
import torch
from torch.utils.data import DataLoader

from .dataset.rag_train_dataset import RAGTrainDataset, rag_collate_fn_with_dataset
from .dataset import PanelData, WordVocab
from .model.bert import BERTWithRAG
from .model.foundation_model import BERTFoundationModel
from .main.pretrain_with_val_enhanced import BERTTrainerWithValidationEnhanced


def main():
    parser = argparse.ArgumentParser()

    # 数据路径
    parser.add_argument("--train_dataset", required=True, type=str, help="训练H5文件")
    parser.add_argument("--train_panel", required=True, type=str, help="训练panel文件")
    parser.add_argument("--val_dataset", type=str, default=None, help="验证H5文件")
    parser.add_argument("--val_panel", type=str, default=None, help="验证panel文件")

    # RAG参数
    parser.add_argument("--refpanel_path", required=True, type=str, help="参考面板VCF")
    parser.add_argument("--freq_path", required=True, type=str, help="频率文件")
    parser.add_argument("--window_path", required=True, type=str, help="窗口文件")
    parser.add_argument("--type_path", required=True, type=str, help="type映射文件")
    parser.add_argument("--pop_path", required=True, type=str, help="pop映射文件")
    parser.add_argument("--pos_path", required=True, type=str, help="pos映射文件")

    # 模型参数
    parser.add_argument("--dims", type=int, default=128, help="模型维度")
    parser.add_argument("--layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("--attn_heads", type=int, default=4, help="注意力头数")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--train_batch_size", type=int, default=64, help="训练batch size")
    parser.add_argument("--val_batch_size", type=int, default=128, help="验证batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=20000, help="warmup步数")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数")

    # Early stopping参数
    parser.add_argument("--patience", type=int, default=5, help="early stopping耐心值")
    parser.add_argument("--val_metric", type=str, default='f1', help="验证指标")
    parser.add_argument("--min_delta", type=float, default=0.001, help="最小改进阈值")

    # RAG参数
    parser.add_argument("--rag_k", type=int, default=1, help="RAG检索K值")

    # GPU参数
    parser.add_argument("--cuda_devices", type=int, default=0, help="GPU设备")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载worker数")

    # 输出参数
    parser.add_argument("--output_path", required=True, type=str, help="模型保存路径")
    parser.add_argument("--log_freq", type=int, default=1000, help="日志打印频率")

    # 增强输出参数 (新增)
    parser.add_argument("--rare_threshold", type=float, default=0.05, help="Rare变异MAF阈值")
    parser.add_argument("--metrics_csv", type=str, default=None, help="指标CSV输出路径")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Enhanced Training with Rare/Common F1 Breakdown")
    print(f"{'='*60}")
    print(f"Rare threshold: MAF < {args.rare_threshold}")
    if args.metrics_csv:
        print(f"Metrics CSV: {args.metrics_csv}")
    print(f"{'='*60}\n")

    # 加载panel和vocab
    print(f"{'='*60}")
    print(f"Loading Data...")
    print(f"{'='*60}")

    # 加载panel
    panel = PanelData.from_file(args.train_panel)
    print("✓ Panel loaded")

    # 初始化词表
    print("Initializing Vocab...")
    vocab = WordVocab(list(panel.pop_class_dict.keys()))
    print(f"✓ Vocab size: {len(vocab)}")

    # 加载训练数据
    print("\nLoading Training Dataset...")
    rag_train_loader = RAGTrainDataset.from_file(
        vocab,
        args.train_dataset,
        args.train_panel,
        args.freq_path,
        args.window_path,
        args.type_path,
        args.pop_path,
        args.pos_path,
        args.refpanel_path,
        build_ref_data=True,
        n_gpu=1
    )

    train_dataloader = DataLoader(
        rag_train_loader,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda batch: rag_collate_fn_with_dataset(batch, rag_train_loader, args.rag_k),
        shuffle=True,
        pin_memory=True
    )

    print(f"✓ Training dataset: {len(rag_train_loader)} samples, {len(train_dataloader)} batches")

    # 加载验证数据
    rag_val_loader = None
    val_dataloader = None

    if args.val_dataset and args.val_panel:
        print("\nLoading Validation Dataset...")
        rag_val_loader = RAGTrainDataset.from_file(
            vocab,
            args.val_dataset,
            args.val_panel,
            args.freq_path,
            args.window_path,
            args.type_path,
            args.pop_path,
            args.pos_path,
            args.refpanel_path,
            build_ref_data=True,
            n_gpu=1
        )

        val_dataloader = DataLoader(
            rag_val_loader,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            collate_fn=lambda batch: rag_collate_fn_with_dataset(batch, rag_val_loader, args.rag_k),
            shuffle=False,
            pin_memory=True
        )

        print(f"✓ Validation dataset: {len(rag_val_loader)} samples, {len(val_dataloader)} batches")

    # 构建模型
    print(f"\n{'='*60}")
    print(f"Building Model...")
    print(f"{'='*60}")

    bert_model = BERTWithRAG(
        vocab_size=len(vocab),
        dims=args.dims,
        n_layers=args.layers,
        attn_heads=args.attn_heads
    )

    model = BERTFoundationModel(bert_model)

    print(f"✓ Model built: dims={args.dims}, layers={args.layers}, heads={args.attn_heads}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建trainer (增强版)
    print(f"\n{'='*60}")
    print(f"Initializing Trainer...")
    print(f"{'='*60}")

    trainer = BERTTrainerWithValidationEnhanced(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        vocab=vocab,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        with_cuda=True,
        cuda_devices=args.cuda_devices,
        log_freq=args.log_freq,
        grad_accum_steps=args.grad_accum_steps,
        patience=args.patience,
        val_metric=args.val_metric,
        min_delta=args.min_delta,
        # 增强输出参数
        rare_threshold=args.rare_threshold,
        output_csv=args.metrics_csv
    )

    # 训练循环
    print(f"\n{'='*60}")
    print(f"Starting Training...")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # 训练
        train_metrics = trainer.train(epoch)

        # 验证
        if val_dataloader:
            val_metrics = trainer.validate(epoch)

            # 保存模型
            is_best = (trainer.epochs_no_improve == 0)
            trainer.save(epoch, args.output_path, is_best=is_best)

            # 检查early stopping
            if trainer.should_stop_early(val_metrics, epoch):
                print(f"\n✓ Training stopped at epoch {epoch+1}")
                break
        else:
            # 无验证集,每个epoch都保存
            trainer.save(epoch, args.output_path)

        # 增加难度
        if rag_train_loader:
            rag_train_loader.add_level()
        if rag_val_loader:
            rag_val_loader.add_level()

    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    if trainer.best_model_path:
        print(f"Best model: {trainer.best_model_path}")
        print(f"Best {args.val_metric}: {trainer.best_val_metric:.4f}")
    if args.metrics_csv:
        print(f"Metrics saved to: {args.metrics_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
