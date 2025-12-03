"""
Embedding RAG训练入口 - 端到端可学习版本

核心改进:
1. 检索在embedding space进行 (端到端可学习)
2. Reference embeddings每个epoch刷新 (保持最新)
3. 内存优化: 只需过一次Transformer
4. 速度提升: 1.8x faster
"""

import argparse
import torch
from torch.utils.data import DataLoader

from .dataset.embedding_rag_dataset import EmbeddingRAGDataset, embedding_rag_collate_fn
from .dataset import PanelData, WordVocab
from .model.bert import BERTWithEmbeddingRAG
from .model.foundation_model import BERTFoundationModel
from .main.pretrain_with_val_optimized import BERTTrainerWithValidationOptimized


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
    parser.add_argument("--dims", type=int, default=192, help="模型维度")
    parser.add_argument("--layers", type=int, default=10, help="Transformer层数")
    parser.add_argument("--attn_heads", type=int, default=6, help="注意力头数")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--train_batch_size", type=int, default=32, help="训练batch size")
    parser.add_argument("--val_batch_size", type=int, default=64, help="验证batch size")
    parser.add_argument("--lr", type=float, default=7.5e-5, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=15000, help="warmup步数")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="梯度累积步数")

    # 优化参数
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal Loss gamma")
    parser.add_argument("--use_recon_loss", type=str, default="false",
                       choices=["true", "false"],
                       help="是否使用reconstruction loss")

    # Early stopping参数
    parser.add_argument("--patience", type=int, default=5, help="early stopping耐心值")
    parser.add_argument("--val_metric", type=str, default='f1', help="验证指标")
    parser.add_argument("--min_delta", type=float, default=0.001, help="最小改进阈值")

    # RAG参数
    parser.add_argument("--rag_k", type=int, default=1, help="RAG检索K值")

    # GPU参数
    parser.add_argument("--cuda_devices", type=int, default=0, help="GPU设备")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载worker数 (V18必须为0，避免CUDA fork error)")

    # 输出参数
    parser.add_argument("--output_path", required=True, type=str, help="模型保存路径")
    parser.add_argument("--log_freq", type=int, default=500, help="日志打印频率")

    # 增强输出参数
    parser.add_argument("--rare_threshold", type=float, default=0.05, help="Rare变异MAF阈值")
    parser.add_argument("--metrics_csv", type=str, default=None, help="指标CSV输出路径")

    args = parser.parse_args()

    # 转换boolean参数
    use_recon_loss = (args.use_recon_loss.lower() == "true")

    print(f"\n{'='*80}")
    print(f"Embedding RAG Training Configuration")
    print(f"{'='*80}")
    print(f"Model: Embedding RAG (End-to-End Learnable)")
    print(f"  - Retrieval in embedding space")
    print(f"  - Reference embeddings refreshed every epoch")
    print(f"  - Only pass Transformer ONCE")
    print(f"")
    print(f"Model Architecture:")
    print(f"  - Dims: {args.dims}")
    print(f"  - Layers: {args.layers}")
    print(f"  - Heads: {args.attn_heads}")
    print(f"")
    print(f"Training Config:")
    print(f"  - Batch size: {args.train_batch_size}")
    print(f"  - Grad accum: {args.grad_accum_steps}")
    print(f"  - Effective batch: {args.train_batch_size * args.grad_accum_steps}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Warmup steps: {args.warmup_steps}")
    print(f"  - Focal gamma: {args.focal_gamma}")
    print(f"  - Use recon loss: {use_recon_loss}")
    print(f"")
    print(f"Output:")
    print(f"  - Model: {args.output_path}")
    if args.metrics_csv:
        print(f"  - Metrics CSV: {args.metrics_csv}")
    print(f"{'='*80}\n")

    # 加载panel和vocab
    print(f"{'='*80}")
    print(f"Loading Data...")
    print(f"{'='*80}")

    panel = PanelData.from_file(args.train_panel)
    print("✓ Panel loaded")

    vocab = WordVocab(list(panel.pop_class_dict.keys()))
    print(f"✓ Vocab size: {len(vocab)}")

    # === 关键: 先构建模型获取embedding layer ===
    print(f"\n{'='*80}")
    print(f"Building Model...")
    print(f"{'='*80}")

    bert_model = BERTWithEmbeddingRAG(
        vocab_size=len(vocab),
        dims=args.dims,
        n_layers=args.layers,
        attn_heads=args.attn_heads
    )

    model = BERTFoundationModel(bert_model)

    # 设置设备
    device = torch.device(f'cuda:{args.cuda_devices}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"✓ Model built: dims={args.dims}, layers={args.layers}, heads={args.attn_heads}")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Device: {device}")

    # === 关键: 获取embedding layer用于预编码 ===
    embedding_layer = bert_model.embedding
    print(f"✓ Embedding layer obtained for pre-encoding")

    # 加载训练数据 (使用EmbeddingRAGDataset)
    print(f"\n{'='*80}")
    print(f"Loading Training Dataset with Embedding RAG...")
    print(f"{'='*80}")

    rag_train_loader = EmbeddingRAGDataset.from_file(
        vocab,
        args.train_dataset,
        args.train_panel,
        args.freq_path,
        args.window_path,
        args.type_path,
        args.pop_path,
        args.pos_path,
        args.refpanel_path,
        embedding_layer=embedding_layer,  # 传入embedding layer
        build_ref_data=True,
        n_gpu=1,
        use_dynamic_mask=True  # V18优势: 支持dynamic mask! 每个epoch索引会刷新
    )

    train_dataloader = DataLoader(
        rag_train_loader,
        batch_size=args.train_batch_size,
        num_workers=0,  # V18: 必须为0，避免CUDA fork error (collate_fn使用GPU)
        collate_fn=lambda batch: embedding_rag_collate_fn(
            batch, rag_train_loader, embedding_layer, args.rag_k
        ),
        shuffle=True,
        pin_memory=False  # num_workers=0时pin_memory无效
    )

    print(f"✓ Training dataset: {len(rag_train_loader)} samples, {len(train_dataloader)} batches")

    # 加载验证数据
    rag_val_loader = None
    val_dataloader = None

    if args.val_dataset and args.val_panel:
        print(f"\n{'='*80}")
        print(f"Loading Validation Dataset with Embedding RAG...")
        print(f"{'='*80}")

        rag_val_loader = EmbeddingRAGDataset.from_file(
            vocab,
            args.val_dataset,
            args.val_panel,
            args.freq_path,
            args.window_path,
            args.type_path,
            args.pop_path,
            args.pos_path,
            args.refpanel_path,
            embedding_layer=embedding_layer,
            build_ref_data=True,
            n_gpu=1,
            use_dynamic_mask=True
        )

        val_dataloader = DataLoader(
            rag_val_loader,
            batch_size=args.val_batch_size,
            num_workers=0,  # V18: 必须为0，避免CUDA fork error
            collate_fn=lambda batch: embedding_rag_collate_fn(
                batch, rag_val_loader, embedding_layer, args.rag_k
            ),
            shuffle=False,
            pin_memory=False  # num_workers=0时pin_memory无效
        )

        print(f"✓ Validation dataset: {len(rag_val_loader)} samples, {len(val_dataloader)} batches")

    # 创建trainer
    print(f"\n{'='*80}")
    print(f"Initializing Trainer...")
    print(f"{'='*80}")

    trainer = BERTTrainerWithValidationOptimized(
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
        focal_gamma=args.focal_gamma,
        use_recon_loss=use_recon_loss,
        rare_threshold=args.rare_threshold,
        output_csv=args.metrics_csv
    )

    # 训练循环
    print(f"\n{'='*80}")
    print(f"Starting Training...")
    print(f"{'='*80}\n")

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")

        # 更新epoch计数器
        if rag_train_loader:
            rag_train_loader.current_epoch = epoch
        if rag_val_loader:
            rag_val_loader.current_epoch = epoch

        # === 关键修改1: 每个epoch开始时刷新mask和索引 ===
        if epoch > 0:  # 第一个epoch使用初始化的mask
            print(f"\n{'='*80}")
            print(f"▣ Epoch {epoch}: 刷新Mask和索引 (数据增强)")
            print(f"{'='*80}")

            # 1. 重新生成mask pattern
            if rag_train_loader:
                rag_train_loader.regenerate_masks(seed=epoch)
            if rag_val_loader:
                rag_val_loader.regenerate_masks(seed=epoch)

            # 2. 用新mask和最新模型重建FAISS索引
            if rag_train_loader:
                rag_train_loader.rebuild_indexes(embedding_layer, device=device)
            if rag_val_loader:
                rag_val_loader.rebuild_indexes(embedding_layer, device=device)

            print(f"✓ Mask和索引刷新完成!\n")

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
            trainer.save(epoch, args.output_path)

        # === 修改: refresh_complete_embeddings已删除 ===
        # 现在使用按需编码，不预存储complete embeddings
        # 每个batch在collate_fn中调用encode_complete_embeddings()
        # 这样既节省内存，又确保使用最新模型
        pass  # 不需要额外操作

        # 增加难度 (Curriculum Learning)
        if rag_train_loader:
            rag_train_loader.add_level()
        if rag_val_loader:
            rag_val_loader.add_level()

    print(f"\n{'='*80}")
    print(f"Training Completed!")
    print(f"{'='*80}")
    print(f"✓ Best model saved to: {args.output_path}")
    if args.metrics_csv:
        print(f"✓ Metrics saved to: {args.metrics_csv}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
