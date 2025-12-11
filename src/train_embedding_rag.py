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
from .dataset.sampler import WindowGroupedSampler  # ✅ 导入Window-Grouped Sampler
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
    parser.add_argument("--window_size", type=int, default=511, help="固定窗口大小 (默认511,加SOS后512)")
    parser.add_argument("--type_path", required=True, type=str, help="type映射文件")
    parser.add_argument("--pop_path", required=True, type=str, help="pop映射文件")
    parser.add_argument("--pos_path", required=True, type=str, help="pos映射文件")

    # 模型参数
    parser.add_argument("--dims", type=int, default=384, help="模型维度 (RAG任务推荐384+)")
    parser.add_argument("--layers", type=int, default=12, help="Transformer层数")
    parser.add_argument("--attn_heads", type=int, default=12, help="注意力头数")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--train_batch_size", type=int, default=24, help="训练batch size (384维推荐24)")
    parser.add_argument("--val_batch_size", type=int, default=48, help="验证batch size")
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
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载worker数 (重构后支持多worker)")

    # Checkpoint恢复参数
    parser.add_argument("--resume_path", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--resume_epoch", type=int, default=0, help="恢复的起始epoch (用于课程学习)")

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

    # === Checkpoint恢复: 加载预训练权重 ===
    start_epoch = 0
    if args.resume_path:
        print(f"\n{'='*80}")
        print(f"Resuming from Checkpoint...")
        print(f"{'='*80}")
        print(f"Loading weights from: {args.resume_path}")

        checkpoint = torch.load(args.resume_path, map_location=device)

        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            # 格式1: {'state_dict': OrderedDict(...), ...}
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            # 格式2: 直接是 state_dict (OrderedDict)
            else:
                state_dict = checkpoint
        elif hasattr(checkpoint, 'state_dict'):
            # 格式3: checkpoint 是模型对象本身
            print(f"✓ Checkpoint is a model object, extracting state_dict...")
            state_dict = checkpoint.state_dict()
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

        # 移除 'module.' 前缀 (如果存在，DataParallel模型会有这个前缀)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print(f"✓ Weights loaded successfully")

        # 设置起始epoch
        start_epoch = args.resume_epoch
        print(f"✓ Resuming from epoch {start_epoch}")
        print(f"{'='*80}\n")

    # 加载训练数据 (使用EmbeddingRAGDataset)
    print(f"\n{'='*80}")
    print(f"Loading Training Dataset with Embedding RAG...")
    print(f"{'='*80}")

    rag_train_loader = EmbeddingRAGDataset.from_file(
        vocab,
        args.train_dataset,
        args.train_panel,
        args.freq_path,
        args.window_size,  # 使用固定窗口大小替代window_path
        args.type_path,
        args.pop_path,
        args.pos_path,
        args.refpanel_path,
        embedding_layer=embedding_layer,  # 传入embedding layer
        build_ref_data=True,
        n_gpu=1,
        use_dynamic_mask=False,  # 关键修复: 必须False，确保Query Mask与索引Mask一致
        name='train'  # 关键修复: 指定训练集名称，避免与验证集索引冲突
    )

    # === 性能优化: 使用Window-Grouped Sampler (减少磁盘I/O) ===
    # 将同一窗口的样本聚类在一起训练，配合单槽位缓存实现零I/O
    train_sampler = WindowGroupedSampler(rag_train_loader, shuffle=True, seed=42)

    train_dataloader = DataLoader(
        rag_train_loader,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,  # 重构后支持多worker (纯CPU collate_fn)
        collate_fn=embedding_rag_collate_fn,  # 简化collate_fn，不传参数
        sampler=train_sampler,  # ✅ 使用Window-Grouped Sampler (不能与shuffle=True同时使用)
        pin_memory=True  # 加速CPU->GPU传输
    )

    print(f"✓ Training dataset: {len(rag_train_loader)} samples, {len(train_dataloader)} batches")
    print(f"✓ Using WindowGroupedSampler for optimal I/O performance")

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
            args.window_size,  # 使用固定窗口大小替代window_path
            args.type_path,
            args.pop_path,
            args.pos_path,
            args.refpanel_path,
            embedding_layer=embedding_layer,
            build_ref_data=True,
            n_gpu=1,
            use_dynamic_mask=False,  # 关键修复: 必须False，确保Query Mask与索引Mask一致
            name='val'  # 关键修复: 指定验证集名称，使用独立的索引目录
        )

        # === 性能优化: 验证集也使用Window-Grouped Sampler ===
        # 避免验证时频繁切换窗口导致的缓存未命中和磁盘颠簸
        val_sampler = WindowGroupedSampler(rag_val_loader, shuffle=False)  # shuffle=False保证确定性

        val_dataloader = DataLoader(
            rag_val_loader,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,  # 重构后支持多worker
            collate_fn=embedding_rag_collate_fn,  # 简化collate_fn
            sampler=val_sampler,  # ✅ 使用Window-Grouped Sampler
            pin_memory=True  # 加速CPU->GPU传输
        )

        print(f"✓ Validation dataset: {len(rag_val_loader)} samples, {len(val_dataloader)} batches")
        print(f"✓ Using WindowGroupedSampler for validation (shuffle=False)")

        # === 关键修改: 固定验证集难度为50% (level=4) ===
        # 验证集不参与课程学习，保持固定难度以便公平比较不同epoch的性能
        print(f"\n{'='*80}")
        print(f"Setting Validation Mask Level to 50%...")
        print(f"{'='*80}")
        for _ in range(4):  # 从level=0提升到level=4 (50% mask)
            rag_val_loader.add_level()

        # [FIX] 立即刷新 Mask 和索引，确保 Epoch 1 即为 50% 难度，与后续 Epoch 保持一致
        print(f"Applying 50% mask immediately for consistency...")
        rag_val_loader.regenerate_masks(seed=2024)  # 使用固定种子确保可复现
        rag_val_loader.rebuild_indexes(embedding_layer, device=device)

        print(f"✓ Validation mask level set to 50%")
        print(f"✓ Validation difficulty is now FIXED for all epochs")
        print(f"{'='*80}\n")

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

    # === 关键: 传递RAG相关信息给trainer ===
    # 让trainer能够在训练循环中调用process_batch_retrieval
    trainer.rag_train_dataset = rag_train_loader
    trainer.rag_val_dataset = rag_val_loader
    trainer.embedding_layer = embedding_layer
    trainer.rag_k = args.rag_k

    # === 关键修改: 恢复训练时，同步训练集的mask level ===
    if start_epoch > 0 and rag_train_loader:
        print(f"\n{'='*80}")
        print(f"Restoring Training Mask Level for Epoch {start_epoch}...")
        print(f"{'='*80}")
        # 课程学习策略: 每2个epoch增加一次难度
        target_level = min(start_epoch // 2, 7)  # level最大为7 (80% mask)
        for _ in range(target_level):
            rag_train_loader.add_level()
        current_mask_rate = rag_train_loader._TrainDataset__mask_rate[rag_train_loader._TrainDataset__level]
        print(f"✓ Training mask level restored to: {current_mask_rate*100:.0f}%")
        print(f"{'='*80}\n")

    # 训练循环
    print(f"\n{'='*80}")
    print(f"Starting Training...")
    print(f"{'='*80}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")

        # [FIX C] 更新Sampler的随机种子，确保每个epoch的batch顺序不同
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
            print(f"✓ Train sampler epoch set to {epoch}")

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

            # 1. 重新生成mask pattern（仅训练集）
            if rag_train_loader:
                rag_train_loader.regenerate_masks(seed=epoch)
                print(f"✓ 训练集 Mask 已刷新（数据增强）")

            # [VALIDATION STRATEGY FIX] 验证集 Mask 必须固定，严禁刷新！
            # 原因：
            # - 验证集 Mask（题目）变化会导致 Loss 不可比，干扰 Early Stopping
            # - 验证集应评估"模型在固定任务上的表现"，而非"模型对新任务的泛化"
            # if rag_val_loader:
            #     rag_val_loader.regenerate_masks(seed=epoch)  # ← 已禁用！保持题目固定
            print(f"✓ 验证集 Mask 保持固定（50%），确保评估基准一致")

            # 2. 标记索引为无效 (懒惰重建模式 - JIT并行处理)
            # 关键改进: 不立即重建索引,避免串行等待5分钟
            # 实际重建会在训练时由DataLoader并行触发
            if rag_train_loader:
                rag_train_loader.invalidate_indexes()
                print(f"✓ 训练集索引已标记为无效 (将在训练时 JIT 重建)")

            if rag_val_loader:
                # 验证集索引也需要标记无效（答案随 Embedding Layer 变化）
                rag_val_loader.invalidate_indexes()
                print(f"✓ 验证集索引已标记为无效 (将在验证时 JIT 重建)")

            print(f"\n✓ Mask 刷新和索引标记完成! Epoch 切换延迟 <1秒\n")

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

        # === 关键修改: 课程学习策略优化 ===
        # 1. 训练集: 每2个epoch增加一次难度 (给模型更多时间收敛)
        # 2. 验证集: 固定50%难度，不再增加 (保持评估标准一致)
        if (epoch + 1) % 2 == 0 and rag_train_loader:
            # 只在偶数epoch增加训练难度
            current_level = rag_train_loader._TrainDataset__level
            max_level = len(rag_train_loader._TrainDataset__mask_rate) - 1

            if current_level < max_level:
                rag_train_loader.add_level()
                new_mask_rate = rag_train_loader._TrainDataset__mask_rate[rag_train_loader._TrainDataset__level]
                print(f"\n{'='*80}")
                print(f"▣ Curriculum Learning: Training Mask Rate → {new_mask_rate*100:.0f}%")
                print(f"{'='*80}\n")
            else:
                print(f"\n▣ Curriculum Learning: Maximum mask rate reached (80%)")

        # 验证集保持固定难度 (50%)
        # ❌ 已禁用: rag_val_loader.add_level()
        # 原因: 验证集必须在整个训练过程中保持固定难度，以便Loss和F1在不同epoch间可比较

    print(f"\n{'='*80}")
    print(f"Training Completed!")
    print(f"{'='*80}")
    print(f"✓ Best model saved to: {args.output_path}")
    if args.metrics_csv:
        print(f"✓ Metrics saved to: {args.metrics_csv}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
