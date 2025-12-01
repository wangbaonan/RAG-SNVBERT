"""
训练脚本 - 带Validation支持
基于原有train.py，添加验证集和Early Stopping
"""

import argparse
import random
import torch
from torch.utils.data import DataLoader

from .model import BERT, BERTWithRAG
from .main import BERTTrainerWithValidation  # 使用新的Trainer
from .dataset import PanelData, TrainDataset, WordVocab, RAGTrainDataset
from .dataset.rag_train_dataset import rag_collate_fn_with_dataset

# 设置随机种子
seed = 1234
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)


def train():
    parser = argparse.ArgumentParser(description='RAG-SNVBERT Training with Validation')

    # 数据集参数
    parser.add_argument("--train_dataset", type=str, required=True, help="训练数据集H5文件")
    parser.add_argument("--train_panel", type=str, required=True, help="训练集panel文件")

    # 新增：验证集参数
    parser.add_argument("--val_dataset", type=str, default=None, help="验证数据集H5文件")
    parser.add_argument("--val_panel", type=str, default=None, help="验证集panel文件")

    # 共享数据文件
    parser.add_argument("-f", "--freq_path", type=str, required=True, help="频率数据文件")
    parser.add_argument("-w", "--window_path", type=str, required=True, help="窗口数据CSV文件")
    parser.add_argument("--type_path", type=str, required=True, help="基因型索引映射文件")
    parser.add_argument("--pop_path", type=str, required=True, help="群体索引映射文件")
    parser.add_argument("--pos_path", type=str, required=True, help="位点索引映射文件")
    parser.add_argument("--refpanel_path", type=str, required=True, help="RAG参考面板路径")

    # 模型参数
    parser.add_argument("-c", "--check_point", type=str, default=None, help="检查点路径")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="输出模型路径")
    parser.add_argument("-d", "--dims", type=int, default=128, help="隐藏层维度")
    parser.add_argument("-l", "--layers", type=int, default=8, help="Transformer层数")
    parser.add_argument("-a", "--attn_heads", type=int, default=4, help="注意力头数")

    # 训练参数
    parser.add_argument("-b", "--train_batch_size", type=int, default=64, help="训练batch size")
    parser.add_argument("--val_batch_size", type=int, default=128, help="验证batch size（可以更大）")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="数据加载workers数量")

    # 新增：RAG参数
    parser.add_argument("--rag_k", type=int, default=3, help="FAISS检索Top-K数量（默认3，建议1-2节省显存）")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Adam权重衰减")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="梯度累积步数")

    # 新增：Validation & Early Stopping参数
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--val_metric", type=str, default='f1', choices=['f1', 'accuracy', 'loss'],
                       help="验证监控指标")
    parser.add_argument("--min_delta", type=float, default=0.001, help="最小改进阈值")

    # 其他参数
    parser.add_argument("--with_cuda", type=bool, default=True, help="是否使用CUDA")
    parser.add_argument("--log_freq", type=int, default=1000, help="日志打印频率")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA设备ID")

    args = parser.parse_args()

    # ========== 加载数据 ==========
    print("="*60)
    print("Loading Data...")
    print("="*60)

    # 加载panel
    panel = PanelData.from_file(args.train_panel)
    print("✓ Panel loaded")

    # 初始化词表
    print("Initializing Vocab...")
    vocab = WordVocab(list(panel.pop_class_dict.keys()))
    print(f"✓ Vocab size: {len(vocab)}")

    # 加载训练集
    print("\nLoading Training Dataset...")
    train_dataset = TrainDataset.from_file(
        vocab, args.train_dataset, args.train_panel, args.freq_path,
        args.window_path, args.type_path, args.pop_path, args.pos_path
    )

    rag_train_dataset = RAGTrainDataset.from_file(
        vocab, args.train_dataset, args.train_panel, args.freq_path,
        args.window_path, args.type_path, args.pop_path, args.pos_path,
        args.refpanel_path, build_ref_data=True, n_gpu=1
    )

    rag_train_loader = DataLoader(
        rag_train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: rag_collate_fn_with_dataset(batch, rag_train_dataset, args.rag_k)
    )
    print(f"✓ Training dataset: {len(rag_train_dataset)} samples, {len(rag_train_loader)} batches")

    # 加载验证集（可选）
    rag_val_loader = None
    if args.val_dataset and args.val_panel:
        print("\nLoading Validation Dataset...")

        val_panel = PanelData.from_file(args.val_panel)

        rag_val_dataset = RAGTrainDataset.from_file(
            vocab, args.val_dataset, args.val_panel, args.freq_path,
            args.window_path, args.type_path, args.pop_path, args.pos_path,
            args.refpanel_path, build_ref_data=True, n_gpu=1
        )

        rag_val_loader = DataLoader(
            rag_val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,  # 验证不需要shuffle
            num_workers=args.num_workers,
            collate_fn=lambda batch: rag_collate_fn_with_dataset(batch, rag_val_dataset, args.rag_k)
        )
        print(f"✓ Validation dataset: {len(rag_val_dataset)} samples, {len(rag_val_loader)} batches")
    else:
        print("\n⚠ No validation dataset provided, training without validation")

    # ========== 构建模型 ==========
    print("\n" + "="*60)
    print("Building Model...")
    print("="*60)

    rag_bert = BERTWithRAG(len(vocab), dims=args.dims, n_layers=args.layers, attn_heads=args.attn_heads)
    print(f"✓ Model: BERTWithRAG(dims={args.dims}, layers={args.layers}, heads={args.attn_heads})")

    # 加载检查点
    state_dict = None
    if args.check_point is not None:
        print(f"Loading checkpoint: {args.check_point}")
        state_dict = torch.load(args.check_point)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        print("✓ Checkpoint loaded")

    # ========== 创建Trainer ==========
    print("\n" + "="*60)
    print("Creating Trainer...")
    print("="*60)

    trainer = BERTTrainerWithValidation(
        rag_bert,
        train_dataloader=rag_train_loader,
        val_dataloader=rag_val_loader,  # 新增
        vocab=vocab,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        with_cuda=args.with_cuda,
        cuda_devices=args.cuda_devices,
        log_freq=args.log_freq,
        state_dict=state_dict,
        grad_accum_steps=args.grad_accum_steps,
        # Validation参数
        patience=args.patience,
        val_metric=args.val_metric,
        min_delta=args.min_delta
    )

    print("✓ Trainer created")
    print(f"  - RAG TopK: {args.rag_k}")
    print(f"  - Gradient Accumulation Steps: {args.grad_accum_steps}")
    if rag_val_loader:
        print(f"  - Early Stopping: patience={args.patience}, metric={args.val_metric}")

    # ========== 开始训练 ==========
    print("\n" + "="*60)
    print("Training Start!")
    print("="*60)

    for epoch in range(args.epochs):
        # 训练
        train_metrics = trainer.train(epoch)

        # 验证（如果有验证集）
        if rag_val_loader:
            val_metrics = trainer.validate(epoch)

            # 保存最佳模型
            is_best = (trainer.epochs_no_improve == 0)
            trainer.save(epoch, args.output_path, is_best=is_best)

            # 检查Early Stopping
            if trainer.should_stop_early(val_metrics, epoch):
                print(f"\n{'='*60}")
                print(f"Training stopped early at epoch {epoch+1}")
                print(f"Best {args.val_metric}: {trainer.best_val_metric:.4f}")
                print(f"Best model saved: {trainer.best_model_path}")
                print(f"{'='*60}")
                break
        else:
            # 没有验证集，每个epoch都保存
            trainer.save(epoch, args.output_path)

        # 增加训练难度
        rag_train_loader.dataset.add_level()

    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    if rag_val_loader and trainer.best_model_path:
        print(f"Best model: {trainer.best_model_path}")
        print(f"Best {args.val_metric}: {trainer.best_val_metric:.4f}")


if __name__ == "__main__":
    train()
