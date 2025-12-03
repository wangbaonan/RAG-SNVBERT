import argparse
import random
import torch


from torch.utils.data import DataLoader


from .model import BERT, BERTWithRAG
from .main import BERTTrainer
from .dataset import PanelData, TrainDataset, WordVocab, RAGTrainDataset
from .dataset.rag_train_dataset import rag_collate_fn_with_dataset


# 设置随机种子
seed = 1234

# 设置 PyTorch 的种子
torch.manual_seed(seed)

# 如果使用 GPU，设置 CUDA 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 设置 cuDNN 后端的随机性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置 Python 的随机数生成器
random.seed(seed)


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str, help="train dataset for train bert")

    parser.add_argument("--train_panel", type=str, help="load population for train data")
    
    parser.add_argument("-f", "--freq_path", type=str, help="file which saves all frequency data")
    
    parser.add_argument("-w", "--window_path", type=str, help=".csv file which saves all [Window] data")

    parser.add_argument("--type_path", type=str, help="file mapping genotype to index.")
    parser.add_argument("--pop_path", type=str, help="file mapping population to index.")
    parser.add_argument("--pos_path", type=str, help="file mapping position to index.")
    parser.add_argument("--refpanel_path", type=str, help="rag panel path.")

    
    parser.add_argument("-c", "--check_point", type=str, default=None, help="output/bert.model")

    parser.add_argument("-o", "--output_path", required=True, type=str, help="output/bert.model")

    parser.add_argument("-d", "--dims", type=int, default=512, help="hidden dimension of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")

    parser.add_argument("-b", "--train_batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    panel = PanelData.from_file(args.train_panel)

    print("Initializing Vocab")
    vocab = WordVocab(list(panel.pop_class_dict.keys()))

    print("Loading Train Dataset...")
    train_dataset = TrainDataset.from_file(vocab, args.train_dataset, args.train_panel, args.freq_path, args.window_path,
                                           args.type_path, args.pop_path, args.pos_path)

    #  构建 RAGTrainDataset (多索引) NEWADD
    rag_train_dataset = RAGTrainDataset.from_file(
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

    print("Creating Dataloader")

    # RAG Version NEWADD
    rag_train_data_loader = DataLoader(
        rag_train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch_list: rag_collate_fn_with_dataset(batch_list, rag_train_dataset, 3)# FAISS TopK K=5
    )

    print("Building BERT model")
    rag_bert = BERTWithRAG(len(vocab), dims=args.dims, n_layers=args.layers, attn_heads=args.attn_heads)

    if args.check_point is not None:
        state_dict = torch.load(args.check_point)
        state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    else:
        state_dict = None

    print("Creating BERT Trainer")
    trainer = BERTTrainer(rag_bert, train_dataloader=rag_train_data_loader, vocab=vocab,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, state_dict=state_dict)

    print("Training Start")

    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)
        rag_train_data_loader.dataset.add_level()


if __name__ == "__main__":
    train()
