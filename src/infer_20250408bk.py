import argparse
import random
import torch


from torch.utils.data import DataLoader


from .model import BERT,BERTWithRAG
from .main import BERTInfer
from .dataset import PanelData, InferDataset, WordVocab, RAGInferDataset
from .dataset.rag_train_dataset import rag_collate_fn_with_dataset

def infer():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ref_panel", type=str, help="reference panel for FAISS index.")
    parser.add_argument("--infer_dataset", type=str, help="infer dataset for infer bert")

    parser.add_argument("--infer_panel", type=str, help="load population for infer data")
    parser.add_argument("-w", "--window_path", type=str, help=".csv file which saves all [Window] data")
    
    parser.add_argument("-f", "--freq_path", type=str, help="file which saves all frequency data")
    
    parser.add_argument("--type_path", type=str, help="file mapping genotype to index.")
    parser.add_argument("--pop_path", type=str, help="file mapping population to index.")
    parser.add_argument("--pos_path", type=str, help="file mapping position to index.")

    
    parser.add_argument("-c", "--check_point", type=str, required=True, help="output/bert.model")

    parser.add_argument("-o", "--output_path", required=True, type=str, help="infer_output/")

    parser.add_argument("-d", "--dims", type=int, default=512, help="hidden dimension of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=12, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")

    parser.add_argument("-b", "--infer_batch_size", type=int, default=16, help="number of batch_size")
    parser.add_argument("-n", "--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="infering with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    args = parser.parse_args()

    panel = PanelData.from_file(args.infer_panel)

    print("Initializing Vocab")
    vocab = WordVocab(list(panel.pop_class_dict.keys()))

    print("Loading infer Dataset...")
    # 控制infer_dataset的构建 多次迭代主动mask

    # 首先直接读取pos+path

    infer_dataset = RAGInferDataset.from_file(vocab, args.infer_dataset, args.infer_panel, args.freq_path, args.window_path, 
                                           args.type_path, args.pop_path, args.pos_path, args.ref_panel)

    print("Creating Dataloader")
    #### For onece version
    # infer_data_loader = DataLoader(infer_dataset, batch_size=args.infer_batch_size, num_workers=args.num_workers,
    #    collate_fn=lambda batch_list: rag_collate_fn_with_dataset(batch_list, infer_dataset, 1))

    ### For iter version
    infer_data_loader = DataLoader(infer_dataset, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                                    collate_fn=lambda batch_list: rag_collate_fn_with_dataset(batch_list, infer_dataset, 5))

    print("Loading BERT model")
    bert_rag = BERTWithRAG(len(vocab), dims=args.dims, n_layers=args.layers, attn_heads=args.attn_heads)
    state_dict = torch.load(args.check_point)
    state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}


    print("Creating BERT Infer")
    inferer = BERTInfer(bert_rag, infer_dataloader=infer_data_loader, vocab=vocab,
                        with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, state_dict=state_dict, output_path=args.output_path)
    
    """
    inferer = RAGBERTInfer(bert_rag, 
                        infer_dataloader=infer_data_loader, 
                        ref_vcf_path=args.ref_panel,
                        vocab=vocab,
                        k_retrieve=5,
                        update_freq=3,
                        with_cuda=args.with_cuda, 
                        cuda_devices=args.cuda_devices, 
                        state_dict=state_dict,
                        output_path=args.output_path)
    """

    print("infering Start!")

    vcf_data = inferer.infer()
    #vcf_data = inferer.progressive_infer(step_ratio=0.5)

if __name__ == "__main__":
    infer()
