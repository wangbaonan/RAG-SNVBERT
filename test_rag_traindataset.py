#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset import PanelData, TrainDataset, WordVocab, RAGTrainDataset
from src.dataset.rag_train_dataset import rag_collate_fn_with_dataset

def calc_diff(hap_pred: torch.Tensor, hap_label: torch.Tensor) -> float:
    """
    简易示例：比较两个hap序列的差异率（0/1 vs 0/1）。
    遇到 -1/缺失时跳过。
    """
    pred_np = hap_pred.cpu().numpy()
    label_np = hap_label.cpu().numpy()

    mask_pred = (pred_np >= 0)
    mask_label = (label_np >= 0)
    valid_mask = mask_pred & mask_label

    valid_pred = pred_np[valid_mask]
    valid_label = label_np[valid_mask]

    if valid_pred.size == 0:
        return 0.0

    diff_num = (valid_pred != valid_label).sum()
    diff_rate = diff_num / valid_pred.size
    return diff_rate

'''
# 添加的校验代码
def validate_consistency(dataset):
    # 随机抽查10个样本
    for _ in range(10):
        idx = np.random.randint(len(dataset))
        sample = dataset[idx]
        win_idx = sample['window_idx'].item()
        
        # 验证mask一致性
        assert torch.all(sample['mask'] == torch.LongTensor(dataset.padded_masks[win_idx])), "Mask不一致"
        
        # 验证物理坐标
        pos_in_vcf = dataset.pos[dataset.window.window_info[win_idx][0]:dataset.window.window_info[win_idx][1]]
        assert dataset.physical_pos[win_idx][0] == pos_in_vcf[0], "起始坐标错误"
        assert dataset.physical_pos[win_idx][1] == pos_in_vcf[-1], "终止坐标错误"
    
    print("✅ 所有一致性检查通过")
'''

def calc_diff(hap_pred: torch.Tensor, hap_label: torch.Tensor) -> float:
    """改进后的差异率计算，忽略填充值"""
    pred_np = hap_pred.cpu().numpy()
    label_np = hap_label.cpu().numpy()
    
    # 有效位点：非填充且非SOS
    valid_mask = (label_np != 0) & (label_np != 2)  # 0=pad, 2=SOS
    
    valid_pred = pred_np[valid_mask]
    valid_label = label_np[valid_mask]
    
    if valid_pred.size == 0:
        return 0.0
    
    return (valid_pred != valid_label).mean()


def main_rag_test(args):
    """
    1) 加载 RAGTrainDataset （多索引或单索引均可）
    2) 建立 DataLoader (collate_fn)
    3) 遍历 dataset, 统计检索耗时 & 误差
    """

    panel = PanelData.from_file(args.train_panel)

    print("Initializing Vocab")
    vocab = WordVocab(list(panel.pop_class_dict.keys()))
    print("Mask Index:")
    print(vocab.mask_index)
    print("Loading RAGTrainDataset ...")
    rag_dataset = RAGTrainDataset.from_file(
        vocab=vocab,  # 如果你需要传 vocab, 在这里修改
        vcfpath=args.train_dataset,
        panelpath=args.train_panel,
        freqpath=args.freq_path,
        windowpath=args.window_path,
        typepath=args.type_path,
        poppath=args.pop_path,
        pospath=args.pos_path,
        ref_vcf_path=args.refpanel_path,
        build_ref_data=True,
        n_gpu=1
    )
    
    # 不做真正训练，此处仅 batch_size=32 测试
    batch_size = 32
    rag_loader = DataLoader(
        rag_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 可根据需求调整
        collate_fn=lambda batch: rag_collate_fn_with_dataset(batch, rag_dataset, 5)
    )

    # 添加词表验证
    sample = rag_dataset[0]
    print("\n=== 词表映射验证 ===")
    print("训练数据示例:")
    print(f"  hap_1[:25]: {sample['hap_1'][:25].numpy().tolist()}")
    print(f"  mask[:25]:   {sample['mask'][:25].numpy().tolist()}")
    
    ref_sample = rag_dataset.ref_data_windows[0][0,:25]
    print("\n参考数据示例:")
    print(f"  ref_data[:25]: {ref_sample.tolist()}")

    total_samples = 0
    total_retrieval_time = 0.0  # 累计检索总耗时
    total_diff_h1 = 0.0
    total_diff_h2 = 0.0
    total_count = 0

    #for batch_idx, batch in enumerate(rag_loader):
    #    # batch 已包含 hap_1, hap_2, 以及 collate_fn 里写回的 rag_seg_h1, rag_seg_h2, ...
    #    B = batch['hap_1'].size(0)
    #    total_samples += B
    #    print("hap1:")
    #    print(batch['hap_1'])
    #    print("rag_seg_h1:")
    #    print(batch['rag_seg_h1'])
    #    print("orig_label:")
    #    print(batch['hap_1_label'])

    for batch_idx, batch in enumerate(rag_loader):
        B = batch['hap_1'].size(0)
        total_samples += B
    
        print("\n" + "="*50)
        print(f"▣ 批次 {batch_idx} 第一个样本前100位点检查")
        print("="*50)
    
        # 选择第一个样本（索引0）
        sample_idx = 0
    
        # 检查 hap_1
        hap1 = batch['hap_1'][sample_idx, :100].cpu().numpy()  # (100,)
        print("\n=== hap_1 (输入单体型) ===")
        print("原始形状", batch['hap_1'].shape)
        print("形状:", hap1.shape)
        print("内容:", hap1.tolist())  # 转换为列表更易读
    
        # 检查 rag_seg_h1
        rag_seg_h1 = batch['rag_seg_h1'][sample_idx][0, :100].cpu().numpy()
        print("\n=== rag_seg_h1 (检索参考序列) ===")
        print("原始形状", batch['rag_seg_h1'].shape)
        print("形状:", rag_seg_h1.shape)
        print("内容:", rag_seg_h1.tolist())

        # 检查 rag_seg_h1
        rag_seg_h1 = batch['rag_seg_h1'][sample_idx][1, :100].cpu().numpy()

        print("\n=== rag_seg_h1 topk no.2 (检索参考序列) ===")
        print("形状:", rag_seg_h1.shape)
        print("内容:", rag_seg_h1.tolist())

        # 检查 rag_seg_h1
        rag_seg_h1 = batch['rag_seg_h1'][sample_idx][2, :100].cpu().numpy()

        print("\n=== rag_seg_h1 topk no.3 (检索参考序列) ===")
        print("形状:", rag_seg_h1.shape)
        print("内容:", rag_seg_h1.tolist())

        # 检查 rag_seg_h1
        rag_seg_h1 = batch['rag_seg_h1'][sample_idx][3, :100].cpu().numpy()

        print("\n=== rag_seg_h1 topk no.4 (检索参考序列) ===")
        print("形状:", rag_seg_h1.shape)
        print("内容:", rag_seg_h1.tolist())

        # 检查 rag_seg_h1
        rag_seg_h1 = batch['rag_seg_h1'][sample_idx][4, :100].cpu().numpy()

        print("\n=== rag_seg_h1 topk no.4 (检索参考序列) ===")
        print("形状:", rag_seg_h1.shape)
        print("内容:", rag_seg_h1.tolist())
    
        # 检查标签
        hap1_label = batch['hap_1_label'][sample_idx, :100].cpu().numpy()
        print("\n=== hap_1_label (真实标签) ===")
        print("形状:", hap1_label.shape)
        print("内容:", hap1_label.tolist())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 这里模拟你 run.py 的常见传参
    parser.add_argument("--train_dataset", type=str, required=True,
                        help="Path to train dataset .h5")
    parser.add_argument("--train_panel", type=str, required=True,
                        help="Path to train panel info file")
    parser.add_argument("--freq_path", type=str, required=True,
                        help="Path to freq npy")
    parser.add_argument("--window_path", type=str, required=True,
                        help="Path to window .csv file")
    parser.add_argument("--type_path", type=str, required=True,
                        help="Path to type_to_idx.bin")
    parser.add_argument("--pop_path", type=str, required=True,
                        help="Path to pop_to_idx.bin")
    parser.add_argument("--pos_path", type=str, required=True,
                        help="Path to pos_to_idx.bin")
    parser.add_argument("--refpanel_path", type=str, required=True,
                        help="Path to reference panel .vcf or .vcf.gz")

    # 你可以再添加更多可选参数，如--batch_size, --num_workers 等
    # parser.add_argument("--batch_size", type=int, default=32)
    # ...

    args = parser.parse_args()

    # 运行测试
    main_rag_test(args)
