import numpy as np
import faiss
import tqdm
import torch
import torch.amp
import torch.nn as nn
from pathlib import Path
import os
import allel
import h5py
from typing import Optional, Dict, Tuple

from torch.utils.data import DataLoader

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from ..dataset import WordVocab, InferDataset, RAGInferDataset
from ..model import BERTFoundationModel, BERT
from ..dataset.utils  import VCFProcessingModule
from ..dataset.rag_train_dataset import rag_collate_fn_with_dataset
INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030

'''
# 辅助类
class ProgressiveInferController:
    """渐进式推理控制器（完全独立类）"""
    def __init__(self, 
                 orig_pos: np.ndarray,
                 initial_pos: np.ndarray,
                 initial_vcf: np.ndarray,
                 step_ratio: float = 0.1):
        """
        Args:
            orig_pos: 全量位点坐标 (n_variants,)
            initial_pos: 初始已知位点 (m_variants,)
            initial_vcf: 初始VCF数据 (n_variants, n_samples, 2)
            step_ratio: 每次新增位点的比例
        """
        # 坐标管理
        self.orig_pos = orig_pos
        self.current_pos = initial_pos
        self.step_ratio = step_ratio
        
        # 数据管理
        self.vcf_data = initial_vcf.copy()
        
    def get_next_positions(self) -> np.ndarray:
        """获取下一批需要开放的位点"""
        remaining = self.orig_pos[~np.isin(self.orig_pos, self.current_pos)]
        n_new = max(1, int(len(remaining) * self.step_ratio))
        return remaining[:n_new]
    
    def update_state(self, new_pos: np.ndarray):
        """更新状态（不修改数据）"""
        self.current_pos = np.union1d(self.current_pos, new_pos)
    
    @property
    def is_complete(self) -> bool:
        # 优先检查长度差异快速返回
        if len(self.current_pos) != len(self.orig_pos):
            return False
        # 精确检查是否包含所有元素
        return np.all(np.isin(self.orig_pos, self.current_pos))
'''

class ProgressiveInferController:
    def __init__(self, 
                 orig_pos: np.ndarray,
                 initial_pos: np.ndarray,
                 initial_vcf: np.ndarray,
                 step_ratio: float = 0.2):
        # 初始化逻辑保持不变
        self.orig_pos = orig_pos
        self.current_pos = initial_pos
        self.step_ratio = step_ratio
        self.vcf_data = initial_vcf.copy()

    """
    def get_next_positions(self) -> np.ndarray:
        ### 按全量位点总数的比例获取下一批位点
        remaining = self.orig_pos[~np.isin(self.orig_pos, self.current_pos)]
        
        # 核心修改点：基于全量位点的比例计算新增数量
        n_new_total = max(1, int(len(self.orig_pos) * self.step_ratio))
        n_new = min(n_new_total, len(remaining))  # 不超过剩余数量
        
        return remaining[:n_new]  # 保留原始顺序
    """

    def get_next_positions(self) -> np.ndarray:
        ### 安全的新增位点选择
        # 精确计算剩余位点
        remaining_mask = ~np.isin(self.orig_pos, self.current_pos)
        remaining = self.orig_pos[remaining_mask]
        
        # 基于剩余位点的实际数量计算新增
        n_new = max(1, int(len(remaining) * self.step_ratio))
        return remaining[:n_new]
    
    """
    def update_state(self, new_pos: np.ndarray):
        # 原有逻辑保持不变
        self.current_pos = np.union1d(self.current_pos, new_pos)
    """

    def update_state(self, new_pos: np.ndarray):
        """带重复校验的状态更新"""
        # 确保新位点存在且唯一
        valid_new = np.intersect1d(new_pos, self.orig_pos, assume_unique=True)
        self.current_pos = np.unique(np.concatenate([self.current_pos, valid_new]))
        
        # 安全截断
        if len(self.current_pos) > len(self.orig_pos):
            self.current_pos = np.intersect1d(self.current_pos, self.orig_pos)
            
    """
    @property
    def is_complete(self) -> bool:
        remaining = len(self.orig_pos) - len(self.current_pos)
        # 当剩余位点 <= 1000 或完全覆盖时终止
        return remaining <= 1000 or remaining == 0 
    """

    @property
    def is_complete(self) -> bool:
        remaining = len(self.orig_pos) - len(self.current_pos)
        return remaining <= 0  # 严格非负终止

class BERTInfer():
    """
    This class contains all the information about inferring.
    """

    def __init__(self,
                 bert: BERT,
                 infer_dataloader: DataLoader = None,
                 vocab : WordVocab = None,
                 with_cuda: bool = True, 
                 cuda_devices = None, 
                 log_freq: int = 10,
                 state_dict = None,
                 output_path = None
                 ):
        """
        Attributes:

            bert : BERT model which you want to infer.
            infer_dataloader : infer dataset data loader.
            with_cuda : traning with cuda.
            log_freq : logging frequency of the batch iteration.
        """
        # Setup cuda device for BERT infering
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:" + str(cuda_devices[0]) if cuda_condition and cuda_devices is not None else "cpu")

        # This BERT model will be saved every epoch.
        self.bert = bert

        self.vocab = vocab

        self.output_path = output_path

        # Initialize the BERT Language Model.
        self.model = BERTFoundationModel(bert)

        # Distributed GPU infering if CUDA can detect more than 1 GPU.
        self.model.load_state_dict(state_dict=state_dict)
        print("Params' loading succeed.")

        # Load model into GPU.
        self.model = self.model.to(self.device)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % len(cuda_devices))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Set infer dataloader.
        self.infer_data = infer_dataloader

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.hap1_prob_mat = np.zeros((infer_dataloader.dataset.ori_pos.shape[0], infer_dataloader.dataset.vcf.shape[1]), dtype=np.float32)
        self.hap2_prob_mat = np.zeros((infer_dataloader.dataset.ori_pos.shape[0], infer_dataloader.dataset.vcf.shape[1]), dtype=np.float32)
        self.gt_prob_mat = np.zeros((infer_dataloader.dataset.ori_pos.shape[0], infer_dataloader.dataset.vcf.shape[1], 4), dtype=np.float32)

    def _core_infer_logic(self, dataloader: DataLoader) -> np.ndarray:
        """封装的原始推理核心逻辑（供infer()和progressive_infer()复用）"""
        self.model.eval()
    
        # 初始化临时存储（避免污染实例变量）
        hap1_prob_mat = np.zeros_like(self.hap1_prob_mat)
        hap2_prob_mat = np.zeros_like(self.hap2_prob_mat)
        gt_prob_mat = np.zeros_like(self.gt_prob_mat)

        data_iter = tqdm.tqdm(enumerate(dataloader),
                            desc="INFER",
                            total=len(dataloader),
                            bar_format="{l_bar}{r_bar}")

        data_in_gpu = ['hap_1', 'hap_2', 'pos', 'af', 'af_p', 'ref', 'het', 'hom', 'rag_seg_h1', 'rag_seg_h2']

        for i, data in data_iter:
            gpu_data = {key: data[key].to(self.device) for key in data_in_gpu}
            with torch.no_grad():
                output = self.model.forward(gpu_data)[:3]

            for idx, tensor in enumerate(output):
                output[idx] = tensor.cpu().numpy()

            sample_idx = data['sample_idx'].numpy()
            start_idx = data['start_idx'].numpy()
            end_idx = data['end_idx'].numpy()

            for idx in range(sample_idx.shape[0]):
                sample_ = sample_idx[idx][0]
                start_ = start_idx[idx][0]
                end_ = end_idx[idx][0]
                len_ = end_ - start_ + 1

                hap1_prob_mat[start_:end_, sample_] = output[0][idx, 1:len_, 1]
                hap2_prob_mat[start_:end_, sample_] = output[1][idx, 1:len_, 1]
                gt_prob_mat[start_:end_, sample_, :] = output[2][idx, 1:len_, :]
                
            self.hap1_prob_mat = hap1_prob_mat
            self.hap2_prob_mat = hap2_prob_mat
            self.gt_prob_mat = gt_prob_mat

        return VCFProcessingModule.process_gt_prob_mat_with_progress(gt_prob_mat)

    """
    def _core_infer_logic(self, dataloader: DataLoader) -> np.ndarray:
        ### 带mask位点检查的核心逻辑
        self.model.eval()
        
        hap1_prob_mat = np.zeros_like(self.hap1_prob_mat)
        hap2_prob_mat = np.zeros_like(self.hap2_prob_mat)
        gt_prob_mat = np.zeros_like(self.gt_prob_mat)

        data_iter = tqdm.tqdm(enumerate(dataloader),
                            desc="INFER",
                            total=len(dataloader),
                            bar_format="{l_bar}{r_bar}")

        data_in_gpu = ['hap_1', 'hap_2', 'pos', 'af', 'af_p', 'ref', 'het', 'hom', 'rag_seg_h1', 'rag_seg_h2']

        for i, data in data_iter:
            gpu_data = {key: data[key].to(self.device) for key in data_in_gpu}
            with torch.no_grad():
                output = self.model.forward(gpu_data)[:3]

            # 转换输出到numpy
            output = [t.cpu().numpy() for t in output]

            for idx in range(data['hap_1'].size(0)):
                sample_id = data['sample_idx'][idx][0].item()
                start_pos = data['start_idx'][idx][0].item()
                end_pos = data['end_idx'][idx][0].item()
                
                # 提取原始序列和mask位置
                h1 = data['hap_1'][idx].cpu().numpy().flatten()
                h2 = data['hap_2'][idx].cpu().numpy().flatten()
                mask_positions = np.where((h1 == 4) | (h2 == 4))[0]
                
                # 新增唯一值检查逻辑
                if len(mask_positions) == 0:
                    unique_h1 = np.unique(h1)
                    unique_h2 = np.unique(h2)
                    #print(f"\n警告：样本 {sample_id} 未检测到mask位点(4)")
                    #print(f"HAP1 唯一值：{unique_h1}")
                    #print(f"HAP2 唯一值：{unique_h2}")
                    #print("请检查输入数据是否包含mask标记（值为4的元素）")
                    continue  # 跳过后续mask处理

                # 原mask处理逻辑保持不变
                if len(mask_positions) > 0:
                    print(f"\n样本 {sample_id} 检测到{len(mask_positions)}个mask位点:")
                    
                    pred_h1 = output[0][idx, 1:, 1]
                    pred_h2 = output[1][idx, 1:, 1]
                    
                    for pos in mask_positions[:5]:
                        if pos >= len(pred_h1):
                            continue
                        
                        start = max(0, pos-5)
                        end = min(len(h1), pos+6)
                        
                        if h1[pos] == 4:
                            print(f"[HAP1] 位点 {pos+start_pos}:")
                            print(f"上下文: {' '.join(map(str, h1[start:end]))}")
                            print(f"预测概率: {pred_h1[pos]:.2f}")
                            print(f"RAG参考: {' '.join(map(str, data['rag_seg_h1'][idx][0].cpu().numpy().flatten()[start:end]))}")
                        
                        if h2[pos] == 4:
                            print(f"[HAP2] 位点 {pos+start_pos}:")
                            print(f"上下文: {' '.join(map(str, h2[start:end]))}")
                            print(f"预测概率: {pred_h2[pos]:.2f}") 
                            print(f"RAG参考: {' '.join(map(str, data['rag_seg_h2'][idx][0].cpu().numpy().flatten()[start:end]))}")

            # 存储预测结果（保持不变）
            sample_idx = data['sample_idx'].numpy()
            start_idx = data['start_idx'].numpy()
            end_idx = data['end_idx'].numpy()

            for idx in range(sample_idx.shape[0]):
                sample_ = sample_idx[idx][0]
                start_ = start_idx[idx][0]
                end_ = end_idx[idx][0]
                len_ = end_ - start_ + 1

                start_ = max(0, start_)
                end_ = min(hap1_prob_mat.shape[0]-1, end_)
                valid_len = end_ - start_ + 1
                
                hap1_prob_mat[start_:end_+1, sample_] = output[0][idx, 1:valid_len+1, 1][:valid_len]
                hap2_prob_mat[start_:end_+1, sample_] = output[1][idx, 1:valid_len+1, 1][:valid_len]
                gt_prob_mat[start_:end_+1, sample_, :] = output[2][idx, 1:valid_len+1, :][:valid_len]

        self.hap1_prob_mat = hap1_prob_mat
        self.hap2_prob_mat = hap2_prob_mat
        self.gt_prob_mat = gt_prob_mat

        return VCFProcessingModule.process_gt_prob_mat_with_progress(gt_prob_mat)
    """



    def infer(self):
        """Loop over the dataloader for infering.
        """
        self.model.eval()
        mode_code = "INFER"

        # Set tqdm bar
        data_iter = tqdm.tqdm(enumerate(self.infer_data),
                              desc="EP_%s" % (mode_code),
                              total=len(self.infer_data),
                              bar_format="{l_bar}{r_bar}")

        # Data in GPU.
        data_in_gpu = ['hap_1', 'hap_2', 'pos', 'af', 'af_p', 'ref', 'het', 'hom', 'rag_seg_h1', 'rag_seg_h2']


        for i, data in data_iter:
            # infer.
            gpu_data = {key: data[key].to(self.device) for key in data_in_gpu}
            with torch.no_grad():
                output  = self.model.forward(gpu_data)[:3]

            for idx, tensor in enumerate(output):
                output[idx] = tensor.cpu().numpy()
            
            sample_idx = data['sample_idx'].numpy()
            start_idx = data['start_idx'].numpy()
            end_idx = data['end_idx'].numpy()

            for idx in range(sample_idx.shape[0]):      # Batch size
                sample_ = sample_idx[idx][0]
                start_ = start_idx[idx][0]
                end_ = end_idx[idx][0]
                
                len_ = end_ - start_ + 1

                self.hap1_prob_mat[start_:end_, sample_] = output[0][idx, 1:len_, 1]
                self.hap2_prob_mat[start_:end_, sample_] = output[1][idx, 1:len_, 1]
                self.gt_prob_mat[start_:end_, sample_, :] = output[2][idx, 1:len_, :]
                #print("\n===== gt_prob_mat 验证 =====")

        vcf_data = VCFProcessingModule.process_gt_prob_mat_with_progress(self.gt_prob_mat)
        self.save_npy_result()
        return vcf_data

    def infer(self):
        """原始全量推理方法（保持完全一致）"""
        # 通过核心逻辑获取结果
        vcf_data = self._core_infer_logic(self.infer_data)
    
        # 以下保持原始后处理逻辑
        self.save_npy_result()
        return vcf_data

    def progressive_infer(self, 
                     step_ratio: float = 0.1,
                     max_iter: int = 100) -> np.ndarray:
        """渐进式推理入口"""
        # 初始化控制器
        iteration = 1
        controller = ProgressiveInferController(
            orig_pos=self.infer_data.dataset.ori_pos.copy(),
            initial_pos=self.infer_data.dataset.pos.copy(),
            initial_vcf=self.infer_data.dataset.vcf.copy(),
            step_ratio=step_ratio
        )
        print(f"🚀 开始渐进式推理 | 总位点: {len(controller.orig_pos)} | 初始已知: {len(controller.current_pos)}")
        # 渐进式循环
        while not controller.is_complete and max_iter > 0:
            # 打印当前状态
            remaining = len(controller.orig_pos) - len(controller.current_pos)
            print(f"🔄 迭代 {iteration} | 剩余位点: {remaining} | 最大剩余迭代: {max_iter}")

            # 动态构建数据集
            new_dataset = RAGInferDataset(
                vocab=self.vocab,
                vcf=controller.vcf_data,
                pos=controller.current_pos,
                panel=self.infer_data.dataset.panel,
                freq=self.infer_data.dataset.freq,
                window=self.infer_data.dataset.window,
                type_to_idx=self.infer_data.dataset.type_to_idx,
                pop_to_idx=self.infer_data.dataset.pop_to_idx,
                pos_to_idx=self.infer_data.dataset.pos_to_idx,
                ref_vcf_path=self.infer_data.dataset.ref_vcf_path,
                build_index=True
            )
            new_dataloader = DataLoader(new_dataset,
                                    batch_size=self.infer_data.batch_size,
                                    num_workers=self.infer_data.num_workers,
                                    collate_fn=lambda batch_list: rag_collate_fn_with_dataset(batch_list, new_dataset, 5))
        
            # 执行推理
            current_vcf = self._core_infer_logic(new_dataloader)
        
            # 更新状态
            new_pos = controller.get_next_positions()
            controller.vcf_data = current_vcf
            controller.update_state(new_pos)
            # 打印进度
            coverage = len(controller.current_pos) / len(controller.orig_pos) * 100
            print(f"✅ 当前覆盖: {coverage:.2f}% | 累计已知位点: {len(controller.current_pos)}\n")
            
            iteration += 1
            max_iter -= 1
        self.save_npy_result()
        print(f"🎉 推理完成！最终覆盖: {len(controller.current_pos)}/{len(controller.orig_pos)} 个位点")
        return controller.vcf_data

    def save_npy_result(self) -> None:
        """Call this func to save results from self.infer().
        """
        np.save(self.output_path + "/HAP1.npy", self.hap1_prob_mat)
        print("HAP1 saved.")

        np.save(self.output_path + "/HAP2.npy", self.hap2_prob_mat)
        print("HAP2 saved.")

        np.save(self.output_path + "/GT.npy", self.gt_prob_mat)
        print("GT saved.")

        np.save(self.output_path + "/POS.npy", self.infer_data.dataset.ori_pos)
        print("POS saved.")

        np.save(self.output_path + "/POS_Flag.npy", self.infer_data.dataset.position_needed)
        print("POS_FLAG saved.")
    
