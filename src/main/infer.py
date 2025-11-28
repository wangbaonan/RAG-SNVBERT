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

class ProgressiveInferController:
    def __init__(self, orig_pos, initial_pos, initial_vcf, step_ratio=0.05):
        self.orig_pos = orig_pos.copy()  # 必须保持原始顺序
        self.current_pos = initial_pos.copy()
        self.step_ratio = step_ratio     # 步长为总位点的比例
        self.vcf_data = initial_vcf.copy()
        self.total_count = len(orig_pos)  # 总位点缓存
        
        # 初始化验证
        if step_ratio <= 0 or step_ratio > 1:
            raise ValueError("step_ratio 必须在 (0, 1] 范围内")
        if not np.array_equal(np.sort(initial_pos), initial_pos):
            raise ValueError("初始位置数组必须已排序")

    def get_next_positions(self) -> np.ndarray:
        """获取下一批位点（严格基于总位点比例）"""
        remaining = self.orig_pos[~np.isin(self.orig_pos, self.current_pos)]
        remaining_count = len(remaining)
        
        # 关键修正：计算每步应填补的绝对数量
        step_count = max(1, int(self.total_count * self.step_ratio))
        # 如果剩余不足步长，全选
        return remaining[:min(step_count, remaining_count)]

    def update_state(self, new_pos: np.ndarray):
        """保持原始顺序合并新位点"""
        combined = np.union1d(self.current_pos, new_pos)
        mask = np.isin(self.orig_pos, combined)
        self.current_pos = self.orig_pos[mask]

    def predict_progress(self, max_iter: int) -> Tuple[int, float]:
        """精确预测最终填补量"""
        simulated_pos = self.current_pos.copy()
        remaining = self.orig_pos[~np.isin(self.orig_pos, simulated_pos)]
        step_count = max(1, int(self.total_count * self.step_ratio))
        
        for _ in range(max_iter):
            fill_num = min(step_count, len(remaining))
            if fill_num == 0:
                break
            simulated_pos = np.union1d(simulated_pos, remaining[:fill_num])
            remaining = remaining[fill_num:]
        
        return len(simulated_pos), len(simulated_pos)/self.total_count*100

    @property
    def is_complete(self) -> bool:
        return np.array_equal(self.current_pos, self.orig_pos)


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

    def _core_infer_logic(self, dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        device = self.device
        
        # 预分配结果缓冲区
        hap1_prob_mat = np.zeros_like(self.hap1_prob_mat)
        hap2_prob_mat = np.zeros_like(self.hap2_prob_mat)
        gt_prob_mat = np.zeros_like(self.gt_prob_mat)
        
        # 混合精度配置
        amp_dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        
        with torch.cuda.amp.autocast(dtype=amp_dtype), torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader), 
                                total=len(dataloader),
                                desc="优化推理进度"):
                                
                # 异步数据传输
                gpu_data = {
                    key: data[key].to(device, non_blocking=True) 
                    for key in data if torch.is_tensor(data[key])
                }
                
                # 直接执行推理（移除CUDA图优化）
                output = self.model(gpu_data)[:3]
                
                # 转换到CPU
                output = [t.cpu().float() if t.dtype == torch.half else t.cpu() 
                        for t in output]
                
                # 索引边界保护
                sample_idx = data['sample_idx'].numpy()
                start_idx = data['start_idx'].numpy()
                end_idx = data['end_idx'].numpy()
                
                for idx in range(sample_idx.shape[0]):
                    sample = sample_idx[idx][0]
                    start = start_idx[idx][0]
                    end = end_idx[idx][0]
                    
                    # 新增索引校验
                    assert start >= 0, f"非法起始索引: {start}"
                    assert end <= hap1_prob_mat.shape[0], (
                        f"结束索引越界: end={end}, 矩阵长度={hap1_prob_mat.shape[0]}")
                    
                    # 使用向量化赋值
                    hap1_prob_mat[start:end, sample] = output[0][idx, 1:(end-start+1), 1].numpy()
                    hap2_prob_mat[start:end, sample] = output[1][idx, 1:(end-start+1), 1].numpy()
                    gt_prob_mat[start:end, sample, :] = output[2][idx, 1:(end-start+1), :].numpy()

        # 原子更新
        with torch.no_grad():
            self.hap1_prob_mat = hap1_prob_mat
            self.hap2_prob_mat = hap2_prob_mat
            self.gt_prob_mat = gt_prob_mat

        return VCFProcessingModule.process_gt_prob_mat_with_progress(gt_prob_mat)



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
        pred_count, pred_percent = controller.predict_progress(max_iter)
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
    
    def progressive_infer(self, step_ratio=0.5, max_iter=100):
        controller = ProgressiveInferController(
            self.infer_data.dataset.ori_pos,
            self.infer_data.dataset.pos,
            self.infer_data.dataset.vcf,
            step_ratio
        )
        
        print(f"🚀 开始渐进式推理 | 总位点: {len(controller.orig_pos)} | 初始已知: {len(controller.current_pos)}")
        
        iteration = 1
        while not controller.is_complete and max_iter > 0:
            remaining = len(controller.orig_pos) - len(controller.current_pos)
            coverage = len(controller.current_pos) / len(controller.orig_pos) * 100
            
            # 保持原有日志格式
            print(f"\n🔄 迭代 {iteration} | 剩余位点: {remaining} | 当前覆盖: {coverage:.2f}%")
            
            new_dataset = RAGInferDataset(
                vocab=self.vocab,
                vcf=controller.vcf_data,
                pos=controller.current_pos,
                panel=self.infer_data.dataset.panel,
                freq=self.infer_data.dataset.freq,
                type_to_idx=self.infer_data.dataset.type_to_idx,
                pop_to_idx=self.infer_data.dataset.pop_to_idx,
                pos_to_idx=self.infer_data.dataset.pos_to_idx,
                ref_vcf_path=self.infer_data.dataset.ref_vcf_path,
                build_index=True
            )
            new_dataloader = DataLoader(
                new_dataset,
                batch_size=self.infer_data.batch_size,
                num_workers=self.infer_data.num_workers,
                collate_fn=lambda batch_list: rag_collate_fn_with_dataset(batch_list, new_dataset, 5)
            )
            
            current_vcf = self._core_infer_logic(new_dataloader)
            
            new_pos = controller.get_next_positions()
            controller.update_state(new_pos)
            
            # 新增关键判断
            if len(new_pos) == 0:
                print("⚠️ 无新位点可添加，提前终止")
                break
                
            print(f"✅ 新增 {len(new_pos)} 个位点 | 累计已知: {len(controller.current_pos)}")
            
            iteration += 1
            max_iter -= 1

        # 最终强制覆盖保障（修正逻辑）
        if not controller.is_complete:
            print("\n🔚 进入最终补全阶段")
            final_pos = controller.orig_pos[~np.isin(controller.orig_pos, controller.current_pos)]
            print(f"🔥 正在加载最后 {len(final_pos)} 个位点")
            
            # 关键修正：必须通过update_state来保证顺序
            controller.update_state(final_pos)
            
            # 重新构建数据集（使用更新后的current_pos）
            final_dataset = RAGInferDataset(
                vocab=self.vocab,
                vcf=controller.vcf_data,
                pos=controller.current_pos,  # 这里已经是合并后的有序数组
                panel=self.infer_data.dataset.panel,
                freq=self.infer_data.dataset.freq,
                type_to_idx=self.infer_data.dataset.type_to_idx,
                pop_to_idx=self.infer_data.dataset.pop_to_idx,
                pos_to_idx=self.infer_data.dataset.pos_to_idx,
                ref_vcf_path=self.infer_data.dataset.ref_vcf_path,
                build_index=True
            )
            final_loader = DataLoader(
                final_dataset,
                batch_size=self.infer_data.batch_size,
                num_workers=self.infer_data.num_workers,
                collate_fn=lambda batch_list: rag_collate_fn_with_dataset(batch_list, final_dataset, 5)
            )
            controller.vcf_data = self._core_infer_logic(final_loader)

        # 新增最终验证（确保完全对齐）
        if not np.array_equal(controller.current_pos, controller.orig_pos):
            missing = len(controller.orig_pos) - len(controller.current_pos)
            raise RuntimeError(f"❌ 最终验证失败！缺失位点数: {missing}")
        print(f"\n✅ 最终验证通过 | 总位点完全对齐")
        
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
    
