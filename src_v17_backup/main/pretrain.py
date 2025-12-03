import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import os

from collections import Counter
import pprint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler 

from ..dataset import WordVocab
from ..model import BERTFoundationModel, BERT, BERTWithRAG
from .optim_schedule import ScheduledOptim, FocalLoss, cal_acc, cal_pr, cal_acc_log

MIN_RECON_LOSS = 1e-6

class AdaptiveMaskScheduler:
    def __init__(self, base_ratio=0.1, max_ratio=0.9, strategy='cosine'):
        self.base_ratio = base_ratio
        self.max_ratio = max_ratio
        self.strategy = strategy
    
    def get_mask_ratio(self, epoch, total_epochs=3):
        progress = epoch / (total_epochs - 1)
        if self.strategy == 'linear':
            return self.base_ratio + (self.max_ratio - self.base_ratio) * progress
        elif self.strategy == 'cosine':
            return self.base_ratio + 0.5*(self.max_ratio - self.base_ratio)*(1 - np.cos(np.pi * progress))
        elif self.strategy == 'exponential':
            return self.base_ratio * ((self.max_ratio/self.base_ratio))**progress
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

class BERTTrainer():
    def __init__(self, 
                 bert: BERTWithRAG,
                 train_dataloader: DataLoader = None,
                 vocab: WordVocab = None,
                 lr: float = 1e-5,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_steps=20000,
                 with_cuda: bool = True,
                 cuda_devices=None,
                 log_freq: int = 10,
                 state_dict=None,
                 mask_scheduler=None,
                 grad_accum_steps: int = 1):
        
        self.with_cuda = with_cuda
        if with_cuda and torch.cuda.is_available():
            self.device = torch.device(
                f"cuda:{cuda_devices[0]}" if cuda_devices else 
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True  
        else:
            self.device = torch.device("cpu")
            self.with_cuda = False

        self.bert = bert
        self.model = BERTFoundationModel(bert)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            print("Params' loading succeed.")
        
        if with_cuda and torch.cuda.device_count() > 1 and cuda_devices:
            print(f"Using {len(cuda_devices)} GPUS for BERT")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.is_parallel = True
        else:
            self.is_parallel = False
            
        self.model = self.model.to(self.device)

        self.train_data = train_dataloader
        self.vocab = vocab

        # 严格匹配Dataset字段类型定义
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, 
                        weight_decay=weight_decay, fused=True)
        self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps)

        self.scaler = GradScaler(enabled=with_cuda)
        self.grad_accum_steps = grad_accum_steps
        self.accum_step = 0

        # 根据Dataset的Long/Float定义初始化损失函数
        self.hap_criterion = FocalLoss(gamma=5, reduction='sum').to(self.device)
        self.gt_criterion = FocalLoss(gamma=5, reduction='sum').to(self.device)
        self.recon_critetion = nn.MSELoss().to(self.device)
        
        self.log_freq = log_freq
        self.mask_scheduler = mask_scheduler
        
        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))

    def train(self, epoch):
        if self.mask_scheduler:
            current_ratio = self.mask_scheduler.get_mask_ratio(epoch)
            print(f"\n=== Epoch {epoch+1} 使用mask比例: {current_ratio:.0%} ===")
            self.train_data.dataset.refresh_indices(epoch)
        else:
            print(f"\n=== Training Epoch {epoch+1} ===")
        
        self.model.train()
        
        data_iter = tqdm.tqdm(enumerate(self.train_data),
                            desc=f"EP_Train:{epoch}",
                            total=len(self.train_data),
                            bar_format="{l_bar}{r_bar}")

        eval_dict = {
            "hap_loss": 0.0, "gt_loss": 0.0, "recon_loss": 0.0,
            "hap_correct": 0, "gt_correct": 0,
            "hap_numbers": 0, "gt_numbers": 0,
            "hap_tp": torch.zeros(2, dtype=torch.long).to(self.device),
            "hap_fp": torch.zeros(2, dtype=torch.long).to(self.device),
            "hap_fn": torch.zeros(2, dtype=torch.long).to(self.device),
            "gt_tp": torch.zeros(4, dtype=torch.long).to(self.device),
            "gt_fp": torch.zeros(4, dtype=torch.long).to(self.device),
            "gt_fn": torch.zeros(4, dtype=torch.long).to(self.device)
        }

        for i, data in data_iter:
            # 精确匹配Dataset的字段类型定义
            gpu_data = {
                # 根据Dataset的long_fields定义
                'hap_1': data['hap_1'].to(self.device, non_blocking=True, dtype=torch.long),
                'hap_2': data['hap_2'].to(self.device, non_blocking=True, dtype=torch.long),
                'rag_seg_h1': data['rag_seg_h1'].to(self.device, non_blocking=True, dtype=torch.long),
                'rag_seg_h2': data['rag_seg_h2'].to(self.device, non_blocking=True, dtype=torch.long),
                
                # 根据Dataset的float_fields定义
                'pos': data['pos'].to(self.device, non_blocking=True, dtype=torch.float),
                'af': data['af'].to(self.device, non_blocking=True, dtype=torch.float),
                'af_p': data['af_p'].to(self.device, non_blocking=True, dtype=torch.float),
                'ref': data['ref'].to(self.device, non_blocking=True, dtype=torch.float),
                'het': data['het'].to(self.device, non_blocking=True, dtype=torch.float),
                'hom': data['hom'].to(self.device, non_blocking=True, dtype=torch.float)
            }

            masks = data['mask'].to(self.device, non_blocking=True, dtype=torch.bool)
            labels = {
                'hap_1': data['hap_1_label'].to(self.device, non_blocking=True),
                'hap_2': data['hap_2_label'].to(self.device, non_blocking=True),
                'gt': data['gt_label'].to(self.device, non_blocking=True)
            }

            with autocast(enabled=self.with_cuda, dtype=torch.float16 if self.with_cuda else torch.bfloat16):
                output = self.model(gpu_data)
                
                # 损失计算与Dataset定义严格一致
                hap_1_loss = self.hap_criterion(output[0][masks], labels['hap_1'][masks])
                hap_2_loss = self.hap_criterion(output[1][masks], labels['hap_2'][masks])
                gt_loss = self.gt_criterion(output[2][masks], labels['gt'][masks])
                
                recon_loss1 = self.recon_critetion(output[3][masks], output[5][masks])
                recon_loss2 = self.recon_critetion(output[4][masks], output[6][masks])
                
                if recon_loss1 > MIN_RECON_LOSS and recon_loss2 > MIN_RECON_LOSS:
                    total_loss = (0.2 * hap_1_loss + 0.2 * hap_2_loss + 0.3 * gt_loss + 
                                0.15 * recon_loss1 + 0.15 * recon_loss2)
                else:
                    total_loss = 3 * hap_1_loss + 3 * hap_2_loss + 4 * gt_loss
                
                total_loss /= self.grad_accum_steps

            self.scaler.scale(total_loss).backward()
            self.accum_step += 1

            if self.accum_step % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim_schedule.step()  # 仅更新学习率
                self.optim_schedule.zero_grad()
                self.accum_step = 0

            with torch.no_grad():
                # 指标计算与Dataset定义严格一致
                hap_1_acc, hap_1_tot = cal_acc(output[0].detach().cpu(), 
                                             labels['hap_1'].cpu(), 
                                             data['mask'].cpu())
                hap_2_acc, hap_2_tot = cal_acc(output[1].detach().cpu(), 
                                             labels['hap_2'].cpu(), 
                                             data['mask'].cpu())
                gt_acc, gt_tot = cal_acc(output[2].detach().cpu(), 
                                       labels['gt'].cpu(), 
                                       data['mask'].cpu())
                
                hap1_stats = cal_pr(output[0].detach().cpu(), 
                                  labels['hap_1'].cpu(), 
                                  data['mask'].cpu(), num_classes=2)
                hap2_stats = cal_pr(output[1].detach().cpu(), 
                                  labels['hap_2'].cpu(), 
                                  data['mask'].cpu(), num_classes=2)
                gt_stats = cal_pr(output[2].detach().cpu(), 
                                labels['gt'].cpu(), 
                                data['mask'].cpu(), num_classes=4)
                
                for c in range(2):
                    eval_dict['hap_tp'][c] += (hap1_stats['tp'][c] + hap2_stats['tp'][c]).to(self.device)
                    eval_dict['hap_fp'][c] += (hap1_stats['fp'][c] + hap2_stats['fp'][c]).to(self.device)
                    eval_dict['hap_fn'][c] += (hap1_stats['fn'][c] + hap2_stats['fn'][c]).to(self.device)

                for c in range(4):
                    eval_dict['gt_tp'][c] += gt_stats['tp'][c].to(self.device)
                    eval_dict['gt_fp'][c] += gt_stats['fp'][c].to(self.device)
                    eval_dict['gt_fn'][c] += gt_stats['fn'][c].to(self.device)

                eval_dict['hap_loss'] += (hap_1_loss.item() + hap_2_loss.item())
                eval_dict['gt_loss'] += gt_loss.item()
                eval_dict['recon_loss'] += (recon_loss1.item() + recon_loss2.item())
                eval_dict['hap_correct'] += (hap_1_acc + hap_2_acc)
                eval_dict['gt_correct'] += gt_acc
                eval_dict['hap_numbers'] += (hap_1_tot + hap_2_tot)
                eval_dict['gt_numbers'] += gt_tot

            if i % self.log_freq == 0:
                self._print_log(epoch, i, eval_dict, data_iter)

        print(
            f"EP{epoch}_Train, avg_hap_loss = {eval_dict['hap_loss'] / len(data_iter):.4f}"
            f", avg_hap_acc = {eval_dict['hap_correct'] / eval_dict['hap_numbers'] * 100:.2f}%"
            f", avg_gt_loss = {eval_dict['gt_loss'] / len(data_iter):.4f}"
            f", avg_gt_acc = {eval_dict['gt_correct'] / eval_dict['gt_numbers'] * 100:.2f}%"
        )

    def _print_log(self, epoch, iter_step, eval_dict, data_iter):
        hap_precision, hap_recall, hap_f1 = self.calculate_metrics(
            eval_dict['hap_tp'].float(),
            eval_dict['hap_fp'].float(),
            eval_dict['hap_fn'].float()
        )
        
        log_dict = {
            "epoch": epoch,
            "iter": iter_step,
            "Precision": hap_precision[1].item(),
            "Recall": hap_recall[1].item(),
            "F1": hap_f1[1].item(),
            "avg_hap_loss": eval_dict['hap_loss'] / (iter_step + 1),
            "avg_hap_acc": eval_dict['hap_correct'] / eval_dict['hap_numbers'] * 100,
            "avg_gt_loss": eval_dict['gt_loss'] / (iter_step + 1),
            "avg_gt_acc": eval_dict['gt_correct'] / eval_dict['gt_numbers'] * 100,
            "avg_recon_loss": eval_dict['recon_loss'] / (iter_step + 1)
        }
        
        data_iter.write(pprint.pformat(log_dict))

    def calculate_metrics(self, tp, fp, fn):
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return precision, recall, f1

    def save(self, epoch, file_path="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_rag_20250411_mafData/rag_bert.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        save_dir = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(save_dir, exist_ok=True)
        output_path = file_path + ".ep%d" % epoch + ".pth"
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path