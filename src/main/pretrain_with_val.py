"""
BERTTrainer with Validation Support
在原有BERTTrainer基础上添加完整的validation支持
"""

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


class BERTTrainerWithValidation():
    """带Validation支持的BERT Trainer"""

    def __init__(self,
                 bert: BERTWithRAG,
                 train_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,  # 新增：验证数据
                 vocab: WordVocab = None,
                 lr: float = 1e-5,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_steps=20000,
                 with_cuda: bool = True,
                 cuda_devices=None,
                 log_freq: int = 10,
                 state_dict=None,
                 grad_accum_steps: int = 1,
                 # Validation相关参数
                 patience: int = 5,           # Early stopping patience
                 val_metric: str = 'f1',     # 监控指标：'f1', 'accuracy', 'loss'
                 min_delta: float = 0.001):  # 最小改进阈值

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
        self.val_data = val_dataloader  # 新增
        self.vocab = vocab

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas,
                        weight_decay=weight_decay, fused=True)
        self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps)

        self.scaler = GradScaler(enabled=with_cuda)
        self.grad_accum_steps = grad_accum_steps
        self.accum_step = 0

        self.hap_criterion = FocalLoss(gamma=5, reduction='sum').to(self.device)
        self.gt_criterion = FocalLoss(gamma=5, reduction='sum').to(self.device)
        self.recon_critetion = nn.MSELoss().to(self.device)

        self.log_freq = log_freq

        # Validation & Early Stopping 参数
        self.patience = patience
        self.val_metric = val_metric
        self.min_delta = min_delta
        self.best_val_metric = -np.inf if val_metric in ['f1', 'accuracy'] else np.inf
        self.epochs_no_improve = 0
        self.best_model_path = None

        print("Total Parameters:", sum(p.nelement() for p in self.model.parameters()))
        if self.val_data:
            print(f"✓ Validation enabled with early stopping (patience={patience}, metric={val_metric})")

    def train(self, epoch):
        """训练一个epoch"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} - TRAINING")
        print(f"{'='*60}")

        return self._run_epoch(epoch, self.train_data, train=True)

    def validate(self, epoch):
        """验证一个epoch"""
        if self.val_data is None:
            return None

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} - VALIDATION")
        print(f"{'='*60}")

        return self._run_epoch(epoch, self.val_data, train=False)

    def _run_epoch(self, epoch, dataloader, train=True):
        """运行一个epoch（训练或验证）"""
        if train:
            self.model.train()
            desc = f"EP_Train:{epoch}"
        else:
            self.model.eval()
            desc = f"EP_Val:{epoch}"

        data_iter = tqdm.tqdm(enumerate(dataloader),
                            desc=desc,
                            total=len(dataloader),
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
            gpu_data = {
                'hap_1': data['hap_1'].to(self.device, non_blocking=True, dtype=torch.long),
                'hap_2': data['hap_2'].to(self.device, non_blocking=True, dtype=torch.long),
                'rag_seg_h1': data['rag_seg_h1'].to(self.device, non_blocking=True, dtype=torch.long),
                'rag_seg_h2': data['rag_seg_h2'].to(self.device, non_blocking=True, dtype=torch.long),
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

            if train:
                # 训练模式：前向+反向传播
                with autocast(enabled=self.with_cuda, dtype=torch.float16 if self.with_cuda else torch.bfloat16):
                    output = self.model(gpu_data)

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
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()
                    self.accum_step = 0
            else:
                # 验证模式：只前向传播
                with torch.no_grad():
                    with autocast(enabled=self.with_cuda, dtype=torch.float16 if self.with_cuda else torch.bfloat16):
                        output = self.model(gpu_data)

                        hap_1_loss = self.hap_criterion(output[0][masks], labels['hap_1'][masks])
                        hap_2_loss = self.hap_criterion(output[1][masks], labels['hap_2'][masks])
                        gt_loss = self.gt_criterion(output[2][masks], labels['gt'][masks])

                        recon_loss1 = self.recon_critetion(output[3][masks], output[5][masks])
                        recon_loss2 = self.recon_critetion(output[4][masks], output[6][masks])

            # 计算指标（训练和验证都需要）
            with torch.no_grad():
                hap_1_acc, hap_1_tot = cal_acc(output[0].detach().cpu(), labels['hap_1'].cpu(), data['mask'].cpu())
                hap_2_acc, hap_2_tot = cal_acc(output[1].detach().cpu(), labels['hap_2'].cpu(), data['mask'].cpu())
                gt_acc, gt_tot = cal_acc(output[2].detach().cpu(), labels['gt'].cpu(), data['mask'].cpu())

                hap1_stats = cal_pr(output[0].detach().cpu(), labels['hap_1'].cpu(), data['mask'].cpu(), num_classes=2)
                hap2_stats = cal_pr(output[1].detach().cpu(), labels['hap_2'].cpu(), data['mask'].cpu(), num_classes=2)
                gt_stats = cal_pr(output[2].detach().cpu(), labels['gt'].cpu(), data['mask'].cpu(), num_classes=4)

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

            if train and i % self.log_freq == 0:
                self._print_log(epoch, i, eval_dict, data_iter, train=True)

        # Epoch结束，打印完整总结
        self._print_epoch_summary(epoch, eval_dict, len(dataloader), train=train)

        return eval_dict

    def _print_log(self, epoch, iter_step, eval_dict, data_iter, train=True):
        """打印训练日志"""
        hap_precision, hap_recall, hap_f1 = self.calculate_metrics(
            eval_dict['hap_tp'].float(),
            eval_dict['hap_fp'].float(),
            eval_dict['hap_fn'].float()
        )

        prefix = "TRAIN" if train else "VAL"
        log_dict = {
            f"{prefix}_epoch": epoch,
            f"{prefix}_iter": iter_step,
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

    def _print_epoch_summary(self, epoch, eval_dict, num_batches, train=True):
        """打印Epoch总结"""
        hap_precision, hap_recall, hap_f1 = self.calculate_metrics(
            eval_dict['hap_tp'].float(),
            eval_dict['hap_fp'].float(),
            eval_dict['hap_fn'].float()
        )

        gt_precision, gt_recall, gt_f1 = self.calculate_metrics(
            eval_dict['gt_tp'].float(),
            eval_dict['gt_fp'].float(),
            eval_dict['gt_fn'].float()
        )

        mode = "TRAIN" if train else "VAL"

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} {mode} Summary")
        print(f"{'='*60}")
        print(f"Avg Loss:      {eval_dict['hap_loss'] / num_batches:.4f}")
        print(f"Avg Accuracy:  {eval_dict['hap_correct'] / eval_dict['hap_numbers']:.4f}")
        print(f"\nHaplotype Metrics:")
        print(f"  - F1:        {hap_f1[1].item():.4f}")
        print(f"  - Precision: {hap_precision[1].item():.4f}")
        print(f"  - Recall:    {hap_recall[1].item():.4f}")
        print(f"\nGenotype Metrics:")
        for c in range(4):
            if gt_f1[c].item() > 0:
                print(f"  - Class {c} F1: {gt_f1[c].item():.4f}")
        print(f"  - Avg F1:    {gt_f1.mean().item():.4f}")
        print(f"{'='*60}\n")

    def calculate_metrics(self, tp, fp, fn):
        """计算P/R/F1"""
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return precision, recall, f1

    def should_stop_early(self, val_metrics, epoch):
        """检查是否应该Early Stopping"""
        if self.val_data is None:
            return False

        # 提取监控指标
        if self.val_metric == 'f1':
            hap_precision, hap_recall, hap_f1 = self.calculate_metrics(
                val_metrics['hap_tp'].float(),
                val_metrics['hap_fp'].float(),
                val_metrics['hap_fn'].float()
            )
            current_metric = hap_f1[1].item()
            is_better = current_metric > self.best_val_metric + self.min_delta
        elif self.val_metric == 'accuracy':
            current_metric = val_metrics['hap_correct'] / val_metrics['hap_numbers']
            is_better = current_metric > self.best_val_metric + self.min_delta
        elif self.val_metric == 'loss':
            current_metric = val_metrics['hap_loss'] / len(self.val_data)
            is_better = current_metric < self.best_val_metric - self.min_delta
        else:
            raise ValueError(f"Unknown val_metric: {self.val_metric}")

        # 更新最佳指标
        if is_better:
            self.best_val_metric = current_metric
            self.epochs_no_improve = 0
            print(f"\n✓ New best {self.val_metric}: {current_metric:.4f}")
            return False
        else:
            self.epochs_no_improve += 1
            print(f"\n⚠ No improvement for {self.epochs_no_improve} epochs (best {self.val_metric}: {self.best_val_metric:.4f})")

            if self.epochs_no_improve >= self.patience:
                print(f"\n⛔ Early stopping triggered! No improvement for {self.patience} epochs.")
                return True
            return False

    def save(self, epoch, file_path, is_best=False):
        """保存模型"""
        save_dir = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(save_dir, exist_ok=True)

        if is_best:
            output_path = file_path + ".best.pth"
            self.best_model_path = output_path
        else:
            output_path = file_path + f".ep{epoch}.pth"

        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print(f"EP:{epoch} Model Saved: {output_path}")
        return output_path
