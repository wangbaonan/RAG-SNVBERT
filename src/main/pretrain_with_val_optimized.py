"""
优化版训练脚本 - 修复训练停滞问题
主要改进:
1. 可配置Focal Loss gamma (默认2.5而不是5)
2. 可选择是否使用reconstruction loss
3. 保留Rare/Common F1分解和详细日志
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from .optim_schedule import ScheduledOptim, FocalLoss, cal_acc, cal_pr
import pprint

MIN_RECON_LOSS = 0.01


class BERTTrainerWithValidationOptimized():
    """
    增强版BERT训练器
    新增功能:
    1. Rare (MAF<0.05) vs Common (MAF>=0.05) F1分解
    2. 每个epoch的详细指标CSV输出
    3. Loss曲线数据保存
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader=None,  # 新增
        vocab=None,
        lr: float = 5e-5,  # 优化: 5e-5 (原版 1e-5)
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        warmup_steps=10000,  # 优化: 10000 (原版 20000)
        with_cuda: bool = True,
        cuda_devices=None,
        log_freq: int = 100,
        grad_accum_steps: int = 1,
        # Early Stopping & Validation参数
        patience: int = 5,
        val_metric: str = 'f1',
        min_delta: float = 0.001,
        # 优化参数 (新增)
        focal_gamma: float = 2.5,  # 优化: 2.5 (原版 5)
        use_recon_loss: bool = False,  # 优化: False (原版 True)
        # 增强输出参数
        rare_threshold: float = 0.05,  # MAF阈值
        output_csv: str = None  # CSV输出路径
    ):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        if cuda_condition and cuda_devices is not None:
            if isinstance(cuda_devices, list):
                device_ids = cuda_devices
            else:
                device_ids = [cuda_devices]
            print(f"Using GPU devices: {device_ids}")
            self.model = nn.DataParallel(model, device_ids=device_ids).to(self.device)
        else:
            self.model = model.to(self.device)

        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.vocab = vocab

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas,
                        weight_decay=weight_decay, fused=True)
        self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps)

        self.scaler = GradScaler(enabled=with_cuda)
        self.grad_accum_steps = grad_accum_steps
        self.accum_step = 0

        # 优化参数
        self.focal_gamma = focal_gamma
        self.use_recon_loss = use_recon_loss

        self.hap_criterion = FocalLoss(gamma=focal_gamma, reduction='sum').to(self.device)
        self.gt_criterion = FocalLoss(gamma=focal_gamma, reduction='sum').to(self.device)
        self.recon_critetion = nn.MSELoss().to(self.device)

        self.log_freq = log_freq

        # Validation & Early Stopping参数
        self.patience = patience
        self.val_metric = val_metric
        self.min_delta = min_delta
        self.best_val_metric = -np.inf if val_metric in ['f1', 'accuracy'] else np.inf
        self.epochs_no_improve = 0
        self.best_model_path = None

        # 增强输出参数
        self.rare_threshold = rare_threshold
        self.output_csv = output_csv

        # 存储每个epoch的指标
        self.epoch_metrics = []

        if val_dataloader:
            print(f"✓ Validation enabled with early stopping (patience={patience}, metric={val_metric})")
            print(f"✓ Enhanced output: Rare (MAF<{rare_threshold}) vs Common F1")

    def train(self, epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} - TRAINING")
        print(f"{'='*60}")
        return self._run_epoch(epoch, self.train_data, train=True)

    def validate(self, epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} - VALIDATION")
        print(f"{'='*60}")
        return self._run_epoch(epoch, self.val_data, train=False)

    def _run_epoch(self, epoch, dataloader, train=True):
        """统一的epoch运行函数"""
        if train:
            self.model.train()
        else:
            self.model.eval()

        # 初始化统计
        eval_dict = {
            'hap_loss': 0.0,
            'gt_loss': 0.0,
            'recon_loss': 0.0,
            'hap_correct': 0,
            'gt_correct': 0,
            'hap_numbers': 0,
            'gt_numbers': 0,
            # TP/FP/FN for F1计算
            'hap_tp': torch.zeros(2, device=self.device),
            'hap_fp': torch.zeros(2, device=self.device),
            'hap_fn': torch.zeros(2, device=self.device),
            'gt_tp': torch.zeros(4, device=self.device),
            'gt_fp': torch.zeros(4, device=self.device),
            'gt_fn': torch.zeros(4, device=self.device),
            # Rare vs Common统计 (新增)
            'rare_tp': torch.zeros(2, device=self.device),
            'rare_fp': torch.zeros(2, device=self.device),
            'rare_fn': torch.zeros(2, device=self.device),
            'common_tp': torch.zeros(2, device=self.device),
            'common_fp': torch.zeros(2, device=self.device),
            'common_fn': torch.zeros(2, device=self.device),
        }

        mode_str = "EP_Train" if train else "EP_Val"
        data_iter = tqdm(enumerate(dataloader), desc=f"{mode_str}:{epoch}",
                        total=len(dataloader), bar_format="{l_bar}{r_bar}")

        for i, data in data_iter:
            # 准备数据
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

            # Forward
            if train:
                with autocast(enabled=self.with_cuda, dtype=torch.float16 if self.with_cuda else torch.bfloat16):
                    output = self.model(gpu_data)

                    hap_1_loss = self.hap_criterion(output[0][masks], labels['hap_1'][masks])
                    hap_2_loss = self.hap_criterion(output[1][masks], labels['hap_2'][masks])
                    gt_loss = self.gt_criterion(output[2][masks], labels['gt'][masks])

                    # 根据配置决定是否使用reconstruction loss
                    if self.use_recon_loss:
                        recon_loss1 = self.recon_critetion(output[3][masks], output[5][masks])
                        recon_loss2 = self.recon_critetion(output[4][masks], output[6][masks])

                        if recon_loss1 > MIN_RECON_LOSS and recon_loss2 > MIN_RECON_LOSS:
                            total_loss = (0.2 * hap_1_loss + 0.2 * hap_2_loss + 0.3 * gt_loss +
                                        0.15 * recon_loss1 + 0.15 * recon_loss2)
                        else:
                            total_loss = 3 * hap_1_loss + 3 * hap_2_loss + 4 * gt_loss
                    else:
                        # 优化版: 不使用recon loss (避免梯度冲突)
                        total_loss = 3 * hap_1_loss + 3 * hap_2_loss + 4 * gt_loss

                    total_loss /= self.grad_accum_steps

                self.scaler.scale(total_loss).backward()
                self.accum_step += 1

                if self.accum_step % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim_schedule.step()
                    self.optim_schedule.zero_grad()
                    self.accum_step = 0
            else:
                with torch.no_grad():
                    with autocast(enabled=self.with_cuda, dtype=torch.float16 if self.with_cuda else torch.bfloat16):
                        output = self.model(gpu_data)

                        hap_1_loss = self.hap_criterion(output[0][masks], labels['hap_1'][masks])
                        hap_2_loss = self.hap_criterion(output[1][masks], labels['hap_2'][masks])
                        gt_loss = self.gt_criterion(output[2][masks], labels['gt'][masks])

                        recon_loss1 = self.recon_critetion(output[3][masks], output[5][masks])
                        recon_loss2 = self.recon_critetion(output[4][masks], output[6][masks])

            # 计算指标 (训练和验证都需要)
            with torch.no_grad():
                hap_1_acc, hap_1_tot = cal_acc(output[0].detach().cpu(), labels['hap_1'].cpu(), data['mask'].cpu())
                hap_2_acc, hap_2_tot = cal_acc(output[1].detach().cpu(), labels['hap_2'].cpu(), data['mask'].cpu())
                gt_acc, gt_tot = cal_acc(output[2].detach().cpu(), labels['gt'].cpu(), data['mask'].cpu())

                hap1_stats = cal_pr(output[0].detach().cpu(), labels['hap_1'].cpu(), data['mask'].cpu(), num_classes=2)
                hap2_stats = cal_pr(output[1].detach().cpu(), labels['hap_2'].cpu(), data['mask'].cpu(), num_classes=2)
                gt_stats = cal_pr(output[2].detach().cpu(), labels['gt'].cpu(), data['mask'].cpu(), num_classes=4)

                # Overall统计
                for c in range(2):
                    eval_dict['hap_tp'][c] += (hap1_stats['tp'][c] + hap2_stats['tp'][c]).to(self.device)
                    eval_dict['hap_fp'][c] += (hap1_stats['fp'][c] + hap2_stats['fp'][c]).to(self.device)
                    eval_dict['hap_fn'][c] += (hap1_stats['fn'][c] + hap2_stats['fn'][c]).to(self.device)

                for c in range(4):
                    eval_dict['gt_tp'][c] += gt_stats['tp'][c].to(self.device)
                    eval_dict['gt_fp'][c] += gt_stats['fp'][c].to(self.device)
                    eval_dict['gt_fn'][c] += gt_stats['fn'][c].to(self.device)

                # ========== 新增: Rare vs Common统计 ==========
                # 获取MAF: af已经在[0,1]范围,取min(af, 1-af)得到MAF
                af_batch = gpu_data['af']  # [B, L]
                maf = torch.min(af_batch, 1 - af_batch)  # [B, L]

                # Rare mask: MAF < threshold
                rare_mask = (maf < self.rare_threshold) & masks  # [B, L]
                common_mask = (maf >= self.rare_threshold) & masks  # [B, L]

                # 分别计算Rare和Common的TP/FP/FN
                if rare_mask.any():
                    rare1_stats = cal_pr(output[0].detach().cpu(), labels['hap_1'].cpu(),
                                        rare_mask.cpu(), num_classes=2)
                    rare2_stats = cal_pr(output[1].detach().cpu(), labels['hap_2'].cpu(),
                                        rare_mask.cpu(), num_classes=2)
                    for c in range(2):
                        eval_dict['rare_tp'][c] += (rare1_stats['tp'][c] + rare2_stats['tp'][c]).to(self.device)
                        eval_dict['rare_fp'][c] += (rare1_stats['fp'][c] + rare2_stats['fp'][c]).to(self.device)
                        eval_dict['rare_fn'][c] += (rare1_stats['fn'][c] + rare2_stats['fn'][c]).to(self.device)

                if common_mask.any():
                    common1_stats = cal_pr(output[0].detach().cpu(), labels['hap_1'].cpu(),
                                          common_mask.cpu(), num_classes=2)
                    common2_stats = cal_pr(output[1].detach().cpu(), labels['hap_2'].cpu(),
                                          common_mask.cpu(), num_classes=2)
                    for c in range(2):
                        eval_dict['common_tp'][c] += (common1_stats['tp'][c] + common2_stats['tp'][c]).to(self.device)
                        eval_dict['common_fp'][c] += (common1_stats['fp'][c] + common2_stats['fp'][c]).to(self.device)
                        eval_dict['common_fn'][c] += (common1_stats['fn'][c] + common2_stats['fn'][c]).to(self.device)
                # =============================================

                eval_dict['hap_loss'] += (hap_1_loss.item() + hap_2_loss.item())
                eval_dict['gt_loss'] += gt_loss.item()
                eval_dict['recon_loss'] += (recon_loss1.item() + recon_loss2.item())
                eval_dict['hap_correct'] += (hap_1_acc + hap_2_acc)
                eval_dict['gt_correct'] += gt_acc
                eval_dict['hap_numbers'] += (hap_1_tot + hap_2_tot)
                eval_dict['gt_numbers'] += gt_tot

            if train and i % self.log_freq == 0:
                self._print_log(epoch, i, eval_dict, data_iter, train=True)

        # Epoch结束,打印完整总结
        self._print_epoch_summary(epoch, eval_dict, len(dataloader), train=train)

        # 保存epoch指标到CSV (新增)
        self._save_epoch_metrics(epoch, eval_dict, len(dataloader), train=train)

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
        """打印Epoch总结 (增强版)"""
        # Overall metrics
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

        # Rare vs Common metrics (新增)
        rare_precision, rare_recall, rare_f1 = self.calculate_metrics(
            eval_dict['rare_tp'].float(),
            eval_dict['rare_fp'].float(),
            eval_dict['rare_fn'].float()
        )

        common_precision, common_recall, common_f1 = self.calculate_metrics(
            eval_dict['common_tp'].float(),
            eval_dict['common_fp'].float(),
            eval_dict['common_fn'].float()
        )

        mode = "TRAIN" if train else "VAL"

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} {mode} Summary")
        print(f"{'='*60}")
        print(f"Avg Loss:      {eval_dict['hap_loss'] / num_batches:.4f}")
        print(f"Avg Accuracy:  {eval_dict['hap_correct'] / eval_dict['hap_numbers']:.4f}")

        print(f"\nHaplotype Metrics (Overall):")
        print(f"  - F1:        {hap_f1[1].item():.4f}")
        print(f"  - Precision: {hap_precision[1].item():.4f}")
        print(f"  - Recall:    {hap_recall[1].item():.4f}")

        # 新增: Rare vs Common breakdown
        print(f"\nRare Variants (MAF<{self.rare_threshold}):")
        print(f"  - F1:        {rare_f1[1].item():.4f}")
        print(f"  - Precision: {rare_precision[1].item():.4f}")
        print(f"  - Recall:    {rare_recall[1].item():.4f}")

        print(f"\nCommon Variants (MAF>={self.rare_threshold}):")
        print(f"  - F1:        {common_f1[1].item():.4f}")
        print(f"  - Precision: {common_precision[1].item():.4f}")
        print(f"  - Recall:    {common_recall[1].item():.4f}")

        print(f"\nGenotype Metrics:")
        for c in range(4):
            if gt_f1[c].item() > 0:
                print(f"  - Class {c} F1: {gt_f1[c].item():.4f}")
        print(f"  - Avg F1:    {gt_f1.mean().item():.4f}")
        print(f"{'='*60}\n")

    def _save_epoch_metrics(self, epoch, eval_dict, num_batches, train=True):
        """保存epoch指标到CSV (新增)"""
        if self.output_csv is None:
            return

        # 计算所有指标
        hap_p, hap_r, hap_f1 = self.calculate_metrics(
            eval_dict['hap_tp'].float(),
            eval_dict['hap_fp'].float(),
            eval_dict['hap_fn'].float()
        )

        rare_p, rare_r, rare_f1 = self.calculate_metrics(
            eval_dict['rare_tp'].float(),
            eval_dict['rare_fp'].float(),
            eval_dict['rare_fn'].float()
        )

        common_p, common_r, common_f1 = self.calculate_metrics(
            eval_dict['common_tp'].float(),
            eval_dict['common_fp'].float(),
            eval_dict['common_fn'].float()
        )

        # 构建一行数据
        mode = "train" if train else "val"
        metric_row = {
            'epoch': epoch + 1,
            'mode': mode,
            'loss': eval_dict['hap_loss'] / num_batches,
            'accuracy': eval_dict['hap_correct'] / eval_dict['hap_numbers'],
            'overall_f1': hap_f1[1].item(),
            'overall_precision': hap_p[1].item(),
            'overall_recall': hap_r[1].item(),
            'rare_f1': rare_f1[1].item(),
            'rare_precision': rare_p[1].item(),
            'rare_recall': rare_r[1].item(),
            'common_f1': common_f1[1].item(),
            'common_precision': common_p[1].item(),
            'common_recall': common_r[1].item(),
        }

        self.epoch_metrics.append(metric_row)

        # 写入CSV
        import csv
        from pathlib import Path
        csv_path = Path(self.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 判断是否需要写header
        write_header = not csv_path.exists()

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metric_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metric_row)

    def calculate_metrics(self, tp, fp, fn):
        """计算Precision, Recall, F1"""
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1

    def should_stop_early(self, val_metrics, epoch):
        """判断是否early stopping"""
        # 提取validation F1
        hap_precision, hap_recall, hap_f1 = self.calculate_metrics(
            val_metrics['hap_tp'].float(),
            val_metrics['hap_fp'].float(),
            val_metrics['hap_fn'].float()
        )

        current_metric = hap_f1[1].item()

        # 判断是否改进
        is_better = current_metric > self.best_val_metric + self.min_delta

        if is_better:
            self.best_val_metric = current_metric
            self.epochs_no_improve = 0
            print(f"✓ New best {self.val_metric}: {current_metric:.4f}")
            return False
        else:
            self.epochs_no_improve += 1
            print(f"⚠ No improvement for {self.epochs_no_improve} epoch(s) "
                  f"(best: {self.best_val_metric:.4f}, current: {current_metric:.4f})")

            if self.epochs_no_improve >= self.patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered!")
                print(f"No improvement for {self.patience} consecutive epochs")
                print(f"Best {self.val_metric}: {self.best_val_metric:.4f}")
                print(f"{'='*60}\n")
                return True

        return False

    def save(self, epoch, file_path="output/bert_trained.model", is_best=False):
        """保存模型"""
        output_path = file_path + f".ep{epoch}"
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.cpu(), output_path)
            self.model.to(self.device)
        else:
            torch.save(self.model.cpu(), output_path)
            self.model.to(self.device)

        if is_best:
            best_path = file_path + ".best.pth"
            if isinstance(self.model, nn.DataParallel):
                torch.save(self.model.module.cpu(), best_path)
                self.model.to(self.device)
            else:
                torch.save(self.model.cpu(), best_path)
                self.model.to(self.device)
            self.best_model_path = best_path
            print(f"✓ Best model saved: {best_path}")

        print(f"EP:{epoch} Model Saved: {output_path}")

        return output_path

    @property
    def with_cuda(self):
        return next(self.model.parameters()).is_cuda
