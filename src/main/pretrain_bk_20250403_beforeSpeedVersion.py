import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.amp
import torch.nn as nn


from collections import Counter
from pprint import pprint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import Adam
from torch.utils.data import DataLoader


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
        """根据epoch计算当前mask比例"""
        progress = epoch / (total_epochs - 1)  # 注意-1确保第5epoch达到max
        
        if self.strategy == 'linear':
            return self.base_ratio + (self.max_ratio - self.base_ratio) * progress
        elif self.strategy == 'cosine':
            return self.base_ratio + 0.5*(self.max_ratio - self.base_ratio)*(1 - np.cos(np.pi * progress))
        elif self.strategy == 'exponential':
            return self.base_ratio * ( (self.max_ratio/self.base_ratio) )**progress
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class BERTTrainer():
    """
    This class contains all the information about training a BERT.
    """

    def __init__(self,
                 bert: BERTWithRAG,
                 train_dataloader: DataLoader = None,
                 vocab : WordVocab = None,
                 lr: float = 1e-5, 
                 betas = (0.9, 0.999),
                 weight_decay: float = 0.01, 
                 warmup_steps = 20000,
                 with_cuda: bool = True, 
                 cuda_devices = None, 
                 log_freq: int = 10,
                 state_dict = None,
                 mask_scheduler=None
                 ):
        """
        Attributes:

            bert : BERT model which you want to train.
            train_dataloader : train dataset data loader.
            test_dataloader : test dataset data loader [can be None].
            lr : learning rate of optimizer.
            betas : Adam optimizer betas.
            weight_decay : Adam optimizer weight decay param.
            with_cuda : traning with cuda.
            log_freq : logging frequency of the batch iteration.
        """
        # Setup cuda device for BERT training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:" + str(cuda_devices[0]) if cuda_condition and cuda_devices is not None else "cpu")

        # This BERT model will be saved every epoch.
        self.bert = bert

        self.vocab = vocab

        # Initialize the BERT Language Model.
        self.model = BERTFoundationModel(bert)

        # Distributed GPU training if CUDA can detect more than 1 GPU.
        if state_dict is not None:
            self.model.load_state_dict(state_dict=state_dict)
            print("Params' loading succeed.")

        # Load model into GPU.
        self.model = self.model.to(self.device)
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % len(cuda_devices))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Set train dataloader.
        self.train_data = train_dataloader

        # Set Adam optimizer with hyper-param.
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, n_warmup_steps=warmup_steps)

        # Using Focal-loss.
        # scale_factor = 4.0
        # weight_vector = torch.tensor([0.0421, 0.3667, 0.6112*scale_factor]).to(self.device)
        # self.criterion = FocalLoss(alpha=weight_vector, gamma=3)
        # self.hap_weight = torch.tensor([0.02, 0.98])
        # self.gt_weight = torch.tensor([0.0004, 0.0196, 0.0196, 0.9604])
        # self.hap_criterion = FocalLoss(gamma=5, alpha=self.hap_weight, reduction='sum')
        # self.gt_criterion = FocalLoss(gamma=5, alpha=self.gt_weight, reduction='sum')
        self.hap_criterion = FocalLoss(gamma=5, reduction='sum')
        self.gt_criterion = FocalLoss(gamma=5, reduction='sum')
        
        
        self.recon_critetion = nn.MSELoss()

        self.log_freq = log_freq

        self.mask_scheduler = mask_scheduler
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))


    def train(self, epoch):
        """Loop over the dataloader for training.
        """
        current_ratio = self.mask_scheduler.get_mask_ratio(epoch)
        print(f"\n=== Epoch {epoch+1} 使用mask比例: {current_ratio:.0%} ===")
        self.train_data.dataset.refresh_indices(epoch)
        
        self.model.train()
        mode_code = "Train"

        # Set tqdm bar
        data_iter = tqdm.tqdm(enumerate(self.train_data),
                              desc="EP_%s:%d" % (mode_code, epoch),
                              total=len(self.train_data),
                              bar_format="{l_bar}{r_bar}")


        # Evaluation metrics.
        eval_dict = {
            "hap_loss" : 0.0,
            "hap_correct" : 0,
            "hap_numbers" : 0,

            "gt_loss" : 0.0,
            "gt_correct" : 0,
            "gt_numbers" : 0,

            "recon_loss" : 0,

            # 新增单体型统计
            "hap_tp": torch.zeros(2, dtype=torch.long),  # 0:REF, 1:ALT
            "hap_fp": torch.zeros(2, dtype=torch.long),
            "hap_fn": torch.zeros(2, dtype=torch.long),

            # 新增基因型统计
            "gt_tp": torch.zeros(4, dtype=torch.long),   # 0:0/0, 1:0/1, 2:1/1, 3:其他
            "gt_fp": torch.zeros(4, dtype=torch.long),
            "gt_fn": torch.zeros(4, dtype=torch.long)
        }

        # Data in GPU.
        data_in_gpu = ['hap_1', 'hap_2', 'pos', 'af', 'af_p', 'ref', 'het', 'hom', 'rag_seg_h1', 'rag_seg_h2']

        scaler = torch.amp.grad_scaler.GradScaler()

        for i, data in data_iter:
            # Train

            gpu_data = {key: data[key].to(self.device) for key in data_in_gpu}
            with torch.cuda.amp.autocast():
                # hap_1_pred, hap_2_pred, gt_pred  = self.model.forward(gpu_data)
                output  = self.model.forward(gpu_data)
            for idx, tensor in enumerate(output):
                output[idx] = tensor.cpu() 


            """ ==================== Loss ==================== """
            # Calculate.
            hap_1_loss = self.hap_criterion(output[0][data['mask'].bool()], data['hap_1_label'][data['mask'].bool()])
            hap_2_loss = self.hap_criterion(output[1][data['mask'].bool()], data['hap_2_label'][data['mask'].bool()])
            gt_loss = self.gt_criterion(output[2][data['mask'].bool()], data['gt_label'][data['mask'].bool()])
            # recon_loss1 = self.recon_critetion(output[3][data['mask'].bool()], output[5][data['mask'].bool()])
            # recon_loss2 = self.recon_critetion(output[4][data['mask'].bool()], output[6][data['mask'].bool()])
            # if recon_loss1 > MIN_RECON_LOSS and recon_loss2 > MIN_RECON_LOSS:
            #     loss = 0.2 * hap_1_loss + 0.2 * hap_2_loss + 0.3 * gt_loss + 0.15 * recon_loss1 + 0.15 * recon_loss2
            # else:
            #     loss = 3 * hap_1_loss + 3 * hap_2_loss + 4 * gt_loss
            loss = hap_1_loss + hap_2_loss + gt_loss

            # Backward.
            self.optim_schedule.zero_grad()
            scaler.scale(loss).backward()
            self.optim_schedule._update_learning_rate()
            scaler.step(self.optim)
            scaler.update()


            """ ==================== ACC ==================== """
            hap_1_acc, hap_1_tot = cal_acc(output[0], data['hap_1_label'], data['mask'])
            hap_2_acc, hap_2_tot = cal_acc(output[1], data['hap_2_label'], data['mask'])

            """
            hap_1_acc, hap_1_tot = cal_acc_log(
                output[0], 
                data['hap_1_label'],
                data['mask'],
                seq_data={
                    'hap_seq': data['hap_1'],    # 原始单体型序列 [B, L]
                    'pos': data['pos'],          # 全局位置信息 [B, L] 
                    'af': data['af']             # 等位频率 [B, L]
                },
                sample_idx=i % len(data)         # 当前样本索引
            )

            hap_2_acc, hap_2_tot = cal_acc_log(
                output[0], 
                data['hap_2_label'],
                data['mask'],
                seq_data={
                    'hap_seq': data['hap_2'],    # 原始单体型序列 [B, L]
                    'pos': data['pos'],          # 全局位置信息 [B, L] 
                    'af': data['af']             # 等位频率 [B, L]
                },
                sample_idx=i % len(data)         # 当前样本索引
            )
            """

            gt_acc, gt_tot = cal_acc(output[2], data['gt_label'], data['mask'])
            
            """ ==================== PR ======================"""
            # 单体型统计（二分类）
            hap1_stats = cal_pr(output[0], data['hap_1_label'], data['mask'], num_classes=2)
            hap2_stats = cal_pr(output[1], data['hap_2_label'], data['mask'], num_classes=2)
            gt_stats = cal_pr(output[2], data['gt_label'], data['mask'], num_classes=4)
            # 合并统计量
            for c in range(2):
                eval_dict['hap_tp'][c] += (hap1_stats['tp'][c] + hap2_stats['tp'][c])
                eval_dict['hap_fp'][c] += (hap1_stats['fp'][c] + hap2_stats['fp'][c])
                eval_dict['hap_fn'][c] += (hap1_stats['fn'][c] + hap2_stats['fn'][c])

            for c in range(4):
                eval_dict['gt_tp'][c] += gt_stats['tp'][c]
                eval_dict['gt_fp'][c] += gt_stats['fp'][c]
                eval_dict['gt_fn'][c] += gt_stats['fn'][c]
            
            hap_precision, hap_recall, hap_f1 = self.calculate_metrics(
                eval_dict['hap_tp'].float(),
                eval_dict['hap_fp'].float(),
                eval_dict['hap_fn'].float()
            )

            gt_metrics = []
            for c in range(4):
                p, r, f = self.calculate_metrics(
                    eval_dict['gt_tp'][c].float(),
                    eval_dict['gt_fp'][c].float(),
                    eval_dict['gt_fn'][c].float()
                )
                gt_metrics.append((p.item(), r.item(), f.item()))

            """ ==================== Updata Eval-info ==================== """
            eval_dict['hap_loss'] += (hap_1_loss.item() + hap_2_loss.item())
            eval_dict['hap_correct'] += (hap_1_acc + hap_2_acc)
            eval_dict['hap_numbers'] += (hap_1_tot + hap_2_tot)
            
            eval_dict['gt_loss'] += gt_loss.item()
            eval_dict['gt_correct'] += gt_acc
            eval_dict['gt_numbers'] += gt_tot

            # eval_dict['recon_loss'] += (recon_loss1.item() + recon_loss2.item())
            

            """ ==================== Log ==================== """
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "Precision": hap_precision[1],
                "Recall":hap_recall[1],
                "F1":hap_f1[1],
                "avg_hap_loss": eval_dict['hap_loss'] / (i + 1),
                "avg_hap_acc": eval_dict['hap_correct'] / eval_dict['hap_numbers'] * 100,
                "avg_gt_loss": eval_dict['gt_loss'] / (i + 1),
                "avg_gt_acc": eval_dict['gt_correct'] / eval_dict['gt_numbers'] * 100,
                "avg_recon_loss": eval_dict['recon_loss'] / (i + 1)
                # "loss": loss.item()
            }

            if i % self.log_freq == 0:
                pprint(post_fix)

        
        print(
            "EP%d_%s, avg_hap_loss =" % (epoch, mode_code), eval_dict['hap_loss'] / len(data_iter),
            " ,avg_hap_acc =", eval_dict['hap_correct'] / eval_dict['hap_numbers'] * 100,
            " ,avg_gt_loss =", eval_dict['gt_loss'] / len(data_iter),
            " ,avg_gt_acc =", eval_dict['gt_correct'] / eval_dict['gt_numbers'] * 100,
            )
        
    def calculate_metrics(self, tp, fp, fn):
        """安全计算precision/recall"""
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return precision, recall, f1

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch + ".pth"
        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
