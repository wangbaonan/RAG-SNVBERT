import torch
import torch.nn as nn
import math

class LDGuidedRetention(nn.Module):
    """基于LD衰减的注意力机制"""
    def __init__(self, dims, window_size=128, gamma=0.9):
        super().__init__()
        self.window_size = window_size
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=True)
        self.qkv = nn.Linear(dims, 3*dims)
        self.proj = nn.Linear(dims, dims)
        
        # 位置衰减矩阵（预计算）
        self.register_buffer('decay_mask', self._create_decay_mask(window_size))
        
    def _create_decay_mask(self, window_size):
        """创建指数衰减掩码"""
        pos = torch.arange(window_size).unsqueeze(0) - torch.arange(window_size).unsqueeze(1)
        mask = self.gamma ** torch.abs(pos.float())
        mask = torch.tril(mask)  # 保持因果性
        return mask.unsqueeze(0)  # [1, L, L]
    
    def forward(self, x, positions=None):
        B, L, D = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # 相对位置调整
        if positions is not None:
            pos_enc = self.position_encoder(positions)  # 需实现位置编码
            q = q + pos_enc[:, :, :D]
            k = k + pos_enc[:, :, D:]
        
        # 局部LD衰减注意力
        attn = torch.einsum('bqd,bkd->bqk', q, k) / math.sqrt(D)
        
        # 应用衰减掩码（模拟LD衰减）
        if L <= self.window_size:
            attn = attn * self.decay_mask[:, :L, :L]
        else:
            # 长序列分块处理
            attn = self._blockwise_ld_mask(attn)
        
        attn = F.softmax(attn, dim=-1)
        return self.proj(torch.einsum('bqk,bkd->bqd', attn, v))
    
    def _blockwise_ld_mask(self, attn):
        """分块应用衰减掩码"""
        L = attn.size(1)
        num_blocks = L // self.window_size
        for i in range(num_blocks):
            start = i * self.window_size
            end = start + self.window_size
            attn[:, start:end, start:end] *= self.decay_mask
        return attn


class CrossAFInteraction(nn.Module):
    """跨层次AF交互模块"""
    def __init__(self, dims):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, dims),
            nn.Sigmoid()
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(2, dims),
            nn.LayerNorm(dims),
            nn.GELU()
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.01)

    def forward(self, global_af, pop_af):
        combined = torch.stack([global_af, pop_af], dim=-1)  # [B, L, 2]
        gate = self.gate_net(combined)                      # [B, L, D]
        encoded = self.joint_encoder(combined)              # [B, L, D]
        return global_af.unsqueeze(-1) + self.res_scale * (gate * encoded)


class EnhancedRareVariantFusion(nn.Module):
    """罕见变异融合模块"""
    def __init__(self, dims):
        super().__init__()
        # 跨层次AF交互模块
        self.af_interaction = CrossAFInteraction(dims)
        
        self.af_adapter = nn.Sequential(
            nn.Linear(dims, 4*dims),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*dims, dims),
            nn.Sigmoid()
        )
        
        # 动态聚合层
        self.pooling = nn.Sequential(
            nn.Linear(dims, 1),  
            nn.Softmax(dim=2)   
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(2*dims, 4*dims),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*dims, dims),
            nn.LayerNorm(dims)
        )
        
        # 残差缩放参数
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)

    def forward(self, orig_feat, rag_feat, global_af, pop_af):
        B, K, L, D = rag_feat.size()
        
        # 跨层次AF融合
        fused_af = self.af_interaction(global_af, pop_af)  # [B, L, D]
        
        # 生成AF权重
        af_weight = self.af_adapter(fused_af)  # [B, L, D]
        
        # 参考特征加权（维度调整）
        weighted_ref = rag_feat * af_weight.unsqueeze(1)  # [B, K, L, D] 
        weighted_ref = weighted_ref.permute(0, 2, 1, 3)   # [B, L, K, D] 
        
        # 动态聚合
        pool_weights = self.pooling(weighted_ref)          # [B, L, K, 1]
        pooled_ref = torch.sum(weighted_ref * pool_weights, dim=2)  # [B, L, D] 
        
        # 特征融合（维度验证）
        assert orig_feat.size() == (B, L, D), f"原始特征维度错误: {orig_feat.size()}"
        assert pooled_ref.size() == (B, L, D), f"聚合特征维度错误: {pooled_ref.size()}"
        
        fused = self.fusion(torch.cat([orig_feat, pooled_ref], dim=-1))  # [B, L, 2D] -> [B, L, D]
        
        # MAF逆向加权 (优化版: 使用log1p平滑处理)
        maf = torch.min(global_af, 1 - global_af).unsqueeze(-1)  # [B, L, 1]

        # 使用log1p平滑处理，避免梯度爆炸
        # log1p(1/x) = -log1p(x-1) ≈ -log(x) for small x
        # 对于小MAF，权重增加；对于大MAF，权重减少
        maf_weight = torch.log1p(1.0 / (maf + 1e-6)).clamp(max=3.0)  # 降低max阈值

        return orig_feat + self.res_scale * (fused * maf_weight)


class RareVariantAwareFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # AF敏感的特征转换
        self.af_transform = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, dims),
            nn.Sigmoid()
        )
        # 高效特征融合
        self.fusion = nn.Sequential(
            nn.Linear(2*dims, dims),
            nn.LayerNorm(dims),
            nn.GELU()
        )
        
    def forward(self, orig_feat, rag_feat, af):
        """
        orig_feat: [B, L, D]
        rag_feat: [B, K, L, D]
        af: [B, L]
        """
        B, K, L, D = rag_feat.shape
        
        # AF加权聚合（对罕见变异赋予更高权重）
        af_weight = self.af_transform(af.unsqueeze(-1))  # [B, L, D]
        weighted_ref = torch.einsum('bkld,blc->bkld', rag_feat, af_weight)  # [B, K, L, D]
        
        # 动态聚合参考特征
        pooled_ref = 0.7 * weighted_ref.mean(dim=1) + 0.3 * weighted_ref.max(dim=1).values
        
        # 残差融合
        fused = self.fusion(torch.cat([orig_feat, pooled_ref], -1))
        return orig_feat + fused * torch.sqrt(af * (1 - af)).unsqueeze(-1)  # MAF加权


class FixedConcatFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2*dims, dims),  
            nn.LayerNorm(dims),       
            nn.GELU()                 
        )
        
    def forward(self, orig_feat, rag_feat):
        # 特征聚合
        pooled_ref = rag_feat.mean(dim=1)  # [B,L,D]
        
        # 拼接融合
        combined = torch.cat([orig_feat, pooled_ref], dim=-1)
        fused = self.fusion(combined)
        
        # 残差
        return orig_feat + 0.1 * fused  # 缩放系数控制梯度


class ConcatFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # 使用高效的一维卷积进行特征融合（等效于全连接但更高效）
        self.fusion_conv = nn.Conv1d(
            in_channels=dims*2,  # 原始特征+聚合后的参考特征
            out_channels=dims,
            kernel_size=1,
            bias=True
        )
        
    def forward(self, orig_feat, rag_feat):
        """
        改进后的拼接融合方式
        orig_feat: [B, L, D] 原始特征
        rag_feat:  [B, K, L, D] 参考特征
        """
        # 特征聚合策略（平均池化+最大池化的混合模式）
        pooled_ref = 0.5 * rag_feat.mean(dim=1) + 0.5 * rag_feat.max(dim=1).values  # [B, L, D]
        
        # 拼接特征（通道维度）
        combined = torch.cat([orig_feat, pooled_ref], dim=-1)  # [B, L, 2D]
        
        # 维度转换用于卷积（Conv1d需要通道在前）
        combined = combined.permute(0, 2, 1)  # [B, 2D, L]
        
        # 融合特征
        fused = self.fusion_conv(combined)  # [B, D, L]
        
        # 恢复原始维度并残差连接
        return orig_feat + fused.permute(0, 2, 1)  # [B, L, D]

class CrossAttentionFusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dims, num_heads=8)
        
    def forward(self, orig_feat, rag_feat):
        """
        orig_feat: [B, L, D] 原始特征
        rag_feat:  [B, K, L, D] 参考特征
        """
        B, K, L, D = rag_feat.shape
        
        # 重组参考特征维度
        rag_feat = rag_feat.permute(1,0,2,3)  # [K, B, L, D]
        
        # 逐参考处理
        outputs = []
        for k in range(K):
            # 计算当前参考的注意力
            attn_out, _ = self.attn(
                query=orig_feat.permute(1,0,2),  # [L, B, D]
                key=rag_feat[k].permute(1,0,2),
                value=rag_feat[k].permute(1,0,2)
            )
            outputs.append(attn_out.permute(1,0,2))  # [B, L, D]
            
        # 聚合所有参考结果
        fused = torch.stack(outputs, dim=1).mean(dim=1)  # [B, L, D]
        return orig_feat + fused

class PositionFeatModule(nn.Module):
    """Process the feature of 'pos'.
    """
    def __init__(self, 
                 hidden_channels : int = 4,
                 kernel_size : int = 9,
                 stride : int = 1,
                 padding : int = 4,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.act1 = nn.LeakyReLU(negative_slope=0.05)

        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)

        self.act2 = nn.LeakyReLU(negative_slope=0.05)

        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=1,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.act3 = nn.LeakyReLU(negative_slope=0.05)
        
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.norm2 = nn.BatchNorm1d(num_features=hidden_channels)

    
    def forward(self, pos : torch.Tensor):
        # pos.shape == (batch, seq_len)
        # 修复说明: 强制禁用 autocast 并使用 FP32 运行 Conv1d
        # 原因: 规避 V18 中因非连续张量输入导致的 CUDNN_STATUS_NOT_SUPPORTED 报错及性能崩塌
        # 背景: A100/H100 GPU 在混合精度训练时，CuDNN 对 FP16 卷积输入的内存布局极其敏感
        #      当输入张量内存不连续（Stride 不规整）时，CuDNN 无法生成高效执行计划
        #      导致回退到未优化的慢速内核（85s/it → <1s/it）
        with torch.cuda.amp.autocast(enabled=False):
            # 确保输入是连续的 float32
            out = pos.float().unsqueeze(1)

            out = self.norm1(self.act1(self.conv1(out)))
            out = self.norm2(self.act2(self.conv2(out)))
            out = self.act3(self.conv3(out))

            return out.squeeze()



class EmbeddingFusionModule(nn.Module):
    """Process the features of 'hap', 'pos' & 'af'
    """

    def __init__(self, emb_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # pos-related.
        self.pos_feat = PositionFeatModule()

        self.fusion = nn.Linear(emb_size + 2, emb_size) # 将维度映射回 emb_size
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.norm = nn.LayerNorm(emb_size)


    def forward(self, emb : torch.Tensor, pos : torch.Tensor, af : torch.Tensor):
        """
        emb.shape == (batch, seq_len, emb_dim).
        
        pos.shape == (batch, seq_len). unsqueeze -> (batch, seq_len, 1)
        
        af.shape == (batch, seq_len), unsqueeze -> (batch, seq_len, 1)
        """

        # pos.
        pos_feat = self.pos_feat(pos)
        pos_feat = pos_feat.unsqueeze(-1)

        af_feat = af.unsqueeze(-1)

        all_feat = torch.cat((emb, pos_feat, af_feat), dim=-1)
        all_feat = self.act(self.fusion(all_feat))

        return self.norm(emb + all_feat)

