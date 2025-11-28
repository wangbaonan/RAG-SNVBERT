import torch
import torch.nn as nn
import math
import torch.nn.functional as F




class DynamicGeneFusion(nn.Module):
    """整合AF信息的动态基因特征融合模块"""
    def __init__(self, dims):
        super().__init__()
        # 基因型感知参数
        self.allele_weights = nn.Parameter(torch.tensor([0.8, 1.2]))  # 对应5和6
        
        # 低秩投影参数
        self.rank = min(4, dims//4)
        self.U = nn.Parameter(torch.randn(self.rank, dims))
        self.V = nn.Parameter(torch.randn(dims, self.rank))
        
        # AF特征处理器
        self.af_processor = nn.Sequential(
            nn.Linear(2, 16),       # 输入global_af和pop_af
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, dims),
            nn.Sigmoid()
        )
        
        # 动态门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(3*dims, dims),  # 主特征+参考特征+AF特征
            nn.LayerNorm(dims),
            nn.GELU(),
            nn.Linear(dims, 1),
            nn.Sigmoid()
        )
        
        # MAF敏感参数
        self.maf_scale = nn.Parameter(torch.tensor(0.1))
        
        # 初始化保障
        self._init_weights()

    def _init_weights(self):
        """专业初始化保障稳定性"""
        # 低秩投影矩阵初始化
        nn.init.orthogonal_(self.U)
        nn.init.kaiming_normal_(self.V)
        
        # AF处理器初始化
        nn.init.xavier_normal_(self.af_processor[0].weight)
        nn.init.constant_(self.af_processor[0].bias, 0.1)
        nn.init.xavier_normal_(self.af_processor[3].weight)
        nn.init.constant_(self.af_processor[3].bias, 0.5)
        
        # 门控网络初始化
        nn.init.kaiming_normal_(self.gate_net[0].weight, nonlinearity='relu')
        nn.init.constant_(self.gate_net[0].bias, 0.1)
        nn.init.xavier_uniform_(self.gate_net[3].weight)
        nn.init.constant_(self.gate_net[3].bias, 0.5)

    def forward(self, orig_feat, rag_feat, global_af, pop_af):
        # 维度校验与调整 (关键修复)
        B, K, rag_L, D = rag_feat.size()
        orig_B, orig_L, orig_D = orig_feat.size()
        
        # 严格维度校验
        assert B == orig_B, f"批次维度不匹配: {B} vs {orig_B}"
        assert rag_L == orig_L, f"序列长度不匹配: {rag_L} vs {orig_L}"
        assert D == orig_D, f"特征维度不匹配: {D} vs {orig_D}"
        
        # 基因型加权（维度安全版）
        weight_mask = torch.where(
            rag_feat == 5, 
            self.allele_weights[0],
            torch.where(rag_feat == 6, self.allele_weights[1], 1.0)
        ).unsqueeze(-1)  # [B,K,L,1]
        
        proj_ref = torch.einsum('bkld,dr->bklr', rag_feat, self.V)  # [B,K,L,r]
        proj_ref = torch.einsum('bklr,rd->bkld', proj_ref, self.U)  # [B,K,L,D]
        
        # ========== AF特征处理 ==========
        af_feat = torch.stack([global_af, pop_af], dim=-1)  # [B, L, 2]
        processed_af = self.af_processor(af_feat)          # [B, L, D]
        
        # ========== 特征聚合 ==========
        agg_ref = proj_ref.mean(dim=1)  # [B, L, D]
        
        # ========== 动态门控 ==========
        gate_input = torch.cat([
            orig_feat,          # 主特征
            agg_ref,            # 聚合参考特征
            processed_af        # AF处理特征
        ], dim=-1)              # [B, L, 3D]
        
        gate = self.gate_net(gate_input)  # [B, L, 1]
        
        # ========== MAF敏感输出 ==========
        maf = torch.min(global_af, 1 - global_af).unsqueeze(-1)
        maf_weight = (1.0 / (maf + 1e-6)).clamp(max=10.0)
        
        # ========== 最终融合 ==========
        fused_feat = orig_feat + self.maf_scale * (
            gate * orig_feat + (1 - gate) * agg_ref
        ) * maf_weight
        
        return fused_feat


class LDGuidedRetention(nn.Module):
    """基于LD衰减的注意力机制（设备安全版）"""
    def __init__(self, dims, window_size=128, gamma_init=0.9):
        super().__init__()
        self.window_size = window_size
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.qkv = nn.Linear(dims, 3*dims)
        self.proj = nn.Linear(dims, dims)

    def _create_decay_mask(self, seq_len, device):
        """动态生成设备感知的衰减掩码"""
        pos = torch.arange(seq_len, device=device).unsqueeze(0) - \
              torch.arange(seq_len, device=device).unsqueeze(1)
        return (self.gamma ** torch.abs(pos.float())).tril()

    def forward(self, x):
        B, L, D = x.size()
        device = x.device
        
        # 动态生成掩码（自动设备对齐）
        decay_mask = self._create_decay_mask(min(L, self.window_size), device)
        
        # 分块处理长序列
        if L > self.window_size:
            decay_mask = self._create_decay_mask(self.window_size, device)
        
        # 注意力计算（保持设备一致性）
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.einsum('bqd,bkd->bqk', q, k) / math.sqrt(D)
        attn = attn * decay_mask[:L, :L].unsqueeze(0)  # 自动广播到批次维度
        
        attn = F.softmax(attn, dim=-1)
        return self.proj(torch.einsum('bqk,bkd->bqd', attn, v))


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
    """设备安全改进版"""
    def __init__(self, dims, ld_window=128):
        super().__init__()
        self.af_interaction = CrossAFInteraction(dims)
        self.context_attention = LDGuidedRetention(dims, window_size=ld_window)
        
        # 确保所有参数在相同设备
        self.fusion = nn.Sequential(
            nn.Linear(2*dims, 4*dims),
            nn.GELU(),
            nn.Linear(4*dims, dims),
            nn.LayerNorm(dims)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 设备感知初始化
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.maf_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.maf_bias = nn.Parameter(torch.tensor(0.0, device=device))

    def forward(self, orig_feat, rag_feat, global_af, pop_af):
        # 显式设备同步
        orig_feat = orig_feat.to(self.maf_scale.device)
        rag_feat = rag_feat.to(orig_feat.device)
        global_af = global_af.to(orig_feat.device)
        pop_af = pop_af.to(orig_feat.device)
        B, K, L, D = rag_feat.size()
        
        # 跨层次AF融合
        fused_af = self.af_interaction(global_af, pop_af)
        
        # 上下文感知的参考序列聚合
        orig_context = self.context_attention(orig_feat)  # 增强上下文表示
        rag_flat = rag_feat.view(B*K, L, D)
        weighted_ref = self.context_attention(rag_flat).view(B, K, L, D)
        
        # 动态注意力聚合（考虑原始特征）
        attn_scores = torch.einsum('bld,bkld->blk', orig_context, weighted_ref)
        attn_weights = F.softmax(attn_scores / math.sqrt(D), dim=-1)  # [B, L, K]
        pooled_ref = torch.einsum('blk,bkld->bld', attn_weights, rag_feat)
        
        # 稳定MAF加权
        maf = torch.min(global_af, 1 - global_af)
        maf_weight = torch.sigmoid(self.maf_scale * (1/(maf + 1e-6)) + self.maf_bias)
        
        # 残差融合
        fused = self.fusion(torch.cat([orig_feat, pooled_ref], dim=-1))
        return orig_feat + fused * maf_weight.unsqueeze(-1)


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
        out = pos.unsqueeze(1)
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

