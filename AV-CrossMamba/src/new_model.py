import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
import copy
from mamba_ssm import Mamba
from modules.mamba.bimamba import Mamba as BiMamba
from modules.mamba_blocks import MambaBlocksSequential


EPS = 1e-8


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN).

    Arguments
    ---------
    channel_size : int
        Number of channels in the normalization dimension (the third dimension).

    Example
    -------
    >>> x = torch.randn(2, 3, 3)
    >>> norm_func = ChannelwiseLayerNorm(3)
    >>> x_normalized = norm_func(x)
    >>> x.shape
    torch.Size([2, 3, 3])
    """

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))  # [1, 1, N]
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters."""
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, K, N], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, K, N]
        """
        mean = torch.mean(y, dim=2, keepdim=True)  # [M, K, 1]
        var = torch.var(y, dim=2, keepdim=True, unbiased=False)  # [M, K, 1]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm,
              self).__init__(dim,
                             elementwise_affine=elementwise_affine,
                             eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512,
                           512,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=512,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class DeepFusionCrossMamba(nn.Module):
    """
    输入:
        audio: [B, T, D]  (D = d_model)
        video: [B, T, D]  (已被投影到与 audio 相同的维度 d_model)
    输出:
        out: [B, T, D]
    说明:
        - 使用两个门控：audio->video（用于精炼 video），video->audio（用于控制 delta）
        - proj_v_to_delta: 将精炼后 video 映射为 delta_audio
        - mamba_layers: 在修正后的 audio 上做深层时序建模
    """
    def __init__(self,
                 d_model,
                 depth=1,
                 dropout=0.3,
                 n_mamba=2,
                 bidirectional=True):
        super(DeepFusionCrossMamba, self).__init__()

        # 视频->音频的残差产生器（可以用 Conv1d 替代 Linear，如果你坚持“卷积”）
        self.proj_v_to_delta = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 门控：控制 delta 的强度（video -> audio gate）
        self.gate_v2a = nn.Conv1d(d_model, d_model, 1)  # 或 nn.Conv1d(d_model, d_model, 1)

        # 门控：audio -> video，用来精炼 video（去噪）
        self.gate_a2v = nn.Conv1d(d_model, d_model, 1)

        # Mamba 层（深度序列建模）
        self.mamba_layers = nn.Sequential(*[
            MambaBlocksSequential(
                n_mamba=n_mamba,
                bidirectional=bidirectional,
                d_model=d_model,
                d_state=16,
                expand=2,
                d_conv=4,
                fused_add_norm=False,
                rms_norm=True,
                residual_in_fp32=False,
                conv_bias=True,
                bias=False
            ) for _ in range(depth)
        ])

        self.norm = ChannelwiseLayerNorm(d_model)

    def forward(self, audio, video):
        """
        audio, video: [B, T, D]
        """
        # ---- 转为 [B, D, T] 以适配卷积 ----
        a = audio.transpose(1, 2)  # [B, D, T]
        v = video.transpose(1, 2)  # [B, D, T]

        # --- Audio -> Video 门控（精炼视觉） ---
        g_a2v = torch.sigmoid(self.gate_a2v(a))  # [B, D, T]
        v_refined = v * g_a2v  # [B, D, T]

        # --- Video -> Audio 修正项 ---
        delta_audio = self.proj_v_to_delta(v_refined)  # [B, D, T]
        g_v2a = torch.sigmoid(self.gate_v2a(v_refined))  # [B, D, T]
        a_corrected = a + g_v2a * delta_audio

        # --- 时序建模 (Mamba expects [B, T, D]) ---
        a_seq = a_corrected.transpose(1, 2)  # [B, T, D]
        y = self.mamba_layers(a_seq)

        # --- 残差 + norm ---
        out = self.norm(y + audio)
        return out


class CrossGatingMambaIIA_Deep(nn.Module):
    """
    将 CrossGatingMambaIIA 与 DeepFusionCrossMamba 结合的增强版模块。
    输入:
        audio: [B, T, d_model]   (batch_first)
        video: [B, T, video_dim]
    输出:
        out: [B, T, d_model]
    流程：
        1) 低层卷积门控 (InterA-B) -> 得到 a_mod, v_mod（在 [B, D, T] 形式内部）
        2) guided enhancement (depthwise conv) -> a_guided
        3) 将 v_mod 投影到 d_model（video->audio proj）
        4) DeepFusionCrossMamba(audio, v_proj) 做残差修正与深度融合
        5) 最终 Mamba 层（可选） + norm 返回
    """
    def __init__(self,
                 d_model=256,
                 video_dim=512,
                 kernel_size=5,
                 n_mamba=2,
                 depth=4,
                 dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.d_model = d_model
        self.video_dim = video_dim

        # InterA-B 风格的双向卷积门控（操作在 [B, C, T]）
        self.v2a_gate = nn.Sequential(
            nn.Conv1d(video_dim, d_model, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(d_model),
            nn.Sigmoid()
        )
        self.a2v_gate = nn.Sequential(
            nn.Conv1d(d_model, video_dim, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(video_dim),
            nn.Sigmoid()
        )

        # 将 video 映射到 d_model（供 DeepFusion 使用）
        # 使用 1x1 conv 来替代 Linear（保持卷积实现）
        self.video_to_audio_proj = nn.Conv1d(video_dim, d_model, kernel_size=1, bias=False)

        # DeepFusion 模块（注意 DeepFusion 的 proj 期望 video 与 audio 同维度）
        self.deepfusion = DeepFusionCrossMamba(d_model=d_model, depth=max(1, depth), dropout=dropout, n_mamba=n_mamba)

        self.final_norm = ChannelwiseLayerNorm(d_model)

    def forward(self, audio, video):
        """
        audio: [B, T, d_model]
        video: [B, T, video_dim]
        return:
            out: [B, T, d_model]
        """
        B, T_a, D_a = audio.shape
        _, T_v, D_v = video.shape

        # 转为 [B, C, T] 做 conv 操作
        a = audio.transpose(1, 2)  # [B, d_model, T_a]
        v = video.transpose(1, 2)  # [B, video_dim, T_v]

        # 若时间步不同，先上/下采样 video 到 audio 时间步（IIANet 的低层精确对齐）
        if v.size(-1) != a.size(-1):
            v = F.interpolate(v, size=a.size(-1), mode='linear', align_corners=False)

        # 双向卷积门控
        g_a = self.v2a_gate(v)   # [B, d_model, T_a]
        g_v = self.a2v_gate(a)   # [B, video_dim, T_a]

        a_mod = a * g_a          # [B, d_model, T_a]
        v_mod = v * g_v          # [B, video_dim, T_a]

        # guided depthwise enhancement + residual (audio path)
        a_guided = a_mod + a  # [B, d_model, T_a]

        # 将 v_mod 投影到 d_model（1x1 conv）
        v_proj = self.video_to_audio_proj(v_mod)  # [B, d_model, T_a]

        # 转回 batch_first 时间维度: [B, T, D]
        audio_seq = a_guided.transpose(1, 2)  # [B, T_a, d_model]
        video_seq = v_proj.transpose(1, 2)    # [B, T_a, d_model]

        # DeepFusion 做残差修正 + mamba 深度建模
        df_out = self.deepfusion(audio_seq, video_seq)  # [B, T_a, d_model]

        # 最终残差 + norm
        out = self.final_norm(df_out + audio)

        return out


class CrossMambaBlock(nn.Module):
    """
    高层跨模态时序融合模块 (IIANet + Mamba)
    ---------------------------------------
    输入:
        x: [B, T, D]       音频特征序列
        video: [B, T, D]   对齐后的视频特征序列 (维度需匹配)
    输出:
        out: [B, T, D]     融合后的音频特征
    """
    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 dropout=0.2,
                 n_mamba=2):
        super(CrossMambaBlock, self).__init__()

        # 使用 IIANet 风格的深度跨模态 Mamba 模块
        self.mdl = CrossGatingMambaIIA_Deep(
            d_model=d_model,
            video_dim=d_model,  # 注意：video 维度与 audio 一致
            depth=num_layers,
            dropout=dropout,
            n_mamba=n_mamba
        )

    def forward(self, x, video):
        """
        x: [B, T, D]
        video: [B, T, D]
        """
        assert x.shape == video.shape, \
            f"Shape mismatch: audio={x.shape}, video={video.shape}"

        out = self.mdl(x, video)
        return out


class BottleneckAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, downsample_factor=4, dropout=0.1):
        super().__init__()
        self.reduce = nn.Conv1d(d_model, d_model, kernel_size=downsample_factor, stride=downsample_factor, groups=1)
        self.expand = nn.ConvTranspose1d(d_model, d_model, kernel_size=downsample_factor, stride=downsample_factor)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D] -> downsample keys/values along T
        B, T, D = x.shape
        # conv expects [B, D, T]
        kv = self.reduce(x.transpose(1,2)).transpose(1,2)  # [B, T', D]
        out, _ = self.attn(x, kv, kv)
        out = self.norm(x + self.dropout(out))
        return out


class DepthwiseSeparableConv(nn.Module):
    """DW + PW conv, 用于轻量增强局部建模"""
    def __init__(self, d_model, kernel_size=5, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        y = x.transpose(1, 2)   # [B, D, T]
        y = self.dw(y)
        y = self.pw(y)
        y = y.transpose(1, 2)   # [B, T, D]
        y = self.act(y)
        y = self.dropout(y)
        return self.norm(x + y)  # 残差 + norm


class IntraMambaBlock(nn.Module):
    def __init__(self, d_model, n_layers=4, mamba_per_layer=1, attn_every=2,
                 use_dwconv=True, kernel_size=5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_dwconv = use_dwconv

        for i in range(n_layers):
            m = MambaBlocksSequential(
                n_mamba=mamba_per_layer,
                bidirectional=True,
                d_model=d_model,
                d_state=16,
                expand=2,
                d_conv=4
            )
            attn = BottleneckAttention(
                d_model, num_heads=4, downsample_factor=4, dropout=0.2
            ) if (i % attn_every == 0) else nn.Identity()

            dwconv = DepthwiseSeparableConv(d_model, kernel_size=kernel_size) if use_dwconv else nn.Identity()

            self.layers.append(nn.ModuleDict({
                'mamba': m,
                'attn': attn,
                'dwconv': dwconv,
                'norm': nn.LayerNorm(d_model)
            }))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y = x
        for layer in self.layers:
            # 1) Mamba
            y = layer['mamba'](y) + y
            y = layer['norm'](y)

            # 2) Bottleneck Attention
            if not isinstance(layer['attn'], nn.Identity):
                y = layer['attn'](y)

            # 3) Depthwise + Pointwise Conv
            if not isinstance(layer['dwconv'], nn.Identity):
                y = layer['dwconv'](y)

        return self.final_norm(y)


class Cross_Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Cross_Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

    def forward(self, x, v):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape

        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        B_v, N_v, S_v = v.shape

        v = v.unsqueeze(-2).repeat(1, 1, K, 1)
        v = v.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)

        inter = self.inter_mdl(inter, v)

        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class Cross_Dual_Path_Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=160,
        num_spks=2,
        skip_around_intra=True,
        max_length=20000,
    ):
        super(Cross_Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        ve_blocks = []
        for _ in range(5):
            ve_blocks += [VisualConv1D()]
        ve_blocks += [nn.Conv1d(512, 256, 1)]
        self.visual_conv = nn.Sequential(*ve_blocks)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Cross_Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                    )))

        self.conv2d = nn.Conv2d(out_channels,
                                out_channels * num_spks,
                                kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x, video):
        # before each line we indicate the shape after executing the line

        video = video.transpose(1, 2)
        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        v = self.visual_conv(video)
        v = F.pad(v, (0, x.shape[-1] - v.shape[-1]), mode='replicate')
        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, v)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (torch.cat([input1, input2], dim=3).view(B, N, -1,
                                                         K).transpose(2, 3))

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Cross_Sepformer(nn.Module):
    def __init__(self,
                 IntraSeparator,
                 InterSeparator,
                 kernel_size=16,
                 N_encoder_out=256,
                 num_spks=1):
        super(Cross_Sepformer, self).__init__()

        self.AudioEncoder = Encoder(kernel_size=kernel_size,
                                    out_channels=N_encoder_out)
        self.AudioDecoder = Decoder(in_channels=N_encoder_out,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    stride=kernel_size // 2,
                                    bias=False)
        
        self.Separator = Cross_Dual_Path_Model(num_spks=num_spks,
                                               in_channels=N_encoder_out,
                                               out_channels=N_encoder_out,
                                               num_layers=3,
                                               K=160,
                                               intra_model=IntraSeparator,
                                               inter_model=InterSeparator,
                                               norm='ln',
                                               skip_around_intra=True)
        self.num_spks = num_spks

    def forward(self, mix, video):
        """
        mix: [B, T_audio]
        video: [B, T_video_i, 512] (每个样本T_video_i不同)
        """

        # --- 音频编码 ---
        mix_w = self.AudioEncoder(mix)  # [B,256,T_audio]

        # --- 分离 ---
        est_mask = self.Separator(mix_w, video)

        # --- 解码 ---
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        est_source = torch.cat(
            [self.AudioDecoder(sep_h[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1
        )

        # --- 修复长度 ---
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source.permute(0, 2, 1).squeeze(1)


def MambaAVSepFormer_warpper(
    kernel_size: int = 16,
    N_encoder_out: int = 256,
    num_spks: int = 1,
) -> nn.Module:
    """
    Constructs an AV-SepFormer where Intra- and Inter-chunk modeling
    are performed by MambaBlocksSequential instead of Transformers.
    """
    # Create Mamba-based intra- and inter-chunk modules
    IntraSeparator = IntraMambaBlock(
        d_model=N_encoder_out,
    )

    InterSeparator = CrossMambaBlock(
        d_model=N_encoder_out,
)
    # Plug into Cross_Sepformer
    model = Cross_Sepformer(
        IntraSeparator=IntraSeparator,
        InterSeparator=InterSeparator,
        kernel_size=kernel_size,
        N_encoder_out=N_encoder_out,
        num_spks=num_spks,
    )
    return model
