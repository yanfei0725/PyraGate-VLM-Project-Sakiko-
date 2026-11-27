"""
Full gated output head for PyraGate Sakiko v7.

This is the "all enhancements on" output head described in the 2026-11-24
architecture note, adapted to the current model instead of the note's RGB
image assumption:

  - input : final token features [B*T, dim, H, W] on the 1x2x2 patch grid
  - output: per-token velocity [B*T, 2*2*48, H, W]
  - the caller unpatchifies to [B, 48, T, H*2, W*2]

The original note's forward had two fatal shape bugs:
  proj_out(Cmax -> in_dim) + pyramid(Cmax), then + f_single(Cmax)
so this implementation keeps everything in Cmax until the final projection.

All modules keep independent boolean switches for ablation.
"""

from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # 包内 / 平铺两种部署方式兼容
    from .band_head_lr import MultiBandLowRankHead
except ImportError:  # pragma: no cover
    from band_head_lr import MultiBandLowRankHead


def _env_float(name: str, default: float) -> float:
    """自动流水线用环境变量覆盖门控固定系数；未设置时用代码默认值。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


class DWModulator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mod = self.dw(x)
        return self.norm(mod.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LocalSSM1D(nn.Module):
    """Causal depthwise diagonal SSM over a 1D axis.

    The note's kernel was not normalized; with A=sigmoid(-2)=0.88 and length 48
    the effective gain is ~6x. Here the kernel is L1-normalized per channel so
    the SSM is a stable weighted scan, not an amplifier.
    """

    def __init__(self, dim: int, max_len: int = 128):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.A_log = nn.Parameter(torch.full((dim,), -3.0))
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)
        self.D = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, modulate: torch.Tensor) -> torch.Tensor:
        # x, modulate: [B, L, C]
        x = self.norm(x)
        A = torch.sigmoid(self.A_log.float())
        Bc = self.B(modulate)
        Cc = self.C(modulate)
        Dc = self.D(modulate)

        Bx = (Bc * x).transpose(1, 2).float()          # [B, C, L]
        seq_len = x.shape[1]
        t_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        A_pow = torch.pow(A.unsqueeze(0), t_idx.unsqueeze(1)).T  # [C, L]
        kernel = A_pow.unsqueeze(1)                    # [C, 1, L]
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)

        h = F.conv1d(Bx, kernel, padding=seq_len - 1, groups=self.dim)
        h = h[..., :seq_len]
        h = h.transpose(1, 2).to(x.dtype)
        y = Cc * h + Dc * x
        return y


class ConvModulatedSpatialSSM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mod = DWModulator(dim)
        self.ssm_h = LocalSSM1D(dim)
        self.ssm_w = LocalSSM1D(dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mod = self.mod(x)

        x_h = x.permute(0, 3, 1, 2).reshape(B * W, H, C)
        mod_h = mod.permute(0, 3, 1, 2).reshape(B * W, H, C)
        x_h = self.ssm_h(x_h, mod_h).reshape(B, W, H, C).permute(0, 3, 2, 1)

        x_w = x_h.permute(0, 2, 1, 3).reshape(B * H, W, C)
        mod_w = mod.permute(0, 2, 1, 3).reshape(B * H, W, C)
        x_w = self.ssm_w(x_w, mod_w).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Residual is deliberate: the note returned proj(x_w) only, which throws
        # away the branch input and makes the first training phase fragile.
        return x + self.proj(x_w)


class GroupMaskGate(nn.Module):
    def __init__(self, dim: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self.mask_pred = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, groups, 1),
        )
        # sigmoid(2.0) ~= 0.88: start close to identity, not a 0.5 hard shrink.
        nn.init.constant_(self.mask_pred[-1].bias, 2.0)

    def mask(self, cond: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mask_pred(cond))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mask = self.mask(cond)
        chunks = torch.chunk(x, self.groups, dim=1)
        out_chunks = [ck * mask[:, idx:idx + 1] for idx, ck in enumerate(chunks)]
        return torch.cat(out_chunks, dim=1)


class ChannelFreqGate(nn.Module):
    """Per-channel low/high gate, no spatial conv.

    low = x * sigmoid(logit); high = x * (1 - sigmoid(logit)).
    bias starts at -2.0 -> low gate ~0.119, i.e. only a small low-frequency
    subset enters neighborhood aggregation at step 0.
    """

    def __init__(self, dim: int, init_bias: float = -2.0):
        super().__init__()
        # 1536×2 标量：通道偏置 + 通道缩放，不用 conv/MLP
        self.bias = nn.Parameter(torch.full((1, dim, 1, 1), init_bias))
        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor):
        g = torch.sigmoid(self.bias) * torch.clamp(self.scale, 0.0, 1.0)
        g = g.to(x.dtype)
        return x * g, x * (1.0 - g), g


class BandFreqNorm(nn.Module):
    """频带感知 RMSNorm：DC 低频 / residual 高频独立归一化，Rezero 恒等。

    前向初始严格等价于输入：rz=0 时 low/high 原样返回。训练后 rz 学会
    只在需要时把各频带拉回稳定统计，不替换主特征。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.dc_gain = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.dc_bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.res_gain = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.res_bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.dc_rz = nn.Parameter(torch.zeros(1))
        self.res_rz = nn.Parameter(torch.zeros(1))
        self.enable_dc_norm = True
        self.enable_res_norm = True

    def _spatial_rms(self, x: torch.Tensor) -> torch.Tensor:
        # 正确频带 norm：逐通道在 H/W 空间上做 RMS，而不是 nn.RMSNorm(dim)
        # 的逐像素跨通道归一化。后者会白化 patch 均值，直接破坏 DC 连续性。
        # 统计固定 fp32，避免 bf16 的 var/rsqrt 舍入把小 Rezero 更新淹没。
        xf = x.float()
        mu = xf.mean(dim=(2, 3), keepdim=True)
        var = xf.var(dim=(2, 3), keepdim=True, unbiased=False)
        return (xf - mu) * torch.rsqrt(var + self.eps)

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        # 禁用支路必须真正旁路：既保证逐位恒等，也避免无效统计 kernel 和计算图。
        if not self.enable_dc_norm and not self.enable_res_norm:
            return low, high
        low_out, high_out = low, high
        if self.enable_dc_norm:
            ld = self._spatial_rms(low)
            dc_new = ld * self.dc_gain.float() + self.dc_bias.float()
            low_out = (low.float() + self.dc_rz.float()
                       * (dc_new - low.float())).to(low.dtype)
        if self.enable_res_norm:
            lr = self._spatial_rms(high)
            res_new = lr * self.res_gain.float() + self.res_bias.float()
            high_out = (high.float() + self.res_rz.float()
                        * (res_new - high.float())).to(high.dtype)
        return low_out, high_out


class EmbeddedDualActGate(nn.Module):
    """内嵌多源 Dual-Act 门控：sigma/邻域DC/局部对比/时序EMA/相干度 -> 两组门。

    全部 8 维低维统计特征进 16 宽 MLP，输出 gate_dc / gate_res 两组通道门。
    Rezero：rz=0 时 gate=1，3200 热启动零扰动。Conditional AdaK 低秩支路
    在 cond=0 时输出严格为 0；训练后才接入 VLM/专家风格向量。
    """

    def __init__(self, dim: int, cond_dim: int, feat_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        self.feat_dim = int(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        # 末层 bias=0、rz 初始 0.10：gate=1+rz*tanh(0)=1 严格恒等；
        # 但 MLP 权重首步梯度=rz≠0，不会像 rz=0+bias=0 那样双死锁。
        nn.init.zeros_(self.mlp[-1].bias)
        # Conditional AdaK：低秩 8 维瓶颈；up 初始 0，self-supervised cond=0
        # 时通路严格零输出。down 用小随机权重，蒸馏阶段 up 获得梯度后可学习。
        self.cond_down = nn.Linear(cond_dim, 8, bias=False)
        nn.init.normal_(self.cond_down.weight, std=0.02)
        self.cond_up = nn.Linear(8, 2 * dim, bias=False)
        nn.init.zeros_(self.cond_up.weight)
        self.rz_dc = nn.Parameter(torch.full((1,), 0.10))
        self.rz_res = nn.Parameter(torch.full((1,), 0.10))
        self.gate_dc_min = _env_float("PYRAGATE_DUALACT_DC_MIN", 0.75)
        self.gate_dc_max = _env_float("PYRAGATE_DUALACT_DC_MAX", 1.20)
        self.gate_res_min = _env_float("PYRAGATE_DUALACT_RES_MIN", 0.90)
        self.gate_res_max = _env_float("PYRAGATE_DUALACT_RES_MAX", 1.10)
        if not (self.gate_dc_min <= 1.0 <= self.gate_dc_max):
            raise ValueError("DualAct DC clamp 必须包含恒等点 1.0")
        if not (self.gate_res_min <= 1.0 <= self.gate_res_max):
            raise ValueError("DualAct residual clamp 必须包含恒等点 1.0")
        self.enable_cond = True
        self.train_dc = True
        self.train_res = True

    def set_trainable_paths(self, dc: bool = True, res: bool = True):
        """隔离两组门的梯度；共享 MLP 只接收未冻结路径的梯度。"""
        self.train_dc, self.train_res = bool(dc), bool(res)
        self.rz_dc.requires_grad_(self.train_dc)
        self.rz_res.requires_grad_(self.train_res)

    def forward(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        sigma: torch.Tensor | None,
        frames: int,
        context_vec: torch.Tensor | None = None,
    ):
        Bt, C, H, W = low.shape
        vf = low.float()
        pm = F.avg_pool2d(vf, 2, 2)
        dc_mean = vf.mean(dim=(1, 2, 3))
        dc_std = vf.std(dim=(1, 2, 3), unbiased=False)
        local_c = ((pm[..., 1:, :] - pm[..., :-1, :]).abs().mean(dim=(1, 2, 3))
                   + (pm[..., :, 1:] - pm[..., :, :-1]).abs().mean(dim=(1, 2, 3))) * 0.5
        # 时序 EMA 软锚：逐帧 DC 均值相对 EMA 的偏差，抑制色块漂移。
        if frames > 1 and Bt % frames == 0:
            dm = dc_mean.view(Bt // frames, frames)
            ema = dm[..., 0].clone()
            devs = []
            for t in range(frames):
                devs.append((dm[..., t] - ema).abs() / (ema.abs() + 1e-4))
                ema = 0.75 * ema + 0.25 * dm[..., t]
            temporal = torch.stack(devs, dim=1).reshape(-1)
        else:
            temporal = torch.zeros_like(dc_mean)
        # 高频相干度：跨 patch 的余弦一致性（轮廓/纹理 vs 噪声）。
        hn = F.normalize(high.float(), dim=1, eps=1e-6)
        coh = ((hn[..., 1:, :] * hn[..., :-1, :]).mean(dim=(1, 2, 3))
               + (hn[..., :, 1:] * hn[..., :, :-1]).mean(dim=(1, 2, 3))) * 0.5
        if sigma is None:
            s = torch.ones(Bt, device=low.device, dtype=torch.float32)
        else:
            s = sigma.reshape(-1).float()
            if s.numel() != Bt:
                raise RuntimeError(f"dualact sigma numel={s.numel()} != Bt={Bt}")
        feats = torch.stack([s, s * s, 1.0 - s, dc_mean, dc_std,
                             local_c, temporal, coh], dim=1)
        w = self.mlp[0].weight
        g = self.mlp(feats.to(device=w.device, dtype=w.dtype))
        g = g.view(Bt, 2, C, 1, 1)
        if self.enable_cond and context_vec is not None:
            cv = context_vec
            if cv.dim() == 3:
                cv = cv.mean(dim=1)
            if cv.shape[-1] == self.cond_dim:
                c = self.cond_up(self.cond_down(cv.to(w.dtype))).view(Bt, 2, C, 1, 1)
                g = g + c
        g_dc = g[:, 0] if self.train_dc else g[:, 0].detach()
        g_res = g[:, 1] if self.train_res else g[:, 1].detach()
        dc = 1.0 + self.rz_dc.float() * torch.tanh(g_dc.float())
        res = 1.0 + self.rz_res.float() * torch.tanh(g_res.float())
        # 两组门都硬限幅：rz/MLP 即使出现尖峰，也不能把任一路放大到失控。
        dc = torch.clamp(dc, self.gate_dc_min, self.gate_dc_max)
        res = torch.clamp(res, self.gate_res_min, self.gate_res_max)
        low_out = (low.float() * dc).to(low.dtype)
        high_out = (high.float() * res).to(high.dtype)
        return low_out, high_out, dc.mean().detach(), res.mean().detach()


class PatchwiseStatNorm(nn.Module):
    """输出端 patch-wise 可学习统计 Norm，替代外部 seam-renorm。

    以 2×2 输出 patch 为单位做均值/方差归一化，再由 Rezero rz 决定注入量。
    rz=0 时严格恒等；训练后可把 patch DC 对比和幅值恢复内化到模型。
    """

    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.rz = nn.Parameter(torch.zeros(1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"PatchwiseStatNorm 期望 [B,C,H,W]，收到 {tuple(x.shape)}")
        B, C, H, W = x.shape
        if H % 2 or W % 2:
            raise ValueError(f"2x2 patch norm 要求 H/W 为偶数，收到 H={H}, W={W}")
        xf = x.float()
        # 显式按每个 2x2 patch 归约，轴为 patch 内 h/w；不跨 patch 混统计。
        xr = xf.reshape(B, C, H // 2, 2, W // 2, 2)
        pm = xr.mean(dim=(3, 5))
        pv = xr.square().mean(dim=(3, 5)) - pm.square()
        ps = torch.sqrt(pv.clamp_min(0.0) + self.eps)
        pm_u = pm.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        ps_u = ps.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        z = (xf - pm_u) / ps_u
        n = z * self.gain.float() + self.bias.float()
        y = xf + self.rz.float() * (n - xf)
        return y.to(x.dtype)


class AltDirectionalSmooth(nn.Module):
    """Alternating one-direction 3-tap smoothing, no conv / no gather.

    even rows -> horizontal neighbor mix; odd rows -> vertical neighbor mix.
    Only low-frequency channels are passed in by the caller.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,C,H,W]; replicate edges
        xu = F.pad(x, (1, 1, 0, 0), mode="replicate")
        yu = F.pad(x, (0, 0, 1, 1), mode="replicate")
        h_mix = 0.25 * xu[..., :-2] + 0.5 * x + 0.25 * xu[..., 2:]
        v_mix = 0.25 * yu[..., :-2, :] + 0.5 * x + 0.25 * yu[..., 2:, :]
        out = x.clone()
        out[:, :, 0::2, :] = h_mix[:, :, 0::2, :]
        out[:, :, 1::2, :] = v_mix[:, :, 1::2, :]
        return out


class CrossResFiLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.cross_mlp_low = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.cross_mlp_high = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        # softplus(-2)=0.127: cross branch is actually on, but not 0.69-strength
        # at init. The scalar stays bf16 in training, so this is effectively a
        # fixed small residual scale rather than a trainable schedule.
        self.cross_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor):
        cr_low = self.cross_mlp_low(feat_high.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        cr_high = self.cross_mlp_high(feat_low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        s = F.softplus(self.cross_scale)
        return feat_low + s * cr_low, feat_high + s * cr_high


class TimeModulator(nn.Module):
    def __init__(self, t_emb_dim: int, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_emb_dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        # zero init -> sl=0, sh=1 exactly at step 0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t_emb: torch.Tensor):
        s = self.mlp(t_emb)
        sl, sh_raw = s[:, 0:1], s[:, 1:2]
        sl = 0.5 * torch.tanh(sl)
        sh = 1.6 * torch.sigmoid(sh_raw) + 0.2
        return 1 + sl, sh


class LocalRPE2D(nn.Module):
    """Learned local relative-position residual inside the output head.

    The note implemented this as a pure weighted average over a 3x3 window
    (weights sum to 1), i.e. an unconditional blur. Here it is a residual:
    x + sum(offset * table), table zero-init, so it starts as identity and can
    learn local offsets without destroying high frequencies.
    """

    def __init__(self, dim: int, win: int = 3):
        super().__init__()
        self.win = win
        self.rpe_table = nn.Parameter(torch.zeros(win * win, dim))
        nn.init.normal_(self.rpe_table, std=0.01)
        self.pad = win // 2
        # 邻域残差系数：init 0.01，clamp_max 0.1（架构先验，不靠大 loss 硬压）
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor | None = None,
                channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        x_pad = F.pad(x, [self.pad] * 4, mode="replicate")
        unfold = F.unfold(x_pad, kernel_size=self.win, stride=1)
        unfold = unfold.view(B, C, self.win * self.win, H, W)
        table = self.rpe_table.view(1, self.win * self.win, C, 1, 1).permute(0, 2, 1, 3, 4)
        res = (unfold * table).sum(dim=2)
        if channel_mask is not None:
            res = res * channel_mask
        if spatial_mask is not None:
            res = res * spatial_mask
        gamma = torch.clamp(self.gamma, 0.0, 0.1)
        # 返回残差本身；由 head 合成时统一乘 alpha，避免重复缩放
        return res * gamma


class SobelGradMod(nn.Module):
    """Sobel gradients as a modulation signal, not an external loss."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.proj = nn.Conv2d(2, dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        gray = feat.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return self.proj(torch.cat([gx, gy], dim=1))


class AdaptiveAntiAlias(nn.Module):
    """Learnable depthwise anti-alias low-pass with a per-channel gate.

    The note created a random depthwise kernel, which is not an anti-alias
    filter at init. This initializes a normalized 3x3 Gaussian and starts the
    gate near identity (sigmoid(-2)=0.119 blur).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        g = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
        with torch.no_grad():
            self.dw.weight.copy_(g.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))
        self.gate = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blur = self.dw(x)
        g = torch.sigmoid(self.gate(x))
        return g * blur + (1.0 - g) * x


class LatentPyramidShortcut(nn.Module):
    """Lightweight cross-scale residual branch with channel compression."""

    def __init__(self, in_dim: int, compress_ratio: int = 4):
        super().__init__()
        c = max(8, in_dim // compress_ratio)
        self.down = nn.Conv2d(in_dim, c, 2, stride=2)
        self.ssm = ConvModulatedSpatialSSM(c)
        self.up = nn.ConvTranspose2d(c, in_dim, 2, stride=2)
        self.alt_smooth = AltDirectionalSmooth()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.down(x)
        d = self.ssm(d)
        # 低频分支：交替单向 1D 平滑，替代全向稠密邻域；仅低通道输入
        d = self.alt_smooth(d)
        u = self.up(d)
        # 固定 stride 反卷积已精确回到 2×；只在奇数分辨率下裁掉多余边，
        # 不再走 F.interpolate（ROCm 上插值会引入额外的 host/device 同步）。
        if u.shape[-2:] != x.shape[-2:]:
            u = u[..., :x.shape[-2], :x.shape[-1]]
            if u.shape[-2:] != x.shape[-2:]:
                u = F.pad(u, [0, x.shape[-1] - u.shape[-1],
                              0, x.shape[-2] - u.shape[-2]])
        return u


def sampled_token_corr_loss(feat: torch.Tensor, channel_mask: torch.Tensor | None = None,
                            ratio: float = 0.2) -> torch.Tensor:
    """20% 位置采样的相邻 token 余弦相关损失，只在低频通道上算。

    feat [B,C,H,W]；channel_mask [B,C,1,1]。随机取 (H-1)*(W-1) 个可向右/向下
    的 token 位置中的 20%，分别算水平/垂直相邻余弦，返回 1 - 平均相关。
    """
    B, C, H, W = feat.shape
    f = feat
    if channel_mask is not None:
        f = f * channel_mask
    f = f.permute(0, 2, 3, 1).float()
    f = F.normalize(f, dim=-1, eps=1e-6)
    flat = f.reshape(B, H * W, C)
    rr = torch.arange(H, device=feat.device).view(H, 1).expand(H, W)
    cc = torch.arange(W, device=feat.device).view(1, W).expand(H, W)
    valid = ((rr[:, :-1] < H - 1) & (cc[:, :-1] < W - 1)).reshape(-1)
    cand = torch.nonzero(valid, as_tuple=False).flatten()
    n = max(1, int(cand.numel() * ratio))
    idx = cand[torch.randperm(cand.numel(), device=feat.device)[:n]]
    a = flat[:, idx]
    b = flat[:, idx + 1]
    c = flat[:, idx + W]
    ch = (a * b).sum(dim=-1).mean()
    cw = (a * c).sum(dim=-1).mean()
    return 1.0 - 0.5 * (ch + cw)


def patch_mean_pearson_corr_defect(patch_mean: torch.Tensor) -> torch.Tensor:
    """与验收口径一致的相邻 patch-mean Pearson 相关缺陷。

    验收 `score_latents.stats` 的口径是：把相邻 patch mean 的两组值各自 flatten，
    减去均值后做 Pearson 相关，再分别算 H/W。旧训练函数 `sampled_token_corr_loss`
    算的是每个位置的通道向量余弦，和验收口径不是同一个量，不能用来优化
    `patch_mean_corr_h/w`。本函数直接复刻验收的 Pearson 口径（逐样本计算后
    取 batch 平均，浮点稳定等价）。

    patch_mean: [B,C,T,H,W] 或 [B*T,C,H,W]；返回标量 defect = 1 - 平均相关。
    相关越高 defect 越小；只应由 `corr_defect_penalty` 做单边上限约束。
    """
    if patch_mean.dim() == 5:
        b, c, t, h, w = patch_mean.shape
        x = patch_mean.reshape(b * t, c, h, w)
    elif patch_mean.dim() == 4:
        x = patch_mean
    else:
        raise ValueError(f"patch_mean 必须是 4D 或 5D，收到 {tuple(patch_mean.shape)}")
    x = x.float()

    def _pearson_adjacent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        den = a.norm(dim=1) * b.norm(dim=1) + 1e-12
        return (a * b).sum(dim=1) / den

    corr_h = _pearson_adjacent(x[..., :-1, :], x[..., 1:, :])
    corr_w = _pearson_adjacent(x[..., :-1], x[..., 1:])
    return (1.0 - 0.5 * (corr_h + corr_w)).mean()


def corr_defect_penalty(corr_defect: torch.Tensor, target_max: float = 0.10) -> torch.Tensor:
    """单边缺陷惩罚：只在 ``1-corr > target_max`` 时产生非零损失。

    ``corr_defect`` 越接近 0 表示相邻 patch 越相关；高相关不是缺陷，绝不能再
    用一个正下界把模型往低相关方向推。保留 target_min 的 CLI 仅为兼容旧命令，
    训练器会拒绝任何非零下界，防止双向 band 回归。
    """
    if not 0.0 <= float(target_max) <= 2.0:
        raise ValueError(f"corr defect target_max 必须在 [0,2]，收到 {target_max}")
    return torch.relu(corr_defect - corr_defect.new_tensor(float(target_max)))


class FixedFreqSplit(nn.Module):
    """固定 5×5 空间深度卷积：分离 patch 网格低频分量与纹理高频分量。

    该卷积没有可学习参数（kernel 是固定高斯低通），只做一次 (1,5,5)
    深度卷积；输入按帧重排为 [B,C,T,H,W]，不混时间。`hp_grid` 为低通
    网格分量，`hp - hp_grid` 为纹理分量。kernel 不进入 state_dict，
    永远由初始化值重建，避免被 optimizer/ckpt 改动。
    """

    def __init__(self, channels: int, kernel: int = 5, sigma: float = 1.0):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError(f"kernel={kernel} 必须是奇数")
        coords = torch.arange(kernel, dtype=torch.float32) - (kernel - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * float(sigma) ** 2))
        g = g / g.sum()
        w = g.view(1, 1, 1, kernel, kernel).expand(channels, 1, 1, kernel, kernel).clone()
        self.channels = int(channels)
        self.kernel = int(kernel)
        self.register_buffer("weight", w, persistent=False)

    def forward(self, x: torch.Tensor, frames: int) -> torch.Tensor:
        Bt, C, H, W = x.shape
        if C != self.channels:
            raise RuntimeError(f"freq_split C={C} != channels={self.channels}")
        if frames < 1 or Bt % frames != 0:
            raise RuntimeError(f"freq_split Bt={Bt} 不能被 frames={frames} 整除")
        w = self.weight.to(device=x.device, dtype=x.dtype)
        x5 = x.reshape(Bt // frames, C, frames, H, W)
        pad = self.kernel // 2
        y5 = F.conv3d(x5, w, bias=None, stride=1, padding=(0, pad, pad),
                      groups=C)
        y5 = y5.permute(0, 2, 1, 3, 4)
        return y5.reshape(Bt, C, H, W)


class DecoupledSigmaGate(nn.Module):
    """双输入频带门控：g_grid 固定 0.15，g_texture 由 σ 与帧间残差控制。

    g_base(σ) = g_min + (g_max-g_min) * sigmoid(k * (σ0 - σ))
    g_texture = g_base * (1 - α * sigmoid(r_gain * (r_t - r_thresh)))

    当前只激活 g_min/g_max/k 三个可学习标量；σ0=0.3、r_thresh=0.90、
    α=0.35、r_gain=5 全部固定。s_struct/s_texture/g_group 保留为休眠
    参数（不参与前向），保证原 9 参数接口不变。g_grid 是固定 buffer。
    r_thresh 按真实 latent 帧残差量级（约 1.0）标定，避免阈值 0.22 时
    sigmoid 长期饱和、g_texture 恒压最低档。
    """

    def __init__(
        self,
        out_dim: int,
        groups: int = 4,
        sigma0: float = 0.3,
        g_min: float = 0.40,
        g_max: float = 0.62,
        k: float = 2.0,
        g_grid: float = 0.15,
        r_thresh: float = 0.90,
        alpha: float = 0.35,
        r_gain: float = 5.0,
    ):
        super().__init__()
        # 自动修复流水线只覆盖非持久固定系数；g_min/g_max/k 存在 ckpt 中，
        # 环境变量不覆盖它们，避免 resume 时被构造初值悄悄改写。
        sigma0 = _env_float("PYRAGATE_GATE_SIGMA0", sigma0)
        r_thresh = _env_float("PYRAGATE_GATE_R_THRESH", r_thresh)
        alpha = _env_float("PYRAGATE_GATE_ALPHA", alpha)
        r_gain = _env_float("PYRAGATE_GATE_R_GAIN", r_gain)
        g_grid = _env_float("PYRAGATE_GATE_G_GRID", g_grid)
        if out_dim % groups != 0:
            raise ValueError(f"out_dim={out_dim} 不能被 groups={groups} 整除")
        self.out_dim = out_dim
        self.groups = groups
        self.sigma0 = float(sigma0)
        self.r_thresh = float(r_thresh)
        self.alpha = float(alpha)
        self.r_gain = float(r_gain)
        self.g_min_raw = nn.Parameter(
            torch.tensor(math.log(g_min / (1.0 - g_min)), dtype=torch.float32))
        self.g_max_raw = nn.Parameter(
            torch.tensor(math.log(g_max / (1.0 - g_max)), dtype=torch.float32))
        self.k_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(k) - 1.0), dtype=torch.float32))
        # 预留接口：本次全部休眠，不参与前向/反向。
        self.s_struct = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.s_texture = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.g_group = nn.Parameter(torch.ones(groups, dtype=torch.float32))
        # 休眠接口不参与反向/优化器；仍保留在 9 参数 key 中。
        self.s_struct.requires_grad_(False)
        self.s_texture.requires_grad_(False)
        self.g_group.requires_grad_(False)
        # 固定网格门，不训练，不进 state_dict。
        self.register_buffer("g_grid", torch.tensor(float(g_grid), dtype=torch.float32),
                             persistent=False)
        self.enable_freq_dual = False
        self.enable_groups = False
        self.enable_single_route = False
        self._last_base = None
        self._last_texture = None
        self._last_residual = None

    def forward(self, sigma: torch.Tensor | None, frames: int,
                frame_residual: torch.Tensor | None = None) -> torch.Tensor:
        if sigma is None:
            s = torch.ones(frames, 1, 1, 1, device=self.g_min_raw.device)
        else:
            s = sigma.reshape(-1).float()
            if s.numel() != frames:
                raise RuntimeError(f"sigma numel={s.numel()} != frames={frames}")
            s = s.view(frames, 1, 1, 1)
        if frame_residual is None:
            r = torch.zeros(frames, 1, 1, 1, device=self.g_min_raw.device)
        else:
            r = frame_residual.reshape(-1).float()
            if r.numel() != frames:
                raise RuntimeError(f"frame_residual numel={r.numel()} != frames={frames}")
            r = r.view(frames, 1, 1, 1)
        g_min = torch.sigmoid(self.g_min_raw)
        g_max = torch.sigmoid(self.g_max_raw)
        k = F.softplus(self.k_raw)
        base = g_min + (g_max - g_min) * torch.sigmoid(k * (self.sigma0 - s))
        suppress = 1.0 - self.alpha * torch.sigmoid(self.r_gain * (r - self.r_thresh))
        texture = base * suppress
        gate = texture
        if self.enable_single_route:
            gate = gate * self.s_struct
        if self.enable_freq_dual:
            gate = gate * self.s_texture
        if self.enable_groups:
            gg = self.g_group.repeat_interleave(self.out_dim // self.groups)
            gate = gate * gg.view(1, self.out_dim, 1, 1)
        self._last_base = base.mean().detach()
        self._last_texture = texture.mean().detach()
        self._last_residual = r.mean().detach()
        return gate


class FullGatedTokenHead(nn.Module):
    """寄生复用低频邻域头：ChannelFreqGate -> Pyramid -> LocalRPE2D -> final_proj。

    x: [B*T, in_dim, H, W] final token features
    out: [B*T, patch_t*patch_h*patch_w*out_channels, H, W]
    """

    # 历史预留但从未进入 forward 的模块。保留 state_dict key 以兼容旧 ckpt，
    # 但必须永久冻结：它们既拿不到梯度，也不应被误报成“可训容量”。
    _DEAD_MODULE_PREFIXES = (
        "ssm_low.", "ssm_high.", "single_branch.", "cross_film.",
        "time_mod.", "ctx_proj.", "gate_base.", "vlm_offset_mlp.",
        "sobel_mod.",
    )

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_emb_dim: int = 256,
        groups: int = 4,
        vlm_dim: int = 4096,
        ctx_dim: int = 128,
        compress_ratio: int = 4,
        max_rank_div: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Cmax = in_dim // max_rank_div
        self.proj_in = nn.Conv2d(in_dim, self.Cmax, 1)
        self.proj_out = nn.Conv2d(self.Cmax, self.Cmax, 1)

        self.ssm_low = ConvModulatedSpatialSSM(self.Cmax)
        self.ssm_high = ConvModulatedSpatialSSM(self.Cmax)
        self.single_branch = nn.Conv2d(self.Cmax, self.Cmax, 1)

        self.group_mask = GroupMaskGate(self.Cmax, groups=groups)
        self.cross_film = CrossResFiLM(self.Cmax)
        self.time_mod = TimeModulator(t_emb_dim, self.Cmax)
        self.channel_gate = ChannelFreqGate(self.Cmax)
        # 2026-11-24 内嵌式 Dual-Act 升级：全部 Rezero 恒等初始化。
        # 不新增独立旁路，不改 Attention/FFN/通道/patch；3200 热启动零扰动。
        self.band_norm = BandFreqNorm(self.Cmax)
        self.dualact_gate = EmbeddedDualActGate(self.Cmax, cond_dim=in_dim)
        self.patch_norm = PatchwiseStatNorm(out_dim)
        # 2026-11-24 支路专属时间缩放 + 独立 RMSNorm：不复用主干 TimeModulator，
        # 也不与主干共享任何 running 统计；只作用于 RPE+Pyramid 的残差支路。
        # 2026-11-25：保留第一次注册的 head_norm/head_scale_emb 定义。
        # 原因：PyTorch 的 module 重新赋值会改变 named_parameters 顺序，而
        # optimizer_state 是按 param_groups 顺序加载的；删除这段会让 3400 的
        # Adafactor 状态整体错位。这里宁可有“被覆盖的旧定义”，也不能改顺序。
        self.head_norm = nn.RMSNorm(self.Cmax)
        self.head_scale_emb = nn.Sequential(
            nn.Linear(t_emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.Cmax),
        )
        # 初始给常数 1.0 的支路缩放，避免新模块随机初始化把支路直接放大/打散。
        nn.init.zeros_(self.head_scale_emb[-1].weight)
        nn.init.constant_(self.head_scale_emb[-1].bias, 1.0)

        self.rpe = LocalRPE2D(self.Cmax, win=3)
        self.sobel_mod = SobelGradMod(self.Cmax)
        self.anti_alias = AdaptiveAntiAlias(self.Cmax)
        self.pyramid_shortcut = LatentPyramidShortcut(self.Cmax, compress_ratio=compress_ratio)

        # 2026-11-24：金字塔/RPE 支路独立时间缩放 + 支路专属 RMSNorm。
        # 不与主干共享 timestep 调制和 Norm；最后线性层零权重、bias 0.02，
        # 初始只给支路 2% 幅度，接续 step1200 时不会推翻已学好的分支。
        self.head_norm = nn.RMSNorm(self.Cmax, eps=1e-6)
        self.head_scale_emb = nn.Sequential(
            nn.Linear(t_emb_dim, 128), nn.SiLU(), nn.Linear(128, self.Cmax)
        )
        nn.init.zeros_(self.head_scale_emb[-1].weight)
        nn.init.constant_(self.head_scale_emb[-1].bias, 0.02)

        # 2026-11-24 A+B+C+D：投影前可学习缩放。只新增 1 个 Cmax 向量，
        # 放在 proj_out 之后、final_proj 之前，给 final_proj 一个幅值条件更好的输入。
        # 初始全 1，接续 step1800 时前向零扰动；训练只学额外增益。
        self.pre_proj_scale = nn.Parameter(torch.ones(self.Cmax))
        self.pre_proj_low_l2 = 1e-5
        self.pre_proj_high_l2 = 1e-4

        self.ctx_proj = nn.Linear(in_dim, ctx_dim)
        self.gate_base = nn.Sequential(
            nn.Conv2d(self.Cmax + ctx_dim, self.Cmax, 1),
            nn.GELU(),
            nn.Conv2d(self.Cmax, 3, 1),
        )
        nn.init.zeros_(self.gate_base[-1].weight)
        nn.init.zeros_(self.gate_base[-1].bias)

        self.vlm_offset_mlp = nn.Sequential(
            nn.Linear(vlm_dim, self.Cmax), nn.GELU(), nn.Linear(self.Cmax, 3)
        )
        nn.init.zeros_(self.vlm_offset_mlp[-1].weight)
        nn.init.zeros_(self.vlm_offset_mlp[-1].bias)

        # alpha=1.0: all enhancement branches are in the residual from step 1.
        # At 0.01 the scalar is bf16 and its per-step update is below one bf16
        # quantum, so the whole fused branch stays effectively disconnected.
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.final_proj = nn.Conv2d(self.Cmax, out_dim, 1)
        # Zero output at step 0, like the old linear head. Gradients still reach
        # every branch through final_proj.weight.
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # 2026-11-24 双流 Head（方案1+2 第一阶段）：低流保留原 token 通路；
        # 高流直接吃 raw xt 的 2x2 patch 内高通，经 2 层 DWConv + 小 MLP，
        # 再由 g(sigma, out_channel) 门控加到 velocity 上。块内回流支路暂不启用。
        # 高流初始化成严格 identity，首步等效 alpha=0.6 的 hp(xt) 直通，避免随机
        # 分支把 2400 的低流权重打散；训练只学如何修正高通。
        self.high_hidden = self.Cmax
        self.high_in = nn.Conv2d(out_dim, self.high_hidden, 1)
        self.high_dw1 = nn.Conv2d(self.high_hidden, self.high_hidden, 3,
                                  padding=1, groups=self.high_hidden, bias=False)
        self.high_dw2 = nn.Conv2d(self.high_hidden, self.high_hidden, 3,
                                  padding=1, groups=self.high_hidden, bias=False)
        self.high_mlp = nn.Sequential(
            nn.Conv2d(self.high_hidden, self.high_hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(self.high_hidden // 2, self.high_hidden, 1),
        )
        self.high_out = nn.Conv2d(self.high_hidden, out_dim, 1)
        self.high_sigma_scale = nn.Sequential(
            nn.Linear(t_emb_dim, 128), nn.SiLU(), nn.Linear(128, self.high_hidden),
        )
        # 固定 5×5 深度卷积：hp 的网格低频 / 纹理高频分离，无学习参数。
        self.freq_split = FixedFreqSplit(out_dim, kernel=5, sigma=1.0)
        # 自动修复档：固定 3×3 带限平滑，压 hp_texture 的 2×2 内噪与 32px
        # 接缝，保留 4–8px 纹理。mix=0 时完全等价于原路径。
        self.tex_smooth = FixedFreqSplit(out_dim, kernel=3, sigma=0.8)
        self.tex_smooth_mix = _env_float("PYRAGATE_TEX_SMOOTH_MIX", 0.0)
        self.out_gain_min = _env_float("PYRAGATE_HEAD_OUT_GAIN_MIN", 0.08)
        # 低频对比度补偿：7×7 低通后按 γ 叠加，只抬低频幅值，
        # 不改 2×2 内噪和 32px 接缝；γ=0 等价于原输出。
        self.lf_split = FixedFreqSplit(out_dim, kernel=7, sigma=2.0)
        self.lf_gain = _env_float("PYRAGATE_LF_GAIN", 0.0)
        self.high_gate = DecoupledSigmaGate(
            out_dim, groups=4, sigma0=0.3, g_min=0.40, g_max=0.62,
            k=2.0, g_grid=0.15, r_thresh=0.90, alpha=0.35, r_gain=5.0,
        )
        with torch.no_grad():
            # 真实 4.05B: out_dim=192 < Cmax=1536；tiny 审计模型可能相反。
            # 只对可一一对应的通道建立恒等映射，避免 tiny/未来裁剪配置越界。
            n_identity = min(out_dim, self.high_hidden)
            idx = torch.arange(n_identity, device=self.high_in.weight.device)
            self.high_in.weight.zero_()
            self.high_in.weight[idx, idx, 0, 0] = 1.0
            self.high_in.bias.zero_()
            self.high_out.weight.zero_()
            self.high_out.weight[idx, idx, 0, 0] = 1.0
            self.high_out.bias.zero_()
            self.high_dw1.weight.zero_()
            self.high_dw2.weight.zero_()
            self.high_dw1.weight[:, 0, 1, 1] = 1.0
            self.high_dw2.weight[:, 0, 1, 1] = 1.0
        nn.init.zeros_(self.high_mlp[-1].weight)
        nn.init.zeros_(self.high_mlp[-1].bias)
        nn.init.zeros_(self.high_sigma_scale[-1].weight)
        nn.init.constant_(self.high_sigma_scale[-1].bias, 1.0)
        self._last_high_gate_mean = None
        self._last_high_rms = None
        self._last_gate_base = None
        self._last_gate_texture = None
        self._last_frame_residual = None
        self._last_gate_grid = None
        self._last_sigma_atten = None
        self._rms_target_ext = None

        # 2026-11-24 输出幅值外挂增益：final_proj 已被训到 std≈1.3e-4，不再
        # 重置它的权重；只新增 1 个标量把输出抬起来。下限 0.08 防再次坍缩。
        self.out_gain = nn.Parameter(torch.tensor(0.25))
        self.vmag_target = 0.15

        # 2026-11-26 A3：三分频独立 denoise 残差头。base final_proj 保留并冻结，
        # 三个 head 的输出以零残差方式叠加到 v 上；初始 state 与 5200 基线逐 bit 一致。
        # 每个 head = 3x3 depthwise（恒等初始化） + SiLU + 1x1 投影（零初始化）。
        # 1x1 单独无法改变空间频带，depthwise 提供空间邻域混频。
        def _make_band_head():
            dw = nn.Conv2d(self.Cmax, self.Cmax, 3, padding=1,
                           groups=self.Cmax, bias=True)
            nn.init.zeros_(dw.weight)
            nn.init.zeros_(dw.bias)
            with torch.no_grad():
                dw.weight[:, 0, 1, 1] = 1.0
            proj = nn.Conv2d(self.Cmax, self.out_dim, 1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            return nn.Sequential(dw, nn.SiLU(), proj)

        self.enable_band_heads = False
        self.enable_band_high = False
        self.register_buffer("band_head_scale_buf", torch.tensor(0.05))
        self.band_head_l = _make_band_head()
        self.band_head_m = _make_band_head()
        self.band_head_h = _make_band_head()
        self._last_band_heads = None
        self._last_band_base_v = None
        self._last_band_head_rms = None

        # 2026-11-26 bandhead 线：多尺度低秩频带头（共享投影 + 频域分频 +
        # 低秩子头）。默认关闭；关闭时前向完全不调用，与基线逐位等价。
        # 新参数 key 前缀 bandlr.，旧 Stage1 ckpt strict=False 可加载。
        self.enable_band_lr = False
        self.enable_band_lr_high = False
        # 延迟构造：默认关时不注册 bandlr.* 参数，state_dict 与旧 ckpt 逐 key
        # 一致；仅在 set_band_lr(True) 时实例化（见该方法）。
        self.bandlr = None
        self._last_bandlr = None
        self._last_bandlr_low = None
        self._last_bandlr_low_gate = None
        self._last_bandlr_structure_reg = None
        self._last_bandlr_rms = None
        # D：sigma 感知自适应 RMS 目标。sigma=1 -> 0.06+0.22*sigmoid(6)=0.2796，
        # sigma=0 -> 0.0711；高噪抬幅值，干净帧自动回落，不再用固定 0.15。
        self.rms_target_min = 0.06
        self.rms_target_span = 0.22
        self.rms_sigma_gain = 6.0
        # A：final_proj 幅值软 clamp（按绝对值保留符号，当前权重 std≈4.4e-4）。
        self.final_w_min = 1e-4
        self.final_w_max = 2e-2

        self.enable_cross = True
        self.enable_group_mask = True
        self.enable_channel_freq = True
        self.enable_head_scale = True
        self.enable_pre_proj_scale = True
        self.enable_final_weight_clamp = True
        self.enable_time_mod = True
        self.enable_rpe = True
        self.enable_sobel_grad = True
        self.enable_antialias = True
        self.enable_pyramid = True
        self.enable_frame_aux = True
        self.enable_vmag_aux = True
        self.enable_dual_high = True
        self.enable_head_scale = True
        self.enable_dualact_norm = True
        self.enable_dualact_gate = True
        self.enable_patch_norm = True
        self.frame_aux_weight = 1.0  # training arg multiplies this

    def set_switches(self, disabled=()):
        disabled = set(disabled or ())
        self.enable_cross = "cross" not in disabled
        self.enable_group_mask = "group_mask" not in disabled
        self.enable_channel_freq = "channel_freq" not in disabled
        self.enable_head_scale = "head_scale" not in disabled
        self.enable_pre_proj_scale = "pre_proj_scale" not in disabled
        self.enable_final_weight_clamp = "final_weight_clamp" not in disabled
        self.enable_time_mod = "time_mod" not in disabled
        self.enable_rpe = "rpe" not in disabled
        self.enable_sobel_grad = "sobel" not in disabled
        self.enable_antialias = "antialias" not in disabled
        self.enable_pyramid = "pyramid" not in disabled
        self.enable_frame_aux = "frame_aux" not in disabled
        self.enable_vmag_aux = "vmag" not in disabled
        self.enable_dual_high = "dual_high" not in disabled
        self.enable_head_scale = "head_scale" not in disabled
        self.enable_dualact_norm = "dualact_norm" not in disabled
        self.enable_dualact_gate = "dualact_gate" not in disabled
        self.enable_patch_norm = "patch_norm" not in disabled
        self.band_norm.enable_dc_norm = "bandnorm_dc" not in disabled
        self.band_norm.enable_res_norm = "bandnorm_res" not in disabled

    def set_band_heads(self, enabled: bool = True, enable_high: bool = False) -> None:
        """Enable A3 split heads and freeze the old final_proj base.

        The three residual heads are zero-init, so enabling them is an exact
        identity at step 0. Freezing final_proj prevents the old projection from
        absorbing the new band supervision and keeps the decomposition clean.

        ``enable_high=False`` also freezes the high head parameters, so the main
        x0 loss cannot leak gradient into the "closed" high branch.
        """
        self.enable_band_heads = bool(enabled)
        self.enable_band_high = bool(enabled and enable_high)
        for p in self.band_head_h.parameters():
            p.requires_grad_(self.enable_band_high)
        if enabled:
            self.final_proj.requires_grad_(False)
        else:
            self.final_proj.requires_grad_(True)
            self._last_band_heads = None
            self._last_band_base_v = None
            self._last_band_head_rms = None

    def set_band_lr(self, enabled: bool = True, enable_high: bool = False) -> None:
        """开关多尺度低秩频带头（``--band-head-lr``）。

        默认关闭：前向不调用 ``self.bandlr``，输出与基线一致。启用时只把
        三分量挂到 ``_last_bandlr*`` 审计缓存，**不自行叠加到 v**；组合方式
        由集成线在 unpatchify 之后接线。``enable_high=False`` 时冻结高频
        子头参数，防止主损失把关闭的高频分支训起来。
        """
        self.enable_band_lr = bool(enabled)
        self.enable_band_lr_high = bool(enabled and enable_high)
        if enabled:
            if self.bandlr is None:
                self.bandlr = MultiBandLowRankHead(
                    in_dim=self.Cmax, out_dim=self.out_dim,
                    rank=8, shared_rank=8)
                # set_band_lr 可能晚于 model.to(device)/.to(dtype) 调用（采样端
                # 在 load_state_dict 之前惰性构造）。这里主动对齐已有参数，
                # 避免新 bandlr 留在 CPU/fp32 导致前向 device mismatch。
                ref = next(self.parameters())
                self.bandlr.to(device=ref.device, dtype=ref.dtype)
            # 先整体恢复可训，再按 enable_high 冻结高频；这样 freeze_backbone
            # 之后再次 set_band_lr 不会把整个新头留在冻结状态。
            for p in self.bandlr.parameters():
                p.requires_grad_(True)
            for p in self.bandlr.u_h.parameters():
                p.requires_grad_(self.enable_band_lr_high)
            for p in self.bandlr.v_h.parameters():
                p.requires_grad_(self.enable_band_lr_high)
        else:
            # 关断即摘除模块：默认关 state_dict 零 bandlr key（硬约束 A）。
            self.bandlr = None
        self._last_bandlr = None
        self._last_bandlr_low = None
        self._last_bandlr_low_gate = None
        self._last_bandlr_structure_reg = None
        self._last_bandlr_rms = None

    def init_bandlr_from_legacy(self) -> dict:
        """热启动：用既有 ``band_head_l/m/h`` 的 1x1 投影 SVD 初始化 bandlr。

        返回 ``MultiBandLowRankHead.load_from_legacy`` 的报告 dict；共享投影
        无旧权重来源，保留零初始化。
        """
        if self.bandlr is None:
            self.set_band_lr(True)
        return self.bandlr.load_from_legacy(
            self.band_head_l, self.band_head_m, self.band_head_h)

    def freeze_disabled_modules(self) -> int:
        """从优化器中排除已禁用的新模块，返回新冻结的标量数。

        只把 requires_grad 从 True 改为 False，不重新启用其他参数，因此可在
        freeze_backbone/train_mod_only 之后安全调用。开关仍保留完整结构，
        checkpoint key 与采样端保持一致。
        """
        n = 0
        for name, p in self.named_parameters():
            disable = False
            if name.startswith("dualact_gate."):
                disable = not self.enable_dualact_gate
            elif name.startswith("patch_norm."):
                disable = not self.enable_patch_norm
            elif name.startswith("band_norm."):
                is_dc = ".dc_" in name
                is_res = ".res_" in name
                disable = (not self.enable_dualact_norm
                           or (is_dc and not self.band_norm.enable_dc_norm)
                           or (is_res and not self.band_norm.enable_res_norm))
            if disable and p.requires_grad:
                p.requires_grad_(False)
                n += p.numel()
        return n

    def freeze_dead_modules(self) -> int:
        """冻结从未进入 forward 的历史预留模块，保留 key 以兼容 ckpt。"""
        n = 0
        for name, p in self.named_parameters():
            if name.startswith(self._DEAD_MODULE_PREFIXES) and p.requires_grad:
                p.requires_grad_(False)
                n += p.numel()
        return n

    def parameter_audit(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        dead = sum(p.numel() for n, p in self.named_parameters()
                   if n.startswith(self._DEAD_MODULE_PREFIXES))
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "dead_reserved": dead,
                "dead_pct": round(100.0 * dead / max(total, 1), 3),
                "trainable": trainable, "trainable_pct": round(
                    100.0 * trainable / max(total, 1), 3)}

    @torch.no_grad()
    def clamp_weights(self):
        """A：optimizer.step 之后把 final_proj 权重幅值投影到 [1e-4, 2e-2]。

        保留符号；绝对值为 0 的权重不投影，避免零初始化参数被永久钉死。
        """
        if not self.enable_final_weight_clamp:
            return
        w = self.final_proj.weight
        aw = w.abs()
        mask = aw > 0
        wc = torch.where(mask, torch.sign(w) * torch.clamp(
            aw, self.final_w_min, self.final_w_max), w)
        w.copy_(wc)
        if os.environ.get("CLAMP_STAT") == "1":
            with torch.no_grad():
                hi = float((aw >= self.final_w_max * 0.999).float().mean())
                lo = float((aw <= self.final_w_min * 1.001).float().mean())
                self._last_clamp_sat = (hi, lo)
                self._last_w_absmean = float(aw.mean())

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context_vec: torch.Tensor,
        frames: int = 1,
        vlm_emb: torch.Tensor | None = None,
        use_vlm: bool = False,
        training: bool = False,
        sigma: torch.Tensor | None = None,
        pixel_hp: torch.Tensor | None = None,
        frame_residual: torch.Tensor | None = None,
    ):
        Bt, C, H, W = x.shape
        x_r = self.proj_in(x)

        # 1) 复用 GroupMaskGate 的语义 mask：组间分歧大的位置视为边界，
        #    RPE / 金字塔残差在这些位置自动关小，平坦连通区保留邻域聚合。
        if self.enable_group_mask:
            gm = self.group_mask.mask(x_r)                     # [Bt,G,H,W]
            spatial = (1.0 - gm.var(dim=1, keepdim=True)).clamp(0.0, 1.0)
        else:
            spatial = x_r.new_ones(Bt, 1, H, W)

        # 2) 1536×2 标量的通道频域门：低频走邻域，高频直通。
        low, high, g = self.channel_gate(x_r)
        # 2b) 内嵌频带感知 Norm + 多源 Dual-Act 门控（Rezero 初始恒等）。
        if self.enable_dualact_norm:
            low, high = self.band_norm(low, high)
        self._last_dualact_dc = None
        self._last_dualact_res = None
        if self.enable_dualact_gate:
            low, high, self._last_dualact_dc, self._last_dualact_res = \
                self.dualact_gate(low, high, sigma, frames, context_vec)

        # 低频支路：anti-alias 只作用于低频通道；RPE/Pyramid 都只吃 low。
        low_base = self.anti_alias(low) if self.enable_antialias else low
        rpe_res = (self.rpe(low_base, spatial_mask=spatial, channel_mask=g)
                   if self.enable_rpe else low_base.new_zeros(()))
        pyr_res = (self.pyramid_shortcut(low_base) * spatial
                   if self.enable_pyramid else low_base.new_zeros(()))

        # 支路独立 norm + 独立 time-scale：切断 sigma 与主干时间缩放的耦合。
        branch = rpe_res + pyr_res
        scale_t = None
        if self.enable_head_scale:
            b = self.head_norm(branch.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            scale_t = self.head_scale_emb(t_emb).view(Bt, self.Cmax, 1, 1)
            branch = b * scale_t

        # 高频支路完全直通，不做任何邻域融合。
        feat = high + low_base + self.alpha * branch
        feat = self.proj_out(feat)
        if self.enable_pre_proj_scale:
            feat = feat * self.pre_proj_scale.view(1, self.Cmax, 1, 1)

        self._last_low_gate = g.detach()
        self._last_spatial_mask = spatial.detach()
        self._last_feat = feat

        # 2026-11-26 bandhead 线：低秩频带头（默认关闭）。关闭时不进入本分支，
        # 不申请额外算子、不改 out/v。启用时三分量只挂审计缓存，叠加由集成线做。
        self._last_bandlr = None
        self._last_bandlr_low = None
        self._last_bandlr_low_gate = None
        self._last_bandlr_structure_reg = None
        self._last_bandlr_rms = None
        if self.enable_band_lr:
            blr = self.bandlr(feat)
            self._last_bandlr = (blr["low"], blr["mid"], blr["high"])
            self._last_bandlr_low = blr["low"]
            self._last_bandlr_low_gate = blr["low_gate"]
            self._last_bandlr_structure_reg = blr["structure_reg"]
            self._last_bandlr_rms = blr["band_rms"]
        self._last_scale_t = None if scale_t is None else scale_t.detach()
        self._last_scale_t_reg = None
        self._last_pre_proj_l2 = None
        self._last_corr_defect = None
        if training:
            # corr loss 已迁到训练器的 x0hat patch-mean 上，并做 sigma 门控；
            # 不再在 velocity 的 token feature 上计算，避免把速度场压成块常量。
            # 呼应 1：scale_t 只做 sigma 时序相对缩放，RMS 限制在 [0.6,1.4]。
            # 注意不能写 clamp(rms,...) 去回归 rms.detach()，那样越界时梯度为 0；
            # 这里用双侧 hinge，越界才有梯度。
            if scale_t is not None:
                s_rms = torch.sqrt(torch.mean(scale_t.float() ** 2, dim=1) + 1e-12)
                s_reg = (torch.relu(0.6 - s_rms).pow(2)
                         + torch.relu(s_rms - 1.4).pow(2)).mean() * 1e-4
                self._last_scale_t_reg = s_reg

            # 呼应 2：pre_proj_scale 按低频/高频通道分组 L2；低频宽松、高频更强。
            if self.enable_pre_proj_scale:
                gm_ch = g.detach().float().mean(dim=(0, 2, 3))       # [C]
                low_pen = (self.pre_proj_scale.float().pow(2) * gm_ch).mean()
                high_pen = (self.pre_proj_scale.float().pow(2) * (1.0 - gm_ch)).mean()
                self._last_pre_proj_l2 = (self.pre_proj_low_l2 * low_pen
                                          + self.pre_proj_high_l2 * high_pen)

        # A：final_proj 权重幅值投影在 optimizer.step 之后执行（见 clamp_weights）。
        # 不在前向动态构造权重，避免 ROCm 1x1 conv 的非参数化 backward 额外
        # 申请大块 workspace。
        raw_out = self.final_proj(feat)
        gain = torch.clamp(self.out_gain, min=self.out_gain_min)
        v = raw_out * gain

        # 双流高流：pixel_hp 是 raw xt 的 2x2 patch 内高通（已按 unpatchify
        # 的 [ph,pw,C] 顺序铺成 out_dim 通道）。固定 5×5 DWConv 把 hp 拆成
        # 网格低频 hp_grid 与纹理高频 hp_texture：
        #   hp_out = 0.15 * hp_grid + g_texture(σ, r_t) * hp_texture
        # hp 先 detach：门控参数/高频支路的梯度不会沿 hp 反向进入主干 DiT。
        # g_texture 只由 σ 和帧间残差决定，不含 hp 特征，因此门控梯度不会
        # 借特征路径污染主干时序梯度。
        self._last_high_gate_mean = None
        self._last_high_rms = None
        self._last_gate_base = None
        self._last_gate_texture = None
        self._last_frame_residual = None
        self._last_gate_grid = None
        if self.enable_dual_high and pixel_hp is not None:
            hp = pixel_hp.to(dtype=v.dtype).detach()
            if hp.shape[1] != self.out_dim:
                raise RuntimeError(f"pixel_hp C={hp.shape[1]} != out_dim={self.out_dim}")
            hp_grid = self.freq_split(hp, frames)
            hp_texture = hp - hp_grid
            if self.tex_smooth_mix > 0.0:
                tex_s = self.tex_smooth(hp_texture, frames)
                hp_texture = (hp_texture * (1.0 - self.tex_smooth_mix)
                              + tex_s * self.tex_smooth_mix)
            gate = self.high_gate(sigma, Bt, frame_residual).to(dtype=v.dtype)
            g_grid = self.high_gate.g_grid.to(dtype=v.dtype)
            # 高流是“细节残差”，不是高噪声端的速度来源。旧实现把 g_grid=0.15
            # 固定灌到 σ=1，等于把纯噪声的 2×2 高通直接塞进 velocity；实测
            # σ=1 时 highpass 与 target 相关 0.94、RMS 0.84/0.90，主路径被短路。
            # 乘 (1-σ)：σ=1 关闭、σ=0.5 半开、σ→0 全开，保留低噪纹理能力。
            if sigma is None:
                # high_gate 对 None 有自己的默认门；这里同步给全开衰减，避免 .reshape 崩。
                sigma_atten = v.new_ones(Bt, 1, 1, 1)
            else:
                sigma_atten = (1.0 - sigma.reshape(-1).float()).clamp(0.0, 1.0)
                sigma_atten = sigma_atten.view(Bt, 1, 1, 1).to(dtype=v.dtype)
            self._last_sigma_atten = sigma_atten.mean().detach()
            hp_mod = (g_grid * sigma_atten) * hp_grid + (gate * sigma_atten) * hp_texture
            h = self.high_in(hp_mod)
            h = self.high_dw1(h)
            h = self.high_dw2(h)
            h = h + self.high_mlp(h)
            h = h * self.high_sigma_scale(t_emb).view(Bt, self.high_hidden, 1, 1)
            h = self.high_out(h)
            v = v + h
            self._last_high_gate_mean = gate.mean().detach()
            self._last_high_rms = h.detach().float().pow(2).mean().sqrt()
            self._last_gate_base = self.high_gate._last_base
            self._last_gate_texture = self.high_gate._last_texture
            self._last_frame_residual = self.high_gate._last_residual
            self._last_gate_grid = self.high_gate.g_grid.detach().float()
        out = v
        if self.lf_gain > 0.0:
            v_lf = self.lf_split(v, frames)
            out = v + self.lf_gain * v_lf

        if self.enable_patch_norm:
            out = self.patch_norm(out)
            # Only sync when this module is actually active.  With patch_norm
            # disabled this float() forced a GPU synchronize every forward and
            # stalled Stage-2 after the first optimizer step.
            self._last_patch_norm_rz = float(self.patch_norm.rz.detach())
        else:
            self._last_patch_norm_rz = 0.0

        # A3-v3：这里只产出三个 raw 残差头，不再在 token 空间相加。
        # 由 UViT3D 在 unpatchify 后做 latent 空间带限、缩放、残差相加；
        # base_v 与三头之和仍精确等于最终 v，带 loss 不需要扣高流/低频支路。
        self._last_band_heads = None
        self._last_band_base_v = None
        self._last_band_head_rms = None
        if self.enable_band_heads:
            b_l = self.band_head_l(feat)
            b_m = self.band_head_m(feat)
            if self.enable_band_high:
                b_h = self.band_head_h(feat)
            else:
                b_h = b_l.new_zeros(b_l.shape)
            self._last_band_base_v = out.detach()
            self._last_band_heads = (b_l, b_m, b_h)
            self._last_band_head_rms = tuple(
                float(x.detach().float().pow(2).mean().sqrt()) for x in (b_l, b_m, b_h))

        # 幅值目标：优先用训练器按当前 batch 真实 target v=noise-x0 算出的
        # per-sample RMS（_rms_target_ext），废除固定 0.06+0.22*sigmoid 曲线。
        self._last_vmag_loss = None
        self._last_out_rms = None
        self._last_rms_target = None
        if training and self.enable_vmag_aux:
            out_rms = torch.sqrt(torch.mean(out.float() ** 2, dim=1, keepdim=True) + 1e-12)
            ext = getattr(self, "_rms_target_ext", None)
            if ext is not None:
                target = ext.to(device=out_rms.device, dtype=torch.float32)
                target = target.view(-1, 1, 1, 1).expand_as(out_rms)
            elif sigma is not None:
                # 兼容旧调用；正式训练器不会走这里。
                sf = sigma.view(Bt, 1, 1, 1).float()
                target = self.rms_target_min + self.rms_target_span * torch.sigmoid(
                    self.rms_sigma_gain * sf)
            else:
                target = torch.full_like(out_rms, self.vmag_target)
            self._last_vmag_loss = F.mse_loss(out_rms, target.expand_as(out_rms))
            self._last_out_rms = out_rms.mean().detach()
            self._last_rms_target = target.mean().detach()

        aux_loss = None
        if training and self.enable_frame_aux and frames > 1:
            B = out.shape[0] // frames
            o = out.view(B, frames, self.out_dim, H, W)
            aux_loss = F.l1_loss(o[:, 1:], o[:, :-1]) * self.frame_aux_weight
        return out, aux_loss
