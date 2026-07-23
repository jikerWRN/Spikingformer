import math
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models import create_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = ['Spikingformer']


# neuronal threshold
thresh = 0.5
# hyper-parameters of approximate function
lens = 0.5
# decay constants
decay = 0.25


class AdaptiveMultiStepLIFNode(MultiStepLIFNode):
    class QuantUpperBoundSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, upper_bound, checkpoint_upper_bound):
            ctx.save_for_backward(input, upper_bound, checkpoint_upper_bound)
            return torch.where(input >= upper_bound * 0.5, upper_bound.to(input.dtype), torch.zeros_like(input))

        @staticmethod
        def backward(ctx, grad_output):
            input, upper_bound, checkpoint_upper_bound = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 2.0 * checkpoint_upper_bound] = 0
            return grad_input, None, None

    def __init__(
        self,
        tau=2.0,
        decay_input=False,
        v_threshold=thresh,
        v_reset=0.0,
        surrogate_function=None,
        detach_reset=True,
        backend="torch",
        store_v_seq=False,
        decay_factor=decay,
        percentile=(0.7, 0.8, 0.9, 0.99),
        spike_percentile=None,
        default_upper_bound=1.0,
        upper_bound_method="input",
        ema_momentum=0.001,
        max_change_ratio=0.05,
        selected_percentile=0.99,
        spike_selected_percentile=None,
        max_quantile_samples=1000000,
    ):
        if spike_percentile is not None:
            percentile = spike_percentile
        if spike_selected_percentile is not None:
            selected_percentile = spike_selected_percentile
        if backend != "torch":
            raise ValueError("AdaptiveMultiStepLIFNode only supports backend='torch'")
        if upper_bound_method not in ("input", "bn"):
            raise ValueError("upper_bound_method must be 'input' or 'bn'")
        if not 0 < ema_momentum <= 1:
            raise ValueError("ema_momentum must be in (0, 1]")
        if max_change_ratio < 0:
            raise ValueError("max_change_ratio must be non-negative")
        if not 0 <= decay_factor <= 1:
            raise ValueError("decay_factor must be in [0, 1]")

        if torch.is_tensor(percentile):
            percentile = percentile.detach().clone().to(dtype=torch.float32).reshape(-1)
        elif isinstance(percentile, Sequence) and not isinstance(percentile, (str, bytes)):
            percentile = torch.tensor(list(percentile), dtype=torch.float32)
        else:
            percentile = torch.tensor([float(percentile)], dtype=torch.float32)
        if percentile.numel() == 0:
            raise ValueError("percentile must contain at least one value")
        if torch.any((percentile < 0) | (percentile > 1)):
            raise ValueError("all percentile values must be in [0, 1]")

        init_kwargs = dict(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            detach_reset=detach_reset,
            backend=backend,
        )
        if surrogate_function is not None:
            init_kwargs["surrogate_function"] = surrogate_function
        super().__init__(**init_kwargs)

        self.decay_factor = float(decay_factor)
        self.store_v_seq = store_v_seq
        self.upper_bound_method = upper_bound_method
        self.ema_momentum = ema_momentum
        self.max_change_ratio = max_change_ratio
        self.selected_percentile = float(selected_percentile)
        self.max_quantile_samples = int(max_quantile_samples)
        if self.max_quantile_samples <= 0:
            raise ValueError("max_quantile_samples must be positive")

        self.register_buffer("percentile", percentile)
        self.register_buffer("upper_bound_default", torch.tensor(float(default_upper_bound)), persistent=False)

        num_percentiles = int(self.percentile.numel())
        default_stats = torch.full((num_percentiles,), float(default_upper_bound), dtype=torch.float32)
        self.register_buffer("checkpoint_upper_bound_mean", default_stats.clone())
        self.register_buffer("upper_bound_mean", default_stats.clone())
        self.register_buffer("upper_bound_count", torch.tensor(0.0))
        self.register_buffer("upper_bound_raw_last", default_stats.clone())
        self.register_buffer("upper_bound_raw_min", torch.full((num_percentiles,), float("inf"), dtype=torch.float32))
        self.register_buffer("upper_bound_raw_max", torch.full((num_percentiles,), float("-inf"), dtype=torch.float32))
        self.register_buffer("upper_bound_raw_sum", torch.zeros(num_percentiles, dtype=torch.float32))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        checkpoint_mean_key = prefix + "checkpoint_upper_bound_mean"
        upper_bound_mean_key = prefix + "upper_bound_mean"
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # if checkpoint_mean_key not in state_dict and upper_bound_mean_key in state_dict:
        
        if upper_bound_mean_key not in state_dict:
            raise("Value Errors!!!")
        
        self.checkpoint_upper_bound_mean.copy_(self.upper_bound_mean)

    def compute_upper_bound(self, x):
        x = x.detach().reshape(-1)
        positive_x = x[x > 0]
        if positive_x.numel() == 0:
            return None
        if positive_x.numel() > self.max_quantile_samples:
            step = math.ceil(positive_x.numel() / self.max_quantile_samples)
            positive_x = positive_x[::step][:self.max_quantile_samples]
        percentile = self.percentile.to(device=positive_x.device, dtype=positive_x.dtype)
        return torch.quantile(positive_x, percentile)

    def compute_bn_upper_bound(self, bn):
        if bn.affine:
            mean = bn.bias.detach()
            std = bn.weight.detach().abs().clamp_min(1e-6)
        else:
            mean = torch.zeros_like(bn.running_mean)
            std = torch.ones_like(bn.running_var)

        normal_0_cdf = 0.5 * (1.0 + torch.erf((0.0 - mean) / (std * math.sqrt(2.0))))
        percentile = self.percentile.to(device=mean.device, dtype=mean.dtype)
        target_cdf = normal_0_cdf.unsqueeze(0) + percentile.unsqueeze(1) * (1.0 - normal_0_cdf.unsqueeze(0))
        target_cdf = target_cdf.clamp(1e-6, 1.0 - 1e-6)
        upper_bound = mean.unsqueeze(0) + std.unsqueeze(0) * math.sqrt(2.0) * torch.erfinv(2.0 * target_cdf - 1.0)
        return upper_bound.mean(dim=1)

    def update_upper_bound_mean(self, upper_bound):
        upper_bound = upper_bound.detach().to(self.upper_bound_mean.device, dtype=self.upper_bound_mean.dtype)
        upper_bound = upper_bound.clamp_min(1e-6)
        self.upper_bound_raw_last.copy_(upper_bound)
        self.upper_bound_raw_min.copy_(torch.minimum(self.upper_bound_raw_min, upper_bound))
        self.upper_bound_raw_max.copy_(torch.maximum(self.upper_bound_raw_max, upper_bound))
        self.upper_bound_raw_sum.add_(upper_bound)

        if self.max_change_ratio > 0:
            lower = self.upper_bound_mean * (1.0 - self.max_change_ratio)
            upper = self.upper_bound_mean * (1.0 + self.max_change_ratio)
            upper_bound = upper_bound.clamp(lower, upper)
        self.upper_bound_mean.mul_(1.0 - self.ema_momentum).add_(upper_bound * self.ema_momentum)
        self.upper_bound_mean.clamp_min_(1e-6)
        self.upper_bound_count.add_(1.0)

    def get_shared_upper_bound(self, x, bn=None):
        if self.training and self.upper_bound_method == "bn" and bn is not None:
            upper_bound = self.compute_bn_upper_bound(bn)
            self.update_upper_bound_mean(upper_bound)
        elif self.training:
            upper_bound = self.compute_upper_bound(x)
            if upper_bound is not None:
                self.update_upper_bound_mean(upper_bound)

        percentile = self.percentile.to(device=self.upper_bound_mean.device)
        selected = torch.tensor(self.selected_percentile, device=percentile.device, dtype=percentile.dtype)
        matches = torch.isclose(percentile, selected, rtol=1e-6, atol=1e-8)
        if not torch.any(matches):
            available = ", ".join("{:g}".format(v.item()) for v in percentile.cpu())
            raise ValueError(
                "selected_percentile={:g} is not in percentile values: {}".format(
                    self.selected_percentile, available
                )
            )
        percentile_idx = torch.nonzero(matches, as_tuple=False)[0, 0]
        # upper_bound = self.checkpoint_upper_bound_mean.reshape(-1)[percentile_idx].clamp_min(1e-6)
        upper_bound = self.upper_bound_mean.reshape(-1)[percentile_idx].clamp_min(1e-6)
        return upper_bound.to(device=x.device, dtype=x.dtype)
        # return self.upper_bound_default.to(device=x.device, dtype=x.dtype)

    def spike_quantize(self, x, upper_bound=None, checkpoint_upper_bound=None, bn=None):
        if upper_bound is None:
            upper_bound = self.get_shared_upper_bound(x, bn)
        else:
            upper_bound = upper_bound.to(device=x.device, dtype=x.dtype)
        # Keep the custom 0 / upper_bound forward amplitude while restoring
        # SpikingJelly's original surrogate-gradient backward behavior.
        # return upper_bound * self.surrogate_function(x - upper_bound * 0.5)
        return self.QuantUpperBoundSpike.apply(x, upper_bound, checkpoint_upper_bound)

    def _init_state_tensor(self, x):
        v = getattr(self, "v", 0.0)
        if torch.is_tensor(v):
            v = v.to(device=x.device, dtype=x.dtype)
            if v.shape == x.shape:
                return v
            if v.numel() == 1:
                return torch.full_like(x, float(v.item()))
            try:
                return torch.zeros_like(x) + v
            except RuntimeError:
                return torch.zeros_like(x)
        return torch.full_like(x, float(v))

    def multi_step_forward(self, x_seq, bn=None):
        state = self._init_state_tensor(x_seq[0])
        shared_upper_bound = self.get_shared_upper_bound(x_seq, bn)
        checkpoint_upper_bound = self.checkpoint_upper_bound_mean.reshape(-1)[2].clamp_min(1e-6)
        spike_seq = torch.zeros_like(x_seq)
        v_seq = [] if self.store_v_seq else None

        for t in range(x_seq.shape[0]):
            state = state * self.decay_factor + x_seq[t]
            spike = self.spike_quantize(state, shared_upper_bound, checkpoint_upper_bound)
            state = state - spike.detach()
            spike_seq[t] = spike
            if v_seq is not None:
                v_seq.append(state.clone())

        self.v = state.detach()
        if v_seq is not None:
            self.v_seq = torch.stack(v_seq, dim=0)
        return spike_seq.contiguous()

    def forward(self, x_seq, bn=None):
        return self.multi_step_forward(x_seq, bn=bn)


class mem_update(nn.Module):
    def __init__(
        self,
        act=False,
        spike_percentile=(0.7, 0.8, 0.9, 0.99),
        upper_bound_method="input",
        spike_selected_percentile=0.99,
        **neuron_kwargs,
    ):
        super().__init__()
        self.act = act
        self.qtrick = AdaptiveMultiStepLIFNode(
            percentile=spike_percentile,
            upper_bound_method=upper_bound_method,
            selected_percentile=spike_selected_percentile,
            **neuron_kwargs,
        )

    def forward(self, x, bn=None):
        return self.qtrick(x, bn=bn)

    
class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.,
        spike_percentile=(0.7, 0.8, 0.9, 0.99),
        spike_selected_percentile=0.99,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1,
        spike_percentile=(0.7, 0.8, 0.9, 0.99),
        spike_selected_percentile=0.99,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )

        # self.attn_lif = mem_update(
        #     spike_percentile=spike_percentile,
        #     spike_selected_percentile=spike_selected_percentile,
        # )
        self.attn_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 1, 3, 2, 4)

        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class SpikingTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        spike_percentile=(0.7, 0.8, 0.9, 0.99),
        spike_selected_percentile=0.99,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


SpikingTransformer1 = SpikingTransformer
SpikingTransformer2 = SpikingTransformer
SpikingTransformer3 = SpikingTransformer
SpikingTransformer4 = SpikingTransformer


class SpikingTokenizer(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        spike_percentile=(0.7, 0.8, 0.9, 0.99),
        spike_selected_percentile=0.99,
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)

        self.block1_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.block2_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)

        self.block3_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims // 1)

        self.block4_lif = mem_update(
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)

        x = self.block1_lif(x).flatten(0, 1)
        x = self.block1_conv(x)
        x = self.block1_bn(x).reshape(T, B, -1, H, W)

        x = self.block2_lif(x).flatten(0, 1)
        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)

        x = self.block3_lif(x).flatten(0, 1)
        x = self.block3_mp(x)
        x = self.block3_conv(x)
        x = self.block3_bn(x).reshape(T, B, -1, int(H / 2), int(W / 2))

        x = self.block4_lif(x).flatten(0, 1)
        x = self.block4_mp(x)
        x = self.block4_conv(x)
        x = self.block4_bn(x).reshape(T, B, -1, int(H / 4), int(W / 4))

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        pretrained_cfg=None,
        spike_percentile=(0.7, 0.8, 0.9, 0.99),
        spike_selected_percentile=0.99,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        patch_embed = SpikingTokenizer(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            spike_percentile=spike_percentile,
            spike_selected_percentile=spike_selected_percentile,
        )

        block = nn.ModuleList()
        block.append(
            SpikingTransformer1(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[0],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
                spike_percentile=spike_percentile,
                spike_selected_percentile=spike_selected_percentile,
            )
        )
        block.append(
            SpikingTransformer2(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[1],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
                spike_percentile=spike_percentile,
                spike_selected_percentile=spike_selected_percentile,
            )
        )
        block.append(
            SpikingTransformer3(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[2],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
                spike_percentile=spike_percentile,
                spike_selected_percentile=spike_selected_percentile,
            )
        )
        block.append(
            SpikingTransformer4(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[3],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios,
                spike_percentile=spike_percentile,
                spike_selected_percentile=spike_selected_percentile,
            )
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    allowed_kwargs = {
        'img_size_h', 'img_size_w', 'patch_size', 'in_channels', 'num_classes',
        'embed_dims', 'num_heads', 'mlp_ratios', 'qkv_bias', 'qk_scale',
        'drop_rate', 'attn_drop_rate', 'drop_path_rate', 'norm_layer',
        'depths', 'sr_ratios', 'T', 'pretrained_cfg',
        'spike_percentile', 'spike_selected_percentile',
    }
    kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
    model = vit_snn(**kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()

    model.eval()
    y = model(input)

    print(model)
    print(y.shape)
    print('Test Good!')
