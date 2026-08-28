# SPDX-License-Identifier: Apache-2.0
"""Export-time, numerically-equivalent rewrites that remove transpose/reshape/
concat overhead from the traced graph (TensorRT reformat + kgen kernels).
Upstream modules are monkeypatched in-process only; weights are untouched.

1. CausalNormalize (num_groups=1): time2batch -> GroupNorm -> batch2time becomes a
   transpose-free per-frame normalization over (C,H,W) on the 5D tensor.
2. CausalConv3d: repeat+cat+F.pad becomes a temporal replicate Pad with the
   spatial zero padding folded into the conv's padding attribute.
3. repeat_interleave-based nearest upsampling becomes F.interpolate (ONNX Resize).
"""

import types

import torch
import torch.nn.functional as F


def install_fusion_friendly_patches(model: torch.nn.Module, which=("norm", "conv")) -> dict:
    from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d
    from cosmos_predict1.tokenizer.modules.utils import CausalNormalize

    counts = {"norm": 0, "conv": 0}

    def norm_forward(self, x):
        if self.num_groups != 1:
            return self.norm(x)
        dims = (1, 3, 4)  # per (b, t): stats over (c, h, w) == GroupNorm(1 group) per frame
        mean = x.mean(dim=dims, keepdim=True)
        var = (x - mean).pow(2).mean(dim=dims, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.norm.eps)
        C = x.shape[1]
        return y * self.norm.weight.view(1, C, 1, 1, 1) + self.norm.bias.view(1, C, 1, 1, 1)

    def conv_forward(self, x):
        if self.time_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0), mode="replicate")
        return self.conv3d(x)

    for m in model.modules():
        if isinstance(m, CausalNormalize) and "norm" in which:
            m.forward = types.MethodType(norm_forward, m)
            counts["norm"] += 1
        elif isinstance(m, CausalConv3d) and "conv" in which:
            assert m.pad_mode == "constant", m.pad_mode
            p = m.spatial_pad  # (w, w, h, h)
            m.conv3d.padding = (0, p[2], p[0])
            m.forward = types.MethodType(conv_forward, m)
            counts["conv"] += 1
    return counts


_orig_repeat_interleave = torch.Tensor.repeat_interleave


def _ri_as_resize(self, repeats, dim=None, *args, **kwargs):
    """Nearest-neighbour duplication along one of the T/H/W dims of a 5D tensor
    == F.interpolate(mode='nearest') with that scale factor; falls back otherwise."""
    if isinstance(repeats, int) and dim is not None and self.dim() == 5 and dim in (2, 3, 4) and not args and not kwargs:
        scale = [1.0, 1.0, 1.0]
        scale[dim - 2] = float(repeats)
        return F.interpolate(self, scale_factor=tuple(scale), mode="nearest")
    return _orig_repeat_interleave(self, repeats, dim, *args, **kwargs)


def install_resize_upsample():
    torch.Tensor.repeat_interleave = _ri_as_resize


def uninstall_resize_upsample():
    torch.Tensor.repeat_interleave = _orig_repeat_interleave
