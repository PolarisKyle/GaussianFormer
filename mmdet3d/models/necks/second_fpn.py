import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SECONDFPN(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_strides, use_conv_for_no_stride=False, **kwargs):
        super().__init__()
        assert len(in_channels) == len(out_channels) == len(upsample_strides)
        self.deblocks = nn.ModuleList()

        for in_c, out_c, stride in zip(in_channels, out_channels, upsample_strides):
            if stride > 1:
                s = int(round(stride))
                block = nn.Sequential(
                    nn.ConvTranspose2d(in_c, out_c, kernel_size=s, stride=s, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            elif stride < 1:
                s = int(round(1 / stride))
                block = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=s, stride=s, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            self.deblocks.append(block)

    def forward(self, x):
        outs = [deblock(feat) for deblock, feat in zip(self.deblocks, x)]
        if len(outs) > 1:
            h = min(o.shape[-2] for o in outs)
            w = min(o.shape[-1] for o in outs)
            outs = [F.interpolate(o, size=(h, w), mode='bilinear', align_corners=False) if o.shape[-2:] != (h, w) else o for o in outs]
            out = torch.cat(outs, dim=1)
        else:
            out = outs[0]
        return [out]
