import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.backbone_end_level = self.num_ins if end_level == -1 else end_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - (self.backbone_end_level - self.start_level)
        if extra_levels > 0:
            for _ in range(extra_levels):
                self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, inputs):
        assert len(inputs) >= self.backbone_end_level

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals))
        ]

        if self.num_outs > len(outs):
            for i in range(self.num_outs - len(outs)):
                if not self.add_extra_convs:
                    outs.append(F.max_pool2d(outs[-1], kernel_size=2, stride=2))
                else:
                    src = outs[-1]
                    if self.relu_before_extra_convs:
                        src = F.relu(src)
                    outs.append(self.fpn_convs[len(laterals) + i](src))

        return tuple(outs)
