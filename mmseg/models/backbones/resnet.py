import torch
import torch.nn as nn

from mmseg.registry import MODELS


@MODELS.register_module()
class ResNet(nn.Module):
    def __init__(
        self,
        depth=50,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        import torchvision.models as tv_models
        if depth == 18:
            net = tv_models.resnet18(weights=None)
        elif depth == 34:
            net = tv_models.resnet34(weights=None)
        elif depth == 50:
            net = tv_models.resnet50(weights=None)
        elif depth == 101:
            net = tv_models.resnet101(weights=None)
        elif depth == 152:
            net = tv_models.resnet152(weights=None)
        else:
            raise ValueError(f'Unsupported ResNet depth: {depth}')

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        if isinstance(pretrained, str) and pretrained:
            try:
                sd = torch.load(pretrained, map_location='cpu')
                sd = sd.get('state_dict', sd)
                self.load_state_dict(sd, strict=False)
            except Exception:
                pass

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1]:
                for p in m.parameters():
                    p.requires_grad = False
        if self.frozen_stages >= 1:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 2:
            for p in self.layer2.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 3:
            for p in self.layer3.parameters():
                p.requires_grad = False
        if self.frozen_stages >= 4:
            for p in self.layer4.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs[i] for i in self.out_indices)
