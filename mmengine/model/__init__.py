import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass


def xavier_init(module, distribution='normal', bias=0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.xavier_normal_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
