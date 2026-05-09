import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def build_dropout(cfg):
    if cfg is None:
        return nn.Identity()
    drop_type = cfg.get('type', 'Dropout')
    if drop_type == 'Dropout':
        p = cfg.get('drop_prob', cfg.get('p', 0.5))
        return nn.Dropout(p)
    if drop_type == 'DropPath':
        p = cfg.get('drop_prob', 0.0)
        return DropPath(p)
    if drop_type == 'Identity':
        return nn.Identity()
    raise KeyError(f'Unsupported dropout layer: {drop_type}')
