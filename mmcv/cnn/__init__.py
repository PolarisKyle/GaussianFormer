import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        if isinstance(scale, (list, tuple)):
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


def build_activation_layer(cfg):
    cfg = cfg or {'type': 'ReLU'}
    layer_type = cfg.get('type', 'ReLU')
    kwargs = {k: v for k, v in cfg.items() if k != 'type'}
    mapping = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'LeakyReLU': nn.LeakyReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'SiLU': nn.SiLU,
        'Identity': nn.Identity,
    }
    if layer_type not in mapping:
        raise KeyError(f'Unsupported activation layer: {layer_type}')
    return mapping[layer_type](**kwargs)


def build_norm_layer(cfg, num_features):
    cfg = cfg or {'type': 'BN'}
    layer_type = cfg.get('type', 'BN')
    kwargs = {k: v for k, v in cfg.items() if k not in {'type', 'requires_grad', 'normalized_shape'}}
    if layer_type in ['BN', 'BN1d']:
        layer = nn.BatchNorm1d(num_features, **kwargs)
        name = 'bn'
    elif layer_type in ['BN2d']:
        layer = nn.BatchNorm2d(num_features, **kwargs)
        name = 'bn'
    elif layer_type in ['LN', 'LayerNorm']:
        normalized_shape = cfg.get('normalized_shape', num_features)
        layer = nn.LayerNorm(normalized_shape, **kwargs)
        name = 'ln'
    elif layer_type in ['GN', 'GroupNorm']:
        num_groups = cfg.get('num_groups', 32)
        layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, **kwargs)
        name = 'gn'
    else:
        raise KeyError(f'Unsupported norm layer: {layer_type}')

    requires_grad = cfg.get('requires_grad', True)
    for p in layer.parameters():
        p.requires_grad = requires_grad
    return name, layer
