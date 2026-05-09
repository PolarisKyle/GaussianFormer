import torch


def build_optim_wrapper(model, cfg):
    if cfg is None:
        raise ValueError('optimizer config is required')

    wrapper_type = cfg.get('type', None)
    if wrapper_type == 'OptimWrapper':
        optimizer_cfg = cfg.get('optimizer', {})
        paramwise_cfg = cfg.get('paramwise_cfg', None)
    elif 'optimizer' in cfg:
        optimizer_cfg = cfg['optimizer']
        paramwise_cfg = cfg.get('paramwise_cfg', None)
    else:
        optimizer_cfg = cfg
        paramwise_cfg = None

    opt_type = optimizer_cfg.get('type')
    if opt_type is None:
        raise KeyError('optimizer.type is required')

    base_lr = optimizer_cfg.get('lr', 1e-3)
    base_wd = optimizer_cfg.get('weight_decay', 0.0)

    params = []
    custom_keys = (paramwise_cfg or {}).get('custom_keys', {})
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = {'params': [p], 'lr': base_lr, 'weight_decay': base_wd}
        matched_key = None
        for key in custom_keys.keys():
            if key in name and (matched_key is None or len(key) > len(matched_key)):
                matched_key = key
        if matched_key is not None:
            key_cfg = custom_keys[matched_key]
            if 'lr_mult' in key_cfg:
                group['lr'] = base_lr * key_cfg['lr_mult']
            if 'decay_mult' in key_cfg:
                group['weight_decay'] = base_wd * key_cfg['decay_mult']
        params.append(group)

    opt_kwargs = {k: v for k, v in optimizer_cfg.items() if k not in {'type'}}
    opt_kwargs.pop('lr', None)
    opt_kwargs.pop('weight_decay', None)
    optimizer_cls = getattr(torch.optim, opt_type)
    optimizer = optimizer_cls(params, lr=base_lr, weight_decay=base_wd, **opt_kwargs)
    return optimizer
