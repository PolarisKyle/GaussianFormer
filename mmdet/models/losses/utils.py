import torch


def weight_reduce_loss(loss: torch.Tensor, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        raise ValueError(f'Unsupported reduction: {reduction}')

    if reduction == 'mean':
        return loss.sum() / avg_factor
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(f'Unsupported reduction: {reduction}')
