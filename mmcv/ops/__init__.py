import torch
import torch.nn.functional as F


def sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25, class_weight=None, reduction='none'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    alpha_factor = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = alpha_factor * (1 - pt).pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1)

    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(f'Unsupported reduction: {reduction}')


def softmax_focal_loss(pred, target, gamma=2.0, alpha=0.25, class_weight=None, reduction='none'):
    ce_loss = F.cross_entropy(pred, target.long(), weight=class_weight, reduction='none')
    prob = F.softmax(pred, dim=1)
    pt = prob.gather(1, target.long().unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    focal = (1 - pt).pow(gamma)
    if alpha is not None:
        focal = focal * alpha
    loss = ce_loss * focal

    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(f'Unsupported reduction: {reduction}')
