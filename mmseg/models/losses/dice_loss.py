import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, class_weight=None, loss_weight=1.0, ignore_index=255, eps=1e-5):
        super().__init__()
        if class_weight is not None:
            self.register_buffer('class_weight', torch.as_tensor(class_weight, dtype=torch.float))
        else:
            self.class_weight = None
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, pred, target):
        if target.ndim == pred.ndim:
            target = target.squeeze(1)

        pred_prob = F.softmax(pred, dim=1)
        num_classes = pred_prob.shape[1]

        valid_mask = target != self.ignore_index
        target = target.clamp(min=0, max=num_classes - 1)
        one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, -1, *range(1, target.ndim)).float()

        valid_mask = valid_mask.unsqueeze(1)
        pred_prob = pred_prob * valid_mask
        one_hot = one_hot * valid_mask

        dims = tuple(range(2, pred_prob.ndim))
        inter = (pred_prob * one_hot).sum(dim=dims)
        den = pred_prob.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2 * inter + self.eps) / (den + self.eps)
        loss = 1 - dice

        if self.class_weight is not None and self.class_weight.numel() == num_classes:
            weight = self.class_weight.view(1, num_classes).to(loss.device)
            loss = loss * weight

        return self.loss_weight * loss.mean()
