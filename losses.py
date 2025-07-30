import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt, label

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'TotalLoss']



class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        self.lambda_csd = 1.0

    def forward(self, input, target):

        logits = input  # shape (B,1,H,W)
        masks = target
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num




        prob = torch.sigmoid(logits)       # (B,1,H,W)
        prob_fg = prob[:, 0, :, :]         # (B,H,W)
        mask_fg = masks[:, 0, :, :]        # (B,H,W)

        csd_losses = []
        for p_map, m in zip(prob_fg, mask_fg):
            m_np = m.detach().cpu().numpy().astype(np.uint8)
            labeled, n_clusters = label(m_np)

            dt_map = np.zeros_like(m_np, dtype=np.float32)
            for k in range(1, n_clusters + 1):
                mask_k  = (labeled == k).astype(np.uint8)
                dist_bg = distance_transform_edt(mask_k == 0)
                dist_fg = dist_bg * mask_k
                maxd    = dist_fg.max() + self.eps
                dt_map  += dist_fg / maxd

            dt_t = torch.from_numpy(dt_map).to(logits.device)
            csd_losses.append((p_map * dt_t).mean())

        csd_loss = torch.stack(csd_losses).mean()

        return bce + dice + csd_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
