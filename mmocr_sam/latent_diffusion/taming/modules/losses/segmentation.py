import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss, {}


class BCELossWithQuant(nn.Module):

    def __init__(self, codebook_weight=1.):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target)
        loss = bce_loss + self.codebook_weight * qloss
        return loss, {
            f'{split}/total_loss': loss.clone().detach().mean(),
            f'{split}/bce_loss': bce_loss.detach().mean(),
            f'{split}/quant_loss': qloss.detach().mean()
        }
