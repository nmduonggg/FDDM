import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-100, reduction='mean'):
        super().__init__()
        # use standard CE loss without reducion as basis
        weight = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5])
        self.CE = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        '''
        input (B, N)
        target (B)
        '''
        minus_logpt = self.CE(input, target)
        pt = torch.exp(-minus_logpt) # don't forget the minus here
        focal_loss = (1-pt)**self.gamma * minus_logpt

        # apply class weights
        if self.alpha != None:
            focal_loss *= self.alpha.gather(0, target)
        
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss