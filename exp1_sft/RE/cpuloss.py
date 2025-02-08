import torch.nn as nn
import torch
import torch.nn.functional as F

class Neg_log_mean_loss(nn.Module):
    def __init__(self):
        super(Neg_log_mean_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, y_true, y_pred):
        #  yt*log(yp) / |y_t|. missing preds
        pos = - torch.sum(y_true * torch.log(y_pred + self.eps), dim=1) / torch.maximum(self.eps, torch.sum(y_true, dim=1))
        # wrong preds
        neg = torch.sum((1-y_true) * (1-y_pred+self.eps), 1)  / torch.maximum(self.eps, torch.sum(1-y_true, 1))
        neg = - K.log(1 - neg + self.eps)
        loss = pos+ 40*neg
        return loss

class Neg_log_mean_mid_loss(nn.Module):
    def __init__(self):
        super(Neg_log_mean_loss, self).__init__()
        self.mid = 1e-1
        self.eps = 1e-6

    def forward(self, y_true, y_pred):
        pos = - torch.sum(y_true * torch.log(y_pred)) / torch.maximum(1e-6, K.sum(y_true))
        neg = torch.sum((1-y_true) * y_pred) / torch.maximum(1e-6, torch.sum(1-y_true))
        x = torch.abs(neg - self.mid)
        neg = - torch.log(1 - x + self.eps)
        return pos + neg

class Margin_loss(nn.Module):
    def __init__(self):
        super(Margin_loss, self).__init__()
        self.eps = 1e-7
        self.posmargin = 0.7
        self.negmargin = 0.6

    def theta(self, t):
        return (torch.sign(t) + 1.)/2.

    def forward(self, y_true, y_pred):
        lpos = 1 - self.theta(y_true - self.posmargin) * self.theta(y_pred - self.posmargin)
        lneg = 1 - self.theta(1 - self.negmargin - y_true) * self.theta(1 - self.negmargin - y_pred)
        pos = - torch.sum(lpos * y_true * torch.log(y_pred + self.eps), dim=1)
        neg = - torch.sum(lneg * (1 - y_pred) * torch.log(1 - y_pred + self.eps), dim=1) / 1.5
        return pos + neg

