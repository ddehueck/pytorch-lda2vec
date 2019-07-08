import torch
import torch.nn.functional as F
import torch.nn as nn

class DirichletLoss(nn.Module):
    LAMBDA = 1
    ALPHA = 1 / 64  # 1 / number of topics

    def __init__(self):
        super(DirichletLoss, self).__init__()

    def forward(self, doc_weights):
        proportions = F.softmax(doc_weights, dim=0)
        return -self.LAMBDA * (self.ALPHA - 1) * torch.sum(torch.log(proportions))
