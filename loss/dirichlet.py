import torch as t
import torch.nn.functional as F
import torch.nn as nn

class DirichletLoss(nn.Module):

    def __init__(self, args):
        super(DirichletLoss, self).__init__()
        self.alpha = 1.0 / args.num_topics
        self.lambda_val = args.lambda_val

    def forward(self, doc_weights):
        log_proportions = F.log_softmax(doc_weights, dim=0)
        avg_log_proportion = t.sum(log_proportions, dim=2).mean()
        return -self.lambda_val * (self.alpha - 1) * avg_log_proportion
