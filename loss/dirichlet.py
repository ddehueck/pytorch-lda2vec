import torch as t
import torch.nn.functional as F
import torch.nn as nn

class DirichletLoss(nn.Module):
    
    def __init__(self, args):
        super(DirichletLoss, self).__init__()
        self.alpha = 1 / args.num_topics
        self.lambda_val = args.lambda_val

    def forward(self, doc_weights):
        # bathc_size x 1 x 32
        proportions = F.softmax(doc_weights, dim=0)
        avg_log_proportion = t.sum(t.log(proportions), dim=2).mean()
        return -self.lambda_val * (self.alpha - 1) * avg_log_proportion
