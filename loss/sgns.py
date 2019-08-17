import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import AliasMultinomial


class SGNSLoss(nn.Module):
    BETA = 0.75  # exponent to adjust sampling frequency

    def __init__(self, dataset, word_embeddings, device):
        super(SGNSLoss, self).__init__()
        self.dataset = dataset
        self.args = self.dataset.args
        self.num_samples = 15
        self.vocab_len = word_embeddings.weight.size()[0]
        self.word_embeddings = word_embeddings
        self.device = device

        # Helpful values for unigram distribution generation
        self.transformed_freq_vec = t.tensor(
                np.array(list(dataset.term_freq_dict.values()))**self.BETA
            )
        self.freq_sum = t.sum(self.transformed_freq_vec)

        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, context, targets):
        # context - [batch_size x embed_size]
        # targets - [batch_size x window_size x embed_size]
        context, targets = context.squeeze(), targets.squeeze()
        
        # Compute non-sampled portion
        dots = (context.unsqueeze(1) * targets).sum(-1)  # [batch_size x window_size]
        log_targets = F.logsigmoid(dots).sum(-1)  # [batch_size]

        # Compute sampled portion
        batch_size = len(log_targets)

        samples = self.get_unigram_samples(bs=batch_size)
        sample_dots = (context.unsqueeze(1).unsqueeze(1).neg() * samples).sum(-1)  # [batch_size x window_size x num_samples]
        log_samples = F.logsigmoid(sample_dots).sum(-1).sum(-1)  # [batch_size]

        return t.add(log_targets, log_samples).mean().neg()  # Negative so goes towards loss of 0

    def get_unigram_samples(self, bs=None):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        batch_size = self.args.batch_size if bs is None else bs
        window_size = self.args.window_size * 2
        embed_len = self.args.embedding_len
        # How many samples are needed
        n = batch_size * self.num_samples * window_size
        # Get indices
        rand_idxs = self.unigram_table.draw(n).to(self.device)
        rand_idxs = rand_idxs.view(batch_size, window_size, self.num_samples)
        # Get Embeddings
        rand_embeddings = self.word_embeddings(rand_idxs).squeeze()
        return rand_embeddings.view(batch_size, window_size, self.num_samples, embed_len)

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        PDF = []
        for token_idx in range(0, self.vocab_len):
            PDF.append(self.get_unigram_prob(token_idx))
        # Generate the table from PDF
        return AliasMultinomial(PDF, self.device)

