import torch
import torch.nn as nn
import numpy as np


class SGNSLoss(nn.Module):
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 15  # Taken from Moody's OG code
    UNIGRAM_TABLE_SIZE = 10**5
    EPSILON = 1e-9  # value to lower bound clamp to avoid -inf

    def __init__(self, dataset, word_embeddings, device):
        super(SGNSLoss, self).__init__()
        self.dataset = dataset
        self.vocab_len = len(dataset.term_freq_dict)
        self.word_embeddings = word_embeddings
        self.device = device

        # Helpful values for unigram distribution generation
        self.transformed_freq_vec = torch.tensor(
                list(dataset.term_freq_dict.values())
            ).pow(self.BETA)
        self.freq_sum = torch.sum(self.transformed_freq_vec)

        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, context, target):
        context, target = context.squeeze(), target.squeeze()
        print(f'CONTEXT S: {context.size()}')
        print(f'CONTEXT NAN: {torch.isnan(context).any()}')
        print(f'TARGET S: {target.size()}')
        print(f'TARGET NAN: {torch.isnan(target).any()}')
        # compute non-sampled portion
        dots = (context * target).sum(-1)
        print(f'DOTS: {dots}')
        print(f'DOTS S: {dots.size()}')
        log_targets = torch.log(torch.sigmoid(dots).clamp(self.EPSILON))
        print(f'LOG TARGETS S: {log_targets.size()}')
        print(f'LOG TARGETS NAN: {torch.isnan(log_targets).any()}')
        log_samples = []
        for l in range(self.NUM_SAMPLES):
            # Could probably optimize this by optimizing self.get_unigram
            # _sample() - maybe return all the samples needed as specified by
            # self.NUM_SAMPLES? Would probably by faster than random.choice *
            # self.NUM_SAMPLES. 
            sample = self.get_unigram_sample()
            dot = (torch.neg(context) * sample).sum(-1)
            log_samples.append(torch.log(torch.sigmoid(dot).clamp(self.EPSILON)))

        log_samples = torch.stack(log_samples).sum(0)
        print(f'LOG SAMPLE S: {log_samples.size()}')
        print(f'LOG SAMPLES NAN: {torch.isnan(log_samples).any()}')
        return torch.add(log_targets, log_samples).sum()  # A loss should return a single value

    def get_unigram_sample(self):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        rand_idx = np.random.choice(self.unigram_table)
        rand_idx = torch.tensor(rand_idx).to(self.device)
        return self.word_embeddings(rand_idx).squeeze()

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        PDF = []
        for token_idx in range(0, self.vocab_len):
            PDF.append(self.get_unigram_prob(token_idx))
        # Generate the table from PDF
        return np.random.choice(self.vocab_len, self.UNIGRAM_TABLE_SIZE, p=PDF)

