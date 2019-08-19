import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Lda2vec(nn.Module):

    def __init__(self, vocab_size, num_docs, args, pretrained_vecs=None, docs_init=None):
        super(Lda2vec, self).__init__()
        self.args = args
        self.topic_embeds = nn.Parameter(_orthogonal_matrix((args.num_topics, args.embedding_len)))

        if pretrained_vecs is not None:
            self.word_embeds = nn.Embedding(vocab_size, args.embedding_len).from_pretrained(pretrained_vecs, freeze=False)
        else:
            # TODO: scale_grad_by_freq (boolean, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False
            # TODO: sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients
            self.word_embeds = nn.Embedding(vocab_size, args.embedding_len)

        if docs_init is not None:
            docs_init = t.log(docs_init + 1e-5)
            temperature = 7.0
            docs_init /= temperature

            self.doc_weights = nn.Embedding(num_docs, args.num_topics).from_pretrained(docs_init, freeze=False)
        else:
            self.doc_weights = nn.Embedding(num_docs, args.num_topics)
            self.doc_weights.weight.data /= np.sqrt(num_docs + args.num_topics)
        

    def forward(self, center_id, doc_id):
        # x should take the form of: (center word, doc_id)
        # Get word vector
        center_id, doc_id = center_id.squeeze(1), doc_id.squeeze(1)
        word_vecs = self.word_embeds(center_id)
        
        # Get document vector
        # 1. Softmax document weights to get proportions
        doc_weights = self.doc_weights(doc_id)
        proportions = F.softmax(doc_weights, dim=1).squeeze().unsqueeze(dim=2)

        # 2. Multiply by topic embeddings to get doc vector
        topic_vecs = self.topic_embeds.unsqueeze(dim=0)
        doc_vecs = (proportions * topic_vecs).sum(dim=1)

        # Combine into context vector
        context_vecs = word_vecs + doc_vecs

        return context_vecs

    def get_proportions(self):
        """
        Softmax document weights to get proportions
        """
        return F.softmax(self.doc_weights.weight, dim=1).unsqueeze(dim=2)

    def get_doc_vectors(self):
        """
        Multiply by proportions by topic embeddings to get document vectors
        """
        proportions = self.get_proportions()
        doc_vecs = (proportions * self.topic_embeds.unsqueeze(0)).sum(dim=1)

        return doc_vecs


def _orthogonal_matrix(shape):
    # Stolen from blocks:
    # github.com/mila-udem/blocks/blob/master/blocks/initialization.py
    M1 = np.random.randn(shape[0], shape[0])
    M2 = np.random.randn(shape[1], shape[1])

    # QR decomposition of matrix with entries in N(0, 1) is random
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    # Correct that NumPy doesn't force diagonal of R to be non-negative
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))

    n_min = min(shape[0], shape[1])
    return t.from_numpy(np.dot(Q1[:, :n_min], Q2[:n_min, :])).float().requires_grad_(True)