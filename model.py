import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Lda2vec(nn.Module):

    def __init__(self, vocab_size, embedding_len, num_topics, num_docs, args):
        super(Lda2vec, self).__init__()
        self.args = args
        self.word_embeds = nn.Embedding(vocab_size, embedding_len)
        self.doc_weights = nn.Embedding(num_docs, num_topics)
        self.topic_embeds = nn.Parameter(t.randn((embedding_len, num_topics)), requires_grad=True)
        # Reqularization Layers
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm = nn.BatchNorm1d(embedding_len)

    def forward(self, x):
        # x should take the form of: (center word, doc_id)
        # Get word vector
        word_vecs = self.word_embeds(x[0]) # returns wordvec of index x[0] - the center word

        # Get document vector
        # 1. Softmax document embeddings to get proportions
        doc_weights = self.doc_weights(x[1]) # latent doc embedding of x[1] - doc_id
        proportions = F.softmax(doc_weights, dim=1)

        # 2. Multiply by topic embeddings to get doc vector
        doc_vecs = t.matmul(self.topic_embeds, t.transpose(proportions, 1, 2))
        doc_vecs = t.transpose(doc_vecs, 1, 2)

        # Apply regularization
        if self.args.use_dropout:
            word_vecs = self.dropout(word_vecs)
            doc_vecs = self.dropout(doc_vecs)

        if self.args.use_batchnorm:
            word_vecs = self.batchnorm(t.transpose(word_vecs, 1, 2))
            doc_vecs = self.batchnorm(t.transpose(doc_vecs, 1, 2))

        # Combine into context vector - sum
        context_vecs = t.add(word_vecs, doc_vecs)

        return context_vecs

