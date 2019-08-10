import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import spacy

class Lda2vec(nn.Module):

    def __init__(self, vocab_size, num_docs, args, pretrained_vecs=None):
        super(Lda2vec, self).__init__()
        self.args = args
        #self.topic_embeds = nn.Parameter(t.randn((args.embedding_len, args.num_topics)), requires_grad=True)
        self.topic_embeds = nn.Parameter(t.randn((args.num_topics, args.embedding_len)), requires_grad=True)

        if args.use_pretrained:
            assert pretrained_vecs is not None, "pretrained_vecs cannot be None"
            # Initialize from pretrained
            self.word_embeds = nn.Embedding(vocab_size, args.embedding_len).from_pretrained(pretrained_vecs, freeze=False)
        else:
            self.word_embeds = nn.Embedding(vocab_size, args.embedding_len)

        if args.uni_doc_init:
             # Sample from a uniform distribution betwen ~[-sqrt(3), +sqrt(3)]
             # sqrt(3) chosen from goodfellow's GAN initialization
            uni = Uniform(-1.732, 1.732).sample((num_docs, args.num_topics))
            self.doc_weights = nn.Embedding.from_pretrained(uni)
        else:
            self.doc_weights = nn.Embedding(num_docs, args.num_topics)

        # Reqularization Layer
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x should take the form of: (center word, doc_id)
        # Get word vector
        word_vecs = self.word_embeds(x[0]) # returns wordvec of index x[0] - the center word
        
        # Get document vector
        # 1. Softmax document weights to get proportions
        doc_weights = self.doc_weights(x[1]) # latent doc embedding of x[1] - doc_id
        proportions = F.softmax(doc_weights, dim=1)

        # 2. Multiply by topic embeddings to get doc vector
        # Apply regularization
        if self.args.use_dropout:
            doc_vecs = t.matmul(proportions, self.dropout(self.topic_embeds))
        else:
            doc_vecs = t.matmul(proportions, self.topic_embeds)

        # Combine into context vector - sum
        context_vecs = t.add(word_vecs, doc_vecs)

        return context_vecs

    def get_proportions(self):
        """
        Softmax document weights to get proportions
        """
        return F.softmax(self.doc_weights.weight, dim=1)

    def get_doc_vectors(self):
        """
        Multiply by proportions by topic embeddings to get document vectors
        """
        proportions = self.get_proportions()
        doc_vecs = t.matmul(proportions, self.topic_embeds)

        return doc_vecs
