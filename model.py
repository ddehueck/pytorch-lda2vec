import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import spacy
from scipy.stats import ortho_group

class Lda2vec(nn.Module):

    def __init__(self, vocab_size, num_docs, args, pretrained_vecs=None, docs_init=None):
        super(Lda2vec, self).__init__()
        self.args = args
        self.topic_embeds = nn.Parameter(t.tensor(ortho_group.rvs(args.embedding_len)[:args.num_topics], dtype=t.float))

        if pretrained_vecs is not None:
            self.word_embeds = nn.Embedding(vocab_size, args.embedding_len).from_pretrained(pretrained_vecs, freeze=False)
        else:
            self.word_embeds = nn.Embedding(vocab_size, args.embedding_len)

        if args.uni_doc_init:
             # Sample from a uniform distribution betwen ~[-sqrt(3), +sqrt(3)]
             # sqrt(3) chosen from goodfellow's GAN initialization
            uni = Uniform(-1.732, 1.732).sample((num_docs, args.num_topics))
            self.doc_weights = nn.Embedding.from_pretrained(uni, freeze=False)
        elif docs_init is not None:
            self.doc_weights = nn.Embedding(num_docs, args.num_topics).from_pretrained(docs_init, freeze=False)
        else:
            self.doc_weights = nn.Embedding(num_docs, args.num_topics)
        

    def forward(self, center_id, doc_id):
        # x should take the form of: (center word, doc_id)
        # Get word vector
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
