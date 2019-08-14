import torch as t
import numpy as np
import spacy
import os
from tqdm import tqdm
from gensim import corpora, models


def get_sparsity_score(vec):
    """
    Get Sparsity Score
    
    Computes an normalized sparsity score of a vector of proportions
    that sum to one.

    :param vec: Tensor - one dimensional
    :returns: Float - Normalized score
    """
    K = np.prod(np.array(vec.size()))
    uniform_vec = t.tensor([1/K for _ in range(K)]).float()
    max_sparsity = 2 * ((K - 1) / K) 
    norm_score = sum(abs(vec.float().to('cpu') - uniform_vec.float())) / max_sparsity
    
    return norm_score.item()


def get_pretrained_vecs(dataset):
    """
    Build word embedding weights based on pretrained vectors

    :params dataset: Dataset to base vocab on
    :params nlp: A spaCy NLP pipeline with pretrained vectors - md or lg
    :returns: A tensor of size: vocab_len x embed_len
    """

    nlp = spacy.load('en_core_web_md')
    vocab = list(dataset.term_freq_dict.keys())
    vectors = [nlp.vocab[v].vector for v in vocab]
    return t.tensor(vectors)


def get_doc_vecs_lda_initialization(dataset):
    """
    Runs standard LDA on tokenized docs in dataset

    :params dataset: Dataset to get documents from
    :returns: A tensor of size: num_docs x num_topics
    """

    save_init_file = f'{dataset._get_saved_ds_dir()}lda-doc-init.pth'
    if os.path.exists(save_init_file):
        # Data already exists - load it!
        return torch.load(save_init_file)

    print('Using LDA Document Intializations...')
    # Order docs - generated out of order
    ordered_docs = [dataset.tokenized_docs[k] for k in sorted(dataset.tokenized_docs)]
    # Build inputs for LDA
    dictionary = corpora.Dictionary(ordered_docs)
    corpus = [dictionary.doc2bow(text) for text in ordered_docs]
    # Run LDA and get resulting proportions
    lda = models.LdaModel(corpus, alpha=0.9, id2word=dictionary, num_topics=dataset.args.num_topics)
    corpus_lda = lda[corpus]
    # Build tensor to initialize from
    doc_weights_init = np.zeros((len(corpus_lda), dataset.args.num_topics))
    for doc in tqdm(range(len(corpus_lda))):
        topics = corpus_lda[doc]
        for top, prob in topics:
            doc_weights_init[doc, top] = prob

    # Convert to tensor and save
    doc_weights_init = t.from_numpy(doc_weights_init).float()
    t.save(doc_weights_init, save_init_file)
    return doc_weights_init

