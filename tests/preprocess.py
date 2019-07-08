import numpy as np


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def gen_vocab_dicts(tokenized_corpus):
    vocabulary = []
    for sentence in tokenized_corpus:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    vocabulary_size = len(vocabulary)

    return word2idx, idx2word, vocabulary_size


def get_window_pairs(tokenized_corpus, word2idx, window_size=2):
    """
    Returns: An array of all [center, context] pairs within the wondow size
    """
    idx_pairs = []

    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]

        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make sure to not jump out sentence
                if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                    continue

                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))

    return np.array(idx_pairs) # it will be useful to have this as numpy array


def pred_num_window_pairs(window_size, doc_len):
    full_window_words = (doc_len - (2 * window_size)) * 2 * window_size
    edges = 2 * sum([window_size + k for k in range(0, window_size)])

    return full_window_words + edges

