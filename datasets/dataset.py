import os
import torch
import multiprocessing
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from nltk.tokenize import word_tokenize
from multiprocessing.dummy import Pool as ThreadPool


class LDA2VecDataset(Dataset):
    BETA = 0.75

    def __init__(self, src_dir, device, window_size=5):
        self.vocabulary = []
        self.window_size = window_size
        self.freq = []
        self.files = self._get_files_in_dir(src_dir)
        self.examples = []
        self.n_examples = 0
        self.device = device
        self.idx2doc = dict()

    def __getitem__(self, index):
        return self._example_to_tensor(*self.examples[index])

    def __len__(self):
        # Use a counter so this doesn't have to be computed on every call
        return self.n_examples

    def read_file(self, f):
        """
        Read File
        
        Reads file and returns string. This is to allow different file formats
        to be used.

        :param f: File to be read
        :returns: String
        """
        # Needs to be implemented by child class
        raise NotImplementedError

    def _tokenize_doc(self, doc_str):
        """
        Tokenize Document

        Converts a document string into a list of tokens.

        :params doc_str: String representation of a document
        :returns: A list of tokens
        """

        tokenized = []
        for token in word_tokenize(doc_str):
            for c in token:
                # Only accept tokens with alphanumeric values present
                if c.isalnum():
                    tokenized.append(token)
                    break
        return tokenized

    def generate_examples(self):
        """
        Generate Dataset Examples

        Once self.files is populated and self.read_file is implemented
        then this method can be called and will populate self.examples with
        data examples to be returned by self.__getitem__

        :returns: None
        """
        print('\nGenerating Examples for Dataset...')
        for doc_id, file in enumerate(tqdm(self.files)):
            path, filename = os.path.split(file)
            self.idx2doc[str(doc_id)] = filename

            doc_str = self.read_file(file)
            tokenized_doc = word_tokenize(doc_str)

            for i, token in enumerate(tokenized_doc):
                # Ensure token is recorded
                self._add_token_to_vocab(token)
                # Generate context words for token in this doc
                context_words = self._generate_contexts(
                    token, i, tokenized_doc
                )

                # Form Examples:
                # An example consists of:
                #   center word: token
                #   document id: doc_id
                #   context_word: tokenized_doc[context_word_pos]
                # In the form of:
                # (token, doc_id), context - follows form: (input, target)
                in_tuple = (token, doc_id)
                new_examples = [(in_tuple, ctxt) for ctxt in context_words]

                # Add to class
                self.examples.extend(new_examples)
                self.n_examples += len(new_examples)

    
    def generate_examples_from_file(self, in_tuple):
        """
        Generate all example from a file

        :param in_tuple: Input tuple of form: (file, lock)
        """
        file, lock = in_tuple
        doc_id = self.files.index(file)
        path, filename = os.path.split(file)
        self.idx2doc[str(doc_id)] = filename

        doc_str = self.read_file(file)
        tokenized_doc = word_tokenize(doc_str)

        for i, token in enumerate(tokenized_doc):
            # Ensure token is recorded
            lock.acquire()
            self._add_token_to_vocab(token)
            lock.release()
            # Generate context words for token in this doc
            context_words = self._generate_contexts(
                token, i, tokenized_doc
            )

            # Form Examples:
            # An example consists of:
            #   center word: token
            #   document id: doc_id
            #   context_word: tokenized_doc[context_word_pos]
            # In the form of:
            # (token, doc_id), context - follows form: (input, target)
            in_tuple = (token, doc_id)
            new_examples = [(in_tuple, ctxt) for ctxt in context_words]

            # Add to class
            self.examples.extend(new_examples)
            self.n_examples += len(new_examples)


    def generate_examples_multi(self):
        pool = ThreadPool(multiprocessing.cpu_count())
        lock = multiprocessing.Lock()
        files_w_lock = list(zip(self.files, [lock for _ in range(len(self.files))]))

        for _ in tqdm(
            pool.imap_unordered(
                self.generate_examples_from_file,
                files_w_lock),
            total=len(self.files)):
            pass
        pool.close()
        pool.join()


    def _add_token_to_vocab(self, token):
        """
        Add token to self.vocabulary

        Adds new tokens to the end of self.vocabulary and keeps track of
        frequency of tokens

        :param token: String
        :returns: None
        """
        if token not in self.vocabulary:
            self.vocabulary.append(token)
            self.freq.append(1)
        else:
            # Token in vocab - increase frequency for token
            self.freq[self.vocabulary.index(token)] += 1

    def _generate_contexts(self, token, token_idx, tokenized_doc):
        """
        Generate Token's Context Words

        Generates all the context words within the window size defined
        during initialization around token.

        :param token: String - center token in pairs
        :param token_idx: Index at which token is found in tokenized_doc
        :paran tokenized_doc: List - Document broken into tokens
        :returns: List of context words
        """
        contexts = []
        # Iterate over each position in window
        for w in range(-self.window_size, self.window_size + 1):
            context_pos = token_idx + w

            # Make sure current center and context are valid
            is_outside_doc = context_pos < 0 or context_pos >= len(tokenized_doc)
            center_is_context = token_idx == context_pos

            if is_outside_doc or center_is_context:
                # Not valid skip to next window position
                continue

            contexts.append(tokenized_doc[context_pos])
        return contexts

    def _get_files_in_dir(self, src_dir):
        files = []
        src_dir = os.path.expanduser(src_dir)
        d = os.path.join(src_dir)

        if not os.path.isdir(src_dir):
            raise Exception('Path given is not a directory.')

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                files.append(path)

        return files

    def _example_to_tensor(self, example, target):
        center = torch.tensor([self.vocabulary.index(example[0])]).to(self.device)
        doc_id = torch.tensor([example[1]]).to(self.device)
        target = torch.tensor([self.vocabulary.index(target)]).to(self.device)
        return ((center, doc_id), target)



