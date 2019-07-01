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
        self.device = device
        self.term_freq_dict = dict()
        self.window_size = window_size
        self.files = self._get_files_in_dir(src_dir)
        self.examples = []
        self.n_examples = 0
        self.idx2doc = dict()

    def __getitem__(self, index):
        return self._example_to_tensor(*self.examples[index])

    def __len__(self):
        return len(self.examples)

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


    def generate_examples_from_file(self, file, tf_dict):
        """
        Generate all examples from a file

        :param file: File from self.files
        :param tf_dict: Term frequency dict
        :returns: List of examples
        """
        doc_id = self.files.index(file)
        path, filename = os.path.split(file)
        self.idx2doc[str(doc_id)] = filename

        doc_str = self.read_file(file)
        tokenized_doc = word_tokenize(doc_str)

        examples = []
        for i, token in enumerate(tokenized_doc):
            # Ensure token is recorded
            self._add_token_to_vocab(token, tf_dict)
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
            examples.extend(new_examples)
        return examples


    def generate_examples_multi(self):
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = 250
        file_batches = self._batch_files(batch_size=batch_size)

        print('\nGenerating Examples for Dataset (multi-threaded)...')
        for results in tqdm(
            pool.imap_unordered(
                self._generate_examples_worker,
                file_batches),
            total=len(self.files)//batch_size + 1):

            # Reduce results into final locations
            examples, tf_dict = results
            self.examples.extend(examples)
            self._reduce_tf_dict(tf_dict)

        pool.close()
        pool.join()


    def _generate_examples_worker(self, file_batch):
        """
        Generate examples worker

        Worker to generate examples in a map reduce paradigm

        :param file_batch: List of files - a subset of self.files
        """
        tf_dict = dict()
        examples = []

        for f in file_batch:
            examples.extend(self.generate_examples_from_file(f, tf_dict))
        return examples, tf_dict


    def _batch_files(self, batch_size=250):
        """
        Batch Files

        Seperates self.files into smaller batches of files of
        size batch_size

        :param batch_size: Int - size of each batch
        :returns: Generator of batches
        """
        n_files = len(self.files)
        for b_idx in range(0, n_files, batch_size):
            # min() so we don't index outside of self.files
            yield self.files[b_idx:min(b_idx + batch_size, n_files)]


    def _add_token_to_vocab(self, token, tf_dict):
        """
        Add token to the token frequency dict

        Adds new tokens to the tf_dict  and keeps track of
        frequency of tokens

        :param token: String
        :param tf_dict: A {"token": frequency,} dict
        :returns: None
        """
        if token not in tf_dict.keys():
            tf_dict[token] = 1
        else:
            # Token in vocab - increase frequency for token
            tf_dict[token] += 1

    def _reduce_tf_dict(self, tf_dict):
        """
        Reduce a term frequency dictionary

        Updates self.term_freq_dict with values in tf_dict argument.
        Adds new keys if needed or just sums frequencies

        :param tf_dict: A term frequency dictionary
        :returns: None - updates self.term_freq_dict
        """
        for key in tf_dict:
            if key in self.term_freq_dict.keys():
                # Add frequencies
                self.term_freq_dict[key] += tf_dict[key]
            else:
                # Merge
                self.term_freq_dict[key] = tf_dict[key]

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
        center = torch.tensor([
            list(self.term_freq_dict.keys()).index(example[0])
        ]).to(self.device)
        doc_id = torch.tensor([example[1]]).to(self.device)
        target = torch.tensor([
            list(self.term_freq_dict.keys()).index(target)
        ]).to(self.device)
        return ((center, doc_id), target)



