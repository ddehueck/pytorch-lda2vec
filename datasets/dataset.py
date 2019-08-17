import os
import torch
import uuid
import multiprocessing
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from .preprocess import Tokenizer
from multiprocessing.dummy import Pool as ThreadPool
from collections import Counter


class LDA2VecDataset(Dataset):

    def __init__(self, args):
        self.args = args
        self.term_freq_dict = dict()
        self.files = self._get_files_in_dir(args.dataset_dir)
        self.tokenizer = Tokenizer()
        self.tokenized_docs = []
        self.idx2doc = dict()
        self.name = ''
        # Save example to block files
        self.saved_ds_dir = 'saved_datasets/'
        self.block_files = {}
        self.examples = []

        if args.toy:
            # Turns into a toy dataset
            self.file = self.files[:5]


    def __getitem__(self, index):
        if self.args.read_from_blocks:
            # Get file from index
            num_examples = 0
            for file in self.block_files:
                block_length = self.block_files[file]
                num_examples += block_length
                if index < num_examples:
                    # Example in current file
                    break

            in_file_index = index - (num_examples - block_length)
            examples = torch.load(file)[in_file_index]
        else:
            examples = self.examples[index]
        return self._example_to_tensor(*examples)


    def __len__(self):
        return sum(list(self.block_files.values()))


    def _load_dataset(self):
        print(f'Loading dataset from: {self._get_saved_ds_dir()}')
        # Get block files
        files = self._get_files_in_dir(f'{self._get_saved_ds_dir()}blocks/')
        for f in tqdm(files):
            saved_examples = torch.load(f)
            self.block_files[f] = len(saved_examples)

            if not self.args.read_from_blocks:
                self.examples.extend(saved_examples)

        # Load metadata in
        metadata = torch.load(f'{self._get_saved_ds_dir()}metadata.pth' )
        self.idx2doc = metadata['idx2doc']
        self.term_freq_dict = metadata['term_freq_dict']
        self.term_idx_dict = metadata['term_idx_dict']
        print('Loaded Dataset!')
    

    def _save_example_block(self, examples):
        # Create save to directory
        block_dir = f'{self._get_saved_ds_dir()}blocks/'
        if not os.path.exists(block_dir):
            try:
                os.makedirs(block_dir)
            except:
                pass
        
        # Save to block file   
        filename = f'{block_dir}block-{uuid.uuid4()}.dat'
        torch.save(examples, filename)
        self.block_files[filename] = len(examples)


    def _save_metadata(self):
        doc_lengths = [len(self.read_file(f)) for f in self.files]

        torch.save({
            'idx2doc': self.idx2doc,
            'term_freq_dict': self.term_freq_dict,
            'doc_lengths': doc_lengths,
            'term_idx_dict': self.term_idx_dict
        }, f'{self._get_saved_ds_dir()}metadata.pth')


    def _get_saved_ds_dir(self):
        direc = f'{self.saved_ds_dir}{"_".join(self.name.lower().split(" "))}/'
        if self.args.toy:
            direc = direc[:-1] + '_toy/'
        return direc


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


    def generate(self):
        # Check if there is a saved version of dataset
        if os.path.exists(self._get_saved_ds_dir()):
            self._load_dataset()
            return

        # Generate tokenized docs
        # May end up with less docs than files if files are empty
        # or exactly alike.
        self._generate_tokenized_docs()
        # Clean tokenized docs
        self._clean_tokenized_docs()
        # Generate example file batches
        self._generate_examples()
        # Save Metadata
        self._save_metadata()

    
    def _generate_tokenized_docs(self):
        # self.files length is 7532
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = self.args.file_batch_size
        file_batches = self._batch_list(batch_size, self.files)

        print('Tokenizing Documents...')
        for t_batch in tqdm(
            pool.imap_unordered(
                self._generate_tokenized_docs_worker,
                file_batches),
            total=len(self.files)//batch_size + 1):
            
            # Add tokenized doc to global list
            # Add to file name idx2doc
            for t_doc, fn in t_batch:
                self.tokenized_docs.append(t_doc)
                idx = len(self.tokenized_docs) - 1 
                self.idx2doc[idx] = fn

        pool.close()
        pool.join()

    
    def _generate_tokenized_docs_worker(self, file_batch):
        """
        :returns: list of tuples of form [(tokenized_doc, filename), ..., ...]
        """
        t_docs = []
        for file in file_batch:
            doc_str = self.read_file(file)
            _, filename = os.path.split(file)
            t_docs.append((self.tokenizer.tokenize_doc(doc_str), filename))
        return t_docs

    
    def _clean_tokenized_docs(self):
        # Get total number of tokens
        print('Counting tokens...')
        total_tokens = [t for doc in self.tokenized_docs for t in doc]
        counter = Counter(total_tokens)

        # Get all tokens to remove with a count of less than 20
        to_remove = {}
        for token, freq in counter.most_common():
            to_remove[token] = freq < 20 and not self.args.toy

        # Remove the identified tokens from documents
        print('Removing infrequent tokens...')
        for i, doc in enumerate(tqdm(self.tokenized_docs)):
            self.tokenized_docs[i] = [t for t in doc if not to_remove[t]]

        # Remove docs that are too short
        self.tokenized_docs = [doc for doc in self.tokenized_docs if len(doc) > 2 * self.args.window_size]

        # Create term frequency dict - will generate vocab size from this
        self.term_freq_dict = dict(Counter([t for doc in self.tokenized_docs for t in doc]))

        # Turn documents into lists of token ids in vocab
        # Use dict for index lookup for efficiency when feeding examples
        self.term_idx_dict = {}
        for i, t in enumerate(list(self.term_freq_dict.keys())):
            self.term_idx_dict[t] = i


    def _generate_examples(self):
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = self.args.file_batch_size
        doc_batches = self._batch_list(batch_size, self.tokenized_docs)

        print('Generating Examples...')
        for _ in tqdm(
            pool.imap_unordered(
                self._generate_examples_worker,
                doc_batches),
            total=len(self.tokenized_docs)//batch_size + 1):
            continue

        pool.close()
        pool.join()

    
    def _generate_examples_worker(self, doc_batches):
        """
        Generate examples worker

        :param file_batch: List of docs - subset of self.tokenized_docs
        :returns: None - saves batch to block file
        """
        examples = []
        for doc in doc_batches:
            examples.extend(self._generate_examples_from_doc(doc))

        # Save examples to a block file
        self._save_example_block(examples)

    
    def _generate_examples_from_doc(self, tokenized_doc):
        """
        Generates examples from a tokenized doc

        :param tokenized_doc: A tokenized doc (a list of tokens)
        :returns: List of training examples
        """
        examples = []
        doc_id = self.tokenized_docs.index(tokenized_doc)
        for i, token in enumerate(tokenized_doc):
            # Generate context words for token in this doc
            context_words = self._generate_contexts(i, tokenized_doc)

            # Form Examples:
            # An example consists of:
            #   center word: token
            #   document id: doc_id
            #   context_word: tokenized_doc[context_word_pos]
            new_example = (token, doc_id, [w for w in context_words])
            examples.append(new_example)

        return examples


    def _generate_contexts(self, center_idx, tokenized_doc):
        """
        Generate Token's Context Words

        Generates all the context words within the window size defined
        during initialization around token. Ensures all contexts are
        of 2 * args.window_size.

        :param center_idx: Index at which token is found in tokenized_doc
        :param tokenized_doc: List - Document broken into tokens
        :returns: List of context words
        """
        assert len(tokenized_doc) > 2 * self.args.window_size

        contexts = []
        words_before_needed = 0
        words_after_needed = 0
        # Iterate over each position in window
        for w in range(-self.args.window_size, self.args.window_size + 1):
            context_pos = center_idx + w

            # Make sure current center and context are valid
            if context_pos < 0:
                # At the beginning - use words after to account
                words_after_needed += 1
                continue
            elif context_pos >= len(tokenized_doc):
                # At the ending - use words before to account
                words_before_needed += 1
                continue
            elif center_idx == context_pos:
                # Not valid skip to next window position
                continue

            contexts.append(tokenized_doc[context_pos])

        # Account for if ran into beginning or end of a document
        for i in range(words_before_needed):
            context_pos = center_idx + -self.args.window_size - i - 1
            contexts.append(tokenized_doc[context_pos])

        for i in range(words_after_needed):
            context_pos = center_idx + self.args.window_size + i + 1
            contexts.append(tokenized_doc[context_pos])

        assert len(contexts) == 2 * self.args.window_size

        return contexts

    
    def _batch_list(self, batch_size, list_to_batch):
        """
        Batch List

        Seperates list into smaller batches of its elements of
        size batch_size

        :param batch_size: Int - size of each batch
        :param list_to_batch: List - what to create batches of
        :returns: Generator of batches
        """
        n = len(list_to_batch)
        for b_idx in range(0, n, batch_size):
            # min() so we don't index outside of self.files
            yield list_to_batch[b_idx:min(b_idx + batch_size, n)]


    def _get_files_in_dir(self, src_dir):
        if src_dir is None:
            return []

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


    def _example_to_tensor(self, center, doc_id, targets):
        """
        Takes raw example and turns it into tensor values

        :params center: String of the center word
        :params doc_id: Index to document vector of where training example is from
        :params targets: list of Strings - the target words
        :returns: A tuple of tensors
        """
        center = self.term_idx_dict[center]
        targets = [self.term_idx_dict[t] for t in targets]

        return torch.tensor([center]), torch.tensor([doc_id]), torch.tensor([targets])



