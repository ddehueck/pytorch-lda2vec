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
        self.tokenized_docs = dict()
        self.idx2doc = dict()
        self.name = ''
        # Save example to block files
        self.saved_ds_dir = 'saved_datasets/'
        self.block_files = {}

        if args.toy:
            # Turns into a toy dataset
            self.file = self.files[:5]


    def __getitem__(self, index):
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
        return self._example_to_tensor(*examples)


    def __len__(self):
        return sum(list(self.block_files.values()))


    def _load_dataset(self):
        print(f'Loading dataset from: {self._get_saved_ds_dir()}')
        # Get block files
        files = self._get_files_in_dir(f'{self._get_saved_ds_dir()}blocks/')
        for f in tqdm(files):
            self.block_files[f] = len(torch.load(f))

        # Load metadata in
        metadata = torch.load(f'{self._get_saved_ds_dir()}metadata.pth' )
        self.idx2doc = metadata['idx2doc']
        self.term_freq_dict = metadata['term_freq_dict']
        print('Loaded Dataset!')
    

    def _save_example_block(self, examples):
        # Create save to directory
        block_dir = f'{self._get_saved_ds_dir()}blocks/'
        if not os.path.exists(block_dir):
            os.makedirs(block_dir)
        
        # Save to block file   
        filename = f'{block_dir}block-{uuid.uuid4()}.dat'
        torch.save(examples, filename)
        self.block_files[filename] = len(examples)


    def _save_metadata(self):
        # TODO: More efficient way to load tf dict?
        counter = dict(Counter(self.tokenized_docs))
        for freq in counter:
            for term in counter[freq]:
                self.term_freq_dict[term] = freq

        doc_lengths = [len(self.read_file(f)) for f in self.files]

        torch.save({
            'idx2doc': self.idx2doc,
            'term_freq_dict': self.term_freq_dict,
            'doc_lengths': doc_lengths
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
        self._generate_tokenized_docs()
        assert len(self.tokenized_docs) == len(self.files)
        # Clean tokenized docs
        self._clean_tokenized_docs()
        # Generate example file batches
        self._generate_examples()
        # Save Metadata
        self._save_metadata()

    
    def _generate_tokenized_docs(self):
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = self.args.file_batch_size
        file_batches = self._batch_list(batch_size, self.files)

        print('Tokenizing Documents...')
        for _ in tqdm(
            pool.imap_unordered(
                self._generate_tokenized_docs_worker,
                file_batches),
            total=len(self.files)//batch_size + 1):
            continue

        pool.close()
        pool.join()

    
    def _generate_tokenized_docs_worker(self, file_batch):
        for file in file_batch:
            doc_str = self.read_file(file)
            doc_id = self.files.index(file)
            path, filename = os.path.split(file)
            self.idx2doc[str(doc_id)] = filename
            self.tokenized_docs[doc_id] = self.tokenizer.tokenize_doc(doc_str)

    
    def _clean_tokenized_docs(self):
        # Get total number of tokens
        print('Counting tokens...')
        tokenized_docs = list(self.tokenized_docs.values()) # unpack dict
        total_tokens = [t for doc in tokenized_docs for t in doc]
        counter = Counter(total_tokens)

        # Get all tokens to remove with a count of less than 20
        to_remove = []
        for token, freq in counter.most_common():
            if freq < 20 and not self.args.toy:
                to_remove.append(token)

        # Remove the identified tokens from documents
        print('Removing infrequent tokens...')
        for doc_id in self.tokenized_docs:
            cleaned_doc = [t for t in self.tokenized_docs[doc_id] if t not in to_remove]
            self.tokenized_docs[doc_id] = cleaned_doc

    
    def _generate_examples(self):
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = self.args.file_batch_size
        doc_batches = self._batch_list(batch_size, list(self.tokenized_docs.keys()))

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

        :param file_batch: List of doc ids - subset of tokenized_docs keys
        :returns: None - saves batch to block file
        """
        examples = []
        for doc_id in doc_batches:
            examples.extend(self._generate_examples_from_doc(doc_id))

        # Save examples to a block file
        self._save_example_block(examples)

    
    def _generate_examples_from_doc(self, doc_id):
        """
        Generates examples from a tokenized doc

        :param doc_id: id to tokenized doc (a list of tokens)
        :returns: List of training examples
        """
        tokenized_doc = self.tokenized_docs[doc_id]
        examples = []
        for i, token in enumerate(tokenized_doc):
            # Generate context words for token in this doc
            context_words = self._generate_contexts(
                token, i, tokenized_doc
            )

            # Form Examples:
            # An example consists of:
            #   center word: token
            #   document id: doc_id
            #   context_word: tokenized_doc[context_word_pos]
            new_examples = [(token, doc_id, ctxt) for ctxt in context_words]

            # Add to class
            examples.extend(new_examples)
        return examples


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
        for w in range(-self.args.window_size, self.args.window_size + 1):
            context_pos = token_idx + w

            # Make sure current center and context are valid
            is_outside_doc = context_pos < 0 or context_pos >= len(tokenized_doc)
            center_is_context = token_idx == context_pos

            if is_outside_doc or center_is_context:
                # Not valid skip to next window position
                continue

            contexts.append(tokenized_doc[context_pos])
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

    def _example_to_tensor(self, center, doc_id, target):
        """
        Takes raw example and turns it into tensor values

        :params center: String of the center word
        :params doc_id: Document id training example is from
        :params target: String of the target word
        :returns: A tuple of tensors
        """
        center_idx = list(self.term_freq_dict.keys()).index(center)
        target_idx = list(self.term_freq_dict.keys()).index(target)

        center, doc_id, target = torch.tensor([int(center_idx)]), torch.tensor([int(doc_id)]), torch.tensor([int(target_idx)])
        return center, doc_id, target



