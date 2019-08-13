import os
import torch
import uuid
import multiprocessing
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from .preprocess import Tokenizer
from multiprocessing.dummy import Pool as ThreadPool

class LDA2VecDataset(Dataset):
    BETA = 0.75

    def __init__(self, args, saver):
        self.args = args
        self.saver = saver
        self.term_freq_dict = dict()
        self.files = self._get_files_in_dir(args.dataset_dir)
        self.tokenizer = Tokenizer(args)
        self.removed_infrequent_tokens = False
        self.idx2doc = dict()
        self.name = ''
        self.saved_ds_dir = 'saved_datasets/'
        self.block_files = {}

        if args.toy:
            # Turns into a toy dataset
            self.file = self.files[:5]
            self.removed_infrequent_tokens = True


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

        # Set block files
        metadata_file = f'{self._get_saved_ds_dir()}metadata.pth' 
        files = self._get_files_in_dir(self._get_saved_ds_dir())
        files.remove(metadata_file) 
        for f in tqdm(files):
            self.block_files[f] = len(torch.load(f))

        # Load metadata in
        metadata = torch.load(metadata_file)
        self.idx2doc = metadata['idx2doc']
        self.term_freq_dict = metadata['term_freq_dict']
        print('Loaded Dataset!')
    

    def _save_training_examples(self, examples):
        # Create save to directory
        if not os.path.exists(self._get_saved_ds_dir()):
            os.makedirs(self._get_saved_ds_dir())
        
        # Save to np file   
        filename = f'{self._get_saved_ds_dir()}/block-{uuid.uuid4()}.dat'
        torch.save(examples, filename)
        self.block_files[filename] = len(examples)


    def _save_metadata(self):
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
        try:
            tokenized_doc = self.tokenizer.tokenize_doc(doc_str)
        except Exception as e:
            print(doc_str)
            raise Exception(e)

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
            new_examples = [(token, doc_id, ctxt) for ctxt in context_words]

            # Add to class
            examples.extend(new_examples)
        return examples


    def generate_examples_multi(self):
        if os.path.exists(self._get_saved_ds_dir()):
            # Data already exists - load it!
            self._load_dataset()
            return
        
        pool = ThreadPool(multiprocessing.cpu_count())
        batch_size = self.args.file_batch_size
        file_batches = self._batch_files(batch_size)

        print('\nGenerating Examples for Dataset (multi-threaded)...')
        for tf_dict in tqdm(
            pool.imap_unordered(
                self._generate_examples_worker,
                file_batches),
            total=len(self.files)//batch_size + 1):

            # Reduce each tf_dict into final location
            self._reduce_tf_dict(tf_dict)

        pool.close()
        pool.join()

        # Remove any tokens with a frequency of less than 10
        # Remove examples too by regenerating
        if not self.removed_infrequent_tokens:
            tokens_to_remove = set([k for k in self.term_freq_dict if self.term_freq_dict[k] < 10])
            custom_stop = self.tokenizer.custom_stop.union(tokens_to_remove)
            
            self.tokenizer = Tokenizer(self.args, custom_stop=custom_stop)
            self.removed_infrequent_tokens = True

            # Reset and regenerate examples!
            self.term_freq_dict = dict()
            self.generate_examples_multi()
        
        if self.removed_infrequent_tokens:
            self._save_metadata()


    def _generate_examples_worker(self, file_batch):
        """
        Generate examples worker

        Worker to generate examples in a map reduce paradigm
        Saves the generated example to a batch file

        :param file_batch: List of files - a subset of self.files
        :returns: a term frequency dict for its batch
        """
        tf_dict = dict()
        examples = []

        for f in file_batch:
            examples.extend(self.generate_examples_from_file(f, tf_dict))

        # Save batch to file after infrequent tokens are removed
        if self.removed_infrequent_tokens:
            self._save_training_examples(examples)

        return tf_dict


    def _batch_files(self, batch_size):
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



