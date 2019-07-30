import torch
import ujson
from .dataset import LDA2VecDataset


class FreeTextDataset(LDA2VecDataset):

    def __init__(self, args):
        LDA2VecDataset.__init__(self, args)

        if args.load_dataset is not None:
            # Load dataset from .pth file
            dataset = torch.load(args.load_dataset)

            # Load Dataset Object
            self.examples = dataset['examples']
            self.idx2doc = dataset['idx2doc']
            self.term_freq_dict = dataset['term_freq_dict']

            # Already saved
            args.save_dataset = False
        else:    
            self.generate_examples_multi()

    def read_file(self, file):
        """
        Read In File

        Reads JSON files outputted by the selinon worker which generated this
        free text dataset.

        :param file: JSON file
        :returns: String of f
        """
        with open(file, 'r') as f:
            data = ujson.load(f)
            try:
                doc = data["result"]["content"]
            except Exception as e:
                # Can't build any training examples - skip
                # File will still count toward num_docs in model
                doc = ''
            return doc

