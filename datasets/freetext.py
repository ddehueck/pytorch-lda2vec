import torch
import ujson
from .dataset import LDA2VecDataset


class FreeTextDataset(LDA2VecDataset):

    def __init__(self, args):
        LDA2VecDataset.__init__(self, args)
        self.name = 'PyPI Free Text Dataset'
        self.generate_examples_multi()

        print(f'There were {len(list(self.term_freq_dict.keys()))} tokens generated')

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

