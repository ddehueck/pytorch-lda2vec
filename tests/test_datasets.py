import sys,os
sys.path.append(os.path.realpath('..'))

import torch
import unittest
import random
import preprocess as pre
from datasets.dataset import LDA2VecDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TestDataset(LDA2VecDataset):
   
    def __init__(self):
        LDA2VecDataset.__init__(self, None, device)
        self.files = [
            'file one', 
            'file two', 
            'file three',
            'file four'
            ]

    def read_file(self, file):
        """
        Read In File

        Simply returns the file the was sent in as this
        datasets files are strings

        :param file: String
        :returns: String of file
        """
        return 'This is the test file.'


class TestData(unittest.TestCase):

    def test_tokenize_doc(self):
        dataset = TestDataset()
        doc_str = random.choice(dataset.files)
        tokenized_doc = dataset._tokenize_doc(doc_str)

        self.assertEqual(type(tokenized_doc), type([]))

    def test_generate_examples_multi(self):
        dataset = TestDataset()
        # Run multiprocessing
        dataset.generate_examples_multi(batch_size=1)
        
        expected_len = 0
        for file in dataset.files:
            tokenized_doc = dataset._tokenize_doc(file)
            expected_len += pre.pred_num_window_pairs(dataset.window_size, len(tokenized_doc))

        print(expected_len)
        self.assertEqual(len(dataset.examples), expected_len)


if __name__ == '__main__':
    unittest.main()
