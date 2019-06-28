from .dataset import LDA2VecDataset


class NewsgroupsDataset(LDA2VecDataset):

    def __init__(self, src_dir, device, window_size=5):
        LDA2VecDataset.__init__(self, src_dir, device, window_size)
        self.generate_examples_multi()

    def read_file(self, file):
        """
        Read In File

        Reads in plaintext file in newgroups 20 dataset

        :param file: Plaintext file
        :returns: String of f
        """
        with open(file, 'r', encoding='latin-1') as f:
            return f.read()
