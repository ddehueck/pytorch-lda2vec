from sklearn.datasets import fetch_20newsgroups
from .dataset import LDA2VecDataset
from .preprocess import Tokenizer


class NewsgroupsDataset(LDA2VecDataset):

    def __init__(self, args):
        LDA2VecDataset.__init__(self, args)
        self.name = '20 News Groups Dataset'
        self.files = self.read_files_from_scikit()
        self.tokenizer = Tokenizer(custom_stop={'article', 'writes'})
        self.generate()

    def read_files_from_scikit(self):
        """
        Read files from sckilearn dataset

        Allows is to read it as json and return files as strings

        :returns: List of documents as strings
        """
        newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
        if self.args.toy:
            # Turns into a toy dataset
            return newsgroups_data['data'][:5]
            
        return newsgroups_data['data']

    def read_file(self, file):
        """
        Read In File

        Simply returns the file the was sent in as this
        datasets files are strings

        :param file: String
        :returns: String of file
        """
        return file
