import logging
import sys


class Logger:

    def __init__(self, save_to_dir):
        self.logger = logging.getLogger("Pytorch LDA2Vec")
        self.logger.setLevel(logging.INFO)
        
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler(save_to_dir + "info.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)