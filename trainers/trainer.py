from torch.utils.tensorboard import SummaryWriter
from saver import Saver
from logger import Logger

class LDA2VecTrainer:

    def __init__(self, args):
        self.args = args
        self.saver = Saver(args)
        self.writer = SummaryWriter(log_dir=self.saver.save_to_dir, flush_secs=3)
        self.logger = Logger(self.saver.save_to_dir, args).logger
        self.begin_epoch = 0


    def train(self):
        """
        Training method to be called in train.py

        :returns: None - use saver/logger to save model and other info
        """
        raise NotImplementedError
