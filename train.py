import argparse
from datasets import newsgroups
from datasets import freetext
import torch as t


def str_to_bool(arg):
    if isinstance(arg, bool):
       return arg
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str_to_dataset(arg):
    if arg.lower() == 'freetext':
        return freetext.FreeTextDataset
    elif arg.lower() == 'newsgroups':
        return newsgroups.NewsgroupsDataset
    else:
        raise argparse.ArgumentTypeError('No dataset found by the name:', arg)


def get_args():
    # TODO: Turn some boolean flags into just flag or no flag
    parser = argparse.ArgumentParser(description="PyTorch LDA2Vec Training")

    """
    Data handling
    """
    parser.add_argument('--dataset', type=str_to_dataset, default='freetext',
                        help='dataset to use when training (default: freetext)')
    parser.add_argument('--dataset-dir', type=str, default='data/',
                        help='dataset directory (default: data/)')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                       help='dataloader threads (default: 4)')
    parser.add_argument('--window-size', type=int, default=5, help='Window size\
                        used when generating training examples (default: 5)')
    parser.add_argument('--file-batch-size', type=int, default=250, help='Batch size\
                        used when mult-threading the generation of training examples\
                        (default: 250)')
    parser.add_argument('--toy', type=str_to_bool, default=False, help='Boolean to \
                        use just 5 files as a toy dataset for testing (default: False)')
    parser.add_argument('--read-from-blocks', type=str_to_bool, default=False,
                        help='Boolean to read in files on get_item')

    """
    Model Parameters
    """
    parser.add_argument('--num-topics', type=int, default=20, help='Number of topics\
                        to learn during training (default: 20)')
    parser.add_argument('--embedding-len', type=int, default=128, help='Length of\
                        embeddings in model (default: 128)')
    parser.add_argument('--lda-doc-init', type=str_to_bool, default=False,
                        help='Run LDA on dataset and use probs as initialization (default: False)')
    parser.add_argument('--use-pretrained', type=str_to_bool, default=False, 
                        help='Use GloVe vectors trained on Common Crawl (default: False)')
    parser.add_argument('--lambda-val', type=int, default=100, help='Balancing parameter\
                        for dirichlet loss value (default: 100)')
    parser.add_argument('--uni-doc-init', type=str_to_bool, default=False, help='Have doc\
                        weights be uniformaly distributed to counteract initial undefined\
                         topic structure - experimental (default: False)')
    
    """
    Training Hyperparameters
    """
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train for - iterations over the dataset (default: 15)')
    parser.add_argument('--batch-size', type=int, default=4096,
                        metavar='N', help='number of examples in a training batch (default: 4096)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--use-dropout', type=str_to_bool, default=True,
                        help='Boolean value to apply dropout during training\
                        (default: True)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--clip', type=float, default=5.0, help='Value to keep gradient between (-val, +val)\
                        (default: 5)')
    
    """
    Checkpoint options
    """
    parser.add_argument('--resume', type=str, default=None,
                        help='Put the path to checkpoint file to resume training')
    parser.add_argument('--check-dir', type=str, default='experiments/',
                        help='Set the checkpoint directory name')
    parser.add_argument('--stream-log', type=str_to_bool, default=True,
                        help='Boolean to stream log output to console.')
    parser.add_argument('--save-log', type=str_to_bool, default=True,
                        help='Boolean to save log output to a file.')
    parser.add_argument('--load-dataset', type=str, default=None,
                        help='Put the path to dataset file to use.')
    parser.add_argument('--log-step', type=int, default=250, help='Step at which for every step training info\
                        is logged. (default: 250)')
    parser.add_argument('--save-step', type=int, default=20, help='Epoch step to save a checkpoint - should make\
                         --epochs a multiple of this value. (default: 20)')

    """
    Training Settings
    """
    parser.add_argument('--distributed', type=str_to_bool, default=False, help='Boolean to\
                        use distributed data parallel (default: False)')
    parser.add_argument('--horovod', type=str_to_bool, default=False, help='Boolean to\
                        use horovod distributed training (default: False)')
    parser.add_argument('--compression', type=str_to_bool, default=True, help='Boolean to\
                        use fp16 compression horovod distributed training (default: True)')

    # Can only be used with the default, standard trainer.
    parser.add_argument('--device', type=str, default=t.device("cuda:0" if t.cuda.is_available() else "cpu"),
                        help='device to train on (default: cuda:0 if cuda is available otherwise cpu)')

    """
    Jupyter Notebook Argumenet
    """
    parser.add_argument('-f', type=str, help='Jupyter Notebook Argument - file')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Import with call to trainer to avoid distributed requirments in some cases
    if args.distributed:
        from trainers.dist_trainer import DistTrainer
        trainer = DistTrainer(args)
    elif args.horovod:
        from trainers.horovod_trainer import HorovodTrainer
        trainer = HorovodTrainer(args)
    else:
        from trainers.standard_trainer import Trainer
        trainer = Trainer(args)
    
    # Begin Training!
    trainer.train()
