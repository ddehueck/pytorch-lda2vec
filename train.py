import argparse
from trainer import Trainer
from datasets import newsgroups
from datasets import freetext


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


def main():
    parser = argparse.ArgumentParser(description="PyTorch LDA2Vec Training")

    """
    Data handling
    """
    parser.add_argument('--dataset', type=str_to_dataset, default='freetext',
                        help='dataset to use when training (default: freetext)')
    parser.add_argument('--dataset-dir', type=str, default='data/',
                        help='dataset directory (default: data/)')
    parser.add_argument('--save-dataset', type=str_to_bool, default=False,
                        help='Boolean value to save dataset to a file\
                        (default: False)')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                       help='dataloader threads (default: 4)')
    """
    Training Hyperparameters
    """
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train for - iterations over the dataset (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='number of examples in a training batch (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--use-dropout', type=str_to_bool, default=True,
                        help='Boolean value to apply dropout during training\
                        (default: True)')
    parser.add_argument('--use-batchnorm', type=str_to_bool, default=True,
                              help='Boolean value to apply batch normalization\
                              during training (default: True)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    """
    Checkpoint options
    """
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--check_dir', type=str, default='experiments/',
                        help='set the checkpoint name')

    args = parser.parse_args()
    trainer = Trainer(args)
    # Begin training!
    trainer.train()


if __name__ == '__main__':
    main()
