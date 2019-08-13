import torch
import datetime as dt
import os

class Saver:

    def __init__(self, args):
        self.save_to_dir = args.check_dir + \
        dt.datetime.now().strftime("%H%M%S_%d%m%Y") + '_checkpoints/'

        # Use same experiment file if resuming
        if args.resume is not None:
            self.save_to_dir = args.resume.replace(os.path.basename(args.resume), '')

        # Create save to directory
        if not os.path.exists(self.save_to_dir):
            os.makedirs(self.save_to_dir)

        # Save hyperparameters to txt file
        with open(self.save_to_dir + 'hyperparameters.txt', 'w') as f:
            ret = ''
            for k in vars(args):
                ret += k + '=' + str(vars(args)[k]) + '\n'
            f.write(ret)

        # Create a shell script to run tensorboard to avoid typing weird dir name
        with open(self.save_to_dir + 'tensorboard.sh', 'w') as f:
            f.write('tensorboard --logdir=../' +
                    self.save_to_dir.replace(args.check_dir, ''))

    def save_state(self, state, filename):
        filename = self.save_to_dir + filename
        torch.save(state, filename)

    def save_checkpoint(self, state):
        filename = 'checkpoint_' + str(state['epoch']) + '.pth'
        self.save_state(state, filename)

