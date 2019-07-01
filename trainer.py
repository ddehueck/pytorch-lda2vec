import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Lda2vec
from loss.dirichlet import DirichletLoss
from loss.sgns import SGNSLoss
from saver import Saver
from tqdm import tqdm

class Trainer:

    def __init__(self, args):
        # Load helpers
        self.args = args  # Argument parser results
        # TODO: Add support for multiple GPUs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.saver = Saver(args)
        self.writer = SummaryWriter(log_dir=self.saver.save_to_dir, flush_secs=3)

        # Load data
        self.dataset = args.dataset(args.dataset_dir, self.device)
        print("Finished Loading Dataset!")

        if args.save_dataset:
            self.saver.save_state({'dataset': {
                'examples': self.dataset.examples,
                'idx2doc': self.dataset.idx2doc,
                'files': self.dataset.files,
            }}, 'dataset.pth')
            print("Finished Saving Dataset")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)

        # Load model and training necessities
        self.model = Lda2vec(len(self.dataset.term_freq_dict), 128, 64,
                             len(self.dataset.files), args).to(self.device)
        self.optim = optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum)
        self.sgns = SGNSLoss(self.dataset, self.model.word_embeds, self.device)
        self.dirichlet = DirichletLoss()

        # Add graph to tensorboard
        self.writer.add_graph(self.model, iter(self.dataloader).next()[0])
        # Save metadata
        self.saver.save_metadata({
            'term_freq_dict': self.dataset.term_freq_dict,
            'index_to_document': self.dataset.idx2doc,
        })

    def train(self):
        print('\nTraining on device:', self.device)
        running_loss, sgns_loss, diri_loss, global_step = 0.0, 0.0, 0.0, 0
        for epoch in range(self.args.epochs):
            print("\nBeginning epoch: {}/{}".format(epoch+1, self.args.epochs))
            for i, data in enumerate(tqdm(self.dataloader)):
                # unpack data
                (center, doc_id), target = data
                # Remove accumulated gradients
                self.optim.zero_grad()
                # Get context vector - word + doc
                context = self.model((center, doc_id))
                # Calc loss - SGNS + Dirichlet
                sgns = self.sgns(context, self.model.word_embeds(target))
                diri = self.dirichlet(self.model.doc_weights(doc_id))
                loss = sgns + diri
               # Backprop and update
                loss.backward()
                self.optim.step()

                # Keep track of loss
                running_loss += loss.item()
                sgns_loss += sgns.item()
                diri_loss += diri.item()
                global_step += 1
                if global_step % 500 == 0:
                    norm = global_step * self.args.batch_size
                    self.writer.add_scalar('train_loss', running_loss/norm,
                                           global_step)
                    self.writer.add_scalar('sgns_loss', sgns_loss/norm,
                                           global_step)
                    self.writer.add_scalar('diri_loss', diri_loss/norm,
                                           global_step)

            # Log epoch loss
            print("Training Loss:", running_loss/(global_step*self.args.batch_size))

           # Visualize document embeddings
            self.writer.add_embedding(
                F.softmax(self.model.doc_weights.weight, dim=1),
                global_step=epoch,
                tag='DPEs'
            )

            # Save checkpoint
            self.saver.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': loss,
            })

        self.writer.close()

