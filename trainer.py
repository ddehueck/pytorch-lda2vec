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
from logger import Logger
import utils
from tqdm import tqdm
import numpy as np
import os

class Trainer:

    def __init__(self, args):
        self.args = args  # Argument parser results
        # TODO: Add support for multiple GPUs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.saver = Saver(args)
        self.writer = SummaryWriter(log_dir=self.saver.save_to_dir, flush_secs=3)
        self.logger = Logger(self.saver.save_to_dir).logger

        # Load data
        self.dataset = args.dataset(args, self.device)
        self.logger.info("Finished loading dataset")

        if args.save_dataset:
            self.logger.info("Beginning to save dataset.")
            self.saver.save_dataset(self.dataset)
            self.logger.info("Finished saving dataset")

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        # Load model and training necessities
        self.model = Lda2vec(len(self.dataset.term_freq_dict), len(self.dataset.files), args).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)
        self.sgns = SGNSLoss(self.dataset, self.model.word_embeds, self.device)
        self.dirichlet = DirichletLoss()

        # Add graph to tensorboard
        self.writer.add_graph(self.model, iter(self.dataloader).next()[0])

        # Load checkpoint if need be
        if args.resume is not None:
            self.logger.info('Loaded checkpoint {}'.format(args.resume))
            if not os.path.isfile(args.resume):
                raise Exception("There was no checkpoint found at '{}'" .format(args.resume))
            
            checkpoint = torch.load(args.resume)
            self.args.epochs -= checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.info('Loaded checkpoint {} at epoch {}'.format(args.resume, checkpoint['epoch']))


    def train(self):
        # TODO: Offload logging data into logger class
        self.logger.info('Training on device: {}'.format(self.device))
        running_loss, sgns_loss, diri_loss, global_step = 0.0, 0.0, 0.0, 0
        for epoch in range(self.args.epochs):
            self.logger.info('Beginning epoch: {}/{}'.format(epoch+1, self.args.epochs))
            for data in tqdm(self.dataloader):
                # unpack data
                (center, doc_id), target = data
                # Remove accumulated gradients
                self.optim.zero_grad()
                # Get context vector: word + doc
                context = self.model((center, doc_id))
                # Calc loss: SGNS + Dirichlet
                sgns = self.sgns(context, self.model.word_embeds(target))
                diri = self.dirichlet(self.model.doc_weights(doc_id))
                loss = sgns + diri
               # Backprop and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()

                # Keep track of loss
                running_loss += loss.item()
                sgns_loss += sgns.item()
                diri_loss += diri.item()
                global_step += 1
                if global_step % self.args.log_step == 0:
                    norm = global_step * self.args.batch_size
                    self.writer.add_scalar('train_loss', running_loss/norm,
                                           global_step)
                    self.writer.add_scalar('sgns_loss', sgns_loss/norm,
                                           global_step)
                    self.writer.add_scalar('diri_loss', diri_loss/norm,
                                           global_step)
                    # Log gradients - index select to only view gradients of embeddings in batch
                    self.logger.info('\nLogging Gradient: \nDocument weight gradients at step {}:\n {}'.format(
                        global_step, torch.index_select(self.model.doc_weights.weight.grad, 0, doc_id.squeeze())))
                    self.logger.info('\nLogging Gradient: \nWord embedding gradients at step {}:\n {}'.format(
                        global_step, torch.index_select(self.model.word_embeds.weight.grad, 0, center.squeeze())))

                    # Log document weights - check for sparsity
                    doc_weights =  self.model.doc_weights.weight
                    proportions = F.softmax(doc_weights, dim=1)
                    avg_s_score = np.mean([utils.get_sparsity_score(p) for p in proportions])

                    self.logger.info('\nLogging Proportions: \nDocuments proportions at step {}:\n {}'.format(
                        global_step, proportions))    
                    self.logger.info('\nLogging Proportions: Average Sparsity score is {}\n'.format(avg_s_score))   
                    self.writer.add_scalar('avg_doc_prop_sparsity_score', avg_s_score, global_step)

                    _, max_indices = torch.max(proportions, dim=1)
                    self.logger.info('\nLogging Maximum Indices: {}\n'.format(max_indices))
            
            # Log epoch loss
            self.logger.info("Training Loss: {}".format(running_loss/(global_step*self.args.batch_size)))

           # Visualize document embeddings
            self.writer.add_embedding(
                self.model.get_doc_vectors(),
                global_step=epoch,
                tag='DEs',
                metadata=list(self.dataset.idx2doc.values())
            )

            # Save checkpoint
            self.logger.info("Beginning to save checkpoint")
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': loss,
            })
            self.logger.info("Finished saving checkpoint")

        self.writer.close()

