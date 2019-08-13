import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import Lda2vec
from loss.dirichlet import DirichletLoss
from loss.sgns import SGNSLoss
import utils
from tqdm import tqdm
import numpy as np
import os
from .trainer import LDA2VecTrainer
from collections import Counter

class Trainer(LDA2VecTrainer):

    def __init__(self, args):
        LDA2VecTrainer.__init__(self, args)
        # Load data
        self.dataset = args.dataset(args)
        self.logger.info("Finished loading dataset")

        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        pretrained_vecs = None
        if self.args.use_pretrained:
            pretrained_vecs = utils.get_pretrained_vecs(self.dataset, self.args.nlp)

        # Load model and training necessities
        self.model = Lda2vec(len(self.dataset.term_freq_dict), len(self.dataset.files), args,
            pretrained_vecs=pretrained_vecs)

        print(f'Current size of word embeds weights is {self.model.word_embeds.weight.size()}')

        self.optim = optim.Adam(self.model.parameters(), lr=args.lr)
        self.sgns = SGNSLoss(self.dataset, self.model.word_embeds, self.args.device)
        self.dirichlet = DirichletLoss(self.args)

        # Visualize RANDOM document embeddings
        """print('Adding random embeddings')
        self.writer.add_embedding(
            self.model.get_doc_vectors(),
            global_step=0,
            tag=f'de_epoch_random',
        )
        print('Finished adding random embeddings!')"""

        # Add graph to tensorboard
        # TODO: Get working on multi-gpu stuff
        center_id, doc_id, target_id = iter(self.dataloader).next()
        self.writer.add_graph(self.model, input_to_model=(center_id, doc_id))

        # Load checkpoint if need be
        if args.resume is not None:
            self.logger.info('Loaded checkpoint {}'.format(args.resume))
            if not os.path.isfile(args.resume):
                raise Exception("There was no checkpoint found at '{}'" .format(args.resume))
            
            checkpoint = torch.load(args.resume)
            self.begin_epoch = checkpoint['epoch']  # Already added 1 when saving
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.info('Loaded checkpoint {} at epoch {}'.format(args.resume, checkpoint['epoch']))


    def train(self):
        # TODO: Offload logging data into logger class
        self.model.to(self.args.device)
        self.logger.info('Training on device: {}'.format(self.args.device))
        
        for epoch in range(self.begin_epoch, self.args.epochs):
            
            self.logger.info('Beginning epoch: {}/{}'.format(epoch+1, self.args.epochs))
            running_sgns_loss, running_diri_loss = 0.0, 0.0
            global_step = epoch * len(self.dataloader)
            num_examples = 0
            
            for i, data in enumerate(tqdm(self.dataloader)):
                # unpack data
                center, doc_id, target = data
                center, doc_id, target = center.to(self.args.device), doc_id.to(self.args.device), target.to(self.args.device)
                # Remove accumulated gradients
                self.optim.zero_grad()
                # Get context vector: word + doc
                context = self.model(center, doc_id)  # context - [batch_size x embed_len x 1]
                # Calc loss: SGNS + Dirichlet
                sgns_loss = self.sgns(context, self.model.word_embeds(target))  # target - [batch_size x 1]
                diri_loss = self.dirichlet(self.model.doc_weights(doc_id))  # doc_id - [batch_size x 1]
                loss = sgns_loss + diri_loss
               # Backprop and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()

                # Keep track of loss
                running_sgns_loss += sgns_loss.item()
                running_diri_loss += diri_loss.item()
                global_step += 1
                num_examples += len(data) # Last batch size may not equal args.batch_size
                
                # Log at step
                if global_step % self.args.log_step == 0:
                    norm = num_examples
                    self.log_step(epoch, global_step, running_diri_loss/norm, running_sgns_loss/norm, doc_id, center, target)
            
            self.log_and_save_epoch(epoch, (running_sgns_loss + running_diri_loss)/num_examples)

        self.writer.close()

    def log_and_save_epoch(self, epoch, loss):

       # Visualize document embeddings
        self.writer.add_embedding(
            self.model.get_doc_vectors(),
            global_step=epoch,
            tag=f'de_epoch_{epoch}',
        )

        # Save checkpoint
        self.logger.info(f'Beginning to save checkpoint')
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': loss,
        })
        self.logger.info(f'Finished saving checkpoint')


    def log_step(self, epoch, global_step, diri_loss, sgns_loss, doc_id, center, target):
        self.logger.info(f'#############################################')
        self.logger.info(f'EPOCH: {epoch} | STEP: {global_step} | LOSS {diri_loss+sgns_loss}')
        self.logger.info(f'#############################################\n\n')
        # Log loss
        self.logger.info(f'DIRI LOSS: {diri_loss}')
        self.logger.info(f'SGNS LOSS: {sgns_loss}')
        self.writer.add_scalar('train_loss', diri_loss + sgns_loss, global_step)
        self.writer.add_scalar('diri_loss', diri_loss, global_step)
        self.writer.add_scalar('sgns_loss', sgns_loss, global_step)
        
        # Log gradients - index select to only view gradients of embeddings in batch
        self.logger.info(f'DOCUMENT WEIGHT GRADIENTS:\n\
            {torch.index_select(self.model.doc_weights.weight.grad, 0, doc_id.squeeze())}')
        
        self.logger.info(f'TOPIC GRADIENTS:\n{self.model.topic_embeds.grad}')
        
        self.logger.info(f'WORD EMBEDDING GRADIENTS:\n\
            {torch.index_select(self.model.word_embeds.weight.grad, 0, center.squeeze())}')
        self.logger.info(f'\n{torch.index_select(self.model.word_embeds.weight.grad, 0, target.squeeze())}')

        # Log document weights - check for sparsity
        doc_weights = self.model.doc_weights.weight
        proportions = F.softmax(doc_weights, dim=1)
        avg_s_score = np.mean([utils.get_sparsity_score(p) for p in proportions])

        self.logger.info(f'DOCUMENT PROPORTIIONS:\n {proportions}')    
        self.logger.info(f'AVERAGE SPARSITY SCORE: {avg_s_score}\n')   
        self.writer.add_scalar('avg_doc_prop_sparsity_score', avg_s_score, global_step)

        _, max_indices = torch.max(proportions, dim=1)
        max_indices = list(max_indices.cpu().numpy())
        max_counter = Counter(max_indices)
        
        self.logger.info(f'MAXIMUM TOPICS AT INDICES, FREQUENCY: {max_counter}\n')
        self.logger.info(f'MOST FREQUENCT MAX INDICES: {max_counter.most_common(10)}\n')

