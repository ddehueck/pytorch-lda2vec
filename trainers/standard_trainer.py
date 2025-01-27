import torch
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
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        self.logger.info("Finished loading data")
        self.logger.info(f'Number of training examples: {len(self.dataset)}')
        self.logger.info(f'Number of documents: {len(self.dataset.idx2doc)}')
        self.logger.info(f'There were {len(list(self.dataset.term_freq_dict.keys()))} tokens generated')

        # Get model initializations
        pretrained_vecs = utils.get_pretrained_vecs(self.dataset) if self.args.use_pretrained else None
        docs_init = utils.get_doc_vecs_lda_initialization(self.dataset) if self.args.lda_doc_init else None

        # Load model and training necessities
        self.model = Lda2vec(len(self.dataset.term_freq_dict), len(self.dataset.idx2doc), args,
            pretrained_vecs=pretrained_vecs, docs_init=docs_init)

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
        self.model.to(self.args.device)
        self.logger.info(f'Training on device: {self.args.device}\n')
        
        for epoch in range(self.begin_epoch, self.args.epochs):
            
            self.logger.info('Beginning epoch: {}/{}'.format(epoch+1, self.args.epochs))
            running_sgns_loss, running_diri_loss = 0.0, 0.0
            global_step = epoch * len(self.dataloader)
            num_examples = 0
            
            for i, data in enumerate(tqdm(self.dataloader)):
                # unpack data
                center, doc_id, targets = data
                center, doc_id, targets = center.to(self.args.device), doc_id.to(self.args.device), targets.to(self.args.device)
                # Remove accumulated gradients
                self.optim.zero_grad()
                # Get context vector: word + doc
                context = self.model(center, doc_id)
                # Calc loss: SGNS + Dirichlet
                sgns_loss = self.sgns(context, self.model.word_embeds(targets))
                diri_loss = self.dirichlet(self.model.doc_weights(doc_id))
                loss = sgns_loss + diri_loss
                # Backprop and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()

                # Keep track of loss
                running_sgns_loss += sgns_loss.item()
                running_diri_loss += diri_loss.item()
                global_step += 1
                num_examples += len(data)  # Last batch size may not equal args.batch_size
                
            # Log at epoch step
            norm = num_examples
            self.log_step(epoch, running_diri_loss/norm, running_sgns_loss/norm)
            self.save_epoch(epoch, doc_id, center, running_diri_loss/norm, running_sgns_loss/norm)

        self.writer.close()

    def save_epoch(self, epoch, doc_id, center, diri_loss, sgns_loss):
        epoch += 1
        if epoch % self.args.save_step == 0:

            self.logger.info(f'Retrieving topics...')
            vocab = list(self.dataset.term_freq_dict.keys())
            topics = utils.get_topics(self.model.word_embeds, self.model.topic_embeds, vocab)
            for i, topic in enumerate(topics):
                self.logger.info(f'TOPIC {i + 1}: {topic}')

            self.logger.info(f'\nBeginning to add to tensorboard')
            self.writer.add_embedding(
                self.model.get_doc_vectors(),
                global_step=epoch,
                tag=f'de_epoch_{epoch+1}',
            )
            self.logger.info(f'Finished adding to tensorboard\n')

            # Save checkpoint
            self.logger.info(f'Beginning to save checkpoint')
            self.saver.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': diri_loss + sgns_loss,
            })
            self.logger.info(f'Finished saving checkpoint\n')

            # Log loss
            self.writer.add_scalar('train_loss', diri_loss + sgns_loss, epoch)
            self.writer.add_scalar('diri_loss', diri_loss, epoch)
            self.writer.add_scalar('sgns_loss', sgns_loss, epoch)

            # Log gradients - index select to only view gradients of embeddings in batch
            self.logger.info(f'DOCUMENT WEIGHT GRADIENTS:\n\
                {torch.index_select(self.model.doc_weights.weight.grad, 0, doc_id.squeeze())}')

            self.logger.info(f'TOPIC GRADIENTS:\n{self.model.topic_embeds.grad}')

           # self.logger.info(f'WORD EMBEDDING GRADIENTS:\n\
            #    {torch.index_select(self.model.word_embeds.weight.grad, 0, center.squeeze())}')

            # Log document weights - check for sparsity
            doc_weights = self.model.doc_weights.weight
            proportions = F.softmax(doc_weights, dim=1)
            avg_s_score = np.mean([utils.get_sparsity_score(p) for p in proportions])

            self.logger.info(f'DOCUMENT PROPORTIIONS:\n {proportions}\n')
            self.logger.info(f'AVERAGE SPARSITY SCORE: {avg_s_score}\n')
            self.writer.add_scalar('avg_doc_prop_sparsity_score', avg_s_score, epoch)

            _, max_indices = torch.max(proportions, dim=1)
            max_indices = list(max_indices.cpu().numpy())
            max_counter = Counter(max_indices)

            self.logger.info(f'MAXIMUM TOPICS AT INDICES, FREQUENCY: {max_counter}\n')
            self.logger.info(f'MOST FREQUENCT MAX INDICES: {max_counter.most_common(10)}\n')

    def log_step(self, epoch, diri_loss, sgns_loss):
        self.logger.info(f'\nEPOCH: {epoch + 1} | DIRI LOSS {diri_loss:.4f} | SGNS LOSS: {sgns_loss:.4f}\n')

