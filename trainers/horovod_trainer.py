import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from model import Lda2vec
from loss.dirichlet import DirichletLoss
from loss.sgns import SGNSLoss
import horovod.torch as hvd
from .trainer import LDA2VecTrainer
import numpy as np
import utils
from collections import Counter


class HorovodTrainer:
    """
    Following: https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py 
    """

    def __init__(self, args):
        LDA2VecTrainer.__init__(self, args)
        
    def train(self):
        # Initialize Horovod
        hvd.init()
        # Pin GPU to be used to process local rank (one GPU per process)
        torch.cuda.set_device(hvd.local_rank())

        #setup dataloader
        dataset = self.args.dataset(self.args)
        sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
            shuffle=False, sampler=sampler, num_workers=self.args.workers)
            
        model = Lda2vec(len(dataset.term_freq_dict), len(dataset.files), self.args)
        model.cuda()

        sgns = SGNSLoss(dataset, model.word_embeds, 'cuda')
        dirichlet = DirichletLoss()

        #Distributed optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr * hvd.size())
       
       # Broadcast from rank 0 to all other processes
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        compression = hvd.Compression.fp16 if self.args.compression else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression)
        
        # Log on rank 0 GPU
        if hvd.rank() == 0:
            begin_time = time.perf_counter()
            self.logger.info(f'Began Training At: {begin_time}')
            self.logger.info(f'Using {hvd.size()} GPUs')
            
            if self.args.compression:
                self.logger.info(f'Using compression: {compression}')

        global_step = 0
        for epoch in range(self.args.epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # unpack data
                (center, doc_id), target = data             
                center, doc_id, target = center.cuda(), doc_id.cuda(), target.cuda()
                # Remove accumulated gradients
                optimizer.zero_grad()
                # Get context vector: word + doc
                context = model((center, doc_id))
                # Calc loss: SGNS + Dirichlet
                sgns_loss = sgns(context, model.word_embeds(target))
                diri_loss = dirichlet(model.doc_weights(doc_id))
                loss = sgns_loss + diri_loss
                # Backprop and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                optimizer.step()
                
                # Log only rank 0 GPU results
                if hvd.rank() == 0:
                    global_step += 1
                    running_loss += loss
                    if global_step % self.args.log_step == 0:
                        norm = (i + 1) * self.args.batch_size
                        time_since = time.perf_counter() - begin_time
                        train_loss = running_loss/norm
                        self.log_step(model, epoch, time_since, global_step, train_loss, doc_id, center)

            # Log and save only rank 0 GPU results
            if hvd.rank() == 0:
                self.log_and_save_epoch(model, optimizer, epoch, dataset)

        # Finished - close writer
        self.writer.close()

    def log_and_save_epoch(self, model, optim, epoch, dataset):

       # Visualize document embeddings
        self.writer.add_embedding(
            model.get_doc_vectors(),
            global_step=epoch,
            tag=f'de_epoch_{epoch}',
        )

        # Save checkpoint
        self.logger.info(f'Beginning to save checkpoint')
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        })
        self.logger.info(f'Finished saving checkpoint')


    def log_step(self, model, epoch, time, global_step, train_loss, doc_id, center):
        self.logger.info(f'\n\n#####################################################################')
        self.logger.info(f'EPOCH: {epoch} | STEP: {global_step} | TIME: {time} | LOSS {train_loss}')
        self.logger.info(f'#####################################################################\n\n')
        # Log loss
        self.writer.add_scalar('train_loss', train_loss, global_step)
        
        # Log gradients - index select to only view gradients of embeddings in batch
        self.logger.info(f'\nDOCUMENT WEIGHT GRADIENTS:\n\
            {torch.index_select(model.doc_weights.weight.grad, 0, doc_id.squeeze())}')
        
        self.logger.info(f'\nWORD EMBEDDING GRADIENTS:\n\
            {torch.index_select(model.word_embeds.weight.grad, 0, center.squeeze())}')

        # Log document weights - check for sparsity
        doc_weights = model.doc_weights.weight
        proportions = F.softmax(doc_weights, dim=1)
        avg_s_score = np.mean([utils.get_sparsity_score(p) for p in proportions])

        self.logger.info(f'\nDOCUMENT PROPORTIIONS:\n {proportions}')    
        self.logger.info(f'\nAVERAGE SPARSITY SCORE: {avg_s_score}\n')   
        self.writer.add_scalar('avg_doc_prop_sparsity_score', avg_s_score, global_step)

        _, max_indices = torch.max(proportions, dim=1)
        max_indices = list(max_indices.cpu().numpy())
        max_counter = Counter(max_indices)
        
        self.logger.info(f'\nMAXIMUM TOPICS AT INDICES, FREQUENCY: {max_counter}\n')
        self.logger.info(f'MOST FREQUENCT MAX INDICES: {max_counter.most_common(10)}\n')
