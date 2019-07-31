import os
import time
import torch
import torch.nn as nn
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


class HorovodTrainer(LDA2VecTrainer):
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
        torch.autograd.set_detect_anomaly(True)
        torch.cuda.manual_seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # Setup dataloader with a distributed sampler
        dataset = self.args.dataset(self.args)
        sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,
            shuffle=False, sampler=sampler, num_workers=self.args.workers, pin_memory=True)

        if self.args.save_dataset:
            self.logger.info("Beginning to save dataset.")
            self.saver.save_dataset(dataset)
            self.logger.info("Finished saving dataset")
            
        model = Lda2vec(len(dataset.term_freq_dict), len(dataset.idx2doc), self.args).cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr * hvd.size())

        # Distribute Optimizer
        compression = hvd.Compression.fp16 if self.args.compression else hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression)
       
        # Broadcast from rank 0 to all other processes
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Set up losses
        sgns = SGNSLoss(dataset, model.word_embeds, 'cuda')
        dirichlet = DirichletLoss(self.args)

        if self.args.resume is not None:
            self.logger.info(f'Loading checkpoint {self.args.resumer}')
            if not os.path.isfile(self.args.resume):
                raise Exception(f'There was no checkpoint found at {self.args.resum}')

            checkpoint = torch.load(self.args.resume)
            self.begin_epoch = checkpoint['epoch']  # Already added 1 when saving
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.logger.info(f"Loaded checkpoint {self.args.resume} at epoch {checkpoint['epoch']}")
        
        # Log on rank 0 GPU
        if hvd.rank() == 0:
            begin_time = time.perf_counter()
            self.logger.info(f'Began Training At: {begin_time}')
            self.logger.info(f'Using {hvd.size()} GPUs')
            
            if self.args.compression:
                self.logger.info(f'Using compression: {compression}')

        for epoch in range(self.begin_epoch, self.args.epochs):
            sampler.set_epoch(epoch)
            global_step = (1 + epoch) * len(dataloader)
            running_diri_loss, running_sgns_loss = 0.0, 0.0
            self.logger.info(f'GPU:{hvd.rank()} has {len(dataloader)} batches.')         
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
                    running_sgns_loss += sgns_loss
                    running_diri_loss += diri_loss
                    if global_step % self.args.log_step == 0:
                        norm = (i + 1) * self.args.batch_size
                        time_per_batch = ((time.perf_counter() - begin_time)/global_step)/hvd.size()
                        self.log_step(model, epoch, time_per_batch, global_step, 
                        running_diri_loss/norm, running_sgns_loss/norm, doc_id, center)

            # Log and save only rank 0 GPU results
            if hvd.rank() == 0:
                self.log_and_save_epoch(model, optimizer, epoch, dataset, loss)

        # Finished - close writer
        self.writer.close()

    def log_and_save_epoch(self, model, optim, epoch, dataset, loss):

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


    def log_step(self, model, epoch, time, global_step, diri_loss, sgns_loss, doc_id, center):
        self.logger.info(f'##################################################################################')
        self.logger.info(f'EPOCH: {epoch} | STEP: {global_step} | TIME: {time} (s/batch) | LOSS {diri_loss+sgns_loss}')
        self.logger.info(f'##################################################################################\n\n')
        # Log loss
        self.writer.add_scalar('train_loss', diri_loss + sgns_loss, global_step)
        self.writer.add_scalar('diri_loss', diri_loss, global_step)
        self.writer.add_scalar('sgns_loss', sgns_loss, global_step)
        
        # Log gradients - index select to only view gradients of embeddings in batch
        self.logger.info(f'DOCUMENT WEIGHT GRADIENTS:\n\
            {torch.index_select(model.doc_weights.weight.grad, 0, doc_id.squeeze())}')
        
        self.logger.info(f'WORD EMBEDDING GRADIENTS:\n\
            {torch.index_select(model.word_embeds.weight.grad, 0, center.squeeze())}')

        # Log document weights - check for sparsity
        doc_weights = model.doc_weights.weight
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
