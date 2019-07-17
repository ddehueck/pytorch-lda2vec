import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import distributed.utils as d_utils
from model import Lda2vec
from loss.dirichlet import DirichletLoss
from loss.sgns import SGNSLoss
from torch.nn.parallel import DistributedDataParallel as DDP
import horovod.torch as hvd

class HorovodTrainer:
    """
    Following: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, args):
        self.args = args
        
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
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        
        if self.args.compression:
            compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=compression)
            print('Using compression: {}'.format(compression))
        else:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

        # Broadcast from rank 0 to all other processes
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        # Log time
        begin_time = time.perf_counter()
        print("Began Training At:", begin_time)

        global_step = 0
        for epoch in range(self.args.epochs):
            running_loss = 0.0
            for i, data in tqdm(enumerate(dataloader)):
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
                
                global_step += 1
                running_loss += loss
                if global_step % self.args.log_step == 0:
                    norm = (i + 1) * self.args.batch_size
                    print('EPOCH: {} | STEP: {} | LOSS {}'.format(
                        epoch, global_step, running_loss/norm
                    ))

        # Log time
        end_time = time.perf_counter()
        print("Ended Training At:", end_time)
        print("Training Lasted: {} ms".format(end_time - begin_time))
