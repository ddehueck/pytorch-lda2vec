import os
import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import distributed.utils as d_utils
from model import Lda2vec
from loss.dirichlet import DirichletLoss
from loss.sgns import SGNSLoss
from torch.nn.parallel import DistributedDataParallel as DDP

class DistTrainer:

    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset(args, "cuda")
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        
    def dist_train(self, rank, world_size):
        d_utils.setup(rank, world_size)
            
        # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
        # rank 2 uses GPUs [4, 5, 6, 7].
        n = torch.cuda.device_count() // world_size
        device_ids = list(range(rank * n, (rank + 1) * n))
            
        # create model and move it to device_ids[0]
        model = Lda2vec(len(self.dataset.term_freq_dict), len(self.dataset.files), self.args).to(device_ids[0])
        # output_device defaults to device_ids[0]
        ddp_model = DDP(model, device_ids=device_ids)

        sgns = SGNSLoss(self.dataset, ddp_model.module.word_embeds, 'cuda')
        dirichlet = DirichletLoss()
        optimizer = optim.Adam(ddp_model.parameters(), lr=self.args.lr)
        
        for epoch in range(self.args.epochs):
            print('EPOCH:', epoch)
            for i, data in tqdm(enumerate(self.dataloader)):
                # unpack data
                (center, doc_id), target = data
                # Remove accumulated gradients
                optimizer.zero_grad()
                # Get context vector: word + doc
                context = ddp_model((center, doc_id))
                # Calc loss: SGNS + Dirichlet
                sgns = sgns(context, ddp_model.module.model.word_embeds(target))
                diri = dirichlet(ddp_model.module.doc_weights(doc_id))
                loss = sgns + diri
                # Backprop and update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), self.args.clip)
                optimizer.step()

        d_utils.cleanup()


    def spawn(self, demo_fn, world_size):
        mp.spawn(demo_fn,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )

    def train(self):
        self.spawn(self.dist_train, 2)