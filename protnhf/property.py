import os
import shutil
from datetime import timedelta
import time
import importlib
import sys
import numpy as np
from pathlib import Path

import torch
from torch_scatter import scatter

from .dataset import Dataset, DataLoader, AA_TO_INDEX
from .flow import Flow
from .config import ConfigParams

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class NetCharge:
    def __init__(self, device='cpu'):
        charge_table_dict = {'A': 0,'C': 0,'D': -1,'E': -1,'F': 0,'G': 0,'H': +0.1,'I': 0,'K': 1,'L': 0,'M': 0,'N':0,'P':0,'Q': 0,'R': 1,'S': 0,'T': 0,'V': 0,'W': 0,'Y': 0}
        charge_table_idx = {}
        for key, value in charge_table_dict.items():
           charge_table_idx[AA_TO_INDEX[key]] = value

        self.charge_table = torch.tensor([v for k, v in sorted(charge_table_idx.items())], device=device)
    
    def __call__(self, data):
        return scatter(self.charge_table[data.h], data.batch, dim=0, reduce='sum')

       
class DDPLinearRegression:

    def __init__(self, config, world_size, world_rank, local_rank, num_cpus_per_task):
        self.world_size = int(world_size)
        self.world_rank = int(world_rank)
        self.local_rank = int(local_rank)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.world_rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"
        torch.cuda.set_device(self.local_rank)
        device = torch.cuda.current_device()

        dist.init_process_group('nccl', timeout=timedelta(seconds=7200000), init_method="env://", rank=self.world_rank, world_size=self.world_size)

        self.num_cpus_per_task = int(num_cpus_per_task)
        
        if self.world_rank == 0:
            print(f"Running DDP\nInitialized? {dist.is_initialized()}", flush=True)
        
        dataset = Dataset(config.training.dataset.file, config.training.dataset.limit)

        self.sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)            
        self.loader = DataLoader(dataset, batch_size=config.linreg.batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.sampler, drop_last=False)
        
        self.flow = Flow(config.model.n_types, config.model.hidden_dims, config.model.dt, config.model.niter, config.model.std, config.model.integrator,\
                         config.model.energy.d_model, config.model.energy.ff_dim, config.model.energy.n_heads, config.model.energy.n_layers).to(self.local_rank)
        self.checkpoint_path = config.model.checkpoint
        
        if config.linreg.property == 'net_charge': 
            self.prop = NetCharge(self.local_rank)
        else:
            self.prop = None
        
        if os.path.exists(self.checkpoint_path) and self.checkpoint_path != None:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.flow.load_state_dict(checkpoint['model_state_dict'])
        
        self.num_epochs = config.linreg.num_epochs
        
        self.model = torch.nn.Linear(self.flow.n_types, 1, bias=True).to(self.local_rank)
        
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.linreg.lr)
        self.loss = torch.nn.MSELoss()
        
        if self.world_rank == 0:
            print("Setup done", flush=True)
     
    def train(self):
        if self.world_rank == 0:
            print(f'*** {len(self.loader)} batches per rank ***', flush=True)
            print('Epoch \tTraining Loss \t   Time (s)', flush=True)
            print('      \t              \t           ', flush=True)
            
        for epoch in range(0, self.num_epochs):

            self.sampler.set_epoch(epoch)
            self.model.train()

            if self.world_rank == 0:
                torch.cuda.synchronize()
                start_time = time.time()
            
            losses = 0

            for i, data in enumerate(self.loader):
                with torch.no_grad():
                    data = data.to(self.local_rank)
                    _, q0, _ = self.flow.forward(data, train=False)
                    x = scatter(q0, data.batch, dim=0, reduce='mean').detach()

                pred = self.model(x)

                y = self.prop(data).unsqueeze(1)

                loss = self.loss(pred, y)
        
                losses = loss.item()
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss = torch.tensor(losses/len(self.loader), device=self.local_rank)
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss /= self.world_size
            
            if self.world_rank == 0:
                torch.save(self.model.module.weight.to('cpu').squeeze(), 'net_charge.pt')
    
                torch.cuda.synchronize()
                end_time = time.time()
    
                print('%.5i \t    %.5f \t    %.2f' % (epoch, epoch_loss.item(), end_time - start_time), flush=True)
            
            dist.barrier()
       
    def __del__(self):
        dist.destroy_process_group()

config = Path('config.yaml')
hndl = DDPTrainer(ConfigParams.fromFile(config), world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'),\
             local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
hndl.train()
