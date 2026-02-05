import os
import shutil
from datetime import timedelta
import time
import importlib
import sys
import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup
from .dataset import Dataset, DataLoader
from .flow import Flow
from .metrics import NetCharge

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class DDPBase:
    def __init__(self, world_size, world_rank, local_rank, num_cpus_per_task):
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
   
    def __del__(self):
        dist.destroy_process_group()
    
          
class DDPTrainer(DDPBase):

    def __init__(self, config, world_size, world_rank, local_rank, num_cpus_per_task):
        super().__init__(world_size, world_rank, local_rank, num_cpus_per_task)
        
        full_dataset = Dataset(config.training.dataset.file, config.training.dataset.limit)
        train_val_split = [config.training.dataset.split, 1-config.training.dataset.split]
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, train_val_split)
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)            
        self.train_loader = DataLoader(train_dataset, batch_size=config.training.dataset.batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.train_sampler, drop_last=False)
        
        self.val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=config.training.dataset.batch_size, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.val_sampler, drop_last=False)
        
        self.model = Flow(config.model.n_types, config.model.hidden_dims, config.model.dt, config.model.niter, config.model.std, config.model.integrator,\
                         config.model.energy.d_model, config.model.energy.ff_dim, config.model.energy.n_heads, config.model.energy.n_layers).to(self.local_rank)
        self.checkpoint_path = config.model.checkpoint
            
        if os.path.exists(self.checkpoint_path) and self.checkpoint_path != None:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
        else:
            checkpoint = None
            self.start_epoch = 0
        
        self.num_epochs = config.training.num_epochs
        
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.training.optim.lr, betas=config.training.optim.betas, weight_decay=config.training.optim.weight_decay)
        
        steps_per_epoch = len(self.train_loader)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.optim.warmup_epochs*steps_per_epoch,
            num_training_steps=self.num_epochs*steps_per_epoch
        )
            
        if checkpoint and not config.training.optim.override_cpt:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_log_interval = config.training.train_log.interval
        self.eval_log_interval = config.training.eval_log.interval
        
        self.train_log_exist = os.path.exists(config.training.train_log.file)
            
        self.train_log = open(config.training.train_log.file, 'a')
        
        self.eval_log_exist = os.path.exists(config.training.eval_log.file)
            
        self.eval_log = open(config.training.eval_log.file, 'a')
        
        if self.world_rank == 0:
            print(f"Setup done", flush=True)
    
    def train_print(self, txt):
        txt += '\n'
        self.train_log.write(txt)
        self.train_log.flush()
    
    def eval_print(self, txt):
        txt += '\n'
        self.eval_log.write(txt)
        self.eval_log.flush()
            
    def loss_per_epoch(self, loader, train=True):
        losses = 0
        rank_kl_p = torch.empty(0, device=self.local_rank)
        rank_kl_q = torch.empty(0, device=self.local_rank)

        for i, data in enumerate(loader):
            data = data.to(self.local_rank)
            loss, kl_p_, kl_q_ = self.model(data)
    
            losses += loss.item()
    
            rank_kl_p = torch.cat((rank_kl_p, kl_p_.unsqueeze(0)), dim=0)
            rank_kl_q = torch.cat((rank_kl_q, kl_q_.unsqueeze(0)), dim=0)
            
            if train:
                loss = loss + 0.0 * sum(p.sum() for p in self.model.parameters())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        kl_p = [torch.zeros_like(rank_kl_p) for _ in range(self.world_size)]
        kl_q = [torch.zeros_like(rank_kl_q) for _ in range(self.world_size)]
        dist.all_gather(kl_p, rank_kl_p)
        dist.all_gather(kl_q, rank_kl_q)
        kl_p = torch.cat(kl_p)
        kl_q = torch.cat(kl_q)

        epoch_loss = torch.tensor(losses/len(loader), device=self.local_rank)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss /= self.world_size
    
        return epoch_loss, kl_p, kl_q
     
    def train(self):
        if self.world_rank == 0:
            print(f'*** {len(self.train_loader)} batches per rank ***', flush=True)
            
            if not self.train_log_exist:
                self.train_print('Epoch \tTraining Loss \t   Time (s) \t   LR \t\t\t\t  KL Divergence')
                self.train_print('      \t              \t            \t \t \t \t  P   \t\t\t    Q')
            
            if not self.eval_log_exist:
                self.eval_print('Epoch \t   Eval Loss  \t\t\t\t\t  KL Divergence')
                self.eval_print('      \t              \t\t \t \t  P   \t\t\t    Q')
            
        for epoch in range(self.start_epoch, self.num_epochs):

            self.train_sampler.set_epoch(epoch)
            self.model.train()

            if self.world_rank == 0:
                torch.cuda.synchronize()
                start_time = time.time()
            
            epoch_loss, kl_p, kl_q = self.loss_per_epoch(self.train_loader)
            
            if self.world_rank == 0:
                to_save = {
                       'epoch': epoch,
                       'model_state_dict': self.model.module.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'scheduler_state_dict': self.scheduler.state_dict()
                   }
                
                if self.checkpoint_path != None:
                    if os.path.exists(self.checkpoint_path):
                        shutil.copy(self.checkpoint_path, f"{self.checkpoint_path}.bkp") # copy cpt file
                    torch.save(to_save, self.checkpoint_path)
    
                torch.cuda.synchronize()
                end_time = time.time()
    
                if epoch % self.train_log_interval == 0:
                    self.train_print('%.5i \t    %.5f \t    %.2f \t %.3e \t %.4e +/- %.4e\t    %.4e +/- %.4e\t' % (epoch, epoch_loss.item(), end_time - start_time, self.optimizer.param_groups[0]['lr'],\
                                                                                                                         kl_p.mean().item(), kl_p.std().item(),\
                                                                                                                         kl_q.mean().item(), kl_q.std().item()))
            
            if epoch % self.eval_log_interval == 0:
                self.model.eval()
                self.val_sampler.set_epoch(epoch)
                with torch.no_grad():
                    epoch_loss, kl_p, kl_q = self.loss_per_epoch(self.val_loader, train=False)
                
                if self.world_rank == 0:
                    self.eval_print('%.5i \t    %.5f \t   %.4e +/- %.4e\t    %.4e +/- %.4e' % (epoch, epoch_loss.item(), kl_p.mean().item(), kl_p.std().item(), kl_q.mean().item(), kl_q.std().item()))
              
            dist.barrier()
       
class DDPLinearRegression(DDPBase):

    def __init__(self, config, world_size, world_rank, local_rank, num_cpus_per_task):
        super().__init__(world_size, world_rank, local_rank, num_cpus_per_task)

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
