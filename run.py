import os
import shutil
from datetime import timedelta
import time
import sys
import numpy as np
import torch
from dataset import Dataset, DataLoader
from flow import Flow

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
                
class Runner:

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
            eprint(f"Running DDP\nInitialized? {dist.is_initialized()}", flush=True)
        
        full_dataset = Dataset()
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
        self.train_loader = DataLoader(train_dataset, batch_size=5, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.train_sampler, drop_last=False)
        
        self.test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.world_rank, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=5, num_workers=self.num_cpus_per_task, pin_memory=True, shuffle=False, sampler=self.test_sampler, drop_last=False)
        
        self.model = Flow().to(self.local_rank)
        self.checkpoint_path = 'model.cpt'
            
        if os.path.exists(self.checkpoint_path) and self.checkpoint_path != None:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch']+1
        else:
            checkpoint = None
            self.start_epoch = 0
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-7, total_iters=500)
        if checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.num_epochs = self.start_epoch+50000
        self.log_interval = 1
        self.accum_iter = 100
        self.eval_interval = 1
        if self.world_rank == 0:
            eprint(f"Setup done", flush=True)
     
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

                # weights update
                if ((i + 1) % self.accum_iter == 0) or (i + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
        self.scheduler.step()
        kl_p = [torch.zeros_like(rank_kl_p) for _ in range(self.world_size)]
        kl_q = [torch.zeros_like(rank_kl_q) for _ in range(self.world_size)]
        dist.all_gather(kl_p, rank_kl_p)
        dist.all_gather(kl_q, rank_kl_q)
        kl_p = torch.cat(kl_p)
        kl_q = torch.cat(kl_q)

        epoch_loss = torch.tensor(losses/len(self.train_loader), device=self.local_rank)
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss /= self.world_size
    
        return epoch_loss, kl_p, kl_q
     
    def train(self):
        if self.world_rank == 0:
            eprint(f'*** {len(self.train_loader)} batches per rank ***', flush=True)
            print('Epoch \tTraining Loss \t   Time (s) \t   LR \t\t\t\t  KL Divergence', flush=True)
            print('      \t              \t            \t \t \t \t  P   \t\t\t    Q', flush=True)
            
            eprint('Epoch \t   Eval Loss  \t\t\t\t\t  KL Divergence', flush=True)
            eprint('      \t              \t\t \t \t  P   \t\t\t    Q', flush=True)
            
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
    
                if epoch % self.log_interval == 0:
                    print('%.5i \t    %.5f \t    %.2f \t %.3e \t %.4e +/- %.4e\t    %.4e +/- %.4e\t' % (epoch, epoch_loss.item(), end_time - start_time, self.optimizer.param_groups[0]['lr'],\
                                                                                                                         kl_p.mean().item(), kl_p.std().item(),\
                                                                                                                         kl_q.mean().item(), kl_q.std().item()), flush=True)
            
            if epoch % self.eval_interval == 0:
                self.model.eval()
                self.test_sampler.set_epoch(epoch)
                with torch.no_grad():
                    epoch_loss, kl_p, kl_q = self.loss_per_epoch(self.test_loader, train=False)
                
                if self.world_rank == 0:
                    eprint('%.5i \t    %.5f \t   %.4e +/- %.4e\t    %.4e +/- %.4e\t' % (epoch, epoch_loss.item(), kl_p.mean().item(), kl_p.std().item(), kl_q.mean().item(), kl_q.std().item()), flush=True)
              
            dist.barrier()
       
    def __del__(self):
        dist.destroy_process_group()
        
hndl = Runner(world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'), local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
hndl.train()
