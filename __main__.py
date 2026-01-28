import os
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import numpy as np
import typer
app = typer.Typer()

import yaml
import json

import torch
import random
from torch_scatter import scatter

from protnhf.trainer import DDPTrainer
from protnhf.config import ConfigParams
from protnhf.flow import Flow
from protnhf import metrics
from protnhf.dataset import decode, Dataset, AA_TO_INDEX, DataLoader
from protnhf import bias
from protnhf import property

from tqdm import tqdm

Model = Annotated[Path,
                typer.Argument(help="NN Parameters for generation.")]

@app.command()
def train(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Train (or continue training of) a neural network
        flow model for generating structures.
    """
    hndl = DDPTrainer(ConfigParams.fromFile(config), world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'),\
                 local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
    hndl.train()

def net_charge(seq):
    charge_table_dict = {'A': 0,'C': 0,'D': -1,'E': -1,'F': 0,'G': 0,'H': +0.1,'I': 0,'K': 1,'L': 0,'M': 0,'N':0,'P':0,'Q': 0,'R': 1,'S': 0,'T': 0,'V': 0,'W': 0,'Y': 0}

    charge = 0
    for i in seq:
        charge += charge_table_dict[i]

    return charge

@app.command()
def biastest(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Use a trained model to generate structures.
    """
    config = ConfigParams.fromFile(config)
    model = Flow(config.model.n_types, config.model.hidden_dims, config.model.dt, config.model.niter, config.model.std, config.model.integrator,\
                         config.model.energy.d_model, config.model.energy.ff_dim, config.model.energy.n_heads, config.model.energy.n_layers)
                         
    if os.path.exists(config.model.checkpoint) and config.model.checkpoint != None:
        checkpoint = torch.load(config.model.checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loading from epoch {checkpoint['epoch']}")
         
    dataset = Dataset(config.training.dataset.file)
    dataT = dataset[10]#random.randint(0, len(dataset))
    esm2hndl = metrics.ESM2Handle()
    
    seq = decode(dataT.h)
    print("From dataset:")
    print(seq)
    print(f"Shannon Entropy: {metrics.shannon_entropy(seq)}")
    print(f"Low Sequence Complexity Percentage: {metrics.seg_low_complexity(seq)}")
    print(f"charge: {net_charge(seq)}")
    #print(f"ESM-2 PPPL: {esm2hndl.get_pppl(seq)}")
    
    p0, q0, qT = model.forward(dataT, train=False)
    batch = dataT.batch
    #num = 20
    #shape = (num, model.n_types)
    #p0 = model.prior.sample(sample_shape=shape)
    #q0 = model.prior.sample(sample_shape=shape)
    #batch = torch.zeros(num, dtype=torch.int64)
    
    w = torch.load('net_charge/net_charge.pt')
    get_bias = bias.HarmonicBias(qT, w)
    logits = model.embedd.reverse(model.reverse(p0, q0, batch, get_bias)[1])
    seq = decode(logits)
    esm2hndl = metrics.ESM2Handle()
    
    print("Sampling:")
    print(seq)
    print(f"Shannon Entropy: {metrics.shannon_entropy(seq)}")
    print(f"Low Sequence Complexity Percentage: {metrics.seg_low_complexity(seq)}")
    #print(f"ESM-2 PPPL: {esm2hndl.get_pppl(seq)}")
    print(f"charge: {net_charge(seq)}")
    

@app.command()
def test(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")], num: Annotated[int, typer.Argument(help="Sequence Length.")]):
    """ Test a trained model.
    """
    
    config = ConfigParams.fromFile(config)
    model = Flow(config.model.n_types, config.model.hidden_dims, config.model.dt, config.model.niter, config.model.std, config.model.integrator,\
                         config.model.energy.d_model, config.model.energy.ff_dim, config.model.energy.n_heads, config.model.energy.n_layers)
    esm2hndl = metrics.ESM2Handle()

    while(True):
        if os.path.exists(config.model.checkpoint) and config.model.checkpoint != None:
            checkpoint = torch.load(config.model.checkpoint, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loading from epoch {checkpoint['epoch']}")

        print(f"Max difference of q: {metrics.reversibilty(model)}")
        
        logits = model.sample(num)

        seq = decode(logits)
        print("Sampling:")
        print(seq)
        print(f"Shannon Entropy: {metrics.shannon_entropy(seq)}")
        print(f"Low Sequence Complexity Percentage: {metrics.seg_low_complexity(seq)}")
        print(f"ESM-2 PPPL: {esm2hndl.get_pppl(seq)}")
        print("=================================")
        
@app.command()
def linreg(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Use a trained model to generate structures.
    """
    hndl = property.DDPLinearRegression(ConfigParams.fromFile(config), world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'),\
                 local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
    hndl.train()
                    
app()