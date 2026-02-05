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

from protnhf.trainer import DDPTrainer, DDPLinearRegression
from protnhf.config import ConfigParams
from protnhf.flow import Flow
from protnhf import metrics
from protnhf.dataset import decode, Dataset, AA_TO_INDEX, DataLoader
from protnhf import bias
from protnhf.sampler import Sampler

from tqdm import tqdm

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
def sample(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")], out: Annotated[Path, typer.Argument(help="Output file.")]):
    """ Test a trained model.
    """
    
    sampler = Sampler(config)
    seqs = sampler.sample()
    esm2hndl = metrics.ESM2Handle()
    
    with open(out, 'w') as f:
        for seq in seqs:
            f.write(f"{seq},{metrics.seg_low_complexity(seq)},{esm2hndl.get_pppl(seq)}\n")
        
@app.command()
def linreg(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Use a trained model to generate structures.
    """
    hndl = DDPLinearRegression(ConfigParams.fromFile(config), world_size=os.environ.get('SLURM_NTASKS'), world_rank=os.environ.get('SLURM_PROCID'),\
                 local_rank=os.environ.get('SLURM_LOCALID'), num_cpus_per_task=os.environ.get("SLURM_CPUS_PER_TASK"))
    hndl.train()
                    
app()