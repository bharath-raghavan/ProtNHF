import os
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Optional, List, Union, Dict
from typing_extensions import Annotated

import logging
_logger = logging.getLogger(__name__)
import asyncio

import numpy as np
import typer
app = typer.Typer()

import yaml
import json

import torch

from protnhf.trainer import DDPTrainer
from protnhf.config import ConfigParams
from protnhf.flow import Flow
from protnhf import metrics
from protnhf.dataset import decode

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

@app.command()
def sample(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
    """ Use a trained model to generate structures.
    """
    config = ConfigParams.fromFile(config)
    model = config.get()
    hndl.generate()

@app.command()
def test(config: Annotated[Path, typer.Argument(help="Training parameter yaml file.")]):
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
        
        logits = model.sample(20)

        new_seq = decode(logits)
        print("Sampling:")
        print(new_seq)
        print(esm2hndl.get_pppl(new_seq))
        print("=================================")
        
    
app()