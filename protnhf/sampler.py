import os
import json
import torch
import torch.nn.functional as F

from .config import ConfigParams
from .flow import Flow
from . import metrics
from .bias import *
from .dataset import decode, AA_TO_INDEX, Data

class Sampler:
    def __init__(self, config):
        config = ConfigParams.fromFile(config)
        if config.model.from_huggingface:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            
            repo_id = 'bharathraghavan/ProtNHF'
            
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json"
            )

            # Download weights
            weights_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors"
            )

            # Load architecture config
            with open(config_path) as f:
                arch_config = json.load(f)

            self.model = Flow(arch_config['n_types'], arch_config['hidden_dims'], arch_config['dt'], arch_config['niter'], arch_config['std'], arch_config['integrator'],\
                                arch_config['energy']['d_model'], arch_config['energy']['ff_dim'], arch_config['energy']['n_heads'], arch_config['energy']['n_layers'])

            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)
                    
        else:
            self.model = Flow(config.model.n_types, config.model.hidden_dims, config.model.dt, config.model.niter, config.model.std, config.model.integrator,\
                         config.model.energy.d_model, config.model.energy.ff_dim, config.model.energy.n_heads, config.model.energy.n_layers)
            if os.path.exists(config.model.checkpoint) and config.model.checkpoint != None:
                checkpoint = torch.load(config.model.checkpoint, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loading from epoch {checkpoint['epoch']}")
        
        self.nums = config.sample.lengths
        
        if self.nums is None:
            self.nums = [config.sample.length]*config.sample.num
        
        if config.sample.bias is None:
            self.bias = None
        else:
            self.bias = []
            for bias_params in config.sample.bias:
                params = bias_params.model_dump()
                            
                if params['target'] is None:
                    aa = torch.tensor([AA_TO_INDEX[params['residue']]])
                    xO = F.one_hot(aa, num_classes=20).to(torch.float)
                
                if params['type'] == 'coulomb':
                    bias = CoulombBias(params['k'], xO)
                elif params['type'] == 'gaussian':                
                    bias = GaussianBias(params['k'], params['sigma'], xO)
                elif params['type'] == 'positionrestraint':                
                    bias = PositionRestraint(params['k'], params['i']-1, xO[0])
                elif params['type'] == 'netchargerestraint':                
                    bias = NetChargeRestraint(params['k'], params['target'])
                    
                self.bias.append(bias)
            
            self.bias = CombinedBias(self.bias)
    
    def sample(self):
        p_list = []
        q_list = []
        batch_list = []
        
        for num in self.nums:
            shape = (num, self.model.n_types)
            p_list.append(self.model.prior.sample(sample_shape=shape))
            q_list.append(self.model.prior.sample(sample_shape=shape))
            batch_list.append(torch.zeros(num, dtype=torch.int64))
        
        p = torch.cat(p_list)
        q = torch.cat(q_list) 
        batch = torch.cat([batch+i for i, batch in enumerate(batch_list)]) 
        
        with torch.no_grad():
            q_T = self.model.reverse(p, q, batch, self.bias)[1]
            logits_combined = self.model.embedd.reverse(q_T)
            self.softmax_net_charge = net_charge(q_T, batch)
            num_seq = batch.max().item() + 1
            logits_set = [
                logits_combined[batch == i]
                for i in range(num_seq)
            ]
            seq = [decode(logits) for logits in logits_set]

        return seq
