import os
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
