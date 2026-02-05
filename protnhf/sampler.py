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
        
        self.bias_params = config.sample.bias
        
    
    def sample(self):
        bias_params = None if self.bias_params is None else self.bias_params.model_dump()
        
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
        
        def get_xO():
            if '+' in bias_params['residue']:
                xO_list = []
                for res in  bias_params['residue'].split('+'):
                    aa = torch.tensor([AA_TO_INDEX[res]])
                    xO_list.append(F.one_hot(aa, num_classes=20).to(torch.float))
                xO = torch.cat(xO_list).sum(dim=0)[None, :]
            else:
                aa = torch.tensor([AA_TO_INDEX[bias_params['residue']]])
                xO = F.one_hot(aa, num_classes=20).to(torch.float)
            
            return xO
        
        def get_seq(p, q, batch, bias=None):
            with torch.no_grad():
                logits_combined = self.model.embedd.reverse(self.model.reverse(p, q, batch, bias)[1])
                num_seq = batch.max().item() + 1
                logits_set = [
                    logits_combined[batch == i]
                    for i in range(num_seq)
                ]
                seq = [decode(logits) for logits in logits_set]

            return seq
        
        if bias_params is None: return get_seq(p, q, batch)
        
        if bias_params['type'] == 'coulomb':
            bias = CoulombBias(bias_params['k'], get_xO())
        elif bias_params['type'] == 'gaussian':                
            bias = GaussianBias(bias_params['k'], bias_params['sigma'], get_xO())
            
        return get_seq(p, q, batch, bias)