from typing import Dict, Optional, List
from pydantic import BaseModel
import importlib, yaml, json
from .flow import Flow

class TransformerParams(BaseModel):    
    d_model: int
    ff_dim: int
    n_heads: int
    n_layers: int
                
class FlowParams(BaseModel):
    hidden_dims: int
    dt: float
    niter: int
    std: float
    integrator: str
    n_types: int
    checkpoint: Optional[str] = None
    energy: TransformerParams
    
    def get(self, read_cpt=True):
        model = Flow(self.n_types, self.hidden_dims, self.dt, self.niter, self.std, self.integrator,\
                         self.energy.d_model, self.energy.ff_dim, self.energy.n_heads, self.energy.n_layers)
        if not read_cpt: return model
        if os.path.exists(self.checkpoint) and self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint, weights _only=False)
            model.load_state_dict(self.checkpoint['model_state_dict'])
        return model
    
class TrainingParams(BaseModel):
    dataset: str
    batch_size: int
    train_val_split: List
    num_epochs: int
    lr: float
    scheduler_type: Optional[str] = None
    scheduler_params: Optional[Dict] = None
    train_log_interval: int
    accum_iter: int
    eval_log_interval: int

class ConfigParams(BaseModel):
    model: FlowParams
    training:  Optional[TrainingParams] = None
    
    @staticmethod
    def fromFile(input):
        with open(input, "r", encoding="utf-8") as f:
            if input.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    
        return ConfigParams(**data)

