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
            checkpoint = torch.load(self.checkpoint, weights_only=False)
            model.load_state_dict(self.checkpoint['model_state_dict'])
        return model

class DatasetParams(BaseModel):
    file: str
    batch_size: int
    split: Optional[float] = 0.8

class LoggingParams(BaseModel):
    interval: Optional[int] = 1
    file: str
   
class LRParams(BaseModel):
    start: float
    scheduler: Optional[Dict] = None
                             
class TrainingParams(BaseModel):
    dataset: DatasetParams
    num_epochs: Optional[int] = 5000
    accum_iter: Optional[int] = 0
    train_log: LoggingParams
    eval_log: LoggingParams
    lr: LRParams
               
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

