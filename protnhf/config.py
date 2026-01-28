from typing import Dict, Optional, Tuple
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

class DatasetParams(BaseModel):
    file: str
    batch_size: int
    split: Optional[float] = 0.8
    limit: Optional[int] = -1

class LoggingParams(BaseModel):
    interval: Optional[int] = 1
    file: str
   
class OptimParams(BaseModel):
    override_cpt: Optional[bool] = False
    lr: float
    betas: Tuple
    weight_decay: float
    warmup_epochs: int
                             
class TrainingParams(BaseModel):
    dataset: DatasetParams
    num_epochs: Optional[int] = 500
    train_log: LoggingParams
    eval_log: LoggingParams
    optim: OptimParams

class BiasParams(BaseModel):
    type: str
    k: Optional[float] = None
    x0: Optional[str] = None
    eps: Optional[float] = None

class SampleParams(BaseModel):
    mode: str
    bias: Optional[BiasParams] = None

class LinearRegression(BaseModel):
    property: str
    batch_size: int
    num_epochs: int
    lr: float
               
class ConfigParams(BaseModel):
    model: FlowParams
    training:  Optional[TrainingParams] = None
    sample: Optional[SampleParams] = None
    linreg: Optional[LinearRegression] = None
    
    @staticmethod
    def fromFile(input):
        with open(input, "r", encoding="utf-8") as f:
            if input.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    
        return ConfigParams(**data)

