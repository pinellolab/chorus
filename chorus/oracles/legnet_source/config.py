import json
import sys
import torch 
import torch.nn as nn

from .model import LegNet
from dataclasses import dataclass, asdict
from pathlib import Path
from dataclasses import InitVar
import operator 
from functools import reduce



@dataclass
class TrainingConfig: 
    stem_ch: int
    stem_ks: int
    ef_ks: int
    ef_block_sizes: list[int]
    resize_factor: int
    pool_sizes: list[int]
    reverse_augment: bool
    use_reverse_channel: bool
    use_shift: bool
    max_shift: tuple[int, int] | None
    max_lr: float
    weight_decay: float
    model_dir: str 
    data_path: str
    epoch_num: int 
    device: int  
    seed: int
    train_batch_size: int
    valid_batch_size: int
    num_workers: int
    training: InitVar[bool] 
    
    def __post_init__(self, training: bool):
        self.check_params()
        model_dir = Path(self.model_dir)
        if training:
            model_dir.mkdir(exist_ok=True,
                            parents=True)
            self.dump()
        
    
    def check_params(self): 
        if Path(self.model_dir).exists():
            print(f"Warning: model dir already exists: {self.model_dir}", file=sys.stderr)
        if not self.reverse_augment:
            if self.use_reverse_channel:
                raise Exception("If model use reverse channel"
                                "reverse augmentation must be performed")
        
           
    def dump(self, path: str | Path | None = None):
        if path is None:
            path = Path(self.model_dir) / "config.json"
        self.to_json(path)
        
    def to_dict(self) -> dict:
        dt = asdict(self)
        return dt
    
    def to_json(self, path: str | Path):
        dt = self.to_dict()
        with open(path, 'w') as out:
            json.dump(dt, out, indent=4)
  
    @classmethod
    def from_dict(cls, dt: dict) -> 'TrainingConfig':
        return cls(**dt)
    
          
    @classmethod
    def from_json(cls, path: Path | str, training: bool = False) -> 'TrainingConfig':
        with open(path, 'r') as inp:
            dt = json.load(inp)
        dt['training'] = training
        return cls.from_dict(dt)
  
    @property
    def in_ch(self) -> int:
       return 4 + self.use_reverse_channel
  
    def get_model(self) -> nn.Module:
       return LegNet(in_ch=self.in_ch,
                   stem_ch=self.stem_ch,
                   stem_ks=self.stem_ks,
                   ef_ks=self.ef_ks,
                   ef_block_sizes=self.ef_block_sizes,  
                   resize_factor=self.resize_factor,
                   pool_sizes=self.pool_sizes)

    def get_min_seq_size(self) -> int:
        """
        return minimal size of sequence (without any padding) that can passed to neural network without a error 
        """
        return reduce(operator.mul, self.pool_sizes, 1)

        

    