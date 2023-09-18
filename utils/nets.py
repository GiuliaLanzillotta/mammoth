"""Utils to support network structures ..."""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class DictionaryNet(nn.Module):

    def __init__(self, network, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleDict({n:m for (n,m) in network.named_children()})

    def module_forward(self, x, name:str): 
        return self.layers[name](x)

    def net_forward(self, x): 
        for n,m in self.layers.items(): 
            if n=='fc': x = torch.flatten(x, 1)
            x = m(x)
        return x  

    def forward(self, x, name=None):
         if name is None: 
              return self.net_forward(x)
         return self.module_forward(x, name)
    