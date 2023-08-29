import numpy as np
import torch


def gather(consts: torch.Tensor, t: torch.Tensor):
    
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)