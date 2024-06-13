import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def to_var(x):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = x.to(device)
    return x



def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
