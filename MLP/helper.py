# helper functions and can be reused another project

import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
