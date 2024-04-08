import os
import random
import numpy as np
import torch

def set_random_seeds(seed):
  os.environ["PL_GLOBAL_SEED"] = str(seed)

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  
  torch.backends.cudnn.deterministic = True
