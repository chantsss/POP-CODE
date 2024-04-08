import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ipm.priority_model import IPMPriorityModel

get_model = IPMPriorityModel()
sta = get_model.load_model_from_file()

assert sta == True, "Load the network model fail"

import numpy as np
import torch
get_priority = get_model.forward_mlp(
    torch.from_numpy(np.array([[1.66480677, -27.05429143]])).float()).detach().numpy()
print(get_priority)
