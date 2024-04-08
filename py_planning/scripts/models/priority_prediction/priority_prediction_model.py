import os
import torch
import numpy as npfrom typing import List, Dict
from train_eval.priority_prediction.model import PriorityPrediction

class PriorityPredictionNetworkModel():
  def __init__(self, device:str, cfg: Dict, model_path: str):
    '''
    :param model_path: path to read the trained network model
    '''
    self.netmodel = PriorityPrediction(args=cfg['network'])
    self.netmodel = self.netmodel.float().to(device)

    checkpoint = torch.load(model_path)
    self.netmodel.load_state_dict(checkpoint['model_state_dict'])
    self.netmodel.double()
    self.netmodel.eval()

  def check_result_is_legal(self, inputs: Dict, ground_truth: Dict) -> float:
    '''
    Return 1.0 when result is corrrect, else 0.0
    '''
    outputs = self.netmodel.forward(inputs)
    outputs = outputs.detach().numpy()

    ground_truth = ground_truth.detach().numpy()

    return (np.argmax(outputs) == np.argmax(ground_truth)) * 1.0

  def predict(self, inputs: Dict) -> torch.Tensor:
    '''
    Return prediction results
    '''

    return self.netmodel.forward(inputs)
    
