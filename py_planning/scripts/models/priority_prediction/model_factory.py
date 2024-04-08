import os
import yamlfrom typing import Tuple, Dict
from models.priority_prediction.priority_prediction_model import PriorityPredictionNetworkModel

class PredictionModelFactory:
  @staticmethod
  def produce(device:str, model_path: str, check_point: str) -> Tuple[Dict, PriorityPredictionNetworkModel]:
    '''
    Produce a priority prediction model
    '''
    cfg = {}
    rmodel = None

    # Load config
    with open(os.path.join(model_path, 'config.yaml'), 'r') as yaml_file:
      cfg = yaml.safe_load(yaml_file)
      checkpoint_path = os.path.join(model_path, 'checkpoints/{}.tar'.format(check_point))

      rmodel = PriorityPredictionNetworkModel(device=device, cfg=cfg, model_path=checkpoint_path)

    return cfg, rmodel
