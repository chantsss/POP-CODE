import os

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
## BASIC FUNCTIONS
def get_root2folder(root_dir: str, folder_name: str) -> str:
  folder_dir = os.path.join(root_dir, folder_name)
  flag = os.path.exists(folder_dir)
  if not flag:
    os.makedirs(folder_dir) # create one empty folder

  return folder_dir

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
## ENVIRONMENTS
# COMMON
import envs.module_path
ENVS_ROOT = os.path.dirname(envs.module_path.__file__)

EXP_ROOT = os.getenv('EXP_ROOT')
if EXP_ROOT == None:
  raise ValueError("EXP_ROOT is nan, export EXP_ROOT before run the python code.")
COMMON_EXP_ROOT = get_root2folder(EXP_ROOT, 'common')

AUTHOR_NAME="Abing"
AUTHOR_AFFILIATION="The Hong Kong University of Science and Technology" 

# NUPLAN
NUPLAN_DATA_ROOT=os.getenv('NUPLAN_DATA_ROOT')
NUPLAN_EXP_ROOT=get_root2folder(EXP_ROOT, 'nuplan')

# INTERACTION DATASET
INTERACTION_DATA_PLAN_ROOT=os.getenv('INTERACTION_DATA_PLAN_ROOT')
INTERACTION_DATA_PREDICTION_ROOT=os.getenv('INTERACTION_DATA_PREDICTION_ROOT')
INTERACTION_EXP_ROOT=get_root2folder(EXP_ROOT, 'interaction_dataset')

# L5KIT
L5KIT_DATA_ROOT=os.getenv('L5KIT_DATA_ROOT')
L5KIT_EXP_ROOT=get_root2folder(EXP_ROOT, 'l5kit')

if not L5KIT_DATA_ROOT is None:
  os.environ["L5KIT_DATA_FOLDER"] = L5KIT_DATA_ROOT

# COMMONROAD: In interactive scenarios, other traffic participants react to the behavior of the ego vehicle. 
#             This is achieved by coupling CommonRoad with the traffic simulator SUMO. 
#             In our scenario database, we denote such scenarios by the suffix I in the scenario ID in 
#             contrast to scenarios with a fixed trajectory prediction T. 
#             To run these scenarios, please use the simulation scripts provided below under 
#             Interactive scenarios, which are based on the CommonRoad-SUMO interface.
#   scenario_T like DEU_A9-2_1_T-1 is scenario with true values
#   scenario_I like ZAM_Tjunction-1_65_I-1-1 is scenario supporting interactive simulation
#   scenario_S like ZAM_Urban-6_1_S-1 is scenario supporting set-based prediction
#   @note: we use hand-crafted/ folder in default.
COMMONROAD_DATA_ROOT=os.getenv('COMMONROAD_DATA_ROOT')
COMMONROAD_EXP_ROOT=get_root2folder(EXP_ROOT, 'commonroad')

import envs.commonroad.example_scenarios.module_path
COMMONROAD_EXAMPLE_SCENARIOS_PATH=os.path.dirname(envs.commonroad.example_scenarios.module_path.__file__)

# NUSCNES
NUSCENES_DATA_ROOT=os.getenv('NUSCENES_DATA_ROOT')
NUSCENES_EXP_ROOT=get_root2folder(EXP_ROOT, 'nuScenes')

# FUNCTIONS
def get_dataset_exp_folder(dataset_name: str, folder_name: str):
  if dataset_name == 'nuplan':
    return get_root2folder(NUPLAN_EXP_ROOT, folder_name)
  elif dataset_name == 'interaction_dataset':
    return get_root2folder(INTERACTION_EXP_ROOT, folder_name)
  elif dataset_name == 'l5kit':
    return get_root2folder(L5KIT_EXP_ROOT, folder_name)
  elif dataset_name == 'commonroad':
    return get_root2folder(COMMONROAD_EXP_ROOT, folder_name)
  elif dataset_name == 'nuscenes':
    return get_root2folder(NUSCENES_EXP_ROOT, folder_name)
  else:
    raise ValueError("unsupported dataset name")
