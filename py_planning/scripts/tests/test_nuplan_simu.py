import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import DictConfig, OmegaConf

import argparsefrom pathlib import Path
import hydra
from envs.config import NUPLAN_EXP_ROOT
import envs.nuplan.config.cfg_loader as cfg_loader 
from nuplan.planning.script.run_simulation import main as main_simulationfrom nuplan.planning.script.run_training import main as main_trainfrom nuplan.planning.script.run_nuboard import main as main_nuboard


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--step", type=int, help="which step to run", nargs="?")
  args = parser.parse_args()

  SAVE_DIR = NUPLAN_EXP_ROOT
  CONFIG_PATH = './envs/nuplan/config/simulation'
  PARAM_CHALLENGE='challenge_1_open_loop_boxes' # param challenge_4_closed_loop_reactive_agents, ...
  PARAM_DATASET=[
      'scenario_builder=nuplan_mini',  # use nuplan mini database
      'scenario_builder/nuplan/scenario_filter=all_scenarios',  # initially select all scenarios in the database
      'scenario_builder.nuplan.scenario_filter.scenario_types=[nearby_dense_vehicle_traffic, ego_at_pudo, ego_starts_unprotected_cross_turn, ego_high_curvature]',  # select scenario types
      'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=10',  # use 10 scenarios per scenario type
      'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',  # subsample 20s scenario from 20Hz to 1Hz
    ]

  RUN_STEP = args.step
  # trainning
  if RUN_STEP == 1:
    # Read configuration
    expriment_file = 'training_raster_experiment'
    cfg = cfg_loader.get_train_cfg(param_save_dir=SAVE_DIR, 
                                   param_exp_name=expriment_file)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(OmegaConf.to_yaml(cfg))

    # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
    main_train(cfg)

  # simulation a simple_planner
  if RUN_STEP == 2:
    # Read configuration
    expriment_file = 'simulation_simple_experiment'
    param_planner = 'simple_planner'

    # Location of path with all simulation configs
    CONFIG_NAME = 'default_simulation'
    
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'experiment_name={expriment_file}',
        f'group={SAVE_DIR}',
        f'planner={param_planner}', # use simple_planner.yaml in simulation/planner/
        f'+simulation={PARAM_CHALLENGE}', # use challenge_1_open_loop_boxes.yaml in experiments/simulation 
        *PARAM_DATASET,
    ])
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(OmegaConf.to_yaml(cfg))

    # Run the simulation loop (real-time visualization not yet supported, see next section for visualization)
    main_simulation(cfg)

    # Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
    parent_dir = Path(SAVE_DIR) / expriment_file
    results_dir = list(parent_dir.iterdir())[0]  # get the child dir
    nuboard_file_1 = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]

  # simulation a mpl_planner for comparison
  if RUN_STEP == 3:
    # Read configuration
    training_log_dir = os.path.join(SAVE_DIR, 'training_raster_experiment')
    expriment_file = 'simulation_raster_experiment'

    # Location of path with all simulation configs
    CONFIG_NAME = 'default_simulation'

    # Get the checkpoint of the trained model
    # last_experiment = sorted(os.listdir(training_log_dir))[-1]
    train_experiment_dir = sorted(Path(training_log_dir).iterdir())[-1]
    checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]
    model_path = str(checkpoint).replace("=", "\=")
    # print("train_experiment_dir=", train_experiment_dir)
    # print("checkpoint=", checkpoint)
    # print("model_path=", model_path)

    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'experiment_name={expriment_file}',
        f'group={SAVE_DIR}',
        'planner=ml_planner',
        'model=raster_model',
        'planner.model_config=${model}',  # hydra notation to select model config
        f'planner.checkpoint_path={model_path}',  # this path can be replaced by the checkpoint of the model trained in the previous section
        f'+simulation={PARAM_CHALLENGE}',
        *PARAM_DATASET,
    ])
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print(OmegaConf.to_yaml(cfg))

    # Run the simulation loop
    main_simulation(cfg)

    # Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
    parent_dir = Path(SAVE_DIR) / expriment_file
    results_dir = list(parent_dir.iterdir())[0]  # get the child dir
    nuboard_file_2 = [str(file) for file in results_dir.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]

  # compares and illustrate in the nuboard
  if RUN_STEP == 4:
    CONFIG_PATH = './envs/nuplan/config/nuboard'
    CONFIG_NAME = 'default_nuboard'
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    parent_dir1 = Path(SAVE_DIR + '/simulation_simple_experiment')
    results_dir1 = list(parent_dir1.iterdir())[0]  # get the child dir
    nuboard_file_1 = [str(file) for file in results_dir1.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]
    print("nuboard_file_1={}".format(nuboard_file_1))

    parent_dir2 = Path(SAVE_DIR + '/simulation_raster_experiment')
    results_dir2 = list(parent_dir2.iterdir())[0]  # get the child dir
    nuboard_file_2 = [str(file) for file in results_dir2.iterdir() if file.is_file() and file.suffix == '.nuboard'][0]
    print("nuboard_file_2={}".format(nuboard_file_2))

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
        f'simulation_path={[nuboard_file_1, nuboard_file_2]}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    ])

    main_nuboard(cfg)

