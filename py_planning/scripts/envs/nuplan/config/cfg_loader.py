import os
import hydra

def get_train_cfg(
  param_save_dir,
  param_exp_name,
):
  """
  Fill & return a training configuref
  :param param_save_dir: directory to store the cache and experiment artifacts
  :param param_exp_name: name of the experiment
  :return configuration for trainning.
  """

  # Location of path with all training configs
  CONFIG_PATH = './training' # relative to yaml files
  CONFIG_NAME = 'default_training' # use default_training.yaml

  # Initialize configuration management system
  hydra.core.global_hydra.GlobalHydra.instance().clear()
  hydra.initialize(config_path=CONFIG_PATH)

  # Compose the configuration
  cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
      f'group={str(param_save_dir)}',
      f'cache_dir={str(param_save_dir)}/cache',
      f'experiment_name={param_exp_name}',
      'py_func=train',
      '+training=training_raster_model',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
      'scenario_builder=nuplan_mini',  # use nuplan mini database
      'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=500',  # Choose 500 scenarios to train with
      'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.01',  # subsample scenarios from 20Hz to 0.2Hz
      'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
      'lightning.trainer.params.max_epochs=10',
      'data_loader.params.batch_size=8',
      'data_loader.params.num_workers=8',
  ])

  return cfg

def example_simu_cfg(
    param_save_dir,
    param_exp_name,
    param_planner='simple_planner',
    param_challenge='challenge_1_open_loop_boxes',
    param_dataset=None,
):
  """
  Fill & return a simulation configure
  :param param_save_dir: directory to store the cache and experiment artifacts
  :param param_exp_name: name of the experiment
  :param param_planner: [simple_planner, ml_planner]
  :param param_challenge: [challenge_1_open_loop_boxes, challenge_3_closed_loop_nonreactive_agents, challenge_4_closed_loop_reactive_agents]
  :param param_dataset: parameters for dataset, details are shown in the function
  :return configuration for simulation.
  """

  # Location of path with all simulation configs
  CONFIG_PATH = './simulation'
  CONFIG_NAME = 'default_simulation' # use default_simulation.yaml

  # Fill default values
  if (param_dataset == None):
    param_dataset=[
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_builder/nuplan/scenario_filter=all_scenarios',  # initially select all scenarios in the database
    'scenario_builder.nuplan.scenario_filter.scenario_types=[nearby_dense_vehicle_traffic, ego_at_pudo, ego_starts_unprotected_cross_turn, ego_high_curvature]',  # select scenario types
    'scenario_builder.nuplan.scenario_filter.limit_scenarios_per_type=10',  # use 10 scenarios per scenario type
    'scenario_builder.nuplan.scenario_filter.subsample_ratio=0.05',  # subsample 20s scenario from 20Hz to 1Hz
    ]

  # Initialize configuration management system
  hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
  hydra.initialize(config_path=CONFIG_PATH)

  # Compose the configuration
  cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
      f'experiment_name={param_exp_name}',
      f'group={param_save_dir}',
      f'planner={param_planner}',
      f'+simulation={param_challenge}',
      *param_dataset,
  ])

  return cfg
