import torch
import time

import thirdparty.configfrom l5kit.configs import load_config_datafrom l5kit.data import LocalDataManager, ChunkedDataset, filter_agents_by_framesfrom l5kit.dataset import EgoDatasetfrom l5kit.rasterization import build_rasterizer
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlanfrom l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               SimulatedDrivenMilesMetric, ReplayDrivenMilesMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)from l5kit.cle.composite_metrics import CompositeMetricAggregatorfrom l5kit.cle.validators import RangeValidator, ValidationCountingAggregatorfrom l5kit.simulation.dataset import SimulationConfigfrom l5kit.simulation.unroll import ClosedLoopSimulator

import osfrom collections import defaultdict
import numpy as np
import matplotlib.pyplot as pltfrom prettytable import PrettyTable

import envs.config
import envs.l5kit.func_metrics as func_metricsfrom envs.l5kit.simulation_unroll import ClosedLoopSimulator_Trafrom envs.l5kit.map_api import MapAPI

import bokeh.plottingfrom envs.l5kit.visualizer.zarr_utils import simulation_out_to_visualizer_scenefrom envs.l5kit.visualizer.visualizer import visualize

###############################################################################
## Simulations
def run_simulation(
  simu_dataset,
  ego_model_nn,
  ego_model_tra,
  simu_model,
  sim_cfg,
  max_simulation_step,
  scenes = [],
  scene_batch_num = 1,
  enable_eval = False,
  enable_rviz = False,
):
  # prepare metrics
  metrics = [DisplacementErrorL2Metric(),
             DistanceToRefTrajectoryMetric(),
             SimulatedDrivenMilesMetric(),
             ReplayDrivenMilesMetric(),
             CollisionFrontMetric(),
             CollisionRearMetric(),
             CollisionSideMetric()]
  validators = [RangeValidator("displacement_error_l2_validator", DisplacementErrorL2Metric, max_value=30),
                RangeValidator("distance_ref_trajectory_validator", DistanceToRefTrajectoryMetric, max_value=4),
                RangeValidator("collision_front_validator", CollisionFrontMetric, max_value=0),
                RangeValidator("collision_rear_validator", CollisionRearMetric, max_value=0),
                RangeValidator("collision_side_validator", CollisionSideMetric, max_value=0),
                ]
  intervention_validators = ["displacement_error_l2_validator",
                             "distance_ref_trajectory_validator",
                             "collision_front_validator",
                             "collision_rear_validator",
                             "collision_side_validator"]

  metric_names = [a.metric_name for a in metrics]
  cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                      validators=validators,
                                      composite_metrics=[],
                                      intervention_validators=intervention_validators))

  # prepare simulation
  sim_loop = ClosedLoopSimulator_Tra(sim_cfg, simu_dataset, device, 
                                     model_ego_nn=ego_model_nn,
                                     model_ego_tra=ego_model_tra,
                                     model_agents=simu_model)

  # begin simulations
  scene_num = len(scenes)
  scene_bid = 0
  scenes = np.array(scenes)
  print(">> Begin traverse scenes")
  while scene_bid < scene_num:
    # Extract scenes_to_unroll
    scene_eid = min(scene_bid + scene_batch_num, scene_num)
    scenes_to_unroll = list(scenes[
      np.arange(scene_bid, scene_eid, 1)]
    )

    # Simulate scenes_to_unroll
    print("Simulation scenes id from={}/to={}/toal_num={}.".format(scene_bid, scene_eid-1, scene_num))
    unroll_result = sim_loop.unroll(scenes_to_unroll, max_simulation_step)
    scenes_nav_map = unroll_result[0]
    sim_outs = unroll_result[1]

    # Evaluations
    if enable_eval:
      cle_evaluator.evaluate(sim_outs)
      validation_results = cle_evaluator.validation_results()
      metric_results = cle_evaluator.metric_results()

      metric_agg = func_metrics.aggregate_metrics(metric_names, metric_results)[0]
      val_agg = ValidationCountingAggregator().aggregate(validation_results)
      cle_evaluator.reset() # reset the evaluation results

      # TODO: values sum up of metric_agg, val_agg
      #
      # metric_agg: dict(): metric_name > float
      # val_agg: dict(): val_name > torch.tensor
      #
      # print("metric_agg[{}]={}".format(len(metric_agg), metric_agg))
      # print("val_agg[{}]={}".format(len(val_agg), val_agg))

    # Debug & log
    scene_bid += scene_batch_num

    if enable_rviz:
      # for sim_out in sim_outs: # for each scene
      assert len(sim_outs) == 1, 'visualization require scenes being simulated one by one'
      # plot bokeh.html
      file_root = envs.config.get_dataset_exp_folder('l5kit', 'visuals')

      file_relative_path = os.path.join(file_root, "planning_scene_{}.html".format(scenes_to_unroll[0]))

      bokeh.plotting.output_file(file_relative_path) 
      sim_out = sim_outs[0]
      vis_in = simulation_out_to_visualizer_scene(scenes_nav_map, sim_out, mapAPI)
      # bokeh.plotting.show() can open a website automatically
      bokeh.plotting.save(visualize(sim_out.scene_id, vis_in))

      # plot graph
      graph_file = os.path.join(file_root, "graph")
      scenes_nav_map.plot_and_save2figure(graph_file)

      # time.sleep(0.1) # each 

###############################################################################
if __name__ == '__main__':
  # Get config & Check
  dm = LocalDataManager(None)

  cfg = load_config_data(os.path.join(envs.config.ENVS_ROOT, 'l5kit/config/planning.yaml'))
  rasterizer = build_rasterizer(cfg, dm)
  mapAPI = MapAPI.from_cfg(dm ,cfg)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Torch device={}, version={}".format(device, torch.__version__))

  # Simulation model for other agents in traffic, 
  simulation_model = torch.jit.load(os.path.join(envs.config.L5KIT_EXP_ROOT, 
                                    'pretrained_models/simulation_model.pt')).to(device)
  simulation_model = simulation_model.eval()

  # Design your ego_model_tra
  ego_model_nn = torch.jit.load(os.path.join(envs.config.L5KIT_EXP_ROOT, 
                                'pretrained_models/planning_model.pt')).to(device)
  ego_model_nn = ego_model_nn.eval()

  ego_model_tra = None # TODO: not implemented

  # Set configs
  torch.set_grad_enabled(False)

  ## Datas
  # Prepare train data:
  #   rendering the datset using history_num_frames, future_num_frames, and so on
  dtset_cfg = cfg["data_loader"]
  dtset_zarr = ChunkedDataset(dm.require(dtset_cfg["key"])).open()
  dtset = EgoDataset(cfg, dtset_zarr, rasterizer)
  print("Extract train scene num={};".format(len(dtset.dataset.scenes)))

  # Paramters
  run_cfg = cfg["run_params"]
  sr = run_cfg["scene_range"]

  simu_dataset = dtset
  scenes = np.arange(sr[0], sr[1], sr[2])
  scene_batch_num = run_cfg["scene_batch_num"]
  max_simulation_step = run_cfg["max_simulation_step"]  
  goal_simulation_step = run_cfg["goal_simulation_step"]
  enable_eval = run_cfg["enable_eval"]
  enable_rviz = run_cfg["enable_rviz"]
  distance_th_far = run_cfg["distance_th_far"]
  distance_th_close = run_cfg["distance_th_close"]

  assert max_simulation_step <= goal_simulation_step, \
    "Error, max_simulation_step should <= goal_simulation_step"
  sim_cfg = SimulationConfig(use_ego_gt=False, 
                             use_agents_gt=False, 
                             disable_new_agents=False,
                             distance_th_far=distance_th_far, 
                             distance_th_close=distance_th_close, 
                             start_frame_index=0,
                             num_simulation_steps=goal_simulation_step,
                             show_info=True)

  # Run simulation
  if enable_rviz:
    scene_batch_num = 1 # for rviz convenience

  run_simulation(simu_dataset=simu_dataset, 
                 ego_model_nn=ego_model_nn, 
                 ego_model_tra=ego_model_tra,
                 simu_model=simulation_model,
                 sim_cfg=sim_cfg,
                 max_simulation_step=max_simulation_step,
                 scenes=scenes, scene_batch_num=scene_batch_num,
                 enable_eval=enable_eval,
                 enable_rviz=enable_rviz)
