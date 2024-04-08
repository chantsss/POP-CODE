import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import thirdparty.configfrom l5kit.configs import load_config_datafrom l5kit.data import LocalDataManager, ChunkedDataset, filter_agents_by_framesfrom l5kit.dataset import EgoDatasetfrom l5kit.rasterization import build_rasterizer
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlanfrom l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               SimulatedDrivenMilesMetric, ReplayDrivenMilesMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)from l5kit.cle.composite_metrics import CompositeMetricAggregatorfrom l5kit.cle.validators import RangeValidator, ValidationCountingAggregatorfrom l5kit.simulation.dataset import SimulationConfigfrom l5kit.simulation.unroll import ClosedLoopSimulatorfrom collections import defaultdict
import numpy as np
import matplotlib.pyplot as pltfrom prettytable import PrettyTablefrom l5kit.data import MapAPI

import envs.config

import envs.l5kit.func_metrics as func_metrics

# set env variable for data
dm = LocalDataManager(None)
# get config
cfg = load_config_data(os.path.join(envs.config.ENVS_ROOT, 'l5kit/config/tutor_simulation.yaml'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("torch device={}, version={}".format(device, torch.__version__))

# Load models, where
#   simulation model for other agents in traffic, 
#   planning model for ego vehicle
simulation_model = torch.jit.load(os.path.join(envs.config.L5KIT_EXP_ROOT, 
                                  'pretrained_models/simulation_model.pt')).to(device)
simulation_model = simulation_model.eval()

ego_model = torch.jit.load(os.path.join(envs.config.L5KIT_EXP_ROOT, 
                           'pretrained_models/planning_model.pt')).to(device)
ego_model = ego_model.eval()

torch.set_grad_enabled(False)

# ===== INIT DATASET
eval_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
mapAPI = MapAPI.from_cfg(dm ,cfg)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
print("Data set scene total num={}".format(len(eval_dataset)))

# _mapAPI = dataset.rasterizer.sem_rast.mapAPI

# ===== Simulation
# Only agents in the initial frame are simulated
# Init parameters
scenes_to_unroll = [1598] # np.arange(0, 5000, 100)
num_simulation_step_example1 = 20
num_simulation_step_example2 = 200 # to scene end

# ===== Qualitive Evaluation
import bokeh.plottingfrom l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scenefrom l5kit.visualization.visualizer.visualizer import visualize

# Save as svg, then convert to pdf.
# bokeh.models.Plot.output_backend = "svg"
# bokeh.plotting.Figure.output_backend = "svg"
# >>> unwork, try to using the following commands:
#   using from bokeh.io import export_svgs
#   export_svgs(plot, filename="plot.svg")

# bokeh.plotting.output_notebook()
if False:
  '''
  : param use_xxx_gt: (ground-truth) Single mode future prediction with availabilities 
                      (either 1->available or 0->unavailable). Header fields have these meanings: 
                      timestamp, track_id, avail_time_0, avail_time_1, ..., coord_x_time_0, coord_y_time_0, ...
  : param disable_new_agents: disable simulating agents not occurs in initial frame
  '''
  sim_cfg = SimulationConfig(use_ego_gt=True, use_agents_gt=False, disable_new_agents=True,
                             distance_th_far=500, distance_th_close=50, 
                             num_simulation_steps=num_simulation_step_example1,
                             start_frame_index=0, 
                             show_info=True)

  sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, 
                                model_agents=simulation_model)

  """
  Simulate the dataset for the given scene indices
  :param scene_indices: the scene indices we want to simulate
  :return: the simulated dataset
  """
  sim_outs = sim_loop.unroll(scenes_to_unroll)

  bokeh.plotting.output_file("test_qeval.html") # only last scene will be stored at html
  for sim_out in sim_outs: # for each scene
      vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
      bokeh.plotting.show(visualize(sim_out.scene_id, vis_in))

if True:
  # ===== Quantitative Evaluation
  metrics = [DisplacementErrorL2Metric(),
             DistanceToRefTrajectoryMetric(),
             SimulatedDrivenMilesMetric(),
             ReplayDrivenMilesMetric(),
             CollisionFrontMetric(),
             CollisionRearMetric(),
             CollisionSideMetric()]
  metric_names = [a.metric_name for a in metrics]

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

  cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                      validators=validators,
                                      composite_metrics=[],
                                      intervention_validators=intervention_validators))

  # evaluating 1 > agg
  sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=False, disable_new_agents=False,
                            distance_th_far=30, distance_th_close=15, 
                            num_simulation_steps=num_simulation_step_example2,
                            start_frame_index=0, show_info=True)

  sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=ego_model, model_agents=simulation_model)

  sim_outs = sim_loop.unroll(scenes_to_unroll)

  cle_evaluator.evaluate(sim_outs)
  validation_results = cle_evaluator.validation_results()
  metric_results = cle_evaluator.metric_results()
  agg = ValidationCountingAggregator().aggregate(validation_results)
  agg2 = func_metrics.aggregate_metrics(metric_names, metric_results)[0]
  cle_evaluator.reset() # reset the evaluation results

  # evaluating 2 > agg2
  sim_cfg_log = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=False,
                                 distance_th_far=30, distance_th_close=15, 
                                 num_simulation_steps=num_simulation_step_example2,
                                 start_frame_index=0, show_info=True)

  sim_loop_log = ClosedLoopSimulator(sim_cfg_log, eval_dataset, device, model_ego=ego_model)
  sim_outs_log = sim_loop_log.unroll(scenes_to_unroll)

  cle_evaluator.evaluate(sim_outs_log)
  validation_results_log = cle_evaluator.validation_results()
  metric_results_log = cle_evaluator.metric_results()
  agg_log = ValidationCountingAggregator().aggregate(validation_results_log)
  agg2_log = func_metrics.aggregate_metrics(metric_names, metric_results_log)[0]
  cle_evaluator.reset()

  # comparisons >>>
  fields = ["metric", "log_replayed agents", "simulated agents"]

  # metrics
  print(">>> Metrics / Miles in percentage")
  table1 = PrettyTable(field_names=fields)
  for metric_name in agg2_log.keys():
    table1.add_row([metric_name, agg2_log[metric_name], agg2[metric_name]])
  print(table1)

  # validation
  print(">>> Validations")
  table2 = PrettyTable(field_names=fields)
  for metric_name in agg_log:
      table2.add_row([metric_name, agg_log[metric_name].item(), agg[metric_name].item()])
  print(table2)

  # # qualitive results
  # bokeh.plotting.output_file("test_method1.html") # only last scene will be stored at html
  # for sim_out in sim_outs: # for each scene
  #     vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
  #     bokeh.plotting.show(visualize(sim_out.scene_id, vis_in))

  # bokeh.plotting.output_file("test_method2.html") # only last scene will be stored at html
  # for sim_out in sim_outs_log: # for each scene
  #     vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
  #     bokeh.plotting.show(visualize(sim_out.scene_id, vis_in))
