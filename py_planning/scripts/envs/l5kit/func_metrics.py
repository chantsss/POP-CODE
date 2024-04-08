import torch
import warningsfrom typing import List

'''
Functions for metric recoginitions
'''
import thirdparty.configfrom l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               SimulatedDrivenMilesMetric, ReplayDrivenMilesMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)

_metric_simu_miles = SimulatedDrivenMilesMetric().metric_name
_metric_replay_miles = ReplayDrivenMilesMetric().metric_name
def is_driven_miles(key):
  return ((key == _metric_simu_miles) or (key == _metric_replay_miles))

'''
Aggregate metric values from scenes
:param metric_name_list: list of metric name, e.g., ['ade', 'fde'],
                         where the string should in line with metric_names in scenes_metric
:param scenes_metric: metric results from cle_evaluator.metric_results()
:return get_metric: list of metric values
:return scene_metrics: list of metric values for each  scene
'''
def aggregate_metrics(metric_name_list, scenes_metric):
  # print(">>> Metrics: ")
  # for sid, dta in  scenes_metric.items():
  #   print("scene_id[{}] >>".format(sid))
  #   for k, v in  dta.items():
  #     print("Key({}), TorchShape{}".format(k, v.shape))

  scene_metrics = dict()
  # init metric list
  for mn in metric_name_list:
    scene_metrics[mn]: List[float] = []

  for sid, dta in  scenes_metric.items():
    # for scene indexs
    for k, values in  dta.items():
      if k in scene_metrics:
        if is_driven_miles(k):
          ### use sum() operation
          sumup_miles = values.sum().item()
          scene_metrics[k].append(sumup_miles)
          # print('key={}, miles={}'.format(k, sumup_miles))
        else:
          ### use mean() operation
          scene_metrics[k].append(values[0:].mean().item())
      else:
        raise ValueError("key={} is not in metric_name_list".format(k))

  
  enable_percentage_miles = False
  if _metric_simu_miles in scene_metrics:
    if _metric_replay_miles in scene_metrics:
      enable_percentage_miles = True
    else:
      warnings.warn(
          "[simulated_driven_miles] requires metric of [replay_driven_miles]", RuntimeWarning, stacklevel=2
      )

  get_metric = dict()
  for k, values in scene_metrics.items():
    if len(values) > 0:
      if is_driven_miles(k):
        if (k == _metric_simu_miles):
          if enable_percentage_miles:
            replay_miles = scene_metrics[_metric_replay_miles]
            assert len(values) == len(replay_miles), "Error, scene of simu_miles != replay_miles"

            mile_percentage = [(1e-6+a)/(1e-6+b) for a, b in zip(values, replay_miles)]
            get_metric[k] = sum(mile_percentage) / len(mile_percentage)
          else:
            get_metric[k] = sum(values) / len(values)

        else:
          # (k == _metric_replay_miles):
          get_metric[k] = None
      else:
        get_metric[k] = sum(values) / len(values) 
    else:
      get_metric[k] = None
      warnings.warn(
          "metrics[{}] with values len = 0".format(k), RuntimeWarning, stacklevel=2
      )

  return [get_metric, scene_metrics]

# reference code:
#
# scenes_result = metric_set.evaluator.metric_results()
# scene_ade_list: List[float] = []
# scene_fde_list: List[float] = []
# for _, scene_result in scenes_result.items():
#     scene_ade_list.append(scene_result["displacement_error_l2"][1:].mean().item())
#     scene_fde_list.append(scene_result['displacement_error_l2'][-1].item())
#
# if len(scene_ade_list) == 0:
#     return (0, 0)
#
# average_ade = sum(scene_ade_list) / len(scene_ade_list)
# average_fde = sum(scene_fde_list) / len(scene_fde_list)
#
# return (average_ade, average_fde)
