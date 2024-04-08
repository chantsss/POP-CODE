import os
import argparse

import envs.configfrom envs.config import get_root2folder
import conf.module_pathfrom analysis.config import ProcessConfigfrom envs.directory import ProcessedDatasetDirectoryfrom utils.file_io import extract_folder_file_list, read_dict_from_bin
from envs.format_trajectory import DatasetTrajecotryIOfrom envs.format_interaction import DatasetInteractionIO
from analysis.collector_trajectory import TrajectoryCollectorfrom analysis.collector_kmeans_trajectory import TrajectoryCollectorWithKmeansProcessorfrom analysis.collector_interaction import InteractionCollector

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # python scripts/analyze_dataset.py --datasets 'interaction_dataset 'l5kit'
  parser.add_argument("--datasets", nargs='+', type=str, default=[],
                      help="give the list of dataset to process")
  parser.add_argument("--read_mode", type=str, default='trajectory',
                      help="determine what to read")
  parser.add_argument("--file_num", type=int, default=1e+10,
                      help="limited number of file to read")
  parser.add_argument("--traj_v0_l", type=float, default=0.0,
                      help="v0 lower bound when read_mode == 'trajectory'")
  parser.add_argument("--traj_v0_u", type=float, default=1e+3,
                      help="v0 upper bound when read_mode == 'trajectory'")
  args = parser.parse_args()

  # Preparations
  READER = {
    'trajectory': DatasetTrajecotryIO,
    'kmeans_trajectory': DatasetTrajecotryIO,
    'interaction': DatasetInteractionIO,
    # TODO:
  }

  EXP_ROOT = {
    'interaction_dataset': envs.config.INTERACTION_EXP_ROOT,
    'commonroad': envs.config.COMMONROAD_EXP_ROOT,
  }

  def get_read_folder(dataset_type: str, read_mode: str):
    dict_map = {
      'trajectory':\
        ProcessedDatasetDirectory.get_path(dataset_type, 'trajectory'),
      'kmeans_trajectory':\
        ProcessedDatasetDirectory.get_path(dataset_type, 'trajectory'),
      'interaction':\
        ProcessedDatasetDirectory.get_path(dataset_type, 'trajectory'),
    }
    return dict_map[read_mode]

  def get_collector(dataset_type: str, read_mode: str):
    if read_mode == 'trajectory':
      return TrajectoryCollector(
          key_mode='plot_elements',
          filter_agent_type_ids=[],
          filter_v0_lowerbound=args.traj_v0_l,
          filter_v0_upperbound=args.traj_v0_u,
          cache_dir=None
        )
    elif read_mode == 'kmeans_trajectory':
      return TrajectoryCollectorWithKmeansProcessor(
          key_mode='plot_elements',
          cache_dir=get_root2folder(EXP_ROOT[dataset_type], 'cache_kmeans_trajs'),
          cache_batch_num=1e+10,
        )
    elif read_mode == 'interaction':
      return InteractionCollector(
          key_mode='analyze_patterns', # 'plot_elements', 'analyze_patterns',
          cache_dir=get_root2folder(EXP_ROOT[dataset_type], 'cache_interaction'),
          cache_batch_num=5000,
          config_path=os.path.join(
            os.path.dirname(conf.module_path.__file__), 
            'data_collector.yaml'),
          save_dir=get_root2folder(EXP_ROOT[dataset_type], 'priority_prediction')
        )
    else:
      raise ValueError("get_collector(), unkown read_mode = {}.".format(read_mode))

  # Loop datasets
  for dtset in args.datasets:
    reader = READER[args.read_mode]
    collector = get_collector(dtset, args.read_mode)
    read_folder = get_read_folder(dtset, args.read_mode)
    file_list = extract_folder_file_list(read_folder)
    print("Traverse dataset={}, file num={}.".format(dtset, len(file_list)))
    for fidx, fname in enumerate(file_list):
      fpath = os.path.join(read_folder, fname)
      data = read_dict_from_bin(fpath, verbose=False)
      data = reader.read_data(data)

      if collector.add_data(data) == False:
        break
      collector.debug_print("process file={}/{}".format(fidx, len(file_list)))

      if fidx >= args.file_num:
        print("\nBreak loop to read data with index={}/{}.".format(fidx, args.file_num))
        break

    collector.final_processing_data()
    collector.plot()
