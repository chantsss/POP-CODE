import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as npfrom typing import Dict, List

import envs.configfrom utils.file_io import extract_folder_file_list, read_dict_from_bin

import paper_plot.functions as plot_func
import paper_plot.utils as plot_utilsfrom paper_plot.color_schemes import FIVE_SEQ_COLOR_MAPS, color_from_indexfrom utils.colored_print import ColorPrinter

def smart_mean(data: np.ndarray) -> float:
  if data.shape[0] == 0:
    return 0.0
  else:
    return np.mean(data)

def read_solution_tag(solution_tag: str) -> Dict:
  '''
  extract information from solution_tag
  :param solution_tag: the input solution tag
  :return: the extracted information
  '''
  # Format: method_name_X_pY, e.g., 'exp_izone-s2_0.2_p1'
  #   name: string of the method name, (str, e.g., 'method_ca')
  #   X: percentage of data being trained for prediction module (float, e.g., 'X=0.1 (10%)')
  #   Y: number of multi modal trajectories being used in planning (e.g., 'Y=1 (only the most possible one)')

  cache_tags = solution_tag.split('_')
  assert len(cache_tags) == 4, "format of solution tag disobey 'method_name_X_pY' rule."

  # divide data into fig_tag/color_tag and ticks <format>
  rdict = {}
  cache_str = cache_tags[0] + '_' + cache_tags[1]

  # @note data in one fig_tag will be plotted in one graph
  #       data in one color_tag will have different color
  #       color_label denotes the label for the corresponding color_tag
  if 'paramtest_izone-lrcirule' in cache_str:
    # for example: 'paramtest_izone-lrcirule-15.0[0.5,0.0,-15.0]_1.0_p1',
    params_text = cache_str.split('lrcirule')[1].split(']')[0]
    base_txts = params_text.split('[')
    param0 = base_txts[0]
    param1 = base_txts[1].split(',')[0]
    param2 = base_txts[1].split(',')[1]
    param3 = base_txts[1].split(',')[2]
    
    rdict['fig_tag'] = cache_str.split('irule')[0] + 'irule'           # paramtest_izone-lrcirule
    rdict['color_tag'] = rdict['fig_tag']                              # paramtest_izone-lrcirule
    rdict['color_label'] = params_text                                 # param= -15.0[0.5,0.0,-15.0
    rdict['plot_marker'] = '-'

    rdict['property_value'] = float(param3)                             # property_value= param
  elif 'coeftest_' in cache_str:
    # for example: coeftest_izone-lrcpred0C[-15.0]_1.0_p1
    #              coeftest_izone-lrcirule0C-0.01[1.0,3.0,-15.0]_1.0_p1
    #              coeftest_conti+0C_1.0_p1
    map2coef = {0: 5.0, 1: 1.0, 2: 3.0, 3: 7.0, 4: 9.0}
    get_coef = map2coef[int(cache_str.split('C')[0][-1])]

    rdict['fig_tag'] = cache_str.split('C')[0][:-1]                   # coeftest_izone-lrcpred
    rdict['color_tag'] = "IR-Influ" # rdict['fig_tag']
    rdict['color_label'] = '$w_v$={:.1f}'.format(get_coef)
    
    rdict['plot_marker'] = '-'
    if not 'lrcirule' in cache_str:
      rdict['color_tag'] = "Conti"
      rdict['plot_marker'] = '--'

    rdict['property_value'] = round(get_coef, 1) # property_value= coefficent
  else:
    rdict['fig_tag'] = cache_str
    if 'predexp' in cache_tags[1]:
      rdict['color_tag'] = cache_tags[1]
      rdict['plot_marker'] = '-'
    elif 'lrcirule' in cache_tags[1]:
      rdict['color_tag'] = "IR-Influ"
      rdict['plot_marker'] = '-'
    elif 'lrcpred' in cache_tags[1]:
      rdict['color_tag'] = "IR-Pred"
      rdict['plot_marker'] = '--'
    elif 'conti+' in cache_tags[1]:
      rdict['color_tag'] = "Conti"
      rdict['plot_marker'] = '--'
    else:
      rdict['color_tag'] = cache_tags[0] + '_' + cache_tags[1] + '_' + cache_tags[3]
      rdict['plot_marker'] = '.-'

    rdict['color_label'] = rdict['color_tag']
    
    rdict['property_value'] = round(float(cache_tags[2]) * 100.0, 1) # property_value= percentage
  rdict['prediction_num'] = int(cache_tags[3][1:])
 
  return rdict

def collect_record(record_dict: Dict) -> None:
  # dict_keys(['plan_success_steps', 'total_simu_steps', 'list_of_time_costs'])
  if 'real_num_of_steps' in record_dict.keys():
    return # skip old version data

  global solu_record_data
  solu_record_data['success_num'] += float(record_dict['plan_success_steps'])
  solu_record_data['total_num'] += float(record_dict['total_simu_steps'])

def collect_metric(metric_dict: Dict) -> None:
  # dict_keys(['v_data', 'acc_data', 'jerk_data', 'clearance_data', 
  #            'acc_loss', 'jerk_loss', 'collision_times',
  #            'path_length', 'reaction_dcc_efforts'])
  global solu_record_data, solu_metric_data

  brake_accs = metric_dict['acc_data'][metric_dict['acc_data'] <= -3.9]
  brake_rate = 0.0
  if metric_dict['acc_data'].shape[0] > 0:
    brake_rate = float(brake_accs.shape[0]) / metric_dict['acc_data'].shape[0]

  solu_metric_data['brake_rate'].append(brake_rate)
  solu_metric_data['acc_loss'].append(metric_dict['acc_loss'])
  solu_metric_data['jerk_loss'].append(metric_dict['jerk_loss'])
  solu_metric_data['not_fault_collision_times'].append(
    metric_dict['collision_times']['front'] +\
      metric_dict['collision_times']['side'] +\
        metric_dict['collision_times']['stop_track_collision'])
  solu_metric_data['rear_collision_times'].append(metric_dict['collision_times']['rear'])
  solu_metric_data['path_length'].append(metric_dict['path_length'])

  if metric_dict['v_data'].shape[0] > 0:
    stop_num = metric_dict['v_data'][metric_dict['v_data'] < 1e-1].shape[0]
    stop_rate = float(stop_num) / metric_dict['v_data'].shape[0]
    solu_metric_data['stop_rate'].append(stop_rate)

  if metric_dict['reaction_dcc_efforts'].shape[0] > 0:
    solu_metric_data['reaction_dcc_efforts'] =\
      solu_metric_data['reaction_dcc_efforts'] + metric_dict['reaction_dcc_efforts'].tolist()

import conf.module_path
import math
import yamlfrom sklearn.gaussian_process import GaussianProcessRegressorfrom sklearn.gaussian_process.kernels import RBF

if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default='commonroad', 
    help="corresponding to the dataset paramter in planning_commonroad.yaml")
  args = parser.parse_args()

  cfg = None
  with open(os.path.join(os.path.dirname(conf.module_path.__file__), 
            'compare_tags.yaml')) as config_file:
    cfg = yaml.safe_load(config_file)['compare_config']
  tag_str_list = cfg['comparison_tags'] # get comparison solution tag list

  root_dir = envs.config.get_dataset_exp_folder(
    dataset_name=args.dataset, folder_name='exp_plan')

  debug_enable_save_pdf = True
  debug_fig_size_coef = 1.75
  debug_enable_plot_mean_and_std_area = True
  debug_enable_plot_mean = False
  debug_enable_plot_colorbar = True
  debug_y_plot_range = 0.2 # 0.2
  debug_only_plot1figure = True

  ########################################################
  # preparation
  legend_loc = 'upper left'
  axis_info = {
    'xtick': list(range(0, 6)),
    'xlabel': [r"DIST$\uparrow$", r"FR$\downarrow$",  r"JERK$\downarrow$", 
      r"RC$\downarrow$", r"CT$\downarrow$", r"RCT$\downarrow$"]
  }

  # collect max_values & raw_data_dict
  max_values = np.zeros(6)
  raw_data_dict = {}

  for tag_str in tag_str_list:
    print("Processing ", tag_str)

    ## read files
    solutions_dir = os.path.join(root_dir, '{}/solutions'.format(tag_str))
    metrics_dir = envs.config.get_root2folder(root_dir, '{}/evals'.format(tag_str))

    record_files = extract_folder_file_list(solutions_dir)
    record_files = [fname for fname in record_files if 'plan_records' in fname]

    metric_files = extract_folder_file_list(metrics_dir)

    ## init data
    solu_record_data = {
      'success_num': 0.0,
      'total_num': 0.0
    }
    solu_metric_data = {
      'brake_rate': [],
      'acc_loss': [],
      'jerk_loss': [],
      'not_fault_collision_times': [],
      'rear_collision_times': [],
      'path_length': [],
      'stop_rate': [],
      'reaction_dcc_efforts': [],
    }

    ## collect data
    for ri, rfile in enumerate(record_files):
      rfile_path = os.path.join(solutions_dir, rfile)
      collect_record(record_dict=read_dict_from_bin(rfile_path, verbose=False))

    for mi, mfile in enumerate(metric_files):
      metric_dict = read_dict_from_bin(os.path.join(metrics_dir, mfile), verbose=False)
      collect_metric(metric_dict=metric_dict)

    ## prepare for visualization
    solu_metric_data['brake_rate'] = np.array(solu_metric_data['brake_rate'], dtype=float)
    solu_metric_data['acc_loss'] = np.array(solu_metric_data['acc_loss'], dtype=float)
    solu_metric_data['jerk_loss'] = np.array(solu_metric_data['jerk_loss'], dtype=float)
    solu_metric_data['not_fault_collision_times'] = np.array(solu_metric_data['not_fault_collision_times'], dtype=float)
    solu_metric_data['rear_collision_times'] = np.array(solu_metric_data['rear_collision_times'], dtype=float)
    solu_metric_data['path_length'] = np.array(solu_metric_data['path_length'], dtype=float)
    solu_metric_data['stop_rate'] = np.array(solu_metric_data['stop_rate'], dtype=float)
    solu_metric_data['reaction_dcc_efforts'] = np.array(solu_metric_data['reaction_dcc_efforts'], dtype=float)

    ## plot things
    # mean_loss_acc = smart_mean(solu_metric_data['acc_loss']) 
    mean_brake_rate = smart_mean(solu_metric_data['brake_rate'])
    mean_loss_jerk = smart_mean(solu_metric_data['jerk_loss'])
    sum_front_side_collision_times = np.sum(solu_metric_data['not_fault_collision_times'])
    sum_rear_collision_times = np.sum(solu_metric_data['rear_collision_times'])
    mean_path_l = smart_mean(solu_metric_data['path_length'])
    total_path_l = np.sum(solu_metric_data['path_length'])
    mean_stop_rate = smart_mean(solu_metric_data['stop_rate'])
    # reaction_loss = smart_mean(solu_metric_data['reaction_dcc_efforts'])
    reaction_loss = np.mean(solu_metric_data['reaction_dcc_efforts'])

    # print('Metric of tag_str are: ', mean_brake_rate, mean_loss_jerk)

    fail2calculate_rate = 1.0 - min(1.0, solu_record_data['success_num'] / solu_record_data['total_num'])
    # collision_metric = (sum_front_side_collision_times + sum_rear_collision_times) / total_path_l * 1000.0 # collisions / per km
    collision_metric = sum_front_side_collision_times
    rear_collision_metric = sum_rear_collision_times

    max_values[0] = max(max_values[0], mean_path_l)
    max_values[1] = max(max_values[1], fail2calculate_rate)
    max_values[2] = max(max_values[2], mean_loss_jerk)
    max_values[3] = max(max_values[3], reaction_loss)
    max_values[4] = max(max_values[4], collision_metric)
    max_values[5] = max(max_values[5], rear_collision_metric)

    get_dict = read_solution_tag(tag_str)
    raw_data_dict[tag_str] = {
      'fig_tag': get_dict['fig_tag'],
      'color_tag': get_dict['color_tag'],
      'color_label': get_dict['color_label'],
      'plot_marker': get_dict['plot_marker'],
      'property_value': get_dict['property_value'],
      'prediction_num': get_dict['prediction_num'],
      'values': [mean_path_l, fail2calculate_rate, mean_loss_jerk, 
                 reaction_loss, collision_metric, rear_collision_metric],
    }

  # collect plot_data_records
  max_values[0] = max(0.1, max_values[0])
  # max_values[1] = max(0.0, max_values[1])
  # max_values[2] = max(0.1, max_values[2])
  plot_data_records = {} # method_pX

  min_y = 1e+3
  max_y = -1e+3
  for _tag_str, content in raw_data_dict.items():
    y_values = (np.array(content['values']) + 1e-5) / (max_values + 1e-5)
    min_y = min(min_y, min(y_values))
    max_y = max(max_y, max(y_values))

    get_dict = read_solution_tag(_tag_str)

    fig_tag = content['fig_tag']
    if not fig_tag in plot_data_records.keys():
      plot_data_records[fig_tag] = {}

    color_tag = get_dict['color_tag']
    property_value = get_dict['property_value']
    prediction_num = get_dict['prediction_num']
    if not color_tag in plot_data_records[fig_tag].keys():
      plot_data_records[fig_tag][color_tag] = {
        'fig_tag': fig_tag,
        'color_label': get_dict['color_label'],
        'plot_marker': get_dict['plot_marker'],
        'yvalues': [],
        'properties': [],
      }

    # divide data into fig_tag/color_tag and ticks <format>
    plot_data_records[fig_tag][color_tag]['yvalues'].append(y_values.tolist())
    plot_data_records[fig_tag][color_tag]['properties'].append(property_value)

  for fig_tag, scheme_dict in plot_data_records.items():
    print("\n"+"**"*30)
    print("Category::{}::print()".format(fig_tag))
    for color_tag, content in scheme_dict.items():
      get_data = np.array(content['yvalues']) * max_values
      print("  Scheme::{}::print() with shape= {}.".format(color_tag, get_data.shape))

      if get_data.shape[0] == 1:
        get_values = get_data[0]
        print("metric=", "{:.2f} & {:.2f}\%  & {:.2f} & {:.3f} & {:d} & {:d}".format(
          get_values[0], (get_values[1] * 100.0), get_values[2], 
          get_values[3], int(get_values[4]), int(get_values[5])))
      else:
        y_mean = np.apply_over_axes(func=np.mean, a=get_data, axes=0).squeeze()
        y_std = np.apply_over_axes(func=np.std, a=get_data, axes=0).squeeze()

        print("Mean=", "{:.2f} & {:.2f}\%  & {:.2f} & {:.3f} & {:d} & {:d}".format(
          y_mean[0], (y_mean[1] * 100.0), y_mean[2], 
          y_mean[3], int(y_mean[4]), int(y_mean[5])))
        print("Std=", "{:.2f} & {:.2f}\%  & {:.2f} & {:.3f} & {:d} & {:d}".format(
          y_std[0], (y_std[1] * 100.0), y_std[2], 
          y_std[3], int(y_std[4]), int(y_std[5])))

  ########################################################
  # visualization

  figures = []
  axes = []
  cate_index :int= 0
  plot_min_y = max(0.0, min_y-debug_y_plot_range)

  print("label=", axis_info['xlabel'])
  print("max_values=", max_values)
  ax1 = None
  for fig_tag, scheme_dict in plot_data_records.items():
    flag_avoid_repeat = (debug_only_plot1figure == False) or (cate_index == 0)

    if flag_avoid_repeat:
      figures.append(plt.figure())

      figures[-1].set_size_inches(
        3.5 * debug_fig_size_coef, 2.163 * debug_fig_size_coef) # (3.5, 2.163) > golden ratio= 0.618
      plot_utils.fig_reset()
      ax1 = plt.subplot(1, 1, 1)
      axes.append([ax1])
      plot_utils.subfig_reset()
    else:
      ColorPrinter.print('yellow', "Warning, debug_only_plot1figure is enabled")

    if debug_enable_save_pdf == False:
      plot_utils.axis_set_title(ax1, fig_tag)

    scheme_index :int= 0
    plot_utils.axis_set_ylabel(ax_example=ax1, label='Norm. score')
    print("Proecess color_tag=", color_tag)
    for color_tag, content in scheme_dict.items():
      label_str = content['color_label']
      line_marker = content['plot_marker']
      y_samples = np.array(content['yvalues'])
      properties = content['properties']

      ###### ax1
      enable_plot_mean_std = (y_samples.shape[0] > 1)

      x_array = np.array(axis_info['xtick'], dtype=float)
      get_marker = line_marker
      if enable_plot_mean_std:
        if debug_enable_plot_mean_and_std_area:
          get_marker = line_marker
        else:
          get_marker = line_marker + 'o'
        ax1.grid(True, linestyle='--', linewidth=0.5)

      # Plot the results
      print("LABEL=", label_str)
      default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
      scheme_colors = []
      for yi, y_array in enumerate(y_samples):
        if (yi == 0) and (enable_plot_mean_std == False):
          ax1.plot(x_array, y_array, get_marker, label=label_str)
          ax1.legend()
        else:
          ax1.plot(x_array, y_array, get_marker, color=default_color_cycle[yi])
          o_y_array = y_array * max_values
          print("[{}] value=".format(default_color_cycle[yi]), y_array)

          print("     metric=", "{:.2f} & {:.2f}\%  & {:.2f} & {:.3f} & {:d} & {:d}".format(
            o_y_array[0], (o_y_array[1] * 100.0), o_y_array[2], 
            o_y_array[3], int(o_y_array[4]), int(o_y_array[5])))
          print(" ")

          scheme_colors.append(default_color_cycle[yi])
      print(" properties=", properties)
      print(" scheme_colors[{}]=".format(scheme_index), scheme_colors)

      if enable_plot_mean_std == True:
        y_mean = np.apply_over_axes(func=np.mean, a=y_samples, axes=0).squeeze()
        y_std = np.apply_over_axes(func=np.std, a=y_samples, axes=0).squeeze()

        if debug_enable_plot_mean_and_std_area:
          if debug_enable_plot_mean:
            ax1.plot(x_array, y_mean, '-o', 
                    color='k', label='Mean')
          ax1.fill_between(
              x=x_array, 
              y1=(y_mean - 2.0*y_std), 
              y2=(y_mean + 2.0*y_std), 
              color=color_from_index(cate_index),
              alpha=0.1, label=color_tag # '{}($2\sigma$ area)'.format(color_tag)
          )
          ax1.legend(loc=legend_loc)

          txt_y_mean = np.apply_over_axes(func=np.mean, a=(y_samples * max_values), axes=0).squeeze()
          txt_y_std = np.apply_over_axes(func=np.std, a=(y_samples * max_values), axes=0).squeeze()
          
          if debug_only_plot1figure == False:
            loc_i :int = 0
            for loc_x, get_mean, get_std in zip(axis_info['xtick'], txt_y_mean, txt_y_std):
              if loc_i == 1:
                get_text = '{:.2f}$\pm${:.2f}\%'.format(get_mean * 100.0, get_std * 100.0)
              elif loc_i != 3:
                get_text = '{:.2f}$\pm${:.2f}'.format(get_mean, get_std)
              else:
                get_text = '{:.3f}$\pm${:.3f}'.format(get_mean, get_std)
              plot_utils.ax_set_text(ax1, loc_x, plot_min_y+(debug_y_plot_range * 0.25), get_text)

              loc_i += 1
        else:
          ColorPrinter.print('yellow', "Disable plot mean & std area, if want to enble pls ...")
          # set debug_enable_plot_mean_and_std_area = True

      # for x, y, show_v in zip(axis_info['xtick'], max_values, max_values):
      #   plot_utils.ax_set_text(ax1, x, 1.1, '[{:.2f}]'.format(show_v))  
      plot_utils.axis_set_xticks(ax_example=ax1,
        tick_values=axis_info['xtick'], tick_labels=axis_info['xlabel'])

      # plot color bar
      if flag_avoid_repeat and enable_plot_mean_std and debug_enable_plot_colorbar:
        assert len(properties) == len(scheme_colors), "Unexpected Error at colour bar illustration"
        axis_locs = np.arange(len(scheme_colors) + 1)
        
        cmp = matplotlib.colors.ListedColormap(scheme_colors)
        cmp_norm = matplotlib.colors.BoundaryNorm(axis_locs, cmp.N)
        fcb = figures[-1].colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=cmp),
          ax=ax1, shrink=1.0, orientation='vertical')
   
        tick_locs = []
        tick_labels = []
        for pi, tick_loc in enumerate(axis_locs):
          if pi > 0:
            tick_locs.append(axis_locs[pi-1]*0.5 + tick_loc*0.5)
            tick_labels.append("{}".format(properties[pi-1]))

        print("  tick_locs", tick_locs)
        print("  tick_labels", tick_labels)
        fcb.set_ticks(tick_locs)
        fcb.set_ticklabels(tick_labels)

      ax1.set_xlim([-0.5, max(axis_info['xtick']) + 0.5])
      ax1.set_ylim([plot_min_y, 1.0 + debug_y_plot_range])
      # plot_utils.save_fig(os.path.expanduser('~') + "/tmp/{}".format(color_tag), vdpi=300)
      plot_utils.save_fig(os.getcwd() + "/{}".format(color_tag), vdpi=300)

      scheme_index += 1
    cate_index += 1

  plt.show()
