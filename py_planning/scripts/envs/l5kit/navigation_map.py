import copyfrom operator import imod
import numpy as np
import math
import networkx
import warningsfrom typing import Dict, Listfrom scipy.spatial import KDTree
import shapely.geometry as shapely_geofrom envs.l5kit.config import NavMapConfigfrom envs.l5kit.map_api import MapAPI, InterpolationMethod, TLFacesColorsfrom utils.angle_operation import get_normalized_angle, get_normalized_angles

import thirdparty.configfrom l5kit.simulation.dataset import SimulationDatasetfrom l5kit.data import ChunkedDatasetfrom l5kit.data.filter import (filter_agents_by_frames, filter_agents_by_labels, filter_tl_faces_by_frames,
                               filter_tl_faces_by_status)from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
import matplotlib.pyplot as plt

_param_successor_lane_dist_cond = 0.25
_param_latereral_lane_dist_cond = 0.5

class navigation_map:
  def __init__(self, 
               mapAPI: MapAPI, 
               sim_dataset: SimulationDataset):
    self.mapAPI = mapAPI
    self.sim_dataset = sim_dataset

    ### datas
    self.is_inited = False

    # map objs
    '''
    :data scene_ego_pos:               scene_index: position of AV in first frame
    :data scene_groute_points:         scene_index: global route points
    :data scene_groute_sum_dist:       scene_index: sum distance of global route
    :data scene_groute_max_v:          scene_index: speed limit (float)
    :data scene_lane_idx_set:          scene_index: set of lane_idx (str)
    :data scene_lane_infos:            scene_index: lane_idx(str): dict of lane informations
    :data scene_crosswalk_polys:       scene_index: crosswalk_idx(int): cross_walk polygons
    '''
    self.scene_ego_pos: Dict[int, Any] = {}
    self.scene_groute_points: Dict[int, List[Any]] = {}
    self.scene_groute_sum_dist: float = 0.0
    self.scene_groute_max_v: Dict[int, float] = {}
    self.scene_lane_idx_set: Dict[int, set] = {}
    self.scene_lane_infos: Dict[int, Dict[str, Dict[str, Any]]] = {}
    self.scene_crosswalk_polys: Dict[int, Dict[int, shapely_geo.Polygon]] = {}

    # kd tree
    '''
    :data scene_kdt_lane_points:       scene_index: points of middle_lanes, used to build kd tree
    :data scene_kdt_lane_points_infos: scene_index: list of properties for points in scene_kdt_lane_points
    :data scene_kdtrees:               scene_index: kdtrees of scene_kdt_lane_points

    :data scene_kdt_left_lane_points:  scene_index: points of left_lane, used to build left-kd tree
    :data scene_kdtrees_left:          scene_index: kdtrees of scene_kdt_left_lane_points
    '''
    self.scene_kdt_lane_points: Dict[int, np.ndarray] = {}
    self.scene_kdt_lane_points_infos: Dict[int, List[Any]] = {}
    self.scene_kdtrees: Dict[int, KDTree] = {}

    self.scene_kdt_left_lane_points: Dict[int, np.ndarray] = {}
    self.scene_kdtrees_left: Dict[int, KDTree] = {}
  
    # graph
    '''
    :data scene_graph_lane_num:            scene_index: totoal lane num in a scene
    :data scene_graph_lane_rviz_pos:       scene_index: lane sequence index(int): np.ndarray
    :data scene_graph_map_lane_idx2seq_id: scene_index: lane_idx(str): lane sequence index(int)
    :data scene_graph_map_seq_id2lane_idx: scene_index: lane sequence index(int): lane_idx(str)
    :data scene_graph:                     scene_index: graph using lane as vertexs
    :data scene_graph_edges:               scene_index: vertex_id(lane_sequence_id): edge info

    :data scene_routing_result:            scene_index: [has_path, [node1, node2, ...], [route1_points, ...]]
    '''
    self.scene_graph_lane_num: Dict[int, int] = {}
    self.scene_graph_lane_rviz_pos: Dict[int, Dict[int, np.array()]] = {}
    self.scene_graph_map_lane_idx2seq_id: Dict[int, Dict[str, int]] = {}
    self.scene_graph_map_seq_id2lane_idx: Dict[int, Dict[int, str]] = {}
    self.scene_graph: Dict[int, networkx.Graph] = {}
    self.scene_graph_edges: Dict[int, Dict[int, Any]] = {}

    self.scene_routing_result: Dict[int, [bool, List[int], List[np.ndarray]]] = {}

    ### initializations
    if (sim_dataset != None):
      self._init_map_things(sim_dataset, mapAPI)
      self._init_graph_networks()
      self._init_routing()
      self.is_inited = True

  def inited(self):
    return self.is_inited

  def has_global_route(self, scene_index):
    if not scene_index in self.scene_routing_result:
      return False

    return self.scene_routing_result[scene_index][0]

  def _init_map_things(self, sim_dataset: SimulationDataset, mapAPI: MapAPI):
    # obtain the full route + 
    #            speed limits +
    #            map: lanes + traffic lights + crosswalk
    scene_index_list = []
    route_last_frame_index: int = 0
    for frame_index in range(len(sim_dataset)): # loop: frames
      dtas = sim_dataset.rasterise_frame_batch(frame_index)

      for scene_ego in dtas: # loop: scenes
        scene_index = scene_ego['scene_index']
        scene_dt = sim_dataset.scene_dataset_batch[scene_index]
        frame_array = scene_dt.dataset.frames[frame_index]
        agents = scene_dt.dataset.agents
        tls = scene_dt.dataset.tl_faces

        if not scene_index in scene_index_list:
          scene_index_list.append(scene_index)

        pos = frame_array["ego_translation"]
        pos[2] *= (np.pi / 360.0)
        speed = abs(scene_ego['curr_speed'])
        if not scene_index in self.scene_ego_pos:
          # init dict contents
          route_last_frame_index = frame_index

          self.scene_ego_pos[scene_index] = pos
          self.scene_groute_points[scene_index] = [pos]
          self.scene_groute_sum_dist = 0.0
          self.scene_groute_max_v[scene_index] = max(speed, NavMapConfig.MIN_SPEED_LIMIT)
          self.scene_lane_idx_set[scene_index] = set()
          self.scene_lane_infos[scene_index] = {}
          self.scene_crosswalk_polys[scene_index] = {}
          self.scene_kdt_lane_points[scene_index] = None
          self.scene_kdt_lane_points_infos[scene_index] = []
          self.scene_kdt_left_lane_points[scene_index] = None
          self.scene_graph_lane_num[scene_index] = 0
          self.scene_graph_lane_rviz_pos[scene_index] = {}
          self.scene_graph_map_lane_idx2seq_id[scene_index] = {}
          self.scene_graph_map_seq_id2lane_idx[scene_index] = {}
        else:
          # fill dt: groute_points
          last_pos = self.scene_groute_points[scene_index][-1]
          dist = np.linalg.norm((pos - last_pos)[0:2])
          if (dist >= 0.25) or ((frame_index - route_last_frame_index) >= 10):
            route_last_frame_index = frame_index
            self.scene_groute_points[scene_index].append(pos)
            self.scene_groute_sum_dist += dist
          # fill dt: groute_max_v
          self.scene_groute_max_v[scene_index] = max(self.scene_groute_max_v[scene_index], speed)

        # map info
        pos2d = pos[0:2]
        lane_indices = indices_in_bounds(pos2d, self.mapAPI.bounds_info["lanes"]["bounds"], 
                                         half_extent=NavMapConfig.OBJ_SEARCH_EXTENT)
        active_tl_ids = set(filter_tl_faces_by_status(tls, "ACTIVE")["face_id"].tolist())
        crosswalk_indices = indices_in_bounds(pos2d, self.mapAPI.bounds_info["crosswalks"]["bounds"], 
                                              half_extent=NavMapConfig.OBJ_SEARCH_EXTENT)

        # traverse crosswalks
        for idx in crosswalk_indices:
          '''
          Traverse all crosswalk and extract informations
          :param crosswalk:
                    key=['xyz']: [[x, y, z], ...] >> polygon.
          '''
          # fill dt: crosswalk_polys
          if idx not in self.scene_crosswalk_polys[scene_index]:
            crosswalk = mapAPI.get_crosswalk_coords(mapAPI.bounds_info["crosswalks"]["ids"][idx])
            poly = shapely_geo.Polygon(crosswalk['xyz'][:, 0:2])
            self.scene_crosswalk_polys[scene_index][idx] = poly
            # print(idx, poly.centroid, poly.area, poly.bounds)

        # traverse lanes
        for idx, lane_idx in enumerate(lane_indices):
          '''
          Traverse all lanes and extract informations
          :param idx: int number
          :param lane_idx: a char sequence
          :param lane_coords: 
                    key=['xyz_left']: [[x, y, z], ...]
                    key=['xyz_right']: [[x, y, z], ...]
          '''
          lane_idx = mapAPI.bounds_info["lanes"]["ids"][lane_idx]
          if lane_idx in self.scene_lane_idx_set[scene_index]:
            continue # this lane is added in this scene

          # fill dt: graph_lane_num & graph_lane_map_idx2seq_id
          assert not lane_idx in self.scene_graph_map_lane_idx2seq_id[scene_index], 'Unexpected Error'
          lane_seq_id = copy.copy(self.scene_graph_lane_num[scene_index])
          self.scene_graph_map_lane_idx2seq_id[scene_index][lane_idx] = lane_seq_id
          self.scene_graph_map_seq_id2lane_idx[scene_index][lane_seq_id] = lane_idx
          self.scene_graph_lane_num[scene_index] += 1

          lane_coords = mapAPI.get_lane_as_interpolation(lane_idx, step=NavMapConfig.MAX_POINTS_PER_LANE, 
                                                         method=InterpolationMethod.INTER_ENSURE_LEN)
          left_lane = lane_coords["xyz_left"][:, :2]
          right_lane = lane_coords["xyz_right"][:, :2]
          middle_lane = lane_coords["xyz_midlane"][:, :2]

          mid_lane_len = middle_lane.shape[0]
          lane_length = np.sum(np.linalg.norm(middle_lane[1:]-middle_lane[:-1], axis=1))

          if left_lane.shape[0] == mid_lane_len:
            dist2lane_edge = np.linalg.norm(left_lane-middle_lane, axis=1)
          elif right_lane.shape[0] == mid_lane_len:
            dist2lane_edge = np.linalg.norm(right_lane-middle_lane, axis=1)
          else:
            raise ValueError("lane mid, left, right with different sizes=[{}, {}, {}]".format(
              left_lane.shape[0], middle_lane.shape[0], right_lane.shape[0]))

          lane_tl_ids = set(mapAPI.get_lane_traffic_control_ids(lane_idx))
          lane_with_tl = None # TLFacesColors.GREEN, TLFacesColors.RED, TLFacesColors.YELLOW
          for tl_id in lane_tl_ids.intersection(active_tl_ids):
              # check sets whether has same units (ids) or not
              lane_with_tl = mapAPI.get_color_for_face(tl_id)

          # fill dt: lanes_idx_set + lane_info
          self.scene_lane_idx_set[scene_index].add(lane_idx)
          self.scene_lane_infos[scene_index][lane_idx] = {
            'middle_lane': middle_lane,         # [[x, y], ...]
            'lane_length': lane_length,         # lane length
            'has_tl_info': lane_with_tl,        # None or TLFacesColors.COLOR, 
            'left_lane': left_lane,             # [[x, y], ...]
            'right_lane': right_lane,           # [[x, y], ...]
            'dist2lane_edge': dist2lane_edge,   # [dist, ...]
          }
          # fill dt: kdt lane_points & lane_points_infos
          dt_is_empty = \
            (type(self.scene_kdt_lane_points[scene_index]).__module__ != np.__name__)
          pi_base = 0
          local_points_num = middle_lane.shape[0]
          local_points_num_1 = local_points_num - 1
          if not dt_is_empty:
            pi_base = self.scene_kdt_lane_points[scene_index].shape[0]

          points_info = []
          for pi in np.arange(local_points_num):
            dvector = middle_lane[pi+1] - middle_lane[pi] if pi < local_points_num_1 else\
                                                          middle_lane[pi] - middle_lane[pi-1]
            dradian = math.atan2(dvector[1], dvector[0])
            points_info.append([
              lane_idx,                                     # lane index
              lane_seq_id,                                  # lane sequence id
              [pi_base + pi, pi, local_points_num, 
               dradian],                                    # point infos, 
              lane_with_tl,                                 # has_traffic_light
              [],                                           # intersected crosswalk (@note:
                                                            #   updated in fill dt: update intersected crosswalk info)
            ])

          if dt_is_empty:
            self.scene_kdt_lane_points[scene_index] = middle_lane
            self.scene_kdt_lane_points_infos[scene_index] = points_info

            self.scene_kdt_left_lane_points[scene_index] = left_lane
          else:
            self.scene_kdt_lane_points[scene_index] =\
              np.concatenate((self.scene_kdt_lane_points[scene_index], middle_lane), axis=0)
            self.scene_kdt_lane_points_infos[scene_index] += points_info

            self.scene_kdt_left_lane_points[scene_index] =\
              np.concatenate((self.scene_kdt_left_lane_points[scene_index], left_lane), axis=0)

        # here is inside loop(scenes)
      # here is inside loop(frames)

    for scene_index in scene_index_list:
      # print("scene", scene_index, frame_index, self.scene_kdt_lane_points[scene_index].shape)
      # fill dt: build kdtrees
      self.scene_kdtrees[scene_index] = KDTree(self.scene_kdt_lane_points[scene_index])
      self.scene_kdtrees_left[scene_index] = KDTree(self.scene_kdt_left_lane_points[scene_index])

      # fill dt: update intersected crosswalk info in kdt_lane_points_infos
      results = self.is_inside_crosswalks(scene_index, self.scene_kdt_lane_points[scene_index])
      self.scene_kdt_lane_points_infos[scene_index] =\
          [[dt1[0], dt1[1], dt1[2], dt1[3], dt2] for dt1, dt2 in\
            zip(self.scene_kdt_lane_points_infos[scene_index], results)]

  def _init_graph_networks(self):
    '''
    Build the graph, where node is the lane. Edge info between two lane is
      map:
    '''
    for scene_index, set_dt in self.scene_lane_idx_set.items():
      self.scene_graph_edges[scene_index] = {}
      this_graph = networkx.DiGraph()

      for lane_idx in set_dt:
        info = self.scene_lane_infos[scene_index][lane_idx]
        # info['middle_lane']:    middle_lane,      # [[x, y], ...]
        # info['has_tl_info']:    lane_with_tl,     # None or TLFacesColors.COLOR, 
        # info['left_lane']:      left_lane,        # [[x, y], ...]
        # info['right_lane']:     right_lane,       # [[x, y], ...]
        # info['dist2lane_edge']: dist2lane_edge,   # [dist, ...]
        lane_seq_id = self.scene_graph_map_lane_idx2seq_id[scene_index][lane_idx]
        middle_lane = info['middle_lane']
        lane_length = info['lane_length']
        right_lane = info['right_lane']
        rlane_pnum = right_lane.shape[0]
        rlane_pnum_1 = rlane_pnum - 1

        # add end edges to graph
        end_edge = []
        closed_point_ids =\
          self.scene_kdtrees[scene_index].query_ball_point((middle_lane[-1]), 
            r=_param_successor_lane_dist_cond)
        for pindex in closed_point_ids:
          pinfo = self.scene_kdt_lane_points_infos[scene_index][pindex]
          to_lane_sid = pinfo[1]
          p_indexs = pinfo[2]
          if (to_lane_sid != lane_seq_id) and (p_indexs[1] == 0):
            # end point is matched to start points of other lanes
            end_edge.append(to_lane_sid) 

        for to_lane_sid in end_edge:
          # print(lane_seq_id, to_lane_sid)
          this_graph.add_edge(lane_seq_id, to_lane_sid, color='k', 
                              weight=NavMapConfig.LANE_FOWARD_COST)
          # print("add edge: ({}, {})".format(lane_seq_id, to_lane_sid))

        # add right edge to graph
        check_points = [right_lane[0], right_lane[int(rlane_pnum/2.0)], right_lane[-1]]
        closed_point_ids =\
          self.scene_kdtrees_left[scene_index].query_ball_point((check_points), 
            r=_param_latereral_lane_dist_cond)

        right_edge = None
        if lane_length >= NavMapConfig.LANE_CHANGE_COND_LENGTH:
          for pindexs in closed_point_ids:
            to_lane_sids = []
            for pindex in pindexs:
              pinfo = self.scene_kdt_lane_points_infos[scene_index][pindex]
              to_lane_sid = pinfo[1]
              if (to_lane_sid != lane_seq_id):
                to_lane_sids.append(to_lane_sid)
            if right_edge == None:
              right_edge = to_lane_sids
            else:
              right_edge = list(set(right_edge).intersection(to_lane_sids))
        if right_edge == None:
          right_edge = []

        for to_lane_sid in right_edge:
          # righ-lane changing
          this_graph.add_edge(lane_seq_id, to_lane_sid, color='b', 
                              weight=NavMapConfig.LANE_CHANGE_COST_DEFAULT)
          # left-lane changing
          this_graph.add_edge(to_lane_sid, lane_seq_id, color='y', 
                              weight=NavMapConfig.LANE_CHANGE_COST_DEFAULT)

        # fill dt: graph_edges & scene_graph_lane_rviz_pos
        # @example:
        #   edge_info={'node_id': 14, 'from_to': [...], 
        #              'end_to': [37, 30, 15], 
        #              'left_to': [...], 'right_to': [...]}
        edge_info = {
          'node_id':  lane_seq_id, # the lane's index in graph
          'from_to':  [],          # what lanes are connected to middle_lane[0]?
          'end_to':   end_edge,    # what lanes are connected to middle_lane[-1]?
          'left_to':  [],          # what lanes are left-changeable to middle_lane?
          'right_to': right_edge,  # what lanes are right-changeable to middle_lane
        }
        self.scene_graph_edges[scene_index][lane_seq_id] = edge_info
        self.scene_graph_lane_rviz_pos[scene_index][lane_seq_id] = middle_lane[0]

      # fill dt: graph_edges['from_to'] & ['left_to']
      content = self.scene_graph_edges[scene_index]
      for lane_sid, dt in content.items():
        for end_lane_sid in dt['end_to']:
          self.scene_graph_edges[scene_index][end_lane_sid]['from_to'] =\
            self.scene_graph_edges[scene_index][end_lane_sid]['from_to'] + [lane_sid]
        for right_lane_sid in dt['right_to']:
          self.scene_graph_edges[scene_index][right_lane_sid]['left_to'] =\
            self.scene_graph_edges[scene_index][right_lane_sid]['left_to'] + [lane_sid]
      for lane_sid, dt in content.items():
        self.scene_graph_edges[scene_index][lane_seq_id]['from_to'] =\
          list(set(self.scene_graph_edges[scene_index][lane_seq_id]['from_to']))
        self.scene_graph_edges[scene_index][lane_seq_id]['left_to'] =\
          list(set(self.scene_graph_edges[scene_index][lane_seq_id]['left_to']))

      # fill dt: graph
      self.scene_graph[scene_index] = this_graph

  def _init_routing(self):
    for scene_id, start in self.scene_ego_pos.items():
      goal = self.scene_groute_points[scene_id][-1]
      start_linfo = self.get_nearest_lane(scene_id, start)
      goal_linfo = self.get_nearest_lane(scene_id, goal)

      has_path = networkx.has_path(self.scene_graph[scene_id], 
                                   start_linfo[1], goal_linfo[1])
      path = []
      paths_points = [] # may have lane changing, so piecewise
      if has_path:
        path = networkx.shortest_path(self.scene_graph[scene_id], 
                                      start_linfo[1], goal_linfo[1])
        points = None
        end_is_added = False
        for lane_sid in path:
          lane_idx = self.scene_graph_map_seq_id2lane_idx[scene_id][lane_sid]
          lane_info = self.scene_lane_infos[scene_id][lane_idx]
          lane_points = lane_info['middle_lane']

          end_is_added = False
          if (type(points).__module__ != np.__name__):
            points = lane_points
          else:
            dist = np.linalg.norm(lane_points[0] - points[-1])
            if dist < 0.25:
              points = np.concatenate((points, lane_points), axis=0)
            else:
              paths_points.append(points)
              points = lane_points
              end_is_added = True
        if end_is_added == False:
          paths_points.append(points)
      else:
        warnings.warn(
            "scene[{}]: routing fail".format(scene_id), RuntimeWarning, stacklevel=2
        )
      self.scene_routing_result[scene_id] = [has_path, path, paths_points]
      # print(scene_id, start_linfo[1], goal_linfo[1], path)

  def is_inside_crosswalks(self, scene_index: int, 
                                 check_points: np.ndarray):
    '''
      Check whether check_points is in crosswalks
      :param scene_index: scene index
      :param check_points: points being checked np.array with shape (N, 2)
      :return [[intersected_crosswalk_index, ], ...], len = N
    '''

    id_polys = list(self.scene_crosswalk_polys[scene_index].items())
    def check_func(p):
      crosswalk_ids = []
      for id_poly in id_polys:
        if id_poly[1].contains(shapely_geo.Point(p)):
          crosswalk_ids.append(id_poly[0]) # idx
      return crosswalk_ids
    result = list(map(check_func, check_points))
    return result

  def get_nearest_lane(self, scene_index, check_xyyaw, coef_yaw=57.3/10.0):
    '''
    Calculate the nearest lane info
    :param scene_index: the scene index
    :param check_xyyaw: the input [x, y, yaw]
    :param coef_yaw:    the coefficent to of yaw distance: 57.3/10.0 means
                        radian=1.0 -> (57.3 / 10.0) m
    :return [lane index, lane sequence id]
    '''
    check_xy = check_xyyaw[0:2]
    point1 =\
      self.scene_kdtrees[scene_index].query((check_xy), k=1)
    dist1 = point1[0]

    n_point_dist = np.inf
    n_point_dist2d = None
    n_point_index = point1[1]
    n_point_info = None
    # print("traverse points")
    for index in self.scene_kdtrees[scene_index].query_ball_point((check_xy), 
                                                                  r=(dist1 * 1.25 + 1.0)):
      cp_pos = self.scene_kdt_lane_points[scene_index][index]
      cp_info = self.scene_kdt_lane_points_infos[scene_index][index]
      cp_direct = cp_info[2][3]
      dist2d = np.linalg.norm(check_xy - cp_pos)

      dyaw = get_normalized_angle(cp_direct - check_xyyaw[2])
      dist = dist2d + coef_yaw * math.fabs(dyaw)
      # print(check_xyyaw[2], index, dist2d, dyaw)
      if dist < n_point_dist:
        n_point_dist = dist
        n_point_dist2d = dist2d
        n_point_index = index
        n_point_info = cp_info
    # print("result=", dist1, point1[1], n_point_dist2d, n_point_dist, n_point_index)
    return [n_point_info[0], n_point_info[1]] # [lane index, lane sequence id]

  def plot_and_save2figure(self, file_name):
    if (not self.inited()):
      return
    for scene_id, graph in self.scene_graph.items():
      dt = self.scene_routing_result[scene_id]
      has_path = dt[0]
      path = dt[1]

      _file_name = file_name + "_{}".format(scene_id) + ".png"
      plt.clf()

      # networkx.draw(graph, with_labels=True, font_weight='bold')

      nodes_pos = self.scene_graph_lane_rviz_pos[scene_id]
      edg_colors = []
      edg_widths = []
      if has_path:
        path_edge = [(path[k], path[k+1]) for k in range(0, len(path) - 1)]
        for u,v in graph.edges():
          if (u, v) in path_edge:
            edg_colors.append('r')
            edg_widths.append(1.0)
          else:
            edg_colors.append(graph[u][v]['color'])
            edg_widths.append(0.5)

      if len(edg_colors) == 0:
        print("edge num == 0")
        return

      # draw nodes and edges
      networkx.draw(graph, pos=nodes_pos,
                    node_color='lightgreen', 
                    node_size=2.0,
                    arrowsize=3.5,
                    edge_color=edg_colors,
                    width=edg_widths)

      # networkx.draw_networkx_labels(graph, pos=nodes_pos)
      # edg_labels = networkx.get_edge_attributes(graph, 'weight')
      # networkx.draw_networkx_edge_labels(graph, nodes_pos, edge_labels=edg_labels)

      plt.savefig(_file_name, dpi=300)

  def debug(self):
    if (not self.inited()):
      return
    # print("navigtaion route:", self.scene_ego_pos, self.scene_groute_sum_dist)
    # print(self.scene_groute_points)

    # print("navigtaion map debug:")
    # for scene_index, ddt in self.scene_graph_edges.items():
    #   print("scene[{}] with nodes number={}/{}".format(scene_index, len(ddt), 
    #                                                    self.scene_graph[scene_index].number_of_nodes()))
    #   for lane_sid, dt in ddt.items():
    #     if (len(dt['from_to']) > 0) and (len(dt['end_to']) > 0) and \
    #        (len(dt['left_to']) > 0) and (len(dt['right_to']) > 0):
    #       print(scene_index, lane_sid, dt)
