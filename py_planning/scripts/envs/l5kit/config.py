
class NavMapConfig:
  MIN_SPEED_LIMIT = 5.0      # m/s
  OBJ_SEARCH_EXTENT = 50.0   # m
  MAX_POINTS_PER_LANE = 50   # int

  # routing config for roadmap graph
  LANE_FOWARD_COST = 1.0          # cost metric
  LANE_CHANGE_COST_DEFAULT = 0.5
  LANE_CHANGE_COND_LENGTH = 5.0   # min lane length allow lane-changing
