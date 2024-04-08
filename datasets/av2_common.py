from pathlib import Pathfrom typing import List

import numba
import numpy as np
import pandas as pd
import torchfrom av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)from av2.map.map_api import ArgoverseStaticMapfrom av2.map.lane_segment import LaneMarkType, LaneTypefrom av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory

TRACK_TYPE_TO_COLOR = {
    0: "orange",
    1: "yellow",
    2: "red",
    3: "purple",
    4: "green",
    5: "lightblue",
    6: "lightblue",
    7: "lightblue",
    8: "lightblue",
    9: "lightblue",
}

OBJECT_TYPE_MAP = {
    ObjectType.VEHICLE.value: 0,
    ObjectType.PEDESTRIAN.value: 1,
    ObjectType.MOTORCYCLIST.value: 2,
    ObjectType.CYCLIST.value: 3,
    ObjectType.BUS.value: 4,
    ObjectType.STATIC.value: 5,
    ObjectType.BACKGROUND.value: 6,
    ObjectType.CONSTRUCTION.value: 7,
    ObjectType.RIDERLESS_BICYCLE.value: 8,
    ObjectType.UNKNOWN.value: 9,
}
OBJECT_TYPE_MAP_COMBINED = {
    ObjectType.VEHICLE.value: 0,
    ObjectType.PEDESTRIAN.value: 1,
    ObjectType.MOTORCYCLIST.value: 2,
    ObjectType.CYCLIST.value: 2,
    ObjectType.BUS.value: 0,
    ObjectType.STATIC.value: 3,
    ObjectType.BACKGROUND.value: 3,
    ObjectType.CONSTRUCTION.value: 3,
    ObjectType.RIDERLESS_BICYCLE.value: 3,
    ObjectType.UNKNOWN.value: 3,
    "truck": 0,
    "large_vehicle": 0,
    "vehicular_trailer": 0,
    "truck_cab": 3,
    "school_bus": 0,
}

DYNAMIC_TRACK_TYPE = [0, 1, 2, 3, 4]


LaneTypeMap = {
    LaneType.VEHICLE.value: 1,
    LaneType.BIKE.value: 2,
    LaneType.BUS.value: 3,
}

LaneMarkTypeMap = {
    LaneMarkType.DASH_SOLID_YELLOW: 0,
    LaneMarkType.DASH_SOLID_WHITE: 1,
    LaneMarkType.DASHED_WHITE: 2,
    LaneMarkType.DASHED_YELLOW: 3,
    LaneMarkType.DOUBLE_SOLID_YELLOW: 4,
    LaneMarkType.DOUBLE_SOLID_WHITE: 5,
    LaneMarkType.DOUBLE_DASH_YELLOW: 6,
    LaneMarkType.DOUBLE_DASH_WHITE: 7,
    LaneMarkType.SOLID_YELLOW: 8,
    LaneMarkType.SOLID_WHITE: 9,
    LaneMarkType.SOLID_DASH_WHITE: 10,
    LaneMarkType.SOLID_DASH_YELLOW: 11,
    LaneMarkType.SOLID_BLUE: 12,
    LaneMarkType.NONE: 13,
    LaneMarkType.UNKNOWN: 14,
}

# https://www.joanwallacedrivingschool.com/do-you-really-know-your-road-markings/
LaneMarkTypeMap_COMBINED = {
    LaneMarkType.DASH_SOLID_YELLOW.value: 0,
    LaneMarkType.DASH_SOLID_WHITE.value: 0,
    LaneMarkType.DASHED_WHITE.value: 0,
    LaneMarkType.DASHED_YELLOW.value: 0,
    LaneMarkType.DOUBLE_SOLID_YELLOW.value: 1,
    LaneMarkType.DOUBLE_SOLID_WHITE.value: 1,
    LaneMarkType.DOUBLE_DASH_YELLOW.value: 0,
    LaneMarkType.DOUBLE_DASH_WHITE.value: 0,
    LaneMarkType.SOLID_YELLOW.value: 1,
    LaneMarkType.SOLID_WHITE.value: 1,
    LaneMarkType.SOLID_DASH_WHITE.value: 0,
    LaneMarkType.SOLID_DASH_YELLOW.value: 0,
    LaneMarkType.SOLID_BLUE.value: 2,  # for bus
    LaneMarkType.NONE.value: 3,
    LaneMarkType.UNKNOWN.value: 3,
}


@numba.njit
def to_local_xy(coord: np.ndarray, angle) -> np.ndarray:
    x = coord[..., 0]
    y = coord[..., 1]
    x_transform = np.cos(angle) * x + np.sin(angle) * y
    y_transform = -np.sin(angle) * x + np.cos(angle) * y
    output_coord = np.stack((x_transform, y_transform), axis=-1)

    return output_coord


@numba.njit
def to_local_angle(target: np.ndarray, ref: float) -> np.ndarray:
    return (target - ref + np.pi) % (2 * np.pi) - np.pi


def load_av2_scenario(scenario_file: Path):
    scenario_id = scenario_file.stem.split("_")[-1]
    scenario = load_argoverse_scenario_parquet(scenario_file)
    static_map = ArgoverseStaticMap.from_json(
        scenario_file.parents[0] / f"log_map_archive_{scenario_id}.json"
    )

    return scenario, static_map


def load_av2_from_dir(log_dir: Path):
    scenario_id = log_dir.stem
    df = pd.read_parquet(log_dir / ("scenario_" + scenario_id + ".parquet"))
    static_map = ArgoverseStaticMap.from_json(
        log_dir / f"log_map_archive_{scenario_id}.json"
    )

    return df, static_map, scenario_id


def load_av2_df(scenario_file: Path):
    scenario_id = scenario_file.stem.split("_")[-1]
    df = pd.read_parquet(scenario_file)
    static_map = ArgoverseStaticMap.from_json(
        scenario_file.parents[0] / f"log_map_archive_{scenario_id}.json"
    )

    return df, static_map, scenario_id


def get_transformation(df: pd.DataFrame):
    origin = torch.tensor(
        [df["position_x"].values[49], df["position_y"].values[49]]
    ).float()
    theta = torch.tensor([df["heading"].values[49]]).float()
    rot_mat = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ],
    ).float()

    return origin, theta, rot_mat


def get_agent_features(df: pd.DataFrame, origin, theta, rot_mat):
    xy = torch.from_numpy(
        np.stack([df["position_x"].values, df["position_y"].values], axis=1)
    ).float()
    heading = torch.from_numpy(df["heading"].values).float()
    vel = torch.from_numpy(
        np.stack([df["velocity_x"].values, df["velocity_y"].values], axis=1)
    ).float()

    xy = (xy - origin) @ rot_mat
    vel = vel @ rot_mat
    heading = (heading - theta + np.pi) % (2 * np.pi) - np.pi
    heading_vec = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)

    return torch.cat([xy, heading_vec, vel], dim=1)


@numba.njit
def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_lane(xy: np.ndarray, arc_length: np.ndarray, steps: np.ndarray):
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def split_and_interpolate_lane_segments(
    xy: np.ndarray, max_segment_length: float, point_num: int
) -> List[np.ndarray]:
    arc_length = get_polyline_arc_length(xy)

    num_segments = np.round(arc_length[-1] / max_segment_length).astype(np.int)
    num_segments = max(num_segments, 1)
    interp_points_num = num_segments * (point_num - 1) + 1

    steps = np.linspace(0, arc_length[-1], interp_points_num)
    xy_inter = interpolate_lane(xy, arc_length, steps)

    lane_segments = []
    for i in range(num_segments):
        start = i * (point_num - 1)
        end = start + point_num
        lane_segments.append(xy_inter[start:end])

    return lane_segments
