# is derived from https://github.com/interaction-dataset/interaction-dataset

import csv
from numpy.lib.polynomial import polyfit

import thirdparty.configfrom interaction_dataset.python.utils.dataset_types import MotionState, Track

class Key:
    case_id = "case_id"
    track_id = "track_id"
    frame_id = "frame_id"
    time_stamp_ms = "timestamp_ms"
    agent_type = "agent_type"
    x = "x"
    y = "y"
    vx = "vx"
    vy = "vy"
    psi_rad = "psi_rad"
    length = "length"
    width = "width"


class KeyEnum:
    case_id = 0
    track_id = 1
    frame_id = 2
    time_stamp_ms = 3
    agent_type = 4
    x = 5
    y = 6
    vx = 7
    vy = 8
    psi_rad = 9
    length = 10
    width = 11


# def read_tracks(filename):
#     with open(filename) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         vehicle_track_dict = dict()
#         pedestrian_track_dict = dict()

#         for i, row in enumerate(list(csv_reader)):
#             if i == 0:
#                 # check first line with key names
#                 assert (row[KeyEnum.case_id] == Key.case_id)
#                 assert (row[KeyEnum.track_id] == Key.track_id)
#                 assert (row[KeyEnum.frame_id] == Key.frame_id)
#                 assert (row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
#                 assert (row[KeyEnum.agent_type] == Key.agent_type)
#                 assert (row[KeyEnum.x] == Key.x)
#                 assert (row[KeyEnum.y] == Key.y)
#                 assert (row[KeyEnum.vx] == Key.vx)
#                 assert (row[KeyEnum.vy] == Key.vy)
#                 assert (row[KeyEnum.psi_rad] == Key.psi_rad)
#                 assert (row[KeyEnum.length] == Key.length)
#                 assert (row[KeyEnum.width] == Key.width)
#                 continue        

#             case_id = int(row[KeyEnum.case_id])
#             track_id = int(row[KeyEnum.track_id])
#             agent_type = row[KeyEnum.agent_type]

#             is_car = (agent_type == "car")
#             if is_car:
#                 fill_dict = vehicle_track_dict
#             else:
#                 # is_pedestrian_bicycle
#                 fill_dict = pedestrian_track_dict

#             if not track_id in fill_dict.keys():
#                 # delcare track
#                 track = Track(track_id)

#                 track.agent_type = agent_type
#                 if is_car:
#                     track.length = float(row[KeyEnum.length])
#                     track.width = float(row[KeyEnum.width])
#                 else:
#                     track.length = 0.6
#                     track.width = 0.6
#                 track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
#                 track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
#                 fill_dict[track_id] = track

#             track = fill_dict[track_id]
#             track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
#             ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
#             ms.x = float(row[KeyEnum.x])
#             ms.y = float(row[KeyEnum.y])
#             ms.vx = float(row[KeyEnum.vx])
#             ms.vy = float(row[KeyEnum.vy])
#             if is_car:
#                 ms.psi_rad = float(row[KeyEnum.psi_rad])
#             else:
#                 ms.psi_rad = 0.0
#             track.motion_states[ms.time_stamp_ms] = ms

#         return vehicle_track_dict, pedestrian_track_dict


def read_cases(filename):
    '''
    Read cases of data from interaction data set (prediction part)
    '''
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        cases_dict = dict()

        for i, row in enumerate(list(csv_reader)):
            if i == 0:
                # check first line with key names
                assert (row[KeyEnum.case_id] == Key.case_id)
                assert (row[KeyEnum.track_id] == Key.track_id)
                assert (row[KeyEnum.frame_id] == Key.frame_id)
                assert (row[KeyEnum.time_stamp_ms] == Key.time_stamp_ms)
                assert (row[KeyEnum.agent_type] == Key.agent_type)
                assert (row[KeyEnum.x] == Key.x)
                assert (row[KeyEnum.y] == Key.y)
                assert (row[KeyEnum.vx] == Key.vx)
                assert (row[KeyEnum.vy] == Key.vy)
                assert (row[KeyEnum.psi_rad] == Key.psi_rad)
                assert (row[KeyEnum.length] == Key.length)
                assert (row[KeyEnum.width] == Key.width)
                continue        

            case_id = int(float(row[KeyEnum.case_id]))
            track_id = int(row[KeyEnum.track_id])
            agent_type = row[KeyEnum.agent_type]

            if not case_id in cases_dict:
                cases_dict[case_id] = (dict(), dict())
            fill_case = cases_dict[case_id]

            is_car = (agent_type == "car")
            if is_car:
                fill_dict = fill_case[0]
            else:
                # is_pedestrian_bicycle
                fill_dict = fill_case[1]

            if not track_id in fill_dict.keys():
                # delcare track
                track = Track(track_id)

                track.agent_type = agent_type
                if is_car:
                    track.length = float(row[KeyEnum.length])
                    track.width = float(row[KeyEnum.width])
                else:
                    track.length = 0.6
                    track.width = 0.6
                track.time_stamp_ms_first = int(row[KeyEnum.time_stamp_ms])
                track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
                fill_dict[track_id] = track

            track = fill_dict[track_id]
            track.time_stamp_ms_last = int(row[KeyEnum.time_stamp_ms])
            ms = MotionState(int(row[KeyEnum.time_stamp_ms]))
            ms.x = float(row[KeyEnum.x])
            ms.y = float(row[KeyEnum.y])
            ms.vx = float(row[KeyEnum.vx])
            ms.vy = float(row[KeyEnum.vy])
            if is_car:
                ms.psi_rad = float(row[KeyEnum.psi_rad])
            else:
                ms.psi_rad = 0.0 #!!! human do not have angle..
            track.motion_states[ms.time_stamp_ms] = ms

        return cases_dict
