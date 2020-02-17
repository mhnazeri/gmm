"""General Helpers functions that can be used everywhere"""
import numpy as np
import ujson as json
import torch


def read_file(file: str, feature: str = None):
    with open(file, "r") as f:
        data = json.load(f)

    features = []
    for id in range(len(data[:-1])):
        features.append(data[id][feature])

    appears = []

    for i in range(len(data[-1])):
        # agent_traj = []
        for j in range(len(data) - 1):
            if data[j]["sample_token"] == data[-1][f"agent_{i}"][0]["sample_token"]:
                appears.append((f"agent_{i}", data[j]["frame_id"],
                                 (data[j]["frame_id"] + len(data[-1][f"agent_{i}"]))))

    if feature:
        return data, features, appears

    return data, appears


def create_feature_matrix_for_cae(file):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 560), dtype=np.double)
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    for key, start, stop in appears:
        for i in range(stop - start):
            agent_data = datum[-1][key][i]["translation"]
            agent_data.extend(datum[-1][key][i]["rotation"])
            agent_data.extend(datum[-1][key][i]["velocity"])
            agent_data.extend(datum[-1][key][i]["size"])
            agent_data.extend([datum[-1][key][i]["movable"]])
            # print(f"{key}, start: {start}, stop: {stop}")
            # agents[int(key.split("_")[-1]), (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)] = np.array(agent_data, dtype=np.float64)
            agents[num, (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)] = np.array(agent_data, dtype=np.double)
        num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        # ego vehicle rotation
        ego.extend(datum[id]["ego_pose_rotation"])
        ego.extend(datum[id]["ego_pose_velocity"])
        ego.extend([1.907, 4.727, 1.957])  # size of ego-vehicle
        ego.extend([1])  # movable

    else:
        for i in range(40 - num_frames):
            ego.extend([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    ego = torch.tensor(ego, dtype=torch.double).reshape(-1, 560)
    agents = torch.from_numpy(agents)
    datum = torch.cat((ego, agents), 0)
    # datum = torch.transpose(datum, 1, 0)

    return datum


def create_feature_matrix_for_train(file):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 280), dtype=np.double)
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    for key, start, stop in appears:
        for i in range(stop - start):
            agent_data = datum[-1][key][i]["translation"]
            agent_data.extend(datum[-1][key][i]["rotation"])
            agents[num, (start * 7) + (i * 7): (start + 1) * 7 + (i * 7)] = np.array(agent_data, dtype=np.double)
        num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        # ego vehicle rotation
        ego.extend(datum[id]["ego_pose_rotation"])
    else:
        for i in range(40 - num_frames):
            ego.extend([0., 0., 0., 0., 0., 0., 0.])

    ego = torch.tensor(ego, dtype=torch.double).reshape(-1, 280)
    agents = torch.from_numpy(agents)
    datum = torch.cat((ego, agents), 0)

    return datum


if __name__ == "__main__":
    data = create_feature_matrix_for_train("exported_json_data/scene-0061.json")
    print(data.shape)