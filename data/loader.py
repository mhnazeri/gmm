import json
import numpy as np


def read_file(file: str, feature: str=None):
    with open(file, "r") as f:
        data = json.load(f)

    features = []
    for id in range(len(data[:-1])):
        features.append(data[id][feature])

    appears = []

    for i in range(len(data[-1])):
        agent_traj = []
        for j in range(len(data) - 1):
            if data[j]["sample_token"] == data[-1][f"agent_{i}"][0]["sample_token"]:
                appears.append((f"agent_{i}", data[j]["frame_id"], (data[j]["frame_id"] + len(data[-1][f"agent_{i}"]))))

    if feature:
        return data, features, appears

    return data, appears


def create_feature_matrix(file):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 440), dtype=np.float64)
    # appears = (agent_num, start, stop)
    for key, start, stop in appears:
        for i in range(stop - start):
            agent_data = datum[-1][key][i]["translation"]
            agent_data.extend(datum[-1][key][i]["rotation"])
            agent_data.extend(datum[-1][key][i]["size"])
            agent_data.extend([datum[-1][key][i]["movable"]])
            print(f"{key}, start: {start}, stop: {stop}")
            agents[int(key.split("_")[-1]), (start * 11) + (i * 11): (start + 1) * 11 + (i * 11)] = np.array(agent_data, dtype=np.float64)

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        # ego vehicle rotaion
        ego.extend(datum[id]["ego_pose_rotation"])
        ego.extend([1.907, 4.727, 1.957]) # size of ego-vehicle
        ego.extend([1]) # movable

    else:
        for i in range(40 - num_frames):
            ego.extend([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            # print(str(len(agents)) + "here")
            # for j in range(len(agents)):
            #     agents[j].extend([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    return np.array(ego, dtype=np.float64).reshape(1, 440), agents



ego, agents = create_feature_matrix("exported_json_data/scene-0061.json")
print(ego.shape)