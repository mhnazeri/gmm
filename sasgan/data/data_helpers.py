"""General Helpers functions that can be used everywhere"""
import os
import numpy as np
from PIL import Image
import ujson as json
import torch
from torchvision import transforms


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


def create_feature_matrix(file):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 520))
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    for key, start, stop in appears:
        for i in range(stop - start):
            if datum[-1][key][i]["movable"] != 0:
                agent_data = datum[-1][key][i]["translation"]
                agent_data.extend(datum[-1][key][i]["rotation"])
                agent_data.extend(datum[-1][key][i]["velocity"])
                agent_data.extend(datum[-1][key][i]["size"])
                # agent_data.extend([datum[-1][key][i]["movable"]])
                # print(f"{key}, start: {start}, stop: {stop}")
                # agents[int(key.split("_")[-1]), (start * 14) + (i * 14): (start + 1) * 14 + (i * 14)] = np.array(agent_data, dtype=np.float64)
                agents[num, (start * 13) + (i * 13): (start + 1) * 13 + (i * 13)] = np.array(agent_data)
        num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        # ego vehicle rotation
        ego.extend(datum[id]["ego_pose_rotation"])
        ego.extend(datum[id]["ego_pose_velocity"])
        ego.extend([1.907, 4.727, 1.957])  # size of ego-vehicle
        # ego.extend([1])  # movable

    else:
        for i in range(40 - num_frames):
            ego.extend(torch.zeros(13, dtype=torch.float32))

    ego = torch.tensor(ego).reshape(-1, 520)
    agents = torch.from_numpy(agents).float()
    datum = torch.cat((ego, agents), 0)
    # datum = torch.transpose(datum, 1, 0)

    return datum


def create_feature_matrix_for_viz(file):
    datum, timestamps, appears = read_file(file, "timestamp")
    num_frames = len(timestamps) if len(timestamps) < 40 else 40
    ego = []
    agents = np.zeros((len(datum[-1]), 120))
    calibrated_features = []
    # appears = (agent_num, start, stop)
    # sort by their number of visibilities in frames
    appears = sorted(appears, key=lambda x: x[2] - x[1], reverse=True)
    num = 0
    print("alksjdlkajsd")
    for key, start, stop in appears:
        for i in range(stop - start):
            agent_data = datum[-1][key][i]["translation"]
            agents[num, (start * 3) + (i * 3): (start + 1) * 3 + (i * 3)] = np.array(agent_data)
        num += 1

    for id in range(num_frames):
        # ego vehicle location
        ego.extend(datum[id]["ego_pose_translation"])
        calibrated_features.append((datum[id]["calibrated_translation"],
                                    datum[id]["calibrated_rotation"]))
    else:
        for i in range(40 - num_frames):
            ego.extend(torch.zeros(3, dtype=torch.float32))

    ego = np.array(ego).reshape(-1, 120)
    agents = agents
    datum = np.concatenate((ego, agents), 0)

    return datum, calibrated_features


def save_train_samples(root_dir, save_dir):
    """save each train sample on hdd"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    json_files = os.path.join(root_dir, "exported_json_data")
    image_files = os.path.join(root_dir, "nuScene-mini")
    files = os.listdir(json_files)
    files = [os.path.join(json_files, _path) for _path in files]
    # check if save_dir exists, otherwise create one
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    images = []
    lidar = []
    features = []
    datums = []
    # read the list 13 element at a time
    num_features = [x for x in range(521) if x % 13 == 0]
    start_stop = list(zip(num_features[:], num_features[1:]))
    index = 0

    for file in files:
        # read lidar and camera data locations
        lidar_address, camera_address = [], [] # read_file(file)
        with open(file, "r") as f:
            _data = json.load(f)

        ego = _data[:-1]
        lidar_address = []
        camera_address = []

        for i in range(len(ego)):
            lidar_address.append(ego[i]["lidar"])
            camera_address.append(ego[i]["camera"])
        # for each file (scene), creates corresponding matrix
        features = create_feature_matrix(file)
        # create zero rows to reach the max agent number
        # dummy = torch.zeros(max_agent - len(features), 520)
        # features = torch.cat((features, dummy), 0)
        data = {}
        stamp = 0

        while stamp < 27:
            past = []
            future = []
            image = []

            for j in range(4):
                # 4 frames in the past
                past.append(features[:, start_stop[stamp + j][0]: start_stop[stamp + j][1]])
                # each frame has an image
                image.append(transform(Image.open(os.path.join(image_files, camera_address[stamp + j]))))

            for j in range(4, 14):
                # 10 frames in the future
                future.append(features[:, start_stop[stamp + j][0]: start_stop[stamp + j][1]])

            # calculate background motion by subtracting two consecutive images
            image = [img_2 - img_1 for img_1, img_2 in zip(image[:], image[1:])]
            # we only need 7 first features (translation, rotation) for relative history
            # a helper to slice out the 7 first features from each frame
            rel_past = torch.cat(past, 1)
            rel_past = [rel_past[:, i:i+7] for i in range(0,52,13)]
            rel_past = [past_2 - past_1 for past_1, past_2 in zip(rel_past[:], rel_past[1:])]

            # if frame is at the beginning of a scene, add zero
            if stamp == 0:
                rel_past.insert(0, torch.zeros_like(past[0][:, :7]))
                image.insert(0, torch.zeros_like(image[0]))
            else:
                rel_past.insert(0, (past[0] - datums[-1]["past"][-1])[:, :7])
                image.insert(0, image[0] - datums[-1]["motion"][-1])

            data["past"] = past
            data["future"] = future

            data["rel_past"] = rel_past
            data["motion"] = image
            datums.append(data)
            # save data on hard
            torch.save(data, os.path.join(save_dir, f"{index}.pt"))
            index += 1
            data = {}
            stamp += 1


if __name__ == "__main__":
    # data = create_feature_matrix_for_viz("exported_json_data/scene-0061.json")
    # print(data.shape)
    save_train_samples(".", "train_data")
