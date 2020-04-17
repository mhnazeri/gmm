"""Helper functions for Visualization of trajectories"""
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nuscenes.nuscenes import NuScenes, view_points
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.eval.common.utils import boxes_to_sensor
from data_helpers import create_feature_matrix_for_viz


# Defining Global variables
# matplotlib.use( 'tkagg' )
fig, axes = plt.subplots(figsize=(18, 9))
view = np.eye(4)
ln, = plt.plot([], [], "b.", markersize=1)
nusc = NuScenes(version="v1.0-mini", dataroot="nuScene-mini",verbose=False)


def sample_extractor(nusc, idx_scene):
    """returns sample frames from scene idx"""
    sample = nusc.get("sample", nusc.scene[idx_scene]["first_sample_token"])
    _frames = []
    while sample["next"] != "":
            _frames.append(nusc.get("sample", sample["token"]))
            sample = nusc.get("sample", sample["next"])

    # Get last frame
    _frames.append(nusc.get("sample", sample["token"]))

    return _frames


def frame_idx():
    init = 0
    for _ in range(40):
        yield init
        init += 1


def update(*frames):
    global data_path
    global boxes
    global corners
    # print(len(frames[1]))
    # global frame_indices
    # frame_indices = frame_idx()
    axes.clear()
    # idx_frame = next(frame_indices)
    # feature_matrix = frame_number

    # for i in range(len(feature_matrix)):
    #     agent_past = feature_matrix[i][idx_frame * 3: (idx_frame + 2) * 3]
    #     agent_future = feature_matrix[i][(idx_frame + 2) * 3: (idx_frame + 5) * 3]
    #     axes.scatter(agent_past[::3], agent_past[1::3], marker='d', label="Past")
    #     axes.scatter(agent_future[::3], agent_future[1::3], marker='s', label="Future")
    for lidar, data, calibre in frames:
        pose_record = {"translation": data["now"][0], "rotation": data["now"][0]}
        for i in range(1, len(lidar)):
            # print(feature_matrix.shape)
            # Move box to ego vehicle coord system
            agent_past = data["past"][i] - data["past"][0]
            agent_future = data["future"][i] - data["future"][0]
            agent_now = data["now"][i] - data["now"][0]

            agent_past -= np.tile(np.array(agent_now), 2)
            agent_future -= np.tile(np.array(agent_now), 3)
            # print(agent_past.shape)

            # agent_past -= pose_record["translation"]
            # agent_future -= pose_record["translation"]
            # Move box to sensor coord system
            # agent_past -= np.tile(np.array(calibre[0]), 2)
            # agent_future -= np.tile(np.array(calibre[0]), 3)
            print(agent_past)
            # plotting ego
            axes.scatter(data["past"][0][::3], data["past"][0][1::3], marker='d', label="Past", color="blue")
            axes.scatter(data["future"][0][::3], data["future"][0][1::3], marker='s', label="Future", color="green")
            # plotting other agents
            axes.scatter(agent_past[::3], agent_past[1::3], marker='d', label="Past", color="blue")
            axes.scatter(agent_future[::3], agent_future[1::3], marker='s', label="Future", color="green")

            # axes_limit = 53
            # axes.set_xlim(-axes_limit, axes_limit)
            # axes.set_ylim(-axes_limit, axes_limit)

        for ann in lidar["anns"]:
            data_path, boxes, _ = nusc.get_sample_data(lidar["data"]["LIDAR_TOP"], selected_anntokens=[ann])

            for box in boxes:
                c = np.array(_get_color(box.name)) / 255.0
                box.render(axes, view=view, colors=(c, c, c))

                corners = _view_points(boxes[0].corners(), view)[:2, :]
                axes.set_xlim([np.min(corners[0, :]) - 10, np.max(corners[0, :]) + 10])
                axes.set_ylim([np.min(corners[1, :]) - 10, np.max(corners[1, :]) + 10])
                axes.axis('off')
                axes.set_aspect('equal')

    global frame
    frame = LidarPointCloud.from_file(data_path)
    frame.render_height(axes, view=np.eye(4))

    global data_
    data_ = _view_points(frame.points[:3,:], np.eye(4))

    xdata = data_[0, :]
    ydata = data_[1, :]
    ln.set_data(xdata, ydata)
    return ln,


def _view_points(points, view):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    return points

def init():
    axes.set_xlim(-20, 20)
    axes.set_ylim(-20, 20)
    ln.set_data([], [])
    return ln,

def _get_color(category_name: str):
    """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    if "bicycle" in category_name or "motorcycle" in category_name:
        return 255, 61, 99  # Red
    elif "vehicle" in category_name or category_name in [
        "bus",
        "car",
        "construction_vehicle",
        "trailer",
        "truck",
    ]:
        return 255, 158, 0  # Orange
    elif "pedestrian" in category_name:
        return 0, 0, 230  # Blue
    elif "cone" in category_name or "barrier" in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta


def render_scene_lidar(root, nusc, idx_scene=0, save_path=None, blit=False):
    scene_data = sample_extractor(nusc, idx_scene)
    feature_matrix, calibrated_features = create_feature_matrix_for_viz("exported_json_data/scene-0061.json")
    lidar = []
    num_features = list(range(121))
    start_stop = list(zip(num_features[::3], num_features[3::3]))

    for idx in range(2, 37):
        data = {}
        data["past"] = feature_matrix[:, start_stop[idx-2][0]: start_stop[idx-2][1]]
        data["past"] = np.concatenate((data["past"], feature_matrix[:, start_stop[idx-1][0]: start_stop[idx-1][1]]), 1)
        data["now"] = feature_matrix[:, start_stop[idx][0]: start_stop[idx][1]]
        data["future"] = feature_matrix[:, start_stop[idx+1][0]: start_stop[idx+1][1]]
        data["future"] = np.concatenate((data["future"], feature_matrix[:, start_stop[idx+2][0]: start_stop[idx+2][1]]), 1)
        data["future"] = np.concatenate((data["future"], feature_matrix[:, start_stop[idx+3][0]: start_stop[idx+3][1]]), 1)
        lidar.append((scene_data[idx],
            data,
            calibrated_features[idx])
        )

    # feature_matrix = zip(feature_matrix[:][: 6: 6],
    #                      feature_matrix[:][6: 21: 15])
    # frame_number = range(0, 40)
    ani = FuncAnimation(fig, update, frames=lidar, init_func=init, blit=blit)

    # fig.tight_layout()
    # fig.legend(bbox_to_anchor=(1., 1.), loc="upper left", fontsize=14)

    if save_path:
        ani.save(save_path + str(idx_scene) + ".mp4")
    # commented for computational reasons
    # plt.show()

if __name__ == "__main__":
    # feature_matrix = create_feature_matrix_for_viz("exported_json_data/scene-" + "0061.json").numpy()
    render_scene_lidar("", nusc, save_path="./")

    # feature_matrix = zip(feature_matrix[0][: 6],
    #                      feature_matrix[0][6: 21])
    # feature_matrix = list(feature_matrix)
    # print(len(feature_matrix[0]))