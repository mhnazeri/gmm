"""Helper functions for Visualization of trajectories"""
import os.path as osp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nuscenes.nuscenes import NuScenes, view_points
from nuscenes.utils.data_classes import LidarPointCloud
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
    for lidar, feature_matrix in frames:
        for i in range(len(feature_matrix)):
            agent_past = feature_matrix[i][: 6] - feature_matrix[0][: 6]
            agent_future = feature_matrix[i][6:] - feature_matrix[0][6:]
            axes.scatter(agent_past[::3], agent_past[1::3], marker='d', label="Past")
            axes.scatter(agent_future[::3], agent_future[1::3], marker='s', label="Future")

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

    global data
    data = _view_points(frame.points[:3,:], np.eye(4))

    xdata = data[0, :]
    ydata = data[1, :]
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
    feature_matrix = create_feature_matrix_for_viz("exported_json_data/scene-0061.json")
    lidar = []
    for idx, sample_frame in enumerate(scene_data):
        lidar.append((sample_frame, feature_matrix[:, (idx * 3) - 6: (idx * 3) + 15]))

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