import json


def read_file(file):
    with open(file, "r") as f:
        data = json.load(f)

        appears = []

    for i in range(len(data[-1])):
        agent_traj = []
        for j in range(len(data) - 1):
            if data[j]["sample_token"] == data[-1][f"agent_{i}"][0]["sample_token"]:
                appears.append((f"agent_{i}", data[j]["frame_id"], (data[j]["frame_id"] + len(data[-1][f"agent_{i}"]))))

    return data, appears


data, start_end = read_file("exported_json_data/scene-0061.json")

print(start_end)