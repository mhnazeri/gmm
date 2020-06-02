import os
import random
from typing import Tuple


def total_samples(root_folder: str) -> Tuple:
    files = os.listdir(root_folder)
    return (files, len(files))


def move_samples(source: str, dest: str, portion: float, seed: int = 42):
    """randomly selects portion of data and move them to test directory
    args:
        str source: source folder
        str dest: destination folder to move selected samples
        float portion: how much of the data you want as test
        int seed: if you don't want reproducibility set seed to `None`
    """
    if seed:
        random.seed(seed)

    files, total = total_samples(source)
    test_portion = int(total * portion)
    selected_files = []
    # selects samples randomly
    for _ in range(test_portion):
        file = random.choice(files)
        selected_files.append(file)
        # to prevent selecting the sample multiple times
        del files[files.index(file)]

    if os.path.exists(dest):
        for file in selected_files:
            os.rename(os.path.join(source, file), os.path.join(dest, file))

    else:
        os.mkdir(dest)
        for file in selected_files:
            os.rename(os.path.join(source, file), os.path.join(dest, file))


if __name__ == "__main__":
    move_samples("train_data", "test_data", 0.2)