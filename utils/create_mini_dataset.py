"""
Preprocess the mini dataset (Horse, Jaguar, Ostrich) from 3 BVH files.
Produces cond.npy and motions/*.npy under dataset/truebones/zoo/truebones_processed/.
Run from the repo root:
    conda run -n anytop python -m utils.create_mini_dataset
"""
import os
import numpy as np
from os.path import join as pjoin

from data_loaders.truebones.truebones_utils.motion_process import process_object
from data_loaders.truebones.truebones_utils.param_utils import (
    DATASET_DIR, MOTION_DIR, ANIMATIONS_DIR, BVHS_DIR
)

ANIMALS = ["Horse", "Jaguar", "Ostrich"]


def main():
    os.makedirs(pjoin(DATASET_DIR, MOTION_DIR), exist_ok=True)
    os.makedirs(pjoin(DATASET_DIR, ANIMATIONS_DIR), exist_ok=True)
    os.makedirs(pjoin(DATASET_DIR, BVHS_DIR), exist_ok=True)

    files_counter = 0
    frames_counter = 0
    max_joints = 23
    squared_positions_error = {}
    cond = {}

    for object_type in ANIMALS:
        prev = files_counter
        files_counter, frames_counter, max_joints, object_cond = process_object(
            object_type, files_counter, frames_counter, max_joints, squared_positions_error)
        cond[object_type] = object_cond
        print(f"{object_type}: {files_counter - prev} clips processed")

    print(f"\nTotal clips: {files_counter}, max_joints: {max_joints}")
    np.save(pjoin(DATASET_DIR, "cond.npy"), cond)
    print(f"Saved cond.npy to {DATASET_DIR}")


if __name__ == "__main__":
    main()
