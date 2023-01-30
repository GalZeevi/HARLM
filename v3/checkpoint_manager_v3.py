import os
import pickle

import numpy as np


class CheckpointManager:  # TODO: add support for .npz and save_compressed
    basePath = 'checkpoints'

    if not os.path.exists(basePath):
        os.makedirs(basePath)

    @staticmethod
    def get_checkpoints():
        if not os.path.exists(CheckpointManager.basePath):
            return []
        return [f for f in os.listdir(CheckpointManager.basePath)]

    @staticmethod
    def get_all_versions():
        all_checkpoints = CheckpointManager.get_checkpoints()
        return [int(f) for f in all_checkpoints]

    @staticmethod
    def get_max_version():
        return max(CheckpointManager.get_all_versions())

    @staticmethod
    def save(name, content, append_to_last=True, numpy=False, should_print=True):
        if not os.path.exists(CheckpointManager.basePath):
            os.makedirs(CheckpointManager.basePath)
        all_checkpoints = CheckpointManager.get_checkpoints()
        all_versions = [int(f) for f in all_checkpoints]
        if len(all_versions) == 0:
            next_version = 1
        elif append_to_last:
            next_version = max(all_versions)
        else:
            next_version = max(all_versions) + 1

        suffix = 'pkl' if numpy is False else 'npy'
        file_path = f"./{CheckpointManager.basePath}/{next_version}"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        should_print and print(f'Saving checkpoint {name}.{suffix} to: [{file_path}]')
        if numpy is False:
            with open(f"{file_path}/{name}.{suffix}", 'wb') as f:
                pickle.dump(content, f)
        else:
            np.save(f"{file_path}/{name}.{suffix}", content)

    @staticmethod
    def load(name, version=None, numpy=False):
        if version is None:
            all_checkpoints = CheckpointManager.get_checkpoints()
            all_versions = [int(f) for f in all_checkpoints]
            version = max(all_versions)

        suffix = 'pkl' if numpy is False else 'npy'
        file_path = f"./{CheckpointManager.basePath}/{version}"

        if not os.path.exists(f"{file_path}/{name}.{suffix}"):
            return None

        if numpy is False:
            with open(f"{file_path}/{name}.{suffix}", 'rb') as f:
                return pickle.load(f)
        else:
            return np.load(f"{file_path}/{name}.{suffix}")

    @staticmethod
    def start_new_version():
        max_version = CheckpointManager.get_max_version()
        os.mkdir(f"./{CheckpointManager.basePath}/{max_version + 1}")
