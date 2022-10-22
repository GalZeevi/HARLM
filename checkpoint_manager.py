import os
import pickle


class CheckpointManager:
    basePath = 'checkpoints'

    if not os.path.exists(basePath):
        os.makedirs(basePath)

    @staticmethod
    def get_checkpoints(name):
        path = f"{CheckpointManager.basePath}/{name}"
        if not os.path.exists(path):
            return []
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    @staticmethod
    def save(name, content):
        checkpointPath = f"{CheckpointManager.basePath}/{name}"
        if not os.path.exists(checkpointPath):
            os.makedirs(checkpointPath)
        all_checkpoints = CheckpointManager.get_checkpoints(name)
        all_versions = [int(f.split('.')[0].split('_')[1]) for f in all_checkpoints]
        next_version = 0 if len(all_versions) == 0 else max(all_versions) + 1
        with open(f"{checkpointPath}/checkpoint_{next_version}.pkl", 'wb') as f:
            pickle.dump(content, f)

    @staticmethod
    def load(name, version=None):
        if version is None:
            all_checkpoints = CheckpointManager.get_checkpoints(name)
            all_versions = [int(f.split('_')[1]) for f in all_checkpoints]
            version = max(all_versions)
        with open(f"{CheckpointManager.basePath}/{name}/checkpoint_{version}", 'w') as f:
            return pickle.load(f)

    @staticmethod
    def exists(name):
        return len(CheckpointManager.get_checkpoints(name)) > 0
