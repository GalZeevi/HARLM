import os
import pickle


class CheckpointManager:
    basePath = 'checkpoints'

    if not os.path.exists(basePath):
        os.makedirs(basePath)

    @staticmethod
    def get_checkpoints(sub_dir=None):
        path = f"{CheckpointManager.basePath}{'' if sub_dir is None else '/' + sub_dir}"
        if not os.path.exists(path):
            return []
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    @staticmethod
    def get_all_versions():
        all_checkpoints = CheckpointManager.get_checkpoints()
        return [int(f) for f in all_checkpoints]

    @staticmethod
    def save(name, content, append_to_last=True):
        if not os.path.exists(CheckpointManager.basePath):
            os.makedirs(CheckpointManager.basePath)
        all_checkpoints = CheckpointManager.get_checkpoints()
        all_versions = [int(f) for f in all_checkpoints]
        if len(all_versions) == 0:
            next_version = 1
        elif append_to_last is None:
            next_version = max(all_versions)
        else:
            next_version = max(all_versions) + 1
        if not os.path.exists(f"./{CheckpointManager.basePath}/{next_version}"):
            os.makedirs(f"./{CheckpointManager.basePath}/{next_version}")
        with open(f"./{CheckpointManager.basePath}/{next_version}/{name}.pkl", 'wb') as f:
            pickle.dump(content, f)

    @staticmethod
    def load(name, version=None):
        if version is None:
            all_checkpoints = CheckpointManager.get_checkpoints()
            all_versions = [int(f) for f in all_checkpoints]
            version = max(all_versions)
        with open(f"./{CheckpointManager.basePath}/{version}/{name}.pkl", 'rb') as f:
            return pickle.load(f)
