import os
import pickle


class CheckpointManager:
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
    def save(name, content, append_to_last=True):
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
        print(f'Saving checkpoint {name}.pkl to: [./{CheckpointManager.basePath}/{next_version}]')
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
        if not os.path.exists(f"./{CheckpointManager.basePath}/{version}/{name}.pkl"):
            return None
        with open(f"./{CheckpointManager.basePath}/{version}/{name}.pkl", 'rb') as f:
            return pickle.load(f)
