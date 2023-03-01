from tqdm import tqdm
from checkpoint_manager_v3 import CheckpointManager
from score_calculator import get_score

sample_200000 = CheckpointManager.load('200000-500-50000_0.5_mab_sample')[0]

if __name__ == '__main__':
    k_list = [10_000, 50_000, 100_000, 150_000]
    # k_list = [10_000]
    for k in tqdm(k_list):
        k_sample = sample_200000[:k]
        k_score = get_score(k_sample)
        CheckpointManager.save(f'{k}-500-50000_0.5_mab_sample', [k_sample, k_score])
