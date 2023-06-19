import numpy as np
from checkpoint_manager_v3 import CheckpointManager
import argparse


def get_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=int, default=CheckpointManager.get_max_version(), help="The checkpoint to use."
    )

    parser.add_argument(
        "--name", type=str, default=CheckpointManager.get_max_version(), help="The trial name."
    )

    cli_args = parser.parse_args()
    print(f"Running with following CLI args: {cli_args}")
    return cli_args


args = get_args()

if __name__ == '__main__':
    all_scores = CheckpointManager.load(name=f'{args.name}/trial_scores', version=args.checkpoint)
    test_scores = [scores_dict.get('test_score', 0) for scores_dict in all_scores]
    threshold_scores = [scores_dict.get('test_threshold_score', 0) for scores_dict in all_scores]
    diversity_scores = [scores_dict.get('diversity_score', 0) for scores_dict in all_scores]
    print(f'############### Sample score: '
          f'[min: {round(np.min(test_scores), ndigits=4)}, '
          f'avg: {round(np.average(test_scores), ndigits=4)}, '
          f'max: {round(np.max(test_scores), ndigits=4)}] ###############')
    print(f'############### Sample 0.25-score: '
          f'[min: {round(np.min(threshold_scores), ndigits=4)}, '
          f'avg: {round(np.average(threshold_scores), ndigits=4)}, '
          f'max: {round(np.max(threshold_scores), ndigits=4)}] ###############')
    print(f'############### Sample diversity score: '
          f'[min: {round(np.min(diversity_scores), ndigits=4)}, '
          f'avg: {round(np.average(diversity_scores), ndigits=4)}, '
          f'max: {round(np.max(diversity_scores), ndigits=4)}] ###############')
