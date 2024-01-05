import argparse
import multiprocessing

import numpy as np
from tqdm import tqdm

import train_test_utils
from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing
from query_similarity import QuerySimilarity
from score_calculator import get_score2


def get_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=str, default="new",
        help="Checkpoint folder to use"
    )

    parser.add_argument(
        "--queries_file", type=str, default='queries.txt',
        help="Path to queries file"
    )

    parser.add_argument(
        "--num_queries_to_execute", type=int, default=ConfigManager.get_config('samplerConfig.testSize'),
        help="How many queries to execute (train only)"
    )
    parser.add_argument('--all_queries', action='store_true', default=False)

    parser.add_argument(
        "--sample_path", type=str, default=None,
        help="A sample to evaluate"
    )

    parser.add_argument('--evaluate', action='store_true', default=False)

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}", flush=True)
    return args


def executed_queries_path(checkpoint):
    return f'{CheckpointManager.basePath}/{checkpoint}/executed_queries.txt'


def get_test_sqls(checkpoint):
    with open(executed_queries_path(checkpoint), 'r') as file:
        return file.readlines()[:ConfigManager.get_config('samplerConfig.testSize')]


def get_train_sqls(checkpoint):
    with open(executed_queries_path(checkpoint), 'r') as file:
        return file.readlines()[ConfigManager.get_config('samplerConfig.testSize'):]


def choose_queries_and_execute(queries_file_path, num_train_queries_to_choose, checkpoint):
    with open(queries_file_path) as file:
        queries = [line.rstrip() for line in file]

    qs = QuerySimilarity()

    results = []
    test_queries = queries[:ConfigManager.get_config('samplerConfig.testSize')]
    train_queries = queries[ConfigManager.get_config('samplerConfig.testSize'):]
    if num_train_queries_to_choose == 'all':
        chosen_queries = [*test_queries, *train_queries]
    else:
        chosen_queries = test_queries + qs.k_means(train_queries, num_train_queries_to_choose)

    for query in tqdm(chosen_queries):
        try:
            query_result = DataAccess.select(query)
            results.append(query_result)
            queries.append(query)
        except Exception as e:
            print(f'Error in query: [{query}]. Error: [{str(e)}]')

    CheckpointManager.save(f'results_0-{len(chosen_queries)}', results, version=checkpoint)
    CheckpointManager.save(f'queries_0-{len(chosen_queries)}', chosen_queries, version=checkpoint)
    # Writing the list to a file line by line
    with open(executed_queries_path(checkpoint), 'w') as file:
        for q in chosen_queries:
            file.write("%s\n" % q)


def init_preprocessing(checkpoint):
    Preprocessing.init(checkpoint)


if __name__ == "__main__":
    args = get_args()

    # Create checkpoint
    if args.checkpoint == "new":
        checkpoint = CheckpointManager.start_new_version()
    else:
        checkpoint = int(args.checkpoint)

    if not args.evaluate:
        assert args.num_queries_to_execute > 0 or args.num_queries_to_execute == 'all'
        process1 = multiprocessing.Process(target=choose_queries_and_execute, args=(args.queries_file,
                                                                                    'all' if args.all_queries else
                                                                                    args.num_queries_to_execute,
                                                                                    checkpoint))
        process2 = multiprocessing.Process(target=init_preprocessing, args=(checkpoint,))

        # Start both processes
        process1.start()
        process2.start()

        # Wait for both processes to finish
        process1.join()
        process2.join()

    # TODO: here you should go and use ray_sampler and measure the train time (for example) - for now it is manually

    # Evaluate
    if args.evaluate and args.sample_path:
        sample_ids = CheckpointManager.load(args.sample_path, args.checkpoint)[0]
        train_query_scores = get_score2(sample_ids, 'train', checkpoint, average=False)
        print(train_query_scores)
        test_query_scores = get_score2(sample_ids, 'test', checkpoint, average=False)
        print(test_query_scores)

        test_results = train_test_utils.get_test_queries(checkpoint)
        qs = QuerySimilarity()

        for i, test_query in enumerate(get_test_sqls(checkpoint)):
            query_similarity_to_train = [qs.sim(test_query, train_query) for train_query in get_train_sqls(checkpoint)]
            predicted_score = max(query_similarity_to_train) * train_query_scores[np.argmax(query_similarity_to_train)]
            actual_score = test_query_scores[i]
            print(f'TEST query no. {i}. Predicted score: {predicted_score}. Actual score: {actual_score}')
