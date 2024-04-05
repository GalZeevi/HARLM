import argparse
import multiprocessing
import os.path
import re
from copy import copy

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

import train_test_utils
from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from ids_to_tuples import select_tuples
from ipf import main as ipf
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
        "--num_queries_to_execute", type=int, default=0,
        help="How many queries to execute (for train only)"
    )
    parser.add_argument('--all_queries', action='store_true', default=False)

    parser.add_argument("--sample_path", type=str, default=None, help="A sample to evaluate")

    parser.add_argument('--evaluate', action='store_true', default=False)

    parser.add_argument('--aqp', action='store_true', default=False)
    parser.add_argument('--use_ipf', action='store_true', default=True)
    parser.add_argument("--full_data_len", type=int, default=None,
                        help="Full data length for IPF")
    parser.add_argument("--marginals_idx", type=str, default=None,
                        help="Marginals ids for IPF of the format: '7, 8, 10, 11, (7, 10), (8, 10), (10, 11)'")
    parser.add_argument("--marginals_folder", type=str, default='flights',
                        help="the folder under /ipf/marginals to read the marginal files from")

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}", flush=True)
    return args


def executed_queries_path(checkpoint):
    return f'{CheckpointManager.basePath}/{checkpoint}/executed_queries.txt'


def aqp_queries_path(checkpoint):
    return f'{CheckpointManager.basePath}/{checkpoint}/aqp_queries.txt'


def get_test_sqls(checkpoint):
    with open(executed_queries_path(checkpoint), 'r') as file:
        return file.readlines()[:ConfigManager.get_config('samplerConfig.testSize')]


def get_test_aqp_sqls(checkpoint):
    with open(aqp_queries_path(checkpoint), 'r') as file:
        return file.readlines()[:ConfigManager.get_config('samplerConfig.testSize')]


def get_train_sqls(checkpoint):
    with open(executed_queries_path(checkpoint), 'r') as file:
        return file.readlines()[ConfigManager.get_config('samplerConfig.testSize'):]


def choose_queries_and_execute(queries_file_path, num_train_queries_to_choose, checkpoint, aqp=False):
    with open(queries_file_path) as file:
        queries = [line.rstrip() for line in file]

    if aqp:
        aqp_test_queries = copy(queries)[:ConfigManager.get_config('samplerConfig.testSize')]
        with open(aqp_queries_path(checkpoint), 'w') as file:
            for q in aqp_test_queries:
                file.write("%s\n" % q)

        for i in range(len(queries)):
            queries[i] = re.sub(re.compile(r"SELECT.+FROM", re.IGNORECASE), "SELECT _id FROM", queries[i])

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


def modify_sql_aggregate(query_to_run):
    weight_str = 'weight'
    if query_to_run.lower().find('count') != -1:
        return query_to_run.replace("COUNT(*)", "SUM({:s})".format(weight_str))
    if query_to_run.lower().find('avg') != -1:
        agg_column = re.search(re.compile(r"AVG\(([^)]+)\)", re.IGNORECASE), query_to_run).group(1)
        return query_to_run.replace("AVG({:s})".format(agg_column),
                                    "SUM({0:s}*{1:s})/SUM({0:s})".format(weight_str, agg_column))
    if query_to_run.lower().find('sum') != -1:
        agg_column = re.search(re.compile(r"SUM\(([^)]+)\)", re.IGNORECASE), query_to_run).group(1)
        return query_to_run.replace("SUM({:s})".format(agg_column), "SUM({0:s}*{1:s})".format(weight_str, agg_column))


def run_sql_on_df(sql: str, df_name, conn):
    if sql.find(
            f'{ConfigManager.get_config("queryConfig.schema")}.{ConfigManager.get_config("queryConfig.table")}') != -1:
        new_sql = sql.replace(
            f'{ConfigManager.get_config("queryConfig.schema")}.{ConfigManager.get_config("queryConfig.table")}',
            df_name)
    else:
        new_sql = sql.replace(ConfigManager.get_config("queryConfig.table"), df_name)
    print(new_sql, flush=True)
    return pd.read_sql_query(new_sql, conn)


def get_non_group_relative_error(i, query, use_weight, engine,  df_name):
    query_truth = DataAccess.select_one(query)
    if query_truth is not None:
        query_truth = float(query_truth)

    if query_truth == 0:
        print(f'############# query no. {i} returned zero results #############')
        query_err = np.nan
    else:
        new_test_query = modify_sql_aggregate(test_query) if use_weight else test_query

        query_pred = run_sql_on_df(new_test_query, df_name, engine).iloc[0].item()
        print(f'Predicted: {query_pred}, Actual: {query_truth}')
        query_err = 5.
        if query_pred is not None:
            query_err = float(abs(query_pred - query_truth) / query_truth)
            query_err = abs(query_err)
    return query_err


if __name__ == "__main__":
    args = get_args()

    # Create checkpoint
    if args.checkpoint == "new":
        checkpoint = CheckpointManager.start_new_version()
    else:
        checkpoint = int(args.checkpoint)

    if not args.evaluate:
        assert args.num_queries_to_execute > 0 or args.all_queries
        process1 = multiprocessing.Process(target=choose_queries_and_execute, args=(args.queries_file,
                                                                                    'all' if args.all_queries else
                                                                                    args.num_queries_to_execute,
                                                                                    checkpoint,
                                                                                    args.aqp))
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
        if not args.aqp:  # Evaluate spj
            train_query_scores = get_score2(sample_ids, 'train', checkpoint, average=False)
            print(train_query_scores)
            test_query_scores = get_score2(sample_ids, 'test', checkpoint, average=False)
            print(test_query_scores)

            test_results = train_test_utils.get_test_queries(checkpoint)
            qs = QuerySimilarity()

            for i, test_query in enumerate(get_test_sqls(checkpoint)):
                print(f'TESTING SPJ query no. {i}.')
                query_similarity_to_train = [qs.sim(test_query, train_query) for train_query in
                                             get_train_sqls(checkpoint)]
                predicted_score = max(query_similarity_to_train) * train_query_scores[
                    np.argmax(query_similarity_to_train)]
                actual_score = test_query_scores[i]
                print(f'TEST query no. {i}. Predicted score: {predicted_score}. Actual score: {actual_score}')
        else:  # Evaluate aqp
            marginals_idx = {4, 5, 6, 7, (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)}  # TODO: get from args
            # marginals_idx = {7, 11, (8, 10), (10, 11)}  # TODO: get from args

            ipf_path = f'{CheckpointManager.basePath}/{checkpoint}/{args.sample_path}_ipf.csv'
            if not (os.path.isfile(ipf_path)):
                tuples = select_tuples(sample_ids)
                sample_df = pd.DataFrame(tuples, index=[t['_id'] for t in tuples])
                sample_ipf_df = ipf(args.marginals_folder, marginals_idx, sample_df, args.full_data_len)
                sample_ipf_df.to_csv(ipf_path, index=False)

            engine = create_engine(f'sqlite:///{CheckpointManager.basePath}/{checkpoint}/saqp_aqp_{checkpoint}')
            # TODO: creating the table using to_sql does not work with float(s)!
            pd.read_csv(ipf_path).to_sql(args.sample_path, engine, if_exists='replace', index=False)

            csv_data = {}
            all_errs = []
            for i, test_query in enumerate(get_test_aqp_sqls(checkpoint)):
                print(f'TESTING AQP query no. {i}.')
                new_test_query = modify_sql_aggregate(test_query)
                query_truth = DataAccess.select_one(test_query)
                if 'group by' in test_query.lower():  # Group By query
                    err = np.nan
                    print('GROUP BY is currently not supported for AQP')
                else:  # simple query
                    err = get_non_group_relative_error(i, test_query, args.use_ipf, engine, args.sample_path)
                all_errs += [err]
                csv_data[f'q{i}'] = err
                print(f'TEST query no. {i}. Relative error with ipf: {err}')
            csv_data['err_avg'] = np.average(all_errs)
            csv_data['err_std'] = np.std(all_errs)

            results_file_name = f'{args.sample_path}_{"ipf" if args.use_ipf else "no_ipf"}_results.csv'
            pd.DataFrame([csv_data]) \
                .to_csv(f'{CheckpointManager.basePath}/{checkpoint}/{results_file_name}',
                        index=False, mode='a', header=False)
