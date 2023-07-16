from tqdm import tqdm
import os
import csv
import numpy as np
import pandas as pd

from checkpoint_manager_v3 import CheckpointManager
from data_access_v3 import DataAccess

AQP_QUERIES_PATH = 'datasets/flights2/generated_queries/aqp/flights_aqp_queries_test.sql'
OUTFILE_PATH = 'datasets/flights2/generated_queries'
SAMPLE_PATH = '10000_CHOOSE_K_PPO_2023-07-07-19-49-37/sample'
VERSION = 12
DEEPDB_FMT = True

sample_ids = CheckpointManager.load(name=SAMPLE_PATH, version=VERSION)[0]
sample_ids_db_fmt = ','.join([str(idx) for idx in sample_ids])

outfile_rows = []


def handle_query(query):
    print(f'Processing query no: {i}')
    query_sample = query.replace('WHERE', f'WHERE _id IN ({sample_ids_db_fmt}) AND')
    if 'GROUP BY' in query:
        query_truth = DataAccess.select(query)
        if len(query_truth) == 0:
            return
        query_pred = DataAccess.select(query_sample)
        truth_num_groups = len(query_truth)
        pred_num_groups = len(query_pred)

        query_err = 0.
        for pred_group in query_pred:
            pred_col_value = pred_group['col']
            pred_agg_value = float(pred_group['agg'])
            truth_agg_value = None
            for truth_group in query_truth:
                if pred_col_value == truth_group['col']:
                    truth_agg_value = float(truth_group['agg'])
                    query_err += abs(pred_agg_value - truth_agg_value) / truth_agg_value
                    break
            if truth_agg_value is None:  # group from pred is not in truth
                pred_num_groups -= 1

        if DEEPDB_FMT is False:
            query_err += (truth_num_groups - pred_num_groups)
        query_err /= truth_num_groups
    else:
        query_truth = DataAccess.select_one(query)
        if query_truth is not None:
            query_truth = float(query_truth)
        query_pred = DataAccess.select_one(query_sample)
        if query_pred is not None:
            query_pred = float(query_pred)
        if query_truth == 0:
            print(f'############# query no. {i} returned zero results #############')
            query_err = np.nan
        else:
            if query_pred is None:
                query_pred = 0.
            query_err = float(abs(query_pred - query_truth) / query_truth)
            if DEEPDB_FMT:
                query_err = abs(query_err)
            print(f'############# query no. {i} error: {query_err} #############')
    outfile_rows.append({'query_no': i,
                         'average_relative_error': query_err,
                         'query': query})


def save_csv(csv_rows, target_csv_path):
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
    print(f"Saving results to {target_csv_path}", flush=True)

    with open(target_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)


def csv_to_scores(csv_path):
    data = pd.read_csv(csv_path)
    data = data[data['average_relative_error'] <= 1.]
    # data['average_relative_error'] = 1 - data['average_relative_error']
    groupbys = data[data['query'].str.contains("GROUP BY")]
    group_sum = groupbys[groupbys['query'].str.contains("SUM")]
    group_avg = groupbys[groupbys['query'].str.contains("AVG")]
    group_cnt = groupbys[groupbys['query'].str.contains("COUNT")]
    non_groupbys = data[~data['query'].str.contains("GROUP BY")]
    nongroup_sum = non_groupbys[non_groupbys['query'].str.contains("SUM")]
    nongroup_avg = non_groupbys[non_groupbys['query'].str.contains("AVG")]
    nongroup_cnt = non_groupbys[non_groupbys['query'].str.contains("COUNT")]

    print('GROUP BY')
    df = group_sum
    print(
        f'op: SUM, rel_score = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = group_avg
    print(
        f'op: AVG, rel_score = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = group_cnt
    print(
        f'op: CNT, rel_score = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')

    print('GROUP BY')
    df = nongroup_sum
    print(
        f'op: SUM, rel_score = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = nongroup_avg
    print(
        f'op: AVG, rel_score = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = nongroup_cnt
    print(
        f'op: CNT, rel_score = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')


with open(AQP_QUERIES_PATH) as file:
    queries = [line.rstrip() for line in file]
DataAccess()

total_query_score = 0.
for i, query in tqdm(enumerate(queries)):
    try:
        handle_query(query)
    except Exception as e:
        print(f'Error in query {i}')
save_csv(outfile_rows, OUTFILE_PATH)

if __name__ == "__main__":
    csv_to_scores('datasets/flights2/generated_queries/aqp/gaqp_model3_results.csv')
