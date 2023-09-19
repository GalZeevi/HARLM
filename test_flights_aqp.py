from tqdm import tqdm
import os
import csv
import numpy as np
import pandas as pd

from checkpoint_manager_v3 import CheckpointManager
from data_access_v3 import DataAccess
from create_histograms import create_flights_histograms, get_cnt_query_answer, get_group_cnt_query_answer

USE_HISTOGRAM = True
AQP_QUERIES_PATH = 'datasets/flights2/generated_queries/aqp/3/flights_aqp_queries_test.sql'
# AQP_QUERIES_PATH = 'datasets/flights2/generated_queries/aqp/3/test.sql'
OUTFILE_PATH = f'datasets/flights2/generated_queries/aqp/3/' \
               f'flights_asqp_2500_{"" if USE_HISTOGRAM else "no_"}hist_results.csv'
SAMPLE_PATH = 'flights_2500_sample'
VERSION = 21
DEEPDB_FMT = True

sample_ids = CheckpointManager.load(name=SAMPLE_PATH, version=VERSION)
# sample_ids = CheckpointManager.load(name=SAMPLE_PATH, version=VERSION)[0]
sample_ids_db_fmt = ','.join([str(idx) for idx in sample_ids])


def handle_query(i, query, outfile_rows):
    print(f'Processing query no: {i}')
    op = ''
    if 'GROUP BY' in query:
        if 'SUM(' in query:
            op = 'G-SUM'
        elif 'COUNT(' in query:
            op = 'G-CNT'
        elif 'AVG(' in query:
            op = 'G-AVG'
        query_err = handle_groupby_query(i, query)
    else:
        if 'SUM(' in query:
            op = 'SUM'
        elif 'COUNT(' in query:
            op = 'CNT'
        elif 'AVG(' in query:
            op = 'AVG'

        if 'SUM(' in query and USE_HISTOGRAM:
            query_err = handle_simple_sum_query(i, query)
        elif 'COUNT(' in query and USE_HISTOGRAM:
            query_err = handle_simple_cnt_query(i, query)
        else:
            query_err = handle_simple_query(i, query)
    print(f'############# query no. {i} , op: {op}, error: {query_err} #############')
    outfile_rows.append({'query_no': i,
                         'average_relative_error': query_err,
                         'operator': op,
                         'query': query})


def handle_simple_query(i, query):
    query_sample = query.replace('WHERE', f'WHERE _id IN ({sample_ids_db_fmt}) AND')
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
    return query_err


def handle_simple_sum_query(i, query):
    avg_query = query.replace('SUM(', 'AVG(')
    cnt_query = query.replace('SUM(', 'COUNT(')
    query_sample = avg_query.replace('WHERE', f'WHERE _id IN ({sample_ids_db_fmt}) AND')
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

        cnt_ans = get_cnt_query_answer(cnt_query, hist_data)
        query_pred *= cnt_ans
        query_err = float(abs(query_pred - query_truth) / query_truth)
        if DEEPDB_FMT:
            query_err = abs(query_err)
    return query_err


def handle_simple_cnt_query(i, query):
    query_truth = DataAccess.select_one(query)
    if query_truth is not None:
        query_truth = float(query_truth)

    if query_truth == 0:
        print(f'############# query no. {i} returned zero results #############')
        query_err = np.nan
    else:
        query_pred = get_cnt_query_answer(query, hist_data)
        query_err = float(abs(query_pred - query_truth) / query_truth)
        if DEEPDB_FMT:
            query_err = abs(query_err)
    return query_err


def handle_groupby_query(i, query):
    query_sample = query.replace('WHERE', f'WHERE _id IN ({sample_ids_db_fmt}) AND')
    query_truth = DataAccess.select(query)
    if len(query_truth) == 0:
        return
    if 'COUNT(' in query and USE_HISTOGRAM:
        query_pred = get_group_cnt_query_answer(query, hist_data)
    elif 'SUM(' in query and USE_HISTOGRAM:
        cnt_query_pred = get_group_cnt_query_answer(query.replace('SUM(', 'COUNT('), hist_data)
        avg_query_pred = DataAccess.select(query_sample.replace('SUM(', 'AVG('))
        if len(cnt_query_pred) > 0:
            cnt_df = pd.DataFrame(cnt_query_pred)
        else:
            cnt_df = pd.DataFrame([], columns=['col', 'agg'])
        cnt_df['agg'] = cnt_df['agg'].astype(float)
        if len(avg_query_pred) > 0:
            avg_df = pd.DataFrame(avg_query_pred)
        else:
            avg_df = pd.DataFrame([], columns=['col', 'agg'])
        avg_df['agg'] = avg_df['agg'].astype(float)
        query_df = pd.merge(cnt_df, avg_df, how='left', on='col', suffixes=('_cnt', '_avg')).fillna(0)
        query_df['agg'] = query_df['agg_cnt'] * query_df['agg_avg']
        query_df = query_df[['col', 'agg']]
        query_pred = query_df.to_dict('records')
    else:
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
    # if query_err == 0:
    #     print('##############################')
    #     print('pred')
    #     print(pd.DataFrame(query_pred))
    #     print('truth')
    #     print(pd.DataFrame(query_truth))
    #     print('##############################')
    return query_err


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
    # data = data[data['average_relative_error'] <= 1.]
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
        f'op: SUM, rel_error = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = group_avg
    print(
        f'op: AVG, rel_error = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = group_cnt
    print(
        f'op: CNT, rel_error = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')

    print('SIMPLE')
    df = nongroup_sum
    print(
        f'op: SUM, rel_error = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = nongroup_avg
    print(
        f'op: AVG, rel_error = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')
    df = nongroup_cnt
    print(
        f'op: CNT, rel_error = {df.loc[:, "average_relative_error"].mean()}, std = {df.loc[:, "average_relative_error"].std()}')


if __name__ == "__main__":
    DataAccess()
    with open(AQP_QUERIES_PATH) as file:
        queries = [line.rstrip() for line in file]

    if USE_HISTOGRAM:
        hist_data = create_flights_histograms(num_bins=100)
    else:
        hist_data = None

    total_query_score = 0.
    outfile_rows = []
    for i, query in tqdm(enumerate(queries)):
        try:
            handle_query(i, query, outfile_rows)
        except Exception as e:
            print(f'Error in query {i}: {str(e)}, {query}')
    save_csv(outfile_rows, OUTFILE_PATH)
    csv_to_scores(OUTFILE_PATH)
    # csv_to_scores(f'../deepdb-public/baselines/aqp/results/deepDB/flights5M_deepdb.csv')
