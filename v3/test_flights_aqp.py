from tqdm import tqdm

from checkpoint_manager_v3 import CheckpointManager
from data_access_v3 import DataAccess

AQP_QUERIES_PATH = 'datasets/flights2/deepdb_aqp_queries.sql'
# SAMPLE_PATH = '1000_CHOOSE_K_PPO_2023-06-17-14-06-14/sample'
# SAMPLE_PATH = '5000_CHOOSE_K_PPO_2023-06-17-16-02-36/sample'
# SAMPLE_PATH = '10000_CHOOSE_K_PPO_2023-06-17-10-42-05/sample'
SAMPLE_PATH = '50000_CHOOSE_K_A3C_2023-06-20-00-03-05/sample'
VERSION = 11

sample_ids = CheckpointManager.load(name=SAMPLE_PATH, version=VERSION)[0]
sample_ids_db_fmt = ','.join([str(idx) for idx in sample_ids])

with open(AQP_QUERIES_PATH) as file:
    queries = [line.rstrip() for line in file]
DataAccess()

total_query_score = 0.
for i, query in tqdm(enumerate(queries)):
    # print(f'Processing query: {query}')
    print(f'Processing query no: {i}')
    query_sample = query.replace('WHERE', f'WHERE _id IN ({sample_ids_db_fmt}) AND')
    if 'GROUP BY' in query:
        query_truth = DataAccess.select(query)
        if len(query_truth) == 0:
            continue
        query_pred = DataAccess.select(query_sample)
        truth_num_groups = len(query_truth)
        pred_num_groups = len(query_pred)

        query_score = 0.
        for pred_group in query_pred:
            pred_col_value = pred_group['col']
            pred_agg_value = float(pred_group['agg'])
            truth_agg_value = None
            for truth_group in query_truth:
                if pred_col_value == truth_group['col']:
                    truth_agg_value = float(truth_group['agg'])
                    query_score += abs(pred_agg_value - truth_agg_value) / truth_agg_value
                    break
            if truth_agg_value is None:  # group from pred is not in truth
                pred_num_groups -= 1

        query_score += (truth_num_groups - pred_num_groups)
        query_score /= truth_num_groups
    else:
        query_truth = DataAccess.select_one(query)
        query_pred = DataAccess.select_one(query_sample)
        query_score = float(abs(query_pred - query_truth) / query_truth)
    print(f'############# query no. {i} score: {query_score} #############')
    total_query_score += query_score

# total_query_score /= len(queries)
# print(f'############# total query score: {total_query_score} #############')
