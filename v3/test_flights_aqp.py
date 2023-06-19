from tqdm import tqdm
from data_access_v3 import DataAccess
from checkpoint_manager_v3 import CheckpointManager

AQP_QUERIES_PATH = 'datasets/flights2/deepdb_aqp_queries.sql'
# SAMPLE_PATH = '1000_CHOOSE_K_PPO_2023-06-17-14-06-14/sample'
SAMPLE_PATH = '5000_CHOOSE_K_PPO_2023-06-17-16-02-36/sample'
# SAMPLE_PATH = '10000_CHOOSE_K_PPO_2023-06-17-10-42-05/sample'
VERSION = 11

sample_ids = CheckpointManager.load(name=SAMPLE_PATH, version=VERSION)[0]
sample_ids_db_fmt = ','.join([str(idx) for idx in sample_ids])

with open(AQP_QUERIES_PATH) as file:
    queries = [line.rstrip() for line in file]
DataAccess()

total_query_score = 0.
for i, query in tqdm(enumerate(queries)):
    print(f'Processing query: {query}')
    query_sample = query.replace('WHERE', f'WHERE _id IN ({sample_ids_db_fmt}) AND')
    if 'GROUP BY' in query:
        query_truth = DataAccess.select(query)
        if len(query_truth) == 0:
            continue
        query_pred = DataAccess.select(query_sample)
        query_score = 0.
        for group in query_truth:
            col_value = group['col']
            agg_value_truth = float(group['agg'])
            agg_value_pred = None
            for group_pred in query_pred:
                if col_value == group_pred['col']:
                    agg_value_pred = float(group_pred['agg'])
                    query_score += abs(agg_value_pred - agg_value_truth) / abs(agg_value_truth)
                    break
        query_score /= len(query_truth)
    else:
        query_truth = DataAccess.select_one(query)
        query_pred = DataAccess.select_one(query_sample)
        query_score = float(abs(query_pred - query_truth) / abs(query_truth))
    print(f'############# query no. {i} score: {query_score} #############')
    total_query_score += query_score

# total_query_score /= len(queries)
# print(f'############# total query score: {total_query_score} #############')



