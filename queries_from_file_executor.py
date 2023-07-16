import numpy as np
from data_access_v3 import DataAccess
from checkpoint_manager_v3 import CheckpointManager
from tqdm import tqdm
from random import shuffle


def start():
    with open('datasets/mas/queries.sql') as file:
        queries = [line.rstrip() for line in file]

    pbar = tqdm(total=len(queries))
    BATCH_NUM = 1
    batches = np.array_split(queries, BATCH_NUM)
    saved_results = 0
    print(f'Starting to process: {len(batches)} batches')
    for batch in batches:
        results = []
        queries = []
        for query in batch:
            try:
                query_result = DataAccess.select(query)
                if len(query_result) > 0:
                    results.append(query_result)
                    queries.append(query)
            except:
                print(f'error in query: [{query}]')
            pbar.update(1)
        CheckpointManager.save(f'queries_{saved_results}-{saved_results + len(queries)}', queries)
        CheckpointManager.save(f'results_{saved_results}-{saved_results + len(queries)}', results)
        saved_results += len(queries)


def shuffle_queries(version=8, steps=(0, 18, 35)):
    queries = np.concatenate((CheckpointManager.load(f'queries_{steps[0]}-{steps[1]}', version),
                              CheckpointManager.load(f'queries_{steps[1]}-{steps[2]}', version)))
    results = CheckpointManager.load(f'results_{steps[0]}-{steps[1]}', version) + \
              CheckpointManager.load(f'results_{steps[1]}-{steps[2]}', version)

    queries_with_results = [*zip(queries, results)]
    shuffle(queries_with_results)
    queries = [x[0] for x in queries_with_results]
    results = [x[1] for x in queries_with_results]

    CheckpointManager.save(f'queries_{steps[0]}-{steps[1]}', queries[:steps[1]])
    CheckpointManager.save(f'results_{steps[0]}-{steps[1]}', results[:steps[1]])
    CheckpointManager.save(f'queries_{steps[1]}-{steps[2]}', queries[steps[1]:])
    CheckpointManager.save(f'results_{steps[1]}-{steps[2]}', results[steps[1]:])


def shuffle_queries2(version=11, steps=(0, 11)):
    queries = CheckpointManager.load(f'queries_{steps[0]}-{steps[1]}', version)
    results = CheckpointManager.load(f'results_{steps[0]}-{steps[1]}', version)

    queries_with_results = [*zip(queries, results)]
    shuffle(queries_with_results)
    queries = [x[0] for x in queries_with_results]
    results = [x[1] for x in queries_with_results]

    CheckpointManager.save(f'queries_{steps[0]}-{steps[1]}', queries)
    CheckpointManager.save(f'results_{steps[0]}-{steps[1]}', results)


if __name__ == '__main__':
    start()
