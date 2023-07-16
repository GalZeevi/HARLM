import tqdm
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
import time
import numpy as np

SQL_FILE = 'datasets/flights2/generated_queries/selection/flights_select_queries_min.sql'

if __name__ == "__main__":
    with open(SQL_FILE) as file:
        queries = [line.rstrip() for line in file]

    N = len(queries)

    start = time.time()
    print(f'############### Initialising table details... ###############')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    pivot = ConfigManager.get_config('queryConfig.pivot')
    T = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')
    print(f'Initialising table details took: {round(time.time() - start, 2)} sec')

    w_prev = None
    w_curr = np.zeros(T)
    report_interval = 5
    diff_avg = 0.

    for i in tqdm.trange(len(queries)):
        w_prev = np.copy(w_curr)
        query_res = DataAccess.select(queries[i])
        w_curr[np.array(query_res)] += 1
        # print((w_curr - w_prev)/N)

        if i > 1:
            diff = np.average(abs((w_curr / i) - (w_prev / (i - 1))))
            if i % report_interval > 0:
                diff_avg += diff
            if i % report_interval == 0:
                print(f'=============== diff={diff_avg / report_interval} ===============')
                diff_avg = 0
