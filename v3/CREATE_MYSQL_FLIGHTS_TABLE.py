import pandas as pd
from numpy import Inf
from tqdm import tqdm

from data_access_v3 import DataAccess

FROM_YEAR = 2009
TO_YEAR = 2010
DATASETS_FOLDER = './datasets/flights'
SCHEMA = 'datasets'
TABLE = f'flights_{FROM_YEAR}_{TO_YEAR}'
PIVOT_COL = '_id'
COLUMNS = ['FL_DATE', 'OP_CARRIER', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY',
           'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY',
           'CANCELLED', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE']
DATE_COLS = [COLUMNS[0]]
VARCHAR_COLS = [COLUMNS[1], COLUMNS[3], COLUMNS[4]]
MAX_ROWS = 5000


def get_db_type(col):
    if col in DATE_COLS:
        return 'DATE'
    if col in VARCHAR_COLS:
        return 'VARCHAR(25)'
    if col in COLUMNS:
        return 'DECIMAL(12,2)'
    else:
        return 'INT'


def get_db_string(col_num, offset):
    if COLUMNS[col_num] in DATE_COLS:
        return "CAST('{$}' AS DATE)".replace('$', f'{col_num + offset}')
    if COLUMNS[col_num] in VARCHAR_COLS:
        return "'{$}'".replace('$', f'{col_num + offset}')
    else:
        return "{$}".replace('$', f'{col_num + offset}')


if __name__ == '__main__':
    counter = 0
    DataAccess.update(f'DROP TABLE IF EXISTS {SCHEMA}.{TABLE}')
    create_table_columns = [f'{col} {get_db_type(col)}' for col in COLUMNS]
    DataAccess.update(f'CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} ('
                      f'{PIVOT_COL} INT NOT NULL, '
                      f'{",".join(create_table_columns)}, '
                      f'PRIMARY KEY ({PIVOT_COL}))')

    for year in range(FROM_YEAR, TO_YEAR + 1):
        csv_path = f'{DATASETS_FOLDER}/{year}.csv'
        print(f'Start working on {csv_path}!')
        csv_data = pd.read_csv(csv_path, index_col=False, delimiter=',')
        csv_data = csv_data[COLUMNS]
        csv_data.dropna(inplace=True)

        pbar = tqdm(total=MAX_ROWS if MAX_ROWS < Inf else len(csv_data.index))
        for i, row in csv_data.iterrows():
            if counter <= MAX_ROWS:
                values_part = ",".join(['{0}', *[get_db_string(col_num, 1) for col_num in range(len(row))]])
                replacement_string = f"INSERT INTO {SCHEMA}.{TABLE} ({','.join([PIVOT_COL, *COLUMNS])}) " \
                                     f"VALUES ({values_part})"
                sql = replacement_string.format(counter, *tuple(row))
                # print(sql)
                DataAccess.update(sql)
                # print("Record inserted")
                counter += 1
                pbar.update(1)
            else:
                print('done!')
                break
