import pandas as pd
from data_access_v3 import DataAccess

CSV_NAME = 'datasets/flights2/sample.csv'
SCHEMA = 'dbsaqp'
TABLE = 'flights2'
ID_COL = '_id'

if __name__ == '__main__':
    csv_data = pd.read_csv(CSV_NAME, index_col=False, delimiter=',')
    total_rows = len(csv_data.index)

    DataAccess.update(f'DROP TABLE IF EXISTS {TABLE}')
    DataAccess.update(f'CREATE TABLE IF NOT EXISTS {TABLE} ('
                      f'{ID_COL} INT NOT NULL AUTO_INCREMENT, '
                      f'YEAR_DATE integer, '
                      f'UNIQUE_CARRIER varchar(100), '
                      f'ORIGIN varchar(100), '
                      f'ORIGIN_STATE_ABR varchar(2), '
                      f'DEST varchar(100), '
                      f'DEST_STATE_ABR varchar(2), '
                      f'DEP_DELAY decimal, '
                      f'TAXI_OUT decimal, '
                      f'TAXI_IN decimal, '
                      f'ARR_DELAY decimal,'
                      f'AIR_TIME decimal,'
                      f'DISTANCE decimal,'
                      f'PRIMARY KEY ({ID_COL}))')

    DataAccess.update(f'CREATE INDEX YEAR_DATE_idx ON {TABLE}(YEAR_DATE);')
    DataAccess.update(f'CREATE INDEX UNIQUE_CARRIER_idx ON {TABLE}(UNIQUE_CARRIER);')
    DataAccess.update(f'CREATE INDEX ORIGIN_idx ON {TABLE}(ORIGIN);')
    DataAccess.update(f'CREATE INDEX ORIGIN_STATE_ABR_idx ON {TABLE}(ORIGIN_STATE_ABR);')
    DataAccess.update(f'CREATE INDEX DEST_idx ON {TABLE}(DEST);')
    DataAccess.update(f'CREATE INDEX DEST_STATE_ABR_idx ON {TABLE}(DEST_STATE_ABR);')
    DataAccess.update(f'CREATE INDEX DEP_DELAY_idx ON {TABLE}(DEP_DELAY);')
    DataAccess.update(f'CREATE INDEX TAXI_OUT_idx ON {TABLE}(TAXI_OUT);')
    DataAccess.update(f'CREATE INDEX TAXI_IN_idx ON {TABLE}(TAXI_IN);')
    DataAccess.update(f'CREATE INDEX ARR_DELAY_idx ON {TABLE}(ARR_DELAY);')
    DataAccess.update(f'CREATE INDEX AIR_TIME_idx ON {TABLE}(AIR_TIME);')
    DataAccess.update(f'CREATE INDEX DISTANCE_idx ON {TABLE}(DISTANCE);')

    print(f'Finished creating table')

    counter = 0
    with open('datasets/flights2/create_flights2.sql', 'a') as f:
        for i, row in csv_data.iterrows():
                sql = "INSERT INTO {0} (YEAR_DATE, UNIQUE_CARRIER, ORIGIN, ORIGIN_STATE_ABR, DEST, DEST_STATE_ABR, DEP_DELAY, TAXI_OUT, TAXI_IN, ARR_DELAY, AIR_TIME, DISTANCE) VALUES ({1}, '{2}', '{3}', '{4}', '{5}', '{6}', {7}, {8}, {9}, {10}, {11}, {12});".format(
                    TABLE,
                    *tuple(row))
                # print(sql)
                f.write(f'{sql}\n')
                # DataAccess.update(sql)
                counter += 1
                print(f"{counter}/{total_rows} rows inserted")

    #  To start from 0
    # DataAccess.update(f'UPDATE {TABLE} SET {ID_COL} = {ID_COL} -1 WHERE {ID_COL} > 0')
