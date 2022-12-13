import pandas as pd
from data_access import DataAccess

CSV_NAME = 'datasets\\austin_bikes_final.csv'
SCHEMA = 'datasets'
LIMIT = 1000
TABLE = f'austin_bikes_{LIMIT}'
ID_COL = '_id'

if __name__ == '__main__':
    csv_data = pd.read_csv(CSV_NAME, index_col=False, delimiter=',')
    data_access = DataAccess()

    data_access.update(f'DROP TABLE IF EXISTS {SCHEMA}.{TABLE}')
    data_access.update(f'CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} ('
                       f'_id INT NOT NULL AUTO_INCREMENT, '
                       f'bikeid VARCHAR(50), '
                       f'start_time DATETIME, '
                       f'duration_minutes INT, '
                       f'start_station_name VARCHAR(100), '
                       f'end_station_name VARCHAR(100), '
                       f'start_latitude DECIMAL(10,6), '
                       f'start_longitude DECIMAL(10,6), '
                       f'end_latitude DECIMAL(10,6), '
                       f'end_longitude DECIMAL(10,6), '
                       f'PRIMARY KEY (_id))')

    columns = data_access.select(f"SELECT column_name AS col FROM information_schema.columns "
                                  f"WHERE table_schema='{SCHEMA}' AND table_name='{TABLE}' "
                                  f"AND column_name  <> '{ID_COL}'")

    counter = 0
    for i, row in csv_data.sample(frac=(LIMIT / len(csv_data.index))).iterrows():
        if counter <= LIMIT:
            sql = "INSERT INTO {0}.{1} (bikeid, start_time, duration_minutes, start_station_name, end_station_name, start_latitude, start_longitude, end_latitude, end_longitude) VALUES ({2}, CAST('{3}' AS DATETIME), {4}, '{5}', '{6}', {7}, {8}, {9}, {10})".format(
                SCHEMA, TABLE,
                *tuple(row))
            print(sql)
            data_access.update(sql)
            print("Record inserted")
            counter += 1
        else:
            break
