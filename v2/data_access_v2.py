import pandas as pd
import sqlalchemy
from config_manager_v2 import ConfigManager
from db_types import DBTypes


def row2dict(row):
    d = {}
    fields = row._fields

    if len(fields) == 1:
        return row._data[0]

    for field in fields:
        d[field] = row[field]
    return d


class DataAccess:
    conn = None

    def __init__(self):
        if DataAccess.conn is None:
            DataAccess._connect()

    @staticmethod
    def _connect_postgres(params):
        url = 'postgresql://{}:{}@{}:{}/{}'.format(params['user'], params['password'],
                                                   params['host'], params['port'], params['database'])

        print(f'Connecting to the postgresql database...')
        return sqlalchemy.create_engine(url, client_encoding='utf8')

    @staticmethod
    def _connect_mysql(params):
        url = 'mysql+pymysql://{}:{}@{}/{}?charset=utf8mb4'.format(params['user'], params['password'],
                                                                   params['host'], params['database'])

        print(f'Connecting to the mysql database...')
        return sqlalchemy.create_engine(url)

    @staticmethod
    def _connect():
        if DataAccess.conn is not None:
            return

        con = None
        try:
            params = ConfigManager.get_config('dbConfig')
            dbType = str.lower(params['type'])

            if DBTypes.IS_POSTGRESQL(dbType):
                con = DataAccess._connect_postgres(params)

            elif DBTypes.IS_MYSQL(dbType):
                con = DataAccess._connect_mysql(params)

            else:
                raise Exception('Unsupported db type! supported types are "postgresql" or "mysql"')

            print(f'Connected to {params["host"]}:{params["database"]}.')

        except Exception as error:
            print(error)

        DataAccess.conn = con

    @staticmethod
    def disconnect():
        if DataAccess.conn:
            DataAccess.conn.dispose()
        print('Database connection closed.')

    @staticmethod
    def update(query):
        if DataAccess.conn is None:
            DataAccess._connect()
        DataAccess.conn.execute(query)

    @staticmethod
    def select(query):
        if DataAccess.conn is None:
            DataAccess._connect()
        answer = []
        for row in DataAccess.conn.execute(query):
            answer.append(row2dict(row))
        return answer

    @staticmethod
    def select_one(query):
        return DataAccess.select(query)[0]

    @staticmethod
    def select_to_df(query):
        if DataAccess.conn is None:
            DataAccess._connect()
        return pd.read_sql_query(query, DataAccess.conn)

    @staticmethod
    def save_query_to_csv(query, csv_path):
        df = DataAccess.select_to_df(query)
        df.to_csv(csv_path, index=False)
