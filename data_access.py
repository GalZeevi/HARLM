import pandas as pd
import sqlalchemy
from config_manager import ConfigManager
from consts import DBTypes


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
            DataAccess.connect()

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
    def connect():
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

    def update(self, query):
        DataAccess.conn.execute(query)

    def select(self, query):
        answer = []
        for row in DataAccess.conn.execute(query):
            answer.append(row2dict(row))
        return answer

    def select_one(self, query):
        return self.select(query)[0]

    def select_to_df(self, query):
        return pd.read_sql_query(query, self.conn)

    def save_query_to_csv(self, query, csv_path):
        df = self.select_to_df(query)
        df.to_csv(csv_path, index=False)
