import pandas as pd
import sqlalchemy
from config_manager import ConfigManager


def row2dict(row):
    d = {}
    for field in row._fields:
        d[field] = row[field]
    return d


class DataAccess:
    conn = None

    def __init__(self):
        if DataAccess.conn is None:
            DataAccess.connect()

    @staticmethod
    def connect():
        con = None
        try:
            params = ConfigManager.get_config('dbConfig')

            url = 'postgresql://{}:{}@{}:{}/{}'
            url = url.format(params['user'], params['password'], params['host'],
                             params['port'], params['database'])

            print('Connecting to the PostgreSQL database...')
            con = sqlalchemy.create_engine(url, client_encoding='utf8')
            print(f'Connected to {params["host"]}:{params["database"]}.')

        except Exception as error:
            print(error)

        DataAccess.conn = con

    @staticmethod
    def disconnect():
        if DataAccess.conn:
            DataAccess.conn.dispose()
        print('Database connection closed.')

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
