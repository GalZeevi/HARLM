from sentence_transformers import SentenceTransformer
from data_access_v3 import DataAccess
from config_manager_v3 import ConfigManager
import numpy as np


def get_table_details():
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    pivot = ConfigManager.get_config('queryConfig.pivot')
    return schema, table, pivot


def get_numerical_cols():
    schema, table, pivot = get_table_details()
    numeric_data_types = ['smallint', 'integer', 'int', 'bigint', 'decimal', 'numeric', 'real', 'double precision',
                          'smallserial', 'serial', 'bigserial', 'float']
    db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
    columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                f"AND column_name <> '{pivot}'")
    return columns


def get_non_numerical_cols():
    schema, table, pivot = get_table_details()
    numeric_data_types = ['smallint', 'integer', 'int', 'bigint', 'decimal', 'numeric', 'real', 'double precision',
                          'smallserial', 'serial', 'bigserial', 'float']
    db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
    columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                f"AND data_type NOT IN ({' , '.join(db_formatted_data_types)}) " +
                                f"AND column_name <> '{pivot}'")
    return columns


class Embedding:

    def __init__(self):
        self.numeric_cols = get_numerical_cols()
        self.categorical_cols = get_non_numerical_cols()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_words = 300
        self.embedding_size = 384

    def tuples2vector(self, tuples):
        num_numeric_cols = len(self.numeric_cols)
        num_categorical_cols = len(self.categorical_cols)
        max_words_per_categ_col = int(
            (self.max_words - (num_numeric_cols + num_categorical_cols) - num_numeric_cols) / num_categorical_cols)
        _, _, pivot = get_table_details()
        trunc_tuples = \
            [{k: (v if k in self.numeric_cols else f'{v}'[:max_words_per_categ_col]) for k, v in tup.items() if k != pivot} for tup in
             tuples]
        tuples_as_string = [', '.join(f'{key}: {value}' for key, value in tup.items()) for tup in trunc_tuples]
        # print(tuples_as_string)
        embeddings = self.model.encode(tuples_as_string)
        # print(embeddings.shape)
        return embeddings


if __name__ == '__main__':
    tuples = DataAccess.select('SELECT * FROM new_imdb.join_title_companies_keyword ORDER BY RAND() LIMIT 10')
    print(tuples)
    embedding = Embedding()
    print(embedding.tuples2vector(tuples))
