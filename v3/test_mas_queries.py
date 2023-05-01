import os
import json
import re
from tqdm import tqdm
from collections import Counter
from data_access_v3 import DataAccess
import numpy as np

input_dir = 'datasets/mas/setup/queries'
sql_folder = 'sql'
TABLES_TO_JOIN = ['conference', 'domain_conference', 'domain', 'author', 'domain_author',
                  'organization', 'writes', 'publication', 'domain_keyword', 'keyword']
JOINED_TABLE_NAME = 'author_writes_publication_organization_conference_domain_keyword'
final_queries_path = 'datasets/mas/queries.sql'
FINAL_TABLE_NAME = 'mas.mas_full_data'


def json_files2sql_files():
    for filename in tqdm(os.listdir(input_dir)):
        f = os.path.join(input_dir, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith('.json'):
            with open(f) as file:
                query_json = json.load(file)
                query_sql = query_json['query_sql']
                # query_sql = ''.join(query_sql.splitlines())
                query_sql = re.sub(r'\s+', ' ', query_sql)
                query_name = query_json['query_name']
                sql_file = open(os.path.join(input_dir, sql_folder, f'{query_name}.sql'), 'w')
                sql_file.write(query_sql)
                sql_file.close()


def read_from_clause():
    sql_path = os.path.join(input_dir, sql_folder)
    tables = []
    for filename in tqdm(os.listdir(sql_path)):
        f = os.path.join(sql_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith('.sql'):
            with open(f) as file:
                query_sql = file.read()
                from_clause = query_sql.split('FROM')[-1].split('WHERE')[0]
                tables.append(from_clause)

    counter = Counter(tables)
    print(counter.items())


def output_possibly_valid_queries():
    sql_path = os.path.join(input_dir, sql_folder)
    queries = []
    for filename in tqdm(os.listdir(sql_path)):
        f = os.path.join(sql_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith('.sql'):
            with open(f) as file:
                query_sql = file.read()
                from_tables = query_sql.split('FROM')[-1].split('WHERE')[0].split(',')
                from_tables = [table.strip() for table in from_tables]
                if len(from_tables) == len(np.intersect1d(from_tables, TABLES_TO_JOIN)):
                    queries.append(query_sql)

    for query in queries:
        where_clause = query.split(' WHERE ')[-1].split(' GROUP BY ')[0]
        for table in TABLES_TO_JOIN:
            where_clause = where_clause.replace(f'{table}.', f'{table}$')
        new_query = f'SELECT _id FROM {FINAL_TABLE_NAME} WHERE {where_clause};'
        print(new_query)


def create_authors_sample_table(sample_size=100000):
    limits = DataAccess.select_one(
        f'SELECT MIN(aid) as min, MAX(aid) as max FROM mas.author;')
    min_aid = limits['min']
    max_aid = limits['max']
    sampled_aids = np.random.choice(np.arange(min_aid, max_aid + 1), size=sample_size, replace=False)
    select_query = f'SELECT * FROM mas.author WHERE aid IN ({" , ".join([str(aid) for aid in sampled_aids])});'
    create_table_query = f'CREATE TABLE mas.author_sample_{sample_size} AS {select_query}'
    DataAccess.update(create_table_query)

    mandatory_values = ['Tova Milo', 'H. V. Jagadish', 'Alin Deutsch']
    column = 'NAME'
    for value in mandatory_values:
        if DataAccess.select_one(
                f'SELECT COUNT(*) AS count FROM mas.author_sample_{sample_size} WHERE {column}=\'{value}\' ') == 0:
            DataAccess.update(f'INSERT INTO mas.author_sample_{sample_size} '
                              f'SELECT * FROM mas.author WHERE {column}=\'{value}\'')


def create_keyword_sample_table(sample_size=500):
    limits = DataAccess.select_one(
        f'SELECT MIN(kid) as min, MAX(kid) as max FROM mas.keyword;')
    min_kid = limits['min']
    max_kid = limits['max']
    sampled_kids = np.random.choice(np.arange(min_kid, max_kid + 1), size=sample_size, replace=False)
    select_query = f'SELECT * FROM mas.keyword WHERE kid IN ({" , ".join([str(aid) for aid in sampled_kids])});'
    create_table_query = f'CREATE TABLE mas.keyword_sample_{sample_size} AS {select_query}'
    DataAccess.update(create_table_query)

    mandatory_values = ['Machine Learning', 'Natural Language Processing']
    column = 'keyword'
    for value in mandatory_values:
        if DataAccess.select_one(
                f'SELECT COUNT(*) AS count FROM mas.keyword_sample_{sample_size} WHERE {column}=\'{value}\' ') == 0:
            DataAccess.update(f'INSERT INTO mas.keyword_sample_{sample_size} '
                              f'SELECT * FROM mas.author WHERE {column}=\'{value}\'')


def get_columns_to_index():
    columns = []
    with open(final_queries_path) as file:
        for query in file:
            where_clause = query.split('WHERE ')[-1]
            columns += [string for string in where_clause.split() if '$' in string]
    columns = [*set(columns)]
    print(columns)
    for col in columns:
        index_query = f'CREATE INDEX {col}_idx ON {FINAL_TABLE_NAME} ({col});'
        print(index_query)


if __name__ == '__main__':
    get_columns_to_index()
