import argparse
from config_manager_v3 import ConfigManager
from multiprocessing import Process, Manager
import pandas as pd
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--r", type=float, default=0.3, help="sample ratio")
parser.add_argument("--schema", type=str, default=ConfigManager.get_config('queryConfig.schema'),
                    help="Schema to run in")
args = parser.parse_args()
print(f"Running with following args: {args}")


def get_tables_data():
    if args.schema == "imdb":
        return [
            ('title', 'datasets\\imdb-job\\table1\\csv\\title.csv', args.r),
            ('movie_keyword', 'datasets\\imdb-job\\table1\\csv\\movie_keyword.csv', args.r),
            ('movie_companies', 'datasets\\imdb-job\\table1\\csv\\movie_companies.csv', args.r),
        ]
    if args.schema == "imdb2":
        return [
            ('join_title_companies_keyword',
             'datasets\\imdb-job\\table2\\join_title_companies_keyword.csv', args.r)
        ]
    if args.schema == "mas":
        return [('mas_full_data', 'datasets\\mas\\mas_full_data2.csv', args.r)]
    return []


def handle_table(table_name, csv_path, ratio, return_dict):
    # read csv into DataFrame and sample
    print(f'handling table: {table_name} with ratio: {ratio}', flush=True)
    df = pd.read_csv(csv_path)
    if len(df.index) <= 500_000:
        return_dict[table_name] = df
        return
    df_sample = df.sample(frac=ratio)
    del df
    gc.collect()
    return_dict[table_name] = df_sample
    return


def start():
    procs = []
    return_dict = Manager().dict()
    for (table_name, csv_path, ratio) in get_tables_data():
        proc = Process(target=handle_table, args=(table_name, csv_path, ratio, return_dict))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    join_samples(return_dict)


def join_samples(return_dict):
    if args.schema == 'imdb':
        title = return_dict['title'][
            ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
             'episode_nr', 'series_years']]
        movie_companies = return_dict['movie_companies'][['movie_id', 'company_id', 'company_type_id', 'note']]
        movie_keyword = return_dict['movie_keyword'][['movie_id', 'keyword_id']]

        result_df = pd.merge(left=title, right=movie_companies,
                             left_on='id', right_on='movie_id', suffixes=('__t', '__mc'))
        result_df = pd.merge(left=result_df, right=movie_keyword,
                             left_on='id', right_on='movie_id', suffixes=('', '__mk'))

        result_df = result_df[
            ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
             'episode_nr', 'series_years', 'company_id', 'company_type_id', 'note', 'keyword_id']]

        print(f'Num of rows: {len(result_df.index)}')
        # result_df.to_csv(f'datasets\\imdb-job\\table1\\csv\\initial_sample_{args.r}.csv', index=False)  # TODO: write to db?
        result_df.to_csv(f'datasets\\imdb-job\\table1\\csv\\initial_sample.csv', index=True, index_label='_id')

    if args.schema == 'imdb2':
        result_df = return_dict['join_title_companies_keyword'][
            ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
             'episode_nr', 'series_years', 'company_id', 'company_type_id', 'note', 'keyword_id']]
        print(f'Num of rows: {len(result_df.index)}')
        result_df.to_csv(
            f'datasets\\imdb-job\\table2\\join_title_companies_keyword_{args.r}.csv', index=False)

    if args.schema == 'mas':
        result_df = return_dict['mas_full_data']
        print(f'Num of rows: {len(result_df.index)}')
        result_df.to_csv(
            f'datasets\\mas\\initial_sample\\verdict\\verdict_mas_full_data2_{args.r}.csv', index=False)


if __name__ == "__main__":
    start()
