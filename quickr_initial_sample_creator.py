import argparse
from config_manager_v3 import ConfigManager
from multiprocessing import Process, Manager
import pandas as pd
import hashlib
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--schema", type=str, default=ConfigManager.get_config('queryConfig.schema'),
                    help="Schema to run in")
parser.add_argument("--p", type=float, default=35, help="percentage")
args = parser.parse_args()
print(f"Running with following args: {args}")


def my_hash(value):
    string_data = value if isinstance(value, str) else str(value)
    return int(hashlib.shake_256(string_data.encode('utf-8')).hexdigest(4), 16)


def sample_tables(table_names, join_cols, percentage, return_dict):
    # We assume df[join_col] is string/numeric (hashable) values
    # We assume percentage is an integer between 1 and 100 (inclusive)
    # We assume that all the df(s) are joinable using the columns in join col
    csv_paths = get_tables_csv_paths()
    dataframes = []
    for name in table_names:
        dataframes.append(pd.read_csv(csv_paths[name]))
    samples = []
    for i in range(len(dataframes)):
        df = dataframes[i]
        join_col = join_cols[i]
        print(f'========= sampling df: {table_names[i]} on col: {join_col} =========', flush=True)
        print(f'========= len before sample: {len(df.index)} =========', flush=True)
        if len(df.index) > 500_000:
            sample = df[df[join_col].apply(lambda value: my_hash(value) % 100 <= percentage)]
        else:
            sample = df
        print(f'========= len after sample: {len(sample.index)} =========', flush=True)
        samples.append(sample)
    return_dict[tuple(table_names)] = samples


def get_tables_csv_paths():
    if args.schema == "imdb":
        return {
            'title': 'datasets\\imdb-job\\table1\\csv\\title.csv',
            'movie_keyword': 'datasets\\imdb-job\\table1\\csv\\movie_keyword.csv',
            'movie_companies': 'datasets\\imdb-job\\table1\\csv\\movie_companies.csv'
        }
    if args.schema == "imdb2":
        return {
            'imdb_t_mk_mc_data': 'datasets\\imdb-job\\table2\\join_title_companies_keyword.csv'
        }
    if args.schema == "mas":
        return {'mas_full_data': 'datasets\\mas\\mas_full_data2.csv'}
    return {}


def get_tables_pk():
    if args.schema == "imdb":
        return {
            'title': 'id',
            'movie_keyword': 'id',
            'movie_companies': 'id'
        }
    if args.schema == "mas":
        return {'mas_full_data': '_id'}
    if args.schema == "imdb2":
        return {
            'imdb_t_mk_mc_data': '_id'
        }
    return {}


def get_joinable_tables_data():
    if args.schema == "imdb":
        return [(['title', 'movie_keyword', 'movie_companies'], ['id', 'movie_id', 'movie_id'])]
    if args.schema == "mas":
        return [(['mas_full_data'], ['publication$pid'])]
    if args.schema == "imdb2":
        return [(['imdb_t_mk_mc_data'], ['id'])]  # TODO: not _id to simulate the real thing
    return []


def join_return_dict(return_dict):
    print(f'joining results: {return_dict.keys()}', flush=True)
    names_to_all_samples = {}
    for table_names, samples in [*return_dict.items()]:
        names_to_sample = [*zip(table_names, samples)]
        for name, sample in names_to_sample:
            if name not in names_to_all_samples:
                names_to_all_samples[name] = []
            names_to_all_samples[name] = names_to_all_samples[name] + [sample]

    result = {}
    tables_pk = get_tables_pk()
    for name, all_samples in [*names_to_all_samples.items()]:
        result[name] = pd.concat(all_samples).drop_duplicates(subset=tables_pk[name])

    return result


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
        result_df.to_csv(f'datasets\\imdb-job\\initial_sample\\quickr\\quickr_initial_sample_{args.p}.csv', index=True,
                         index_label='_id')

    if args.schema == 'imdb2':
        result_df = return_dict['imdb_t_mk_mc_data'][
            ['_id', 'id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
             'episode_nr', 'series_years', 'company_id', 'company_type_id', 'note', 'keyword_id']]
        print(f'Num of rows: {len(result_df.index)}')
        result_df.to_csv(
            f'datasets\\imdb-job\\initial_sample\\quickr\\quickr_initial_sample2_{args.p}.csv', index=False)

    if args.schema == 'mas':
        result_df = return_dict['mas_full_data']
        print(f'Num of rows: {len(result_df.index)}')
        result_df.to_csv(
            f'datasets\\mas\\initial_sample\\quickr\\quickr_mas_full_data2_{args.p}.csv', index=False)


def start():
    procs = []
    return_dict = Manager().dict()
    for (table_names, col_names) in get_joinable_tables_data():
        proc = Process(target=sample_tables, args=(table_names, col_names, args.p, return_dict))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    # print(return_dict)
    return_dict = join_return_dict(return_dict)
    join_samples(return_dict)


if __name__ == "__main__":
    # Set hash seed and restart interpreter.
    # This will be done only once if the env var is clear.
    # if not os.environ.get('PYTHONHASHSEED'):
    #     os.environ['PYTHONHASHSEED'] = '1234'
    #     os.execv(sys.executable, ['python3'] + sys.argv)
    start()
