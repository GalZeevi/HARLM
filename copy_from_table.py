from data_access_v3 import DataAccess
from tqdm import tqdm

############################################################################################
################### THIS SCRIPT CAN COPY DATA FROM ONE TABLE TO ANOTHER ####################
############################################################################################

SCHEMA = 'imdb_saqp'
SRC_TABLE = 'joined_titles1'
COLUMNS_LIST = ['title$id',
                'title$title',
                'title$imdb_index',
                'title$kind_id',
                'title$production_year',
                'title$imdb_id',
                'title$phonetic_code',
                'title$episode_of_id',
                'title$season_nr',
                'title$episode_nr',
                'title$series_years',
                'title$md5sum',
                'movie_companies$id',
                'movie_companies$movie_id',
                'movie_companies$company_id',
                'movie_companies$company_type_id',
                'movie_companies$note',
                'movie_keyword$id',
                'movie_keyword$movie_id',
                'movie_keyword$keyword_id']
DEST_TABLE = 'join_title_companies_keyword'
CHUNK_SIZE = 1000

row_count = DataAccess.select_one(f'SELECT COUNT(*) AS table_size FROM {SCHEMA}.{SRC_TABLE}')
full_batches = int(row_count / CHUNK_SIZE)
final_batch = row_count % CHUNK_SIZE
completed = 0
total_iters = full_batches + 1 if final_batch > 0 else 0
pbar = tqdm(total=total_iters)

for _ in range(total_iters):
    DataAccess.update(
        f'INSERT INTO {SCHEMA}.{DEST_TABLE}({",".join(COLUMNS_LIST)}) '
        f'SELECT {",".join(COLUMNS_LIST)} FROM {SCHEMA}.{SRC_TABLE} LIMIT {completed}, {CHUNK_SIZE};')
    completed += CHUNK_SIZE
    pbar.update(1)
