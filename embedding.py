import numpy as np
import torch
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess
from preprocessing import Preprocessing


class TuplesDataset(Dataset):
    def __init__(self, tuples_and_labels):
        self.tuples_and_labels = tuples_and_labels

    def __len__(self):
        return len(self.tuples_and_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            return [self.tuples_and_labels[i] for i in idx]
        if isinstance(idx, int):
            return self.tuples_and_labels[idx]
        else:
            return None


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

    def __init__(self, checkpoint_version, model_name='all-MiniLM-L6-v2', use_preprocessing=True):
        print('==================== Reading table columns ====================', flush=True)
        self.numeric_cols = get_numerical_cols()
        self.checkpoint_version = checkpoint_version
        self.categorical_cols = get_non_numerical_cols()
        print('==================== Downloading model ====================', flush=True)
        self.model = SentenceTransformer(model_name)
        self.max_words = 300
        self.embedding_size = 384
        use_preprocessing and print('==================== Init Preprocessing ====================', flush=True)
        use_preprocessing and Preprocessing.init(checkpoint_version)

    def sim(self, tup1, tup2):
        sim = 0
        columns = Preprocessing.columns_repo.get_all_columns()
        for col_name in self.numeric_cols:
            col = columns[col_name]
            if (tup1[col_name] is None and tup2[col_name] is not None) or \
                    (tup2[col_name] is None and tup1[col_name] is not None):
                sim += 0
            else:
                sim += 1 - abs(
                    ((tup1[col_name] - col.min_val) / col.max_val) - ((tup2[col_name] - col.min_val) / col.max_val))
        for col_name in self.categorical_cols:
            col = columns[col_name]
            if max([len(key.split()) for key in col.encodings]) > 1:
                sim += (fuzz.ratio(f'{tup1[col_name]}', f'{tup2[col_name]}') / 100)
            else:
                sim += (1 if tup1[col_name] == tup2[col_name] else 0)

        return float(sim) / (len(columns) - 1)

    # TODO: this currently is not working
    def finetune(self, tuples, num_train_examples, name, epochs=5):
        raise NotImplementedError()
        # print('==================== Preparing training data ====================', flush=True)
        # train_examples = CheckpointManager.load(f'models/{name}_train_data', version=self.checkpoint_version)
        # if train_examples is None:
        #     indices = np.random.choice(len(tuples), size=2 * num_train_examples, replace=True)
        #     pairs = [[tuples[indices[i]], tuples[indices[i + 1]]] for i in range(0, len(indices), 2)]
        #     train_examples = [InputExample(texts=self.tuples2sentences(pair), label=self.sim(*pair)) for pair in pairs]
        #     CheckpointManager.save(f'models/{name}_train_data', train_examples, version=self.checkpoint_version)
        #
        # print('==================== Preparing dataloader ====================', flush=True)
        # train_dataloader = DataLoader(TuplesDataset(train_examples), shuffle=True, batch_size=16)
        # train_loss = losses.CosineSimilarityLoss(self.model)
        #
        # print('==================== Start training ====================', flush=True)
        # self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=10000,
        #                output_path=f'models/{name}', save_best_model=True)

    def tuples2sentences(self, tuples):
        num_numeric_cols = len(self.numeric_cols)
        num_categorical_cols = len(self.categorical_cols)
        max_words_per_categ_col = int(
            (self.max_words - (num_numeric_cols + num_categorical_cols) - num_numeric_cols) / num_categorical_cols)
        _, _, pivot = get_table_details()

        trunc_tuples = \
            [{k: (v if k in self.numeric_cols else f'{v}'[:max_words_per_categ_col]) for k, v in tup.items() if
              k != pivot} for tup in
             tuples]
        return [', '.join(f'{key}: {value}' for key, value in tup.items()) for tup in trunc_tuples]

    def tuples2vector(self, tuples):
        tuples_as_string = self.tuples2sentences(tuples)
        embeddings = self.model.encode(tuples_as_string)
        return embeddings


if __name__ == '__main__':
    print('==================== Reading csv ====================', flush=True)
    df = pd.read_csv('/home/milo/dataset/imdb-job/join_title_companies_keyword.csv')
    tuples = df.to_dict('records')
    embedding = Embedding(42)  # , model_name='models/imdb_finetuned_100k')
    # print(tuples)
    # print(embedding.tuples2sentences(tuples))
    # print(embedding.tuples2vector(tuples[:10]))
    embedding.finetune(tuples, 5 * 1_000, 'imdb_finetuned_5k')
    x = DataAccess.select('SELECT * from mas.mas_full_data2 ORDER BY RAND() LIMIT 2')
    # print(x)
    # embedding = Embedding(17)
    # print(embedding.sim(*x))
