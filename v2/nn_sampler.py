import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
from codetiming import Timer
from torch import nn, optim, squeeze
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import random

from checkpoint_manager_v2 import CheckpointManager
from config_manager_v2 import ConfigManager
from data_access_v2 import DataAccess
from db_types import DBTypes


class RelationalDataset(Dataset):
    TIMER_NAME = 'relational_dataset_timer'
    RESULTS_CHECKPOINT_NAME = 'results'

    def __init__(self, queries, results, tuple_ids):
        timer = Timer(name=RelationalDataset.TIMER_NAME, initial_text='Starting dataset initialization')
        timer.start()
        self.queries = queries
        self.results = RelationalDataset.execute_queries(queries) if results is None else results
        self.table = ConfigManager.get_config('queriesConfig.table')
        self.schema = ConfigManager.get_config('queriesConfig.schema')
        self.pivot = ConfigManager.get_config('queriesConfig.pivot')
        self.db_type = ConfigManager.get_config('dbConfig.type')
        self.encodings = RelationalDataset.get_encodings(self.schema, self.table, self.pivot)
        self.num_tuples = len(tuple_ids)
        self.random_tuples = RelationalDataset.encode_numpy_tuples(
            RelationalDataset.get_tuples(self.schema, self.table, self.pivot, tuple_ids),
            self.encodings)
        self.pivot_col_num_when_sorted = sorted(DataAccess.select(f"SELECT column_name FROM information_schema.columns "
                                                                  f"WHERE table_schema='{self.schema}' "
                                                                  f"AND table_name='{self.table}'")).index(self.pivot)
        timer.stop()

    def __len__(self):
        return len(self.results) * self.num_tuples

    def __getitem__(self, idx):
        # get query num by dividing by self.num_tuples
        query_num = int(idx / self.num_tuples)
        # get tuple num using modulu
        tuple_num = idx % self.num_tuples
        tup = self.random_tuples[tuple_num]
        # calculate label
        label = 1. if tup[self.pivot_col_num_when_sorted] in self.results[query_num] else 0.
        return np.delete(tup, self.pivot_col_num_when_sorted, 0), label

    @staticmethod
    def execute_queries(queries):
        results = []
        for q in tqdm(queries):
            results.append(np.array(DataAccess.select(q)))
        CheckpointManager.save(name=RelationalDataset.RESULTS_CHECKPOINT_NAME, content=results)
        return results

    @staticmethod
    def get_encodings(schema, table, pivot):
        numeric_data_types = ['smallint', 'integer', 'bigint',
                              'decimal', 'numeric', 'real', 'double precision',
                              'smallserial', 'serial', 'bigserial', 'int']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        numeric_cols = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                         f"WHERE table_schema='{schema}' AND table_name='{table}' " +
                                         f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                         f"AND column_name <> '{pivot}'")

        db_formatted_numeric_cols = [f"\'{col}\'" for col in numeric_cols]
        categorical_columns = DataAccess.select(f"SELECT column_name AS col FROM information_schema.columns " +
                                                f"WHERE table_schema='{schema}' AND table_name='{table}' "
                                                f"AND column_name NOT IN ({' , '.join(db_formatted_numeric_cols)}) "
                                                f"AND column_name <> '{pivot}'")
        return {col: RelationalDataset.create_encoding_dict(
            DataAccess.select(f'SELECT DISTINCT {col} as val FROM {schema}.{table}'))
            for col in categorical_columns}

    @staticmethod
    def create_encoding_dict(string_values):
        values, codes = np.unique(string_values, return_inverse=True)
        return {value: code for code, value in zip(codes, values)}

    @staticmethod
    def get_tuples(schema, table, pivot, ids):
        return DataAccess.select(f'SELECT * FROM {schema}.{table} '
                                 f'WHERE {pivot} IN ({",".join([str(idx) for idx in ids])})')

    @staticmethod
    def tuples_list_to_numpy(tuples):
        sorted_columns = sorted(tuples[0].keys())
        project_on_second = (lambda l: [pair[1] for pair in l])
        return np.array([
            # sort by columns, then take only values
            project_on_second(sorted([*tup.items()], key=lambda pair: pair[0]))
            for tup in tuples
        ]), sorted_columns

    @staticmethod
    def encode_numpy_tuples(tuples_list, encodings):

        def encode_column(tuples, col_num_when_sorted):
            col = tuples[:, col_num_when_sorted]

            col_name = sorted_columns[col_num_when_sorted]
            column_mapping = encodings[col_name]

            reduced_mapping = {k: v for k, v in column_mapping.items() if k in col}

            for key in reduced_mapping.keys():
                col[np.where(col == key)] = reduced_mapping[key]
            tuples[:, col_num_when_sorted] = col

            return tuples

        tuples_numpy, sorted_columns = RelationalDataset.tuples_list_to_numpy(tuples_list)

        categorical_columns_nums = \
            [num for (num, col) in enumerate(sorted_columns) if col in encodings.keys()]

        tuples_numpy = np.apply_over_axes(encode_column,
                                          tuples_numpy,
                                          categorical_columns_nums)
        return tuples_numpy.astype(np.float)


class RelationalNetwork(pl.LightningModule):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.table = ConfigManager.get_config('queriesConfig.table')
        self.schema = ConfigManager.get_config('queriesConfig.schema')
        self.pivot = ConfigManager.get_config('queriesConfig.pivot')
        self.num_columns_without_id = self._num_columns_without_id()
        self.model = nn.Sequential(nn.Linear(self.num_columns_without_id, self.num_columns_without_id // 2,
                                             dtype=np.float),
                                   nn.Linear(self.num_columns_without_id // 2, 1,
                                             dtype=np.float),
                                   nn.Sigmoid()) \
            if checkpoint_path is None \
            else RelationalNetwork.load_from_checkpoint(checkpoint_path)

    def _num_columns_without_id(self):
        return DataAccess.select_one(f"SELECT COUNT(column_name) FROM information_schema.columns "
                                     f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' "
                                     f"AND column_name <> '{self.pivot}'")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = RelationalNetwork.loss(squeeze(self.model(x)), y)
        self.log("batch {0} train_loss".format(batch_idx), loss)
        return loss

    @staticmethod
    def loss(y_hat, y):
        return f.mse_loss(y_hat, y)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class NNSampler:
    CHECKPOINT_NAME = 'nn_sample'
    SLICE_SIZE = ConfigManager.get_config('nnConfig.sampleSliceSize')

    def __init__(self, queries, results, checkpoint_path=None):
        self.table = ConfigManager.get_config('queriesConfig.table')
        self.schema = ConfigManager.get_config('queriesConfig.schema')
        self.pivot = ConfigManager.get_config('queriesConfig.pivot')
        self.db_type = ConfigManager.get_config('dbConfig.type')
        self.num_tuples = ConfigManager.get_config('nnConfig.numTuples')
        self.num_queries = ConfigManager.get_config('nnConfig.numQueries')
        self.num_workers = ConfigManager.get_config('cpuConfig.num_workers')
        self.batch_size = ConfigManager.get_config('nnConfig.batchSize')
        self.test_size = ConfigManager.get_config('nnConfig.testSize')
        self.max_epochs = ConfigManager.get_config('nnConfig.maxEpochs')
        self.queries = queries
        self.results = results
        self.pivot_col_num_when_sorted = sorted(DataAccess.select(f"SELECT column_name FROM information_schema.columns "
                                                                  f"WHERE table_schema='{self.schema}' "
                                                                  f"AND table_name='{self.table}'")).index(self.pivot)

        self.network = RelationalNetwork(checkpoint_path=checkpoint_path)
        self.trainer = pl.Trainer(max_epochs=self.max_epochs)

    def fit(self):
        train_ids, test_ids = self.get_train_test_tuple_ids()
        CheckpointManager.save(f'{NNSampler.CHECKPOINT_NAME}_train_test', [train_ids, test_ids])
        train_dataset = RelationalDataset(self.queries[:self.num_queries],
                                          None if self.results is None else self.results[:self.num_queries],
                                          train_ids)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        self.trainer.fit(self.network, train_loader)

        self.trainer.save_checkpoint(
            f'./{CheckpointManager.basePath}/{CheckpointManager.get_max_version()}'
            f'/{NNSampler.CHECKPOINT_NAME}_model.ckpt')

    @staticmethod
    def get_random_tuples(schema, table, db_type, num_tuples):
        if DBTypes.IS_POSTGRESQL(db_type):
            random_statement = 'RANDOM'
        elif DBTypes.IS_MYSQL(db_type):
            random_statement = 'RAND'
        else:
            raise Exception('Unsupported db type! supported types are "postgresql" or "mysql"')

        return DataAccess.select(f'SELECT * FROM {schema}.{table} '
                                 f'ORDER BY {random_statement}() LIMIT {num_tuples}')

    def get_train_test_tuple_ids(self):
        if DBTypes.IS_POSTGRESQL(self.db_type):
            random_statement = 'RANDOM'
        elif DBTypes.IS_MYSQL(self.db_type):
            random_statement = 'RAND'
        else:
            raise Exception('Unsupported db type! supported types are "postgresql" or "mysql"')

        all_tuple_ids = DataAccess.select(f'SELECT {self.pivot} FROM {self.schema}.{self.table} '
                                          f'ORDER BY {random_statement}() LIMIT {self.num_tuples}')
        test_len = int(self.test_size * len(all_tuple_ids))
        return all_tuple_ids[test_len:], all_tuple_ids[:test_len]

    def get_test_error(self, test_ids, num_to_test):
        test_dataset = RelationalDataset(self.queries[:self.num_queries],
                                         None if self.results is None else self.results[:self.num_queries],
                                         test_ids)
        test_samples_id = random.sample(range(len(test_dataset)), num_to_test)
        test_samples = [test_dataset[idx] for idx in test_samples_id]
        losses_tensors = \
            [RelationalNetwork.loss(
                self.network.forward(torch.from_numpy(tup[0])),
                torch.Tensor([tup[1]])) for tup in test_samples]
        return [int(tup[0][self.pivot_col_num_when_sorted]) for tup in test_samples], \
               [l.detach().numpy().item() for l in losses_tensors]

    def get_sample(self, k):
        encodings = RelationalDataset.get_encodings(self.schema, self.table, self.pivot)
        last_tuple_id = -1
        max_tuple_id = DataAccess.select_one(f'SELECT MAX({self.pivot}) AS max_id FROM {self.schema}.{self.table}')
        sample = np.arange(k)
        sample_scores = np.array([np.NINF] * k)

        completed_slices = 0
        pbar = tqdm(total=math.ceil(max_tuple_id / NNSampler.SLICE_SIZE))
        while last_tuple_id < max_tuple_id:
            tuples_list = DataAccess.select(f'SELECT * FROM {self.schema}.{self.table} '
                                            f'WHERE {self.pivot} > {last_tuple_id} '
                                            f'ORDER BY {self.pivot} ASC '
                                            f'LIMIT {NNSampler.SLICE_SIZE}')
            encoded_tuples_list = RelationalDataset.encode_numpy_tuples(tuples_list, encodings)
            # Remove pivot
            encoded_tuples_list = np.delete(encoded_tuples_list, self.pivot_col_num_when_sorted, 1)
            scores = np.array([self.network.forward(torch.from_numpy(tup)).detach().numpy().item()
                               for tup in encoded_tuples_list])

            # Take k elements with maximal score
            top_k_ind = np.argpartition(scores, -k)[-k:]
            top_k_scores = scores[top_k_ind]

            for i in range(k):
                # Try to improve the worst score
                value_to_replace_ind = np.argmin(sample_scores)  # TODO what if argmax returns more then one
                value_to_replace = sample_scores[value_to_replace_ind]

                # Drop anyone that can't improve the worst score
                top_k_scores[np.where(top_k_scores <= value_to_replace)] = np.NINF

                # Take the best improvement
                replacement_ind = np.argmax(top_k_scores)
                replacement = top_k_scores[replacement_ind]
                if np.isneginf(replacement):
                    break  # We cannot improve this value so no point in checking anymore

                # Update sample
                sample[value_to_replace_ind] \
                    = (completed_slices * NNSampler.SLICE_SIZE) + top_k_ind[replacement_ind]
                sample_scores[value_to_replace_ind] = top_k_scores[replacement_ind]

                # Drop the replacement in order to not choose it again
                top_k_scores[replacement_ind] = np.NINF

            completed_slices += 1
            pbar.update(1)
            last_tuple_id = tuples_list[-1][self.pivot]
        pbar.close()
        CheckpointManager.save(NNSampler.CHECKPOINT_NAME, sample)
        return sample
