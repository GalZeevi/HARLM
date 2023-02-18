import math
import random
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as f
from torch import nn, optim, squeeze
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from checkpoint_manager_v3 import CheckpointManager
from config_manager_v3 import ConfigManager
from data_access_v3 import DataAccess, DBTypes
from score_calculator import get_score
from train_test_utils import get_test_queries, get_train_queries

view_size = ConfigManager.get_config('samplerConfig.viewSize')
FRAME_SIZE = 2 * view_size


class RelationalDataset(Dataset):

    def __init__(self, results, tuple_ids):
        self.results = results
        self.table = ConfigManager.get_config('queryConfig.table')
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.db_type = ConfigManager.get_config('dbConfig.type')
        self.view_size = ConfigManager.get_config('samplerConfig.viewSize')
        if len(tuple_ids) < FRAME_SIZE:
            raise Exception(f'Not enough tuples! enter at least {FRAME_SIZE}')
        self.num_frames = ConfigManager.get_config('nnConfig.numFrames')
        self.encodings = RelationalDataset.get_encodings(self.schema, self.table, self.pivot)
        self.num_tuples = len(tuple_ids)
        self.encoded_tuples = RelationalDataset.encode_numpy_tuples(
            RelationalDataset.get_tuples(self.schema, self.table, self.pivot, tuple_ids[:self.num_tuples]),
            self.encodings)
        self.pivot_col_num_when_sorted = sorted(DataAccess.select(f"SELECT column_name FROM information_schema.columns "
                                                                  f"WHERE table_schema='{self.schema}' "
                                                                  f"AND table_name='{self.table}'")).index(self.pivot)

    def __len__(self):
        return len(self.results) * self.num_frames

    def __getitem__(self, idx):
        # get random query
        result_num = np.random.choice(len(self.results))
        # get a random frame
        ids = np.random.choice(len(self.encoded_tuples), size=FRAME_SIZE)
        encoded_tups = self.encoded_tuples[ids]
        # calculate label
        mask = np.isin(ids, self.results[result_num]).astype(float)
        mask_top_n = np.argpartition(mask, -view_size)[-view_size:]
        label = np.zeros_like(mask)
        label[mask_top_n] = mask[mask_top_n]
        x = np.delete(encoded_tups, self.pivot_col_num_when_sorted, 1)
        return x, label

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
        return tuples_numpy.astype(float)


class RelationalNetwork(pl.LightningModule):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.table = ConfigManager.get_config('queryConfig.table')
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.num_columns_without_id = self._num_columns_without_id()
        if checkpoint_path is None:
            self.model = nn.Sequential(nn.Flatten(),
                                       nn.Linear(FRAME_SIZE * self.num_columns_without_id,
                                                 FRAME_SIZE * self.num_columns_without_id // 2,
                                                 dtype=float),
                                       nn.Linear(FRAME_SIZE * self.num_columns_without_id // 2,
                                                 FRAME_SIZE,
                                                 dtype=float),
                                       nn.Softmax())
        else:
            RelationalNetwork.load_from_checkpoint(checkpoint_path)

    def _num_columns_without_id(self):
        return DataAccess.select_one(f"SELECT COUNT(column_name) FROM information_schema.columns "
                                     f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' "
                                     f"AND column_name <> '{self.pivot}'")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = RelationalNetwork.loss(squeeze(y_hat), y)
        self.log("batch {0} train_loss".format(batch_idx), loss)
        return loss

    @staticmethod
    def loss(y_hat, y):
        # TODO
        # print(y_hat, y)
        return 0.001

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class NNSampler:
    SLICE_SIZE = 1000

    def __init__(self, checkpoint_path=None):
        self.table = ConfigManager.get_config('queryConfig.table')
        self.schema = ConfigManager.get_config('queryConfig.schema')
        self.pivot = ConfigManager.get_config('queryConfig.pivot')
        self.db_type = ConfigManager.get_config('dbConfig.type')
        self.num_tuples = ConfigManager.get_config('nnConfig.numTuples')
        self.num_queries = ConfigManager.get_config('nnConfig.numQueries')
        self.num_workers = ConfigManager.get_config('cpuConfig.num_workers')
        self.batch_size = ConfigManager.get_config('nnConfig.batchSize')
        self.test_size = ConfigManager.get_config('nnConfig.numTestTuples')
        self.max_epochs = ConfigManager.get_config('nnConfig.maxEpochs')
        self.pivot_col_num_when_sorted = sorted(DataAccess.select(f"SELECT column_name FROM information_schema.columns "
                                                                  f"WHERE table_schema='{self.schema}' "
                                                                  f"AND table_name='{self.table}'")).index(self.pivot)

        self.checkpoint_path = checkpoint_path
        self.network = RelationalNetwork(checkpoint_path=checkpoint_path)
        self.trainer = pl.Trainer(max_epochs=self.max_epochs)
        self.fitted = False or (checkpoint_path is not None)

    def fit(self):
        if self.checkpoint_path is not None:
            return

        train_ids, test_ids = self.get_train_test_tuple_ids()
        CheckpointManager.save(f'{self.num_tuples}_{self.num_queries}_nn_sample_train_test_tuples',
                               [train_ids, test_ids])

        train_results = get_train_queries()
        train_dataset = RelationalDataset(train_results[:self.num_queries], train_ids)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        self.trainer.fit(self.network, train_loader)

        self.trainer.save_checkpoint(
            f'./{CheckpointManager.basePath}/{CheckpointManager.get_max_version()}'
            f'/{view_size}-{self.num_tuples}_{self.num_queries}_nn_sample_model.ckpt')
        self.fitted = True

        return self

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
        return all_tuple_ids[self.test_size:], all_tuple_ids[:self.test_size]

    def get_test_error(self, test_tuples_ids, num_to_test):
        train_results = get_train_queries()
        test_dataset = RelationalDataset(train_results[:self.num_queries], test_tuples_ids)
        test_samples_id = random.sample(range(len(test_dataset)), num_to_test)

        test_samples = [test_dataset[idx] for idx in test_samples_id]
        losses_tensors = \
            [RelationalNetwork.loss(
                self.network.forward(torch.from_numpy(tup[0])),
                torch.Tensor([tup[1]])) for tup in test_samples]

        return [int(tup[0][self.pivot_col_num_when_sorted]) for tup in test_samples], \
               [l.detach().numpy().item() for l in losses_tensors]

    def get_sample(self, k):
        if not self.fitted:
            raise Exception('Must call fit() first!')

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
            if len(scores) >= k:
                top_k_ind = np.argpartition(scores, -k)[-k:]
                top_k_scores = scores[top_k_ind]
            else:
                top_k_ind = np.arange(len(scores))
                top_k_scores = scores[top_k_ind]

            for i in range(k):
                # Try to improve the worst score
                value_to_replace_ind = np.argmin(sample_scores)  # TODO what if argmin returns more then one
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

        return sample


def get_sample(k, dist=False, nn_sampler=None):
    if nn_sampler is None:
        nn_sampler = NNSampler().fit()

    sample = nn_sampler.get_sample(k)

    score = get_score(sample, dist)
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    CheckpointManager.save(f'{k}-{view_size}-{nn_sampler.num_tuples}_{nn_sampler.num_queries}_nn_sample',
                           [sample, score])
    return sample, score


if __name__ == '__main__':
    checkpoint_path = 'lightning_logs/version_3/checkpoints/epoch=0-step=99800.ckpt'
    nnsampler = NNSampler().fit()

    k_list = [100]
    for k in tqdm(k_list):
        get_sample(k, False, nnsampler)

    # get_sample(100)
