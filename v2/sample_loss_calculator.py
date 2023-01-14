from data_access_v2 import DataAccess
import numpy as np
from scipy.spatial.distance import squareform


class SampleLossCalculator:
    def __init__(self, triu_dist_matrix_npz_path):
        data = np.load(triu_dist_matrix_npz_path)
        key = [*data.keys()][0]
        upper_triangle = data[key]
        self.distance_matrix = squareform(upper_triangle, checks=False)

    def test_query(self, sql, sample):
        # Queries should only select the pivot column
        full_result = DataAccess.select(sql)
        if len(full_result) == 0:
            return -1.
        sample_result = np.intersect1d(full_result, sample)
        if len(sample_result) == 0:
            return 1.
        distances = np.min(self.distance_matrix[full_result, :][:, sample_result], axis=1) / len(full_result)
        result = np.round(distances, 5)
        return result
