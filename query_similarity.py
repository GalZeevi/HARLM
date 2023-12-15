from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np
import torch
from tqdm import tqdm


class QuerySimilarity:

    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def query2vector(self, query):
        from_clause = query.split(' FROM ')[1].split(' WHERE ')[0]
        where_clause_str = query.split(' WHERE ')[1].replace(';', '').replace(f'{from_clause}.', '') \
            .replace('>', ' greater than ').replace('<', ' less than ').replace('=', ' equals ') \
            .replace('>=', ' greater than or equal to ').replace('<=', ' less than or equal to ')
        return self.model.encode(where_clause_str, convert_to_tensor=True).numpy()

    def sim(self, query1, query2):
        emb1 = torch.from_numpy(self.query2vector(query1))
        emb2 = torch.from_numpy(self.query2vector(query2))
        return util.pytorch_cos_sim(emb1, emb2).item()

    def dist(self, query1, query2):
        return 1 - self.sim(query1, query2)

    def k_means(self, queries, k):
        X = np.array([self.query2vector(query) for query in tqdm(queries)])
        kmeans = KMeans(n_clusters=k, verbose=1).fit(X)
        cluster_centers = kmeans.cluster_centers_

        # Get indices of cluster centers in the original array
        indices = []
        for center in cluster_centers:
            distances = np.linalg.norm(X - center, axis=1)
            center_index = np.argmin(distances)
            indices.append(center_index)

        return [queries[i] for i in indices]


if __name__ == '__main__':
    qs = QuerySimilarity()
    print(qs.sim(
        'SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.company_type_id=2 AND imdb_t_mk_mc_data.kind_id=0;',
        'SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.keyword_id=117;'))

    print(qs.query2vector(
        'SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.company_type_id=2 AND imdb_t_mk_mc_data.kind_id=0;'))

    print(qs.k_means(
        ["SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.company_type_id=2 AND imdb_t_mk_mc_data.kind_id=0;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.company_type_id=2 AND imdb_t_mk_mc_data.production_year>2010;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.company_type_id=2 AND imdb_t_mk_mc_data.production_year>2008;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.keyword_id=117;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.production_year>2005 AND imdb_t_mk_mc_data.keyword_id=398;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.production_year>1990 AND imdb_t_mk_mc_data.production_year<=1995;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.production_year>2010 AND imdb_t_mk_mc_data.company_type_id=2;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.production_year>1990 AND imdb_t_mk_mc_data.company_type_id=2 AND imdb_t_mk_mc_data.production_year<=1995;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.production_year>2010 AND imdb_t_mk_mc_data.keyword_id=8200;",
         "SELECT _id FROM imdb_t_mk_mc_data WHERE imdb_t_mk_mc_data.production_year>2014;"
         ], 3))
