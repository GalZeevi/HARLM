import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('../')


# import d6tstack.utils

def ensure_dir(d):
    d = os.path.abspath(d)
    if not os.path.exists(d):
        os.makedirs(d)


# In suciu's code this is not really used - also, it looks like it would hurt the ipf pd.merge
def bucketize_sample(sample_df, marginal_rounding_num, cols_to_ignore):
    df = sample_df.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = [c for c in df.select_dtypes(include=numerics).columns if c not in cols_to_ignore]
    if marginal_rounding_num != 0:
        print("Rounding by", marginal_rounding_num)
        df[numeric_cols] = (df[numeric_cols] / marginal_rounding_num).astype(int) * marginal_rounding_num
    return df


# marginal_columns is a collection of tuples that specifiy which marginals to read
# for example {(1,2),(1)} or {("YEAR","NAME")}
def read_marginals(relation, marginal_columns):
    marginals = {}
    for columns in marginal_columns:
        columns_str = [str(col) for col in columns] if isinstance(columns, tuple) or isinstance(columns, list) else str(
            columns)
        file_in = "ipf/marginals/{}/marginal_{}.csv".format(relation, '-'.join(columns_str)
        if isinstance(columns_str, list) else columns_str)
        print("Opening file", file_in)
        data = pd.read_csv(file_in)
        col_name = tuple(data.columns)
        marginals[col_name] = data
        marginals[col_name].columns = data.columns
    return marginals


def itertive_reweighting(sample_data, marginals, full_data_len):
    print("Starting to build linear model")
    start_time = time.time()
    sample_data['weight'] = 1  # Adding the weight column

    # compute the aggregation on the marginal - TODO: why not save it like this already?
    for key, marginal in marginals.items():
        marginals[key] = marginal.groupby(list(marginal.columns), as_index=False) \
            .agg('size').rename(columns={'size': 'count'})

    all_diffs = []
    max_iter = 5 * len(marginals)
    mk_plot = True
    for itr in range(max_iter):
        sum_scaling_diff = 0
        for agg_cols in marginals:
            print('Agg Cols', agg_cols)
            marginal = marginals[agg_cols]
            # Compute the same aggregation as the marginal on the sample
            aggregated_sample = sample_data.groupby(list(agg_cols))['weight'].agg('sum').reset_index(name='sampCount')
            joined = pd.merge(marginal, aggregated_sample, how='left')
            # Count is the aggregation on marginal, sampCount is on the sample (count! not sum(weight))
            joined['scale'] = joined['count'] / joined['sampCount']

            scaling_diff = np.abs(joined['scale'] - 1.0).sum(skipna=True)
            sum_scaling_diff += scaling_diff

            print(sample_data.iloc[0])
            print(joined.iloc[0])

            # In the original paper this is an inner join but since the marginals have missing values
            # this removes tuples from the sample
            data_pre_weight = pd.merge(sample_data,
                                       joined,
                                       on=[*agg_cols],
                                       how='left')

            data_pre_weight['weight'] = data_pre_weight['weight'] * data_pre_weight['scale']
            # This is because I turned the inner join to left join - replace with a different value?
            data_pre_weight['weight'] = data_pre_weight['weight'].fillna(1)
            # remove columns ['count', 'sampCount', 'scale'] but keep new weight
            sample_data = data_pre_weight[sample_data.columns]

        print('ITER {0} OF {1} SUM SCALING DIFF {2}'.format(itr, max_iter, sum_scaling_diff))
        all_diffs.append(sum_scaling_diff)
        if sum_scaling_diff <= 1e-2:
            mk_plot = False
            break

    if mk_plot:
        print("THE SCALING DID NOT CONVERGE")
        fig, axs = plt.subplots(1, 1)
        plt.plot(all_diffs)
        # rand_time = time.time()
        # print("RAND TIME", rand_time)
        # ensure_dir("Scaling/")
        # plt.savefig("Scaling/scaling_diff_{:.6f}.pdf".format(rand_time))

    if np.any(sample_data["weight"] <= 0):
        print("A WEIGHT WAS ZERO OR NEGATIVE IN RESCALING")
        print(sample_data)
        sys.exit(0)

    data_with_unif_weights = sample_data.copy()
    unif_weight = full_data_len / sample_data.shape[0]
    data_with_unif_weights['weight'] = unif_weight
    data_lin = sample_data.copy()
    data_lin_nm = data_lin.copy()
    lin_sum = data_lin_nm['weight'].sum()
    # Re-normalizing the data
    print('Default uniform weight is', unif_weight)
    print('Multiplicative weight factor', (full_data_len / lin_sum))
    data_lin_nm['weight'] = data_lin_nm['weight'] * (full_data_len / lin_sum)
    end_time = time.time()
    print('Final Learning Time Lin:', end_time - start_time)
    return data_with_unif_weights, data_lin, data_lin_nm


def main(marginals_idx, sample_df, full_data_len, marginal_rounding_num=1):
    pd.set_option('expand_frame_repr', True)
    pd.set_option('max_colwidth', 200)
    pd.set_option('display.max_columns', None)

    marginals = read_marginals('flights', marginals_idx)
    cols = [c for c in sample_df.columns]
    cols.append('weight')
    sample_df['_id'] = sample_df.index
    # sample_rounded = bucketize_sample(sample_df, marginal_rounding_num, ['_id'])
    sample_rounded = sample_df
    data_unif, data_lin, data_lin_nm = itertive_reweighting(sample_rounded.copy(), marginals, full_data_len)
    # assert data_lin_nm.shape[0] == sample_df.shape[0]  # TODO: problem?
    print(data_lin_nm)
    sample_unif = sample_df.copy()
    sample_unif['weight'] = data_unif['weight'].astype('float64')
    sample_unif = sample_unif[cols]
    sample_ipf = sample_df.copy()
    # sample_ipf_rounded gives us the index to sample_ipf
    sample_ipf = pd.merge(sample_ipf, data_lin_nm[['_id', 'weight']], how='inner')
    sample_ipf = sample_ipf[cols]


if __name__ == '__main__':
    # TODO: I think next I should create the pipline itself and look at Suciu's gen_marginals
    full_data_df = pd.read_csv('datasets/flights2/sample.csv')
    sample_df = full_data_df.sample(10000)
    marginals_idx = {7, 8, 10, 11, (7, 10), (8, 10), (10, 11)}
    main(marginals_idx, sample_df, len(full_data_df.index))
