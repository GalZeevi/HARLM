import concurrent.futures
from random import random, sample as rand_sample, choices, randrange
from config_manager_v3 import ConfigManager
from checkpoint_manager_v3 import CheckpointManager
from data_access_v3 import DataAccess
from score_calculator import get_score
from tqdm import tqdm


class SummaryGene:
    """
    A data class for summary gene
    """

    def __init__(self,
                 rows: list = None,
                 cols: list = None):
        self._rows = rows if isinstance(rows, list) else []
        self._cols = cols if isinstance(cols, list) else []

    # getters #

    def get_summary(self,
                    dataset):
        # TODO deprecated
        return self.get_rows()

    def get_rows(self):
        return self._rows

    def get_columns(self):
        return self._cols

    def get_row_count(self):
        return len(self._rows)

    def get_col_count(self):
        return len(self._cols)

    # end - getters #

    # logic #

    def mutation(self,
                 max_row_index: int,
                 max_col_index: int,
                 mutation_rate: float):
        mutation_count = round((len(self._rows) + len(self._cols)) * mutation_rate)
        for mutations_index in range(mutation_count):
            # pick row or column
            is_row = random()
            # if we have all columns, we can only change rows
            if len(self._cols) == max_col_index + 1:
                is_row = 0
            # if row
            if is_row < 0.5:
                # pick index we want to change
                change_index = round(random() * (len(self._rows) - 1))
                # pick new value
                new_val = round(random() * max_row_index)
                while new_val in self._rows:
                    new_val = round(random() * max_row_index)
                self._rows[change_index] = new_val
            else:  # else column
                # pick index we want to change
                change_index = round(random() * (len(self._cols) - 1))
                # pick new value
                new_val = round(random() * max_col_index)
                while new_val in self._cols:
                    new_val = round(random() * max_col_index)
                self._cols[change_index] = new_val

    # end - logic #

    def __repr__(self):
        return "<SummaryGene ({}X{})>".format(len(self._rows), len(self._cols))

    def __str__(self):
        return "SummaryGene:\nrows = {}\ncols = {}>".format(self._rows, self._cols)


class SummaryGenePopulation:
    """
    A data class for population of summary genes with the main GA operations on them
    """

    WORKERS = ConfigManager.get_config('cpuConfig.num_workers')

    def __init__(self,
                 row_count: int,
                 col_count: int,
                 genes: list = None):
        self._row_count = row_count
        self._col_count = col_count
        self._genes = genes if isinstance(genes, list) else []
        self._scores = [random() for _ in range(len(self._genes))]

    @staticmethod
    def random_population(row_count: int,
                          col_count: int,
                          summary_rows: int,
                          summary_cols: int,
                          population_size: int):
        row_set = list(range(row_count))  # all _rows
        col_set = list(range(col_count))  # all columns
        return SummaryGenePopulation(row_count=row_count,
                                     col_count=col_count,
                                     genes=[SummaryGene(rows=rand_sample(row_set, k=summary_rows),
                                                        cols=rand_sample(col_set, k=summary_cols))
                                            for _ in range(population_size)])

    # getters #

    def get_scores(self):
        return self._scores

    def get_best_gene(self) -> SummaryGene:
        return self._genes[self._scores.index(min(self._scores))]

    # end - getters #

    # logic #

    def fitness(self,
                fitness_function):
        scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=SummaryGenePopulation.WORKERS) as executor:
            # Start the load operations and mark each future with its URL
            future_to_score = [executor.submit(fitness_function, gene.get_rows()) for gene in
                               self._genes]
            for future in concurrent.futures.as_completed(future_to_score):
                scores.append(future.result())
        max_score = max(scores)
        scores = [score if score != -1 else max_score + 1 for score in scores]
        self._scores = scores

    def selection(self,
                  royalty_rate: float):
        # convert scores to probability
        max_fitness = max(self._scores)
        reverse_scores = [max_fitness - score for score in self._scores]
        sum_fitness = sum(reverse_scores)
        if sum_fitness > 0:
            fitness_probabilities = [score / sum_fitness for score in reverse_scores]
        else:
            fitness_probabilities = reverse_scores
        # sort the population by fitness
        genes_with_fitness = zip(fitness_probabilities, self._genes)
        genes_with_fitness = sorted(genes_with_fitness, key=lambda x: x[0], reverse=True)
        # pick the best royalty_rate anyway
        royalty = [val[1] for val in genes_with_fitness[:round(len(genes_with_fitness) * royalty_rate)]]
        # tournament around the other genes
        left_genes = [val[1] for val in genes_with_fitness[round(len(genes_with_fitness) * royalty_rate):]]
        left_fitness = [val[0] for val in genes_with_fitness[round(len(genes_with_fitness) * royalty_rate):]]
        pick_genes = []
        left_count = len(self._genes) - len(royalty)
        while len(pick_genes) < left_count:
            pick_gene = choices(left_genes, weights=left_fitness)
            pick_genes.append(pick_gene)
        # add the royalty
        pick_genes = list(pick_genes)
        pick_genes.extend(royalty)
        return pick_genes

    def crossover(self):
        # init source and target lists
        target_length = len(self._genes)
        new_genes = []

        # run over the population and get children
        while len(new_genes) < target_length:
            # pick two random parents in the population
            parent_gene_1 = self._genes.pop(randrange(len(self._genes)))
            parent_gene_2 = self._genes.pop(randrange(len(self._genes)))
            # get their children #
            # pick random split location in the rows and columns
            change_rows_index = round(random() * parent_gene_1.get_row_count())
            change_cols_index = round(random() * parent_gene_1.get_col_count())
            # create rows
            child_gene_1_rows = parent_gene_1.get_rows()[:change_rows_index]
            child_gene_1_rows.extend(parent_gene_2.get_rows()[change_rows_index:])
            child_gene_2_rows = parent_gene_2.get_rows()[:change_rows_index]
            child_gene_2_rows.extend(parent_gene_1.get_rows()[change_rows_index:])
            # create colums
            child_gene_1_cols = parent_gene_1.get_columns()[:change_cols_index]
            child_gene_1_cols.extend(parent_gene_2.get_columns()[change_cols_index:])
            child_gene_2_cols = parent_gene_2.get_columns()[:change_cols_index]
            child_gene_2_cols.extend(parent_gene_1.get_columns()[change_cols_index:])
            # wrap with the classes
            child_gene_1 = SummaryGene(rows=child_gene_1_rows,
                                       cols=child_gene_1_cols)
            child_gene_2 = SummaryGene(rows=child_gene_2_rows,
                                       cols=child_gene_2_cols)
            # add the children to the new population
            new_genes.append(child_gene_1)
            new_genes.append(child_gene_2)

        # replace to new gene population and return answer
        self._genes = new_genes

    def mutation(self,
                 mutation_rate: float):
        with concurrent.futures.ThreadPoolExecutor(max_workers=SummaryGenePopulation.WORKERS) as executor:
            # Start the load operations and mark each future with its URL
            future_to_score = [executor.submit(gene.mutation, self._row_count - 1, self._col_count - 1, mutation_rate)
                               for gene in self._genes]
            for future in concurrent.futures.as_completed(future_to_score):
                future.result()

    # end - logic #

    def __repr__(self):
        return "<Genetic summaries population>"

    def __str__(self):
        return "<Genetic summaries population | size = {}>".format(len(self._genes))


class GeneticSampler:
    """
    This class is a genetic-coding type algorithm for dataset summarization
    """

    # SETTINGS #
    STEPS = 50
    MUTATION_RATE = 0.05
    POPULATION_SIZE = 50
    ROYALTY_RATE = 0.02

    # END - SETTINGS #

    def __init__(self):
        pass

    @staticmethod
    def __num_cols__():
        schema = ConfigManager.get_config('queryConfig.schema')
        table = ConfigManager.get_config('queryConfig.table')
        return DataAccess.select_one(f"SELECT COUNT(column_name) AS num_cols FROM information_schema.columns " +
                                     f"WHERE table_schema='{schema}' AND table_name='{table}' ")

    @staticmethod
    def run(num_rows: int,
            k: int,
            evaluate_score_function,
            max_iter: int = -1):

        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at GeneticSampler.run: the max_iter argument must be larger than 1")

        num_cols = GeneticSampler.__num_cols__()

        # setting the round count to the beginning of the process
        round_count = 1

        # init all the vars we need in the process
        best_score = 9999  # TODO: can be done better
        best_rows = []

        checkpoint_population = CheckpointManager.load(f'{k}_{max_iter}_genetic_sample_population')
        # init random population
        if checkpoint_population is None:
            gene_population = SummaryGenePopulation.random_population(row_count=num_rows,
                                                                      col_count=num_cols,
                                                                      summary_rows=k,
                                                                      summary_cols=num_cols,
                                                                      population_size=GeneticSampler.POPULATION_SIZE)
        else:
            gene_population = checkpoint_population
            best_gene = gene_population.get_best_gene()
            best_rows = best_gene.get_rows()
            return best_rows[:k], 1 - evaluate_score_function(best_rows[:k])

        pbar = tqdm(total=max_iter)
        while round_count <= max_iter:
            # optimize over the columns and rows
            gene_population.selection(royalty_rate=GeneticSampler.ROYALTY_RATE)
            gene_population.crossover()
            gene_population.mutation(mutation_rate=GeneticSampler.MUTATION_RATE)
            gene_population.fitness(fitness_function=evaluate_score_function)
            best_gene = gene_population.get_best_gene()

            round_count % 10 == 0 and CheckpointManager.save(f'{k}-{round_count}_genetic_sample_population',
                                                             gene_population)

            # compute scores
            total_score = evaluate_score_function(best_gene.get_rows())

            # check we this summary is better
            if total_score < best_score:
                best_score = total_score
                best_rows = best_gene.get_rows()

            # count this step
            round_count += 1
            pbar.update(1)

        return best_rows, 1 - evaluate_score_function(best_rows)


def get_sample(k, dist=False, max_iter=GeneticSampler.STEPS):
    view_size = ConfigManager.get_config('samplerConfig.viewSize')
    schema = ConfigManager.get_config('queryConfig.schema')
    table = ConfigManager.get_config('queryConfig.table')
    table_size = DataAccess.select_one(f'SELECT COUNT(1) AS table_size FROM {schema}.{table}')

    sample, score = GeneticSampler.run(table_size, k, lambda s: 1 - get_score(s, dist), max_iter)
    CheckpointManager.save(f'{k}-{view_size}-{max_iter}_genetic_sample', [sample, score])

    return sample, score


if __name__ == '__main__':
    k_list = [10_000, 50_000, 100_000, 150_000, 200_000]
    # k_list = [10, 50, 100, 150, 200]
    k_list.reverse()
    for k in tqdm(k_list):
        get_sample(k)
