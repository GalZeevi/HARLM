from data_access import DataAccess
import random


class QueryGenerator:
    def __init__(self, schema, table, index_col, select_entire_row=False):
        self.data_access = DataAccess()
        self.schema = schema
        self.table = table
        self.index_col = index_col
        self.select_entire_row = select_entire_row
        self.numerical_cols = self.init_numerical_cols()
        self.numerical_vals = self.init_numerical_vals(self.numerical_cols)
        self.categorical_cols = self.init_categorical_cols()
        self.categorical_vals = self.init_categorical_vals(self.categorical_cols)

    def init_numerical_cols(self):
        numeric_data_types = ['smallint', 'integer', 'bigint', 'decimal', 'numeric', 'real', 'double precision',
                              'smallserial', 'serial', 'bigserial']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in numeric_data_types]
        columns = self.data_access.select(f"SELECT column_name FROM information_schema.columns " +
                                          f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                          f"AND data_type IN ({' , '.join(db_formatted_data_types)}) " +
                                          f"AND column_name <> '{self.index_col}'")
        return [col_obj['column_name'] for col_obj in columns]

    def init_categorical_cols(self):
        textual_data_types = ['character varying%%', 'varchar%%', 'text', 'char%%', 'character%%']
        db_formatted_data_types = [f"\'{data_type}\'" for data_type in textual_data_types]
        columns = self.data_access.select(f"SELECT column_name FROM information_schema.columns " +
                                          f"WHERE table_schema='{self.schema}' AND table_name='{self.table}' " +
                                          f"AND ({' OR '.join([f'data_type LIKE {data_type}' for data_type in db_formatted_data_types])})")
        return [col_obj['column_name'] for col_obj in columns]

    def init_numerical_vals(self, numerical_cols):
        # build a dict mapping col -> [min, max]
        min_max_vals = {}
        for col in numerical_cols:
            min_val = self.data_access.select_one(f'SELECT MIN({col}) AS val FROM {self.schema}.{self.table}')
            max_val = self.data_access.select_one(f'SELECT MAX({col}) AS val FROM {self.schema}.{self.table}')
            min_max_vals[col] = [min_val['val'], max_val['val']]
        return min_max_vals

    def init_categorical_vals(self, categorical_columns):
        # build a dict mapping col -> list of values
        vals = {}
        for col in categorical_columns:
            column_values = self.data_access.select(f"SELECT values.val AS val FROM (" +
                                                    f"SELECT DISTINCT {col} AS val FROM {self.schema}.{self.table}"
                                                    f") as values ORDER BY RANDOM() LIMIT 10000")
            vals[col] = [column_value['val'] for column_value in column_values]
        return vals

    def get_query(self, num_of_columns):
        #  making sure no more columns than available are selected
        num_of_columns = min(num_of_columns, len(self.categorical_cols) + len(self.numerical_cols))
        chosen_columns = random.sample(self.categorical_cols + self.numerical_cols, num_of_columns)

        where_clause = []

        for col in chosen_columns:
            if col in self.categorical_cols:
                # categorical column
                # TODO I think I should choose frequency different
                frequency = random.uniform(0, 1) * len(self.categorical_vals[col])
                values = random.sample(self.categorical_vals[col], max(int(frequency), 1))
                values = [val.replace("'", "''") for val in values]
                db_formatted_values = " , ".join([f"\'{value}\'" for value in values])

                where_clause.append(f'{col} IN ({db_formatted_values})')

            elif col in self.numerical_cols:
                # numerical column
                choice1 = random.uniform(float(self.numerical_vals[col][0]), float(self.numerical_vals[col][1]))
                choice2 = random.uniform(float(self.numerical_vals[col][0]), float(self.numerical_vals[col][1]))

                lb = min(choice1, choice2)
                ub = max(choice1, choice2)

                where_clause.append(f'{col} BETWEEN {lb} AND {ub}')

        where_clause_str = ''
        if len(where_clause) > 0:
            where_clause_str = f'WHERE {" AND ".join(where_clause)}'

        query = f"SELECT {'*' if self.select_entire_row else self.index_col} " \
                f"FROM {self.schema}.{self.table} {where_clause_str}"

        print(f"Created query:\n{query}")

        return query
