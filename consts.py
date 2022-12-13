class DBTypes:
    IS_POSTGRESQL = lambda db_type: str.lower(db_type) == 'postgres' or str.lower(db_type) == 'postgresql'
    IS_MYSQL = lambda db_type: str.lower(db_type) == 'mysql' or str.lower(db_type) == 'pymysql'


class GraphNames:
    CLUSTERS = 'clusters'
    DISTRIBUTIONS = 'distributions'


class CheckpointNames:
    CLUSTERS = 'clusters'
    WEIGHTS = 'weights'
    SAMPLE = 'sample'
    LAZY_GREEDY_METADATA = 'lazyGreedyMetadata'
    LAZY_GREEDY_MODEL = 'lazyGreedyModel'
