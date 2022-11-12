class DBTypes:
    IS_POSTGRESQL = lambda db_type: str.lower(db_type) == 'postgres' or str.lower(db_type) == 'postgresql'
    IS_MYSQL = lambda db_type: str.lower(db_type) == 'mysql' or str.lower(db_type) == 'pymysql'


class GraphNames:
    CLUSTERS = 'clusters'
    DISTRIBUTIONS = 'distributions'
