from typing import List, Tuple, Dict

from consts import DBTypes
from data_access import DataAccess
from config_manager import ConfigManager


class Graph:
    def __init__(self, points: List[Tuple[float, float]]):
        self.points = points

    def get_x_axis(self) -> List[float]:
        return [point[0] for point in self.points]

    def get_y_axis(self) -> List[float]:
        return [point[1] for point in self.points]

    def get_points(self) -> List[Tuple[float, float]]:
        return self.points

    def get_point_by_x(self, x: float) -> Tuple[float, float]:
        candidates = [point for point in self.points if point[0] == x]
        if len(candidates) == 0:
            raise Exception(f'No point with x value: {x} was located')
        return candidates[0]

    def add_point(self, point: Tuple[float, float]):
        self.points.append(point)


class GraphsManager:
    graphs: Dict[str, Graph] = {}

    @staticmethod
    def get_graph(name: str) -> Graph:
        return GraphsManager.graphs.get(name, Graph([]))

    @staticmethod
    def add_point(graph_name: str, point: Tuple[float, float]):
        if graph_name not in GraphsManager.graphs:
            GraphsManager.graphs[graph_name] = Graph([])
        GraphsManager.graphs[graph_name].add_point(point)

    @staticmethod
    def get_point_by_x(graph_name: str, x: float) -> Tuple[float, float]:
        if graph_name not in GraphsManager.graphs:
            raise Exception(f'No such graph as: {graph_name}')
        return GraphsManager.graphs[graph_name].get_point_by_x(x)


class MetricsCalculator:
    MAX_TUPLES_TO_CALCULATE = 100

    @staticmethod
    def _get_random_tuples():
        dataAccess = DataAccess()
        indexCol = ConfigManager.get_config('workloadConfig.index_col')
        schema = ConfigManager.get_config('workloadConfig.schema')
        table = ConfigManager.get_config('workloadConfig.table')
        randomStatement = 'RANDOM' if DBTypes.IS_POSTGRESQL(ConfigManager.get_config('dbConfig.type')) else 'RAND'
        return dataAccess.select(
            f"SELECT {indexCol} FROM {schema}.{table} "
            f"ORDER BY {randomStatement}() LIMIT {MetricsCalculator.MAX_TUPLES_TO_CALCULATE}")

    def __init__(self):
        self.tuple_ids = MetricsCalculator._get_random_tuples()

    def calculate_distribution(self, query_clusters):
        distribution = []
        for tupleId in self.tuple_ids:
            distribution.append(
                sum([cluster['frequency'] for cluster in query_clusters if tupleId in cluster['result']]))
        s = sum(distribution)
        return [p / s for p in distribution]
