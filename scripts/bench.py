import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, \
    BarColumn, MofNCompleteColumn, SpinnerColumn,\
    TimeRemainingColumn


from ponca_kdtree import KdTree, KnnGraph
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree as SklearnKDTree

class KDTreeInterface:
    def __init__(self):
        pass

    def build(self, points):
        pass

    def query_radius(self, index, radius):
        return None

    def query_knn(self, index, k):
        return None
    
    def name(self):
        return "KDTree Interface"


class PoncaKDTree(KDTreeInterface):
    def __init__(self):
        super().__init__()

    def build(self, points):
        self.kdtree = KdTree(points)

    def query_radius(self, index, radius):
        return np.asarray(self.kdtree.query_radius_index(index, radius))

    def query_knn(self, index, k):
        return np.asarray(self.kdtree.query_knn_index(index, k))

    def name(self):
        return "Ponca KDTree"


class PoncaKNNGraph(KDTreeInterface):
    def __init__(self):
        super().__init__()

    def build(self, points, graph_k=8):
        self.graph = KnnGraph(points, graph_k)

    def query_radius(self, index, radius):
        return np.asarray(self.graph.query_radius_index(index, radius))

    def query_knn(self, index, k):
        neighbors = np.asarray(self.graph.query_knn_index(index))
        return neighbors
    
    def name(self):
        return "Ponca KNNGraph"


class ScipyKDTree(KDTreeInterface):
    def __init__(self):
        super().__init__()

    def build(self, points):
        self.kdtree = cKDTree(points)

    def query_radius(self, index, radius):
        return np.asarray(self.kdtree.query_ball_point(self.kdtree.data[index], radius))

    def query_knn(self, index, k):
        _, neighbors = self.kdtree.query(self.kdtree.data[index], k=k + 1)
        return neighbors

    def name(self):
        return "SciPy KDTree"


class SkLearnKDTree(KDTreeInterface):
    def __init__(self):
        super().__init__()

    def build(self, points):
        self.points = points
        self.kdtree = SklearnKDTree(points)

    def query_radius(self, index, radius):
        return np.asarray(self.kdtree.query_radius(self.points[index:index + 1], r=radius)[0])

    def query_knn(self, index, k):
        _, neighbors = self.kdtree.query(self.points[index:index + 1], k=k + 1)
        return neighbors[0]

    def name(self):
        return "sklearn KDTree"


def count_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def build_kdtree(kdtree_object: KDTreeInterface, points, graph_k):
    if isinstance(kdtree_object, PoncaKNNGraph):
        return count_time(kdtree_object.build, points, graph_k)
    return count_time(kdtree_object.build, points)


def query_radius(kdtree_object: KDTreeInterface, query_indices: np.ndarray, radius: float):
    return count_time(lambda: [kdtree_object.query_radius(int(index), radius) for index in query_indices])


def query_knn(kdtree_object: KDTreeInterface, query_indices: np.ndarray, k: int):
    return count_time(lambda: [kdtree_object.query_knn(int(index), k) for index in query_indices])


def create_pointcloud(num_points, dim=3):
    return np.random.rand(num_points, dim).astype(np.float32)


@dataclass
class Kdtrees:
    ponca_kdtree: PoncaKDTree = field(default_factory=PoncaKDTree)
    ponca_knngraph: PoncaKNNGraph = field(default_factory=PoncaKNNGraph)
    scipy_kdtree: ScipyKDTree = field(default_factory=ScipyKDTree)
    sklearn_kdtree: SkLearnKDTree = field(default_factory=SkLearnKDTree)

    def __iter__(self):
        return iter([self.ponca_kdtree, self.ponca_knngraph, self.scipy_kdtree, self.sklearn_kdtree])

@dataclass
class BenchmarkResults:
    names: list
    build_times: list
    radius_times: list
    knn_times: list


def run_benchmark(num_points, radius, knn, graph_k):

    pointcloud = create_pointcloud(num_points)
    query_points = np.random.choice(num_points, size=min(100, num_points), replace=False)
    kdtrees = Kdtrees()

    names = []
    build_results = []
    radius_results = []
    knn_results = []

    for kdtree in kdtrees:
        names.append(kdtree.name())
        _, build_time = build_kdtree(kdtree, pointcloud, graph_k)
        radius_queries, radius_time = query_radius(kdtree, query_points, radius)
        knn_queries, knn_time = query_knn(kdtree, query_points, knn)

        build_results.append(build_time)
        radius_results.append(radius_time)
        knn_results.append(knn_time)

    return BenchmarkResults(names, build_results, radius_results, knn_results)


def add_result(results_dict, result: BenchmarkResults):
    for i, name in enumerate(result.names):
        if name not in results_dict:
            results_dict[name] = {
                "build_time": [],
                "radius_time": [],
                "knn_time": []
            }
        results_dict[name]["build_time"].append(result.build_times[i])
        results_dict[name]["radius_time"].append(result.radius_times[i])
        results_dict[name]["knn_time"].append(result.knn_times[i])
    return results_dict

def fmt(val, best):
    ratio = val / best 
    text = f"{val:.4f} [dim]{ratio:.2f}x[/]"
    if val == best:
        text = f"[bold green]{text}[/]"
    return text

def log_results(results_dict):
    console = Console()

    table = Table(title="Benchmark Results", header_style="bold magenta", border_style="dim")
    table.add_column("Lib", style="bold", no_wrap=True)
    table.add_column("Mean Build Time (s)", justify="right")
    table.add_column("Mean Radius Query Time (s)", justify="right")
    table.add_column("Mean KNN Query Time (s)", justify="right")

    best_build_time = min(np.mean(metrics["build_time"]) for metrics in results_dict.values())
    best_radius_time = min(np.mean(metrics["radius_time"]) for metrics in results_dict.values())
    best_knn_time = min(np.mean(metrics["knn_time"]) for metrics in results_dict.values())

    for name, metrics in results_dict.items():
        mean_build_time = float(np.mean(metrics["build_time"]))
        mean_radius_time = float(np.mean(metrics["radius_time"]))
        mean_knn_time = float(np.mean(metrics["knn_time"]))

        table.add_row(
            name,
            fmt(mean_build_time, best_build_time),
            fmt(mean_radius_time, best_radius_time),
            fmt(mean_knn_time, best_knn_time),
        )
    console.print(table)
    console.print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", default=100_000,  type=int )
    parser.add_argument("--radius", default=0.5, type=float )
    parser.add_argument("--knn", default=8, type=int )
    parser.add_argument("--graph-k",default=8, type=int )
    parser.add_argument("--repeats", default=25, type=int )
    parser.add_argument("--seed", default=123, type=int )
    args = parser.parse_args()

    results = {}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("Benchmark"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        for i in progress.track(range(args.repeats)):
            benchResult = run_benchmark(args.num_points, args.radius, args.knn, args.graph_k)
            results = add_result(results, benchResult)
    log_results(results)
    
if __name__ == "__main__":
    main()
