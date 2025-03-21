import abc
import contextlib
import random
import collections
import copy

import numpy as np
import networkx as nx

"""A general Interface"""


class GraphSimilarityDataset(object):
    """Base class for all the graph similarity learning datasets.
  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets.
  """
    @abc.abstractmethod
    def pairs(self, batch_size, communities, G):
        """Create an iterator over pairs.
    Args:
      batch_size: int, number of pairs in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
        pass


"""Graph Edit Distance Task"""


# Graph Manipulation Functions
def permute_graph_nodes(g):
    """Permute node ordering of a graph, returns a new graph."""
    n = g.number_of_nodes()
    new_g = nx.Graph()
    new_g.add_nodes_from(range(n))
    perm = np.random.permutation(n)
    edges = g.edges()
    new_edges = []
    for x, y in edges:
        new_edges.append((perm[x], perm[y]))
    new_g.add_edges_from(new_edges)
    return new_g


def substitute_random_edges(g, n):
    """Substitutes n edges from graph g with another n randomly picked edges."""
    g = copy.deepcopy(g)
    n_nodes = g.number_of_nodes()
    edges = list(g.edges())
    # sample n edges without replacement
    e_remove = [
        edges[i] for i in np.random.choice(np.arange(len(edges)), n, replace=False)
    ]
    edge_set = set(edges)
    e_add = set()
    while len(e_add) < n:
        e = np.random.choice(n_nodes, 2, replace=False)
        # make sure e does not exist and is not already chosen to be added
        if (
                (e[0], e[1]) not in edge_set
                and (e[1], e[0]) not in edge_set
                and (e[0], e[1]) not in e_add
                and (e[1], e[0]) not in e_add
        ):
            e_add.add((e[0], e[1]))

    for i, j in e_remove:
        g.remove_edge(i, j)
    for i, j in e_add:
        g.add_edge(i, j)
    return g


def substitute_random_edges_ig(G, n):
    """在 igraph.Graph (有向图) 中随机替换 `n` 条边"""
    G = G.copy()  # 复制图，避免修改原始图
    n_nodes = G.vcount()  # 获取节点数
    edges = G.get_edgelist()  # 获取所有边的列表

    # 如果图的节点数小于 2，无法替换边，直接返回原图
    if n_nodes < 2:
        print(f"[警告] 图的节点数不足 ({n_nodes})，无法替换边，直接返回原图。")
        return G

    # 如果要替换的边比图中的边还多，直接返回
    if len(edges) < n:
        print(f"[警告] 需要替换 {n} 条边，但图中只有 {len(edges)} 条边，直接返回原图。")
        return G

    # 1、**随机选择 `n` 条边进行删除**
    e_remove_idx = np.random.choice(len(edges), n, replace=False)  # 选 `n` 条边索引
    e_remove = [edges[i] for i in e_remove_idx]  # 获取要删除的边
    edge_set = set(edges)  # 转换为集合，方便查重

    # 2、**随机生成 `n` 条新边**
    e_add = set()
    while len(e_add) < n:
        e = tuple(np.random.choice(n_nodes, 2, replace=False))  # 生成 (src, dst)
        if e not in edge_set and e not in e_add:  # 确保新边不重复
            e_add.add(e)

    # 3、**执行删除和添加**
    G.delete_edges(e_remove)  # 删除选定的 `n` 条边
    G.add_edges(list(e_add))  # 添加 `n` 条新边

    return G  # 返回修改后的图


class GraphEditDistanceDataset(GraphSimilarityDataset):
    """Graph edit distance dataset."""

    def __init__(
            self,
            n_nodes_range,
            p_edge_range,
            n_changes_positive,
            n_changes_negative,
            permute=True,
    ):
        """Constructor.
    Args:
      n_nodes_range: a tuple (n_min, n_max).  The minimum and maximum number of
        nodes in a graph to generate.
      p_edge_range: a tuple (p_min, p_max).  The minimum and maximum edge
        probability.
      n_changes_positive: the number of edge substitutions for a pair to be
        considered positive (similar).
      n_changes_negative: the number of edge substitutions for a pair to be
        considered negative (not similar).
      permute: if True (default), permute node orderings in addition to
        changing edges; if False, the node orderings across a pair or triplet of
        graphs will be the same, useful for visualization.
    """
        self._n_min, self._n_max = n_nodes_range
        self._p_min, self._p_max = p_edge_range
        self._k_pos = n_changes_positive
        self._k_neg = n_changes_negative
        self._permute = permute

    def _get_pair(self, positive, communities, G):
        """Generate one pair of graphs from a given community structure.

        Args:
            positive (bool): 是否是正样本
            communities (dict): {社区ID: [节点列表]}
            G (igraph.Graph): 原始的完整图

        Returns:
            permuted_g (igraph.Graph): 经过节点重排的社区子图
            changed_g (igraph.Graph): 经过边修改的社区子图
        """
        # 从 `communities` 里随机选一个社区**
        community_id = np.random.choice(list(communities.keys()))
        community_nodes = communities[community_id]  # 该社区的节点列表

        # **从原图 `G` 提取该社区的子图**
        g = G.subgraph(community_nodes)

        # **根据 `positive` 选择边的修改数量**
        n_changes = self._k_pos if positive else self._k_neg

        # **对子图 `g` 进行边修改**
        changed_g = substitute_random_edges_ig(g, n_changes)

        return g, changed_g

    def _pairs(self, batch_size, communities, G):
        """Yields batches of pair data."""
        while True:
            batch_graphs = []
            batch_labels = []
            positive = True
            for _ in range(batch_size):
                g1, g2 = self._get_pair(positive, communities, G)
                batch_graphs.append((g1, g2))
                batch_labels.append(1 if positive else -1)
                positive = not positive

            packed_graphs = self._pack_batch(batch_graphs)
            labels = np.array(batch_labels, dtype=np.int32)
            yield packed_graphs, labels

    def _pack_batch(self, graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
    Args:
      graphs: a list of generated networkx graphs.
    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
        Graphs = []
        for graph in graphs:
            for inergraph in graph:
                Graphs.append(inergraph)
        graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.vcount()
            n_edges = g.ecount()
            # **检查是否为空图**
            if n_nodes == 0:
                print(f"[警告] 图 {i} 没有节点，跳过！")
                continue

            if n_edges == 0:
                print(f"[警告] 图 {i} 没有边，跳过！")
                continue

            edges = np.array(g.get_edgelist(), dtype=np.int32)
            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features.
            # setting higher dimension of ones to confirm code functioning
            # with high dimensional features.
            node_features=np.ones((n_total_nodes, 8), dtype=np.float32),
            edge_features=np.ones((n_total_edges, 4), dtype=np.float32),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )


# Use Fixed datasets for evaluation
@contextlib.contextmanager
def reset_random_state(seed):
    """This function creates a context that uses the given seed."""
    np_rnd_state = np.random.get_state()
    rnd_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed + 1)
    try:
        yield
    finally:
        random.setstate(rnd_state)
        np.random.set_state(np_rnd_state)


class FixedGraphEditDistanceDataset(GraphEditDistanceDataset):
    """A fixed dataset of pairs or triplets for the graph edit distance task.
  This dataset can be used for evaluation.
  """

    def __init__(
            self,
            n_nodes_range,
            p_edge_range,
            n_changes_positive,
            n_changes_negative,
            dataset_size,
            permute=True,
            seed=1234,
    ):
        super(FixedGraphEditDistanceDataset, self).__init__(
            n_nodes_range,
            p_edge_range,
            n_changes_positive,
            n_changes_negative,
            permute=permute,
        )
        self._dataset_size = dataset_size
        self._seed = seed

    def pairs(self, batch_size, communities, G):
        """Yield pairs and labels."""

        if hasattr(self, "_pairs") and hasattr(self, "_labels"):
            pairs = self._pairs
            labels = self._labels
        else:
            # get a fixed set of pairs first
            with reset_random_state(self._seed):
                pairs = []
                labels = []
                positive = True
                for _ in range(self._dataset_size):
                    pairs.append(self._get_pair(positive, communities, G))
                    labels.append(1 if positive else -1)
                    positive = not positive
            labels = np.array(labels, dtype=np.int32)

            self._pairs = pairs
            self._labels = labels

        ptr = 0
        while ptr + batch_size <= len(pairs):
            batch_graphs = pairs[ptr: ptr + batch_size]
            packed_batch = self._pack_batch(batch_graphs)
            yield packed_batch, labels[ptr: ptr + batch_size]
            ptr += batch_size
