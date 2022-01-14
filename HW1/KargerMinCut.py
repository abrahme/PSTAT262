import numpy as np
import math
from typing import List, Tuple, Set


def create_random_graph(n: int, p: float) -> List[Tuple[int, int, int]]:
    """

    :param n: number of nodes
    :param p: prob of an edge occurring
    :return: weighted edge list of undirected graph
    """
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < p:
                edges.append((i, j, 1))
    return edges


class Graph(object):
    """
    graph object to do things with
    """

    def __init__(self, edges: List[tuple[int, int, int]], n: int):
        """

        :param edges: list of tuple from node i to node j and weight
        """
        self.n = n
        self.edge_list = edges
        self.nodes = self.create_node_set(edges)
        self.adjacency_matrix = self.create_adjacency_matrix(self.edge_list, self.n)

    @staticmethod
    def create_adjacency_matrix(edges: List[tuple[int, int, int]], n: int) -> np.array:
        """

        :param n: how many nodes
        :param edges: list of 3-tuple objects describing an edge, weight represents number of edges
        :return: n x n binary matrix
        """
        adjacency_matrix = np.zeros((n, n))
        for edge in edges:
            start_node = edge[0]
            end_node = edge[1]
            weight = edge[2]
            adjacency_matrix[end_node, start_node] = weight
            adjacency_matrix[start_node, end_node] = weight

        return adjacency_matrix

    def contract_edge(self, contract_edge: Tuple[int, int, int], edges: List[Tuple[int, int, int]]) -> List[
        Tuple[int, int, int]]:
        """

        :param edges:
        :param contract_edge:  a tuple of edge
        :return: new edge list
        """

        new_edge_list = []
        node_set = self.create_node_set([contract_edge])
        contracted_node_name = min(node_set)

        for node in self.nodes:
            if node not in node_set:
                ### get edges of node
                node_edges = []
                for edge in edges:
                    if node == edge[0] or node == edge[1]:
                        ### gets edges where current node is connected to either endpoint of the contract edge
                        node_edges.append(edge)
                total_weight = 0

                for node_edge in node_edges:
                    parent, child, weight = node_edge
                    if parent in node_set or child in node_set:
                        total_weight += weight
                    else:
                        if node_edge not in new_edge_list and node_edge != contract_edge:
                            new_edge_list.append(node_edge)
                if total_weight > 0:
                    new_edge_list.append(
                        (min(node, contracted_node_name), max(node, contracted_node_name), total_weight))

        return new_edge_list

    @staticmethod
    def create_node_set(edges: List[tuple[int, int, int]]) -> Set:
        """

        :param edges: creates a set
        :return: set of edges
        """

        node_set = set()
        for (start_node, end_node, _) in edges:
            node_set.add(start_node)
            node_set.add(end_node)
        return node_set

    @staticmethod
    def pick_edge(edges: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """

        :param edges: as it seems
        :return: which edge to cut
        """
        weights = [edge[2] for edge in edges]
        probs = np.array(weights) / np.sum(weights)
        return edges[np.random.choice(np.arange(len(edges)), p=probs)]

    def karger_min_cut(self) -> int:
        """

        :return: the minimum cut of the graph
        """
        num_nodes = len(self.nodes)
        edges = self.edge_list
        cuts = []
        while num_nodes > 2:
            cut_edge = self.pick_edge(edges)
            cuts.append(cut_edge)
            edges = self.contract_edge(cut_edge, edges)
            num_nodes = len(self.create_node_set(edges))
        return edges[0][2]


if __name__ == "__main__":
    k = 10000
    p = .9
    for exponent in range(1,5):
        n = 10 * exponent
        edge_list = create_random_graph(n, p)
        graph = Graph(edge_list,n)
        min_val = np.inf
        for _ in range(int(np.log(k)*math.comb(n,2))):
            cut_size = graph.karger_min_cut()
            if cut_size < min_val:
                min_val = cut_size
        print(f"Min cut is {min_val} for N = {n}, p = {p}")
