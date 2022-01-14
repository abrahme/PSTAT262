import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set


class IsingLattice(object):
    """
    ising model lattice
    """

    def __init__(self, lattice: np.array, T: int):
        """

        :param lattice: (n x n) numpy array of starting configuration
        :param T: temperature
        """
        self.T = T
        self.lattice = lattice
        self.neighbor_graph = self.compute_neighbors(self.lattice)
        self.neighbor_list = self.compute_neighbor_list(self.neighbor_graph)

    def update_lattice(self) -> np.array:
        """

        :return: updated numpy array lattice
        """

        ### first we have to pick i,j to flip
        original_hamiltonian = self.compute_hamiltonian(self.lattice, self.neighbor_list)
        parent = random.choice(list(self.neighbor_graph))
        new_lattice = self.lattice.copy()
        new_lattice[parent[0], parent[1]] *= -1
        new_hamiltonian = self.compute_hamiltonian(new_lattice, self.neighbor_list)
        prob = min(1, np.exp(-1 * (new_hamiltonian - original_hamiltonian) / self.T))
        if np.random.rand() < prob:
            self.lattice = new_lattice

    @staticmethod
    def compute_neighbors(lattice: np.array) -> Dict:
        """
        computes neighbors of each site and builds dictionary
        :param lattice: n x n matrix of 1,-1
        :return: hamiltonian
        """

        n = lattice.shape[0]

        ### first we have to find all pair of adjacent sites and we do not want to repeat
        ### j adjacent i means i adjacent j
        neighbor_dict = {}
        for row in range(n):
            for column in range(n):
                site = (row, column)
                children_list = []
                if row < n - 1:
                    children_list.append((row + 1, column))
                if column < n - 1:
                    children_list.append((row, column + 1))
                if 0 < column:
                    children_list.append((row, column - 1))
                if 0 < row:
                    children_list.append((row - 1, column))
                neighbor_dict[site] = children_list
        return neighbor_dict

    @staticmethod
    def compute_neighbor_list(neighbor_graph: Dict) -> List[Set[Tuple]]:
        """

        :param neighbor_graph: dict keying site number to parent node and children
        :return: list of edges in the entire graph
        """

        neighbor_list = []
        for parent in neighbor_graph:
            children = neighbor_graph[parent]
            for child in children:
                if parent in neighbor_graph[child]:
                    edge = {parent, child}
                    if edge not in neighbor_list:
                        neighbor_list.append(edge)
        return neighbor_list

    @staticmethod
    def compute_hamiltonian(lattice: np.array, neighbor_list: List[Set[Tuple]]) -> int:
        """

        :param lattice: n x n array
        :param neighbor_list: list of which edges to compute interaction with
        :return: hamiltonian calculation
        """
        hamiltonian = 0
        for edge in neighbor_list:
            val = 1
            for node in edge:
                val *= lattice[node[0], node[1]]
            hamiltonian += val
        return hamiltonian

    def compute_magnetization(self) -> float:
        """
        computes the magnetization of a lattice

        :return: sum of positive - sum of negatives / total
        """

        num_pos = np.sum(self.lattice == 1)

        num_neg = np.sum(self.lattice == -1)

        return (num_pos - num_neg) / len(self.neighbor_list)


if __name__ == "__main__":
    lattice_size = 9
    lattice = -1 * np.ones((lattice_size, lattice_size))
    time_steps = 200
    results = dict()
    for T in range(1, 9):
        ising = IsingLattice(lattice, T)
        magnetization = []
        for time_step in range(time_steps):
            ising.update_lattice()
            magnetization.append(ising.compute_magnetization())
        results[T] = magnetization

    for T in results:
        plt.plot(range(0, time_steps), results[T], label=f"T = {T}")

    plt.xlabel('Time Steps')
    plt.ylabel('Magnetization')
    plt.title(f'Magnetization over Time, {lattice_size} by {lattice_size} Lattice')
    plt.legend()
    plt.show()
