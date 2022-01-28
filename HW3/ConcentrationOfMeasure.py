import numpy as np


class HyperCube(object):
    """
    hypercube to do things with
    """

    def __init__(self, dim: int):
        """

        :param dim: dimensions of the unit hypercube
        """
        self.dim = dim

    def sample(self, n):
        """

        :param n: number of samples
        :return:
        """
        dim = self.dim
        samples = np.random.uniform(np.zeros((dim,)), np.ones((dim,)), size=(n, dim))
        return samples

    @staticmethod
    def compute_min_distance_boundary(samples: np.array) -> np.array:
        """

        :param samples: n x dim array containing the samples in hypercube
        :return:
        """
        min_dim = np.minimum(samples, 1 - samples)
        min_distance = np.min(min_dim, axis=1)
        return min_distance

    @staticmethod
    def compute_min_distance_equator(samples: np.array, dim: int) -> np.array:
        """

        :param dim: dimensions of the hypercube
        :param samples:  n x dim array of samples
        :return: array of distances for each point
        """
        e = np.ones((dim, 1))
        return np.abs(np.matmul(samples, e) / dim - .5) * np.sqrt(dim)

    def estimate_annulus_volume(self, r: float, n: int) -> np.array:
        """
        :param n: num_samples
        :param r: radius of annulus
        :return: volume of annulus
        """
        ### first we draw n samples from an n dimensional hypercube
        samples = self.sample(n)
        #### then we compute the minimum distance of each sample to the boundary
        min_distance = self.compute_min_distance_boundary(samples)
        return np.mean(min_distance <= r)

    def estimate_meridian_volume(self, r: float, n: int) -> np.array:
        """

        :param r: radius
        :param n: num samples
        :return:
        """
        samples = self.sample(n)
        distance = self.compute_min_distance_equator(samples, self.dim)
        return np.mean(distance <= r * np.sqrt(self.dim))

    def estimate_intersection_meridian_annulus(self, r: float, n: int) -> np.array:
        """

        :param r: radius
        :param n: num samples
        :return:
        """
        samples = self.sample(n)
        min_distance_annulus = self.compute_min_distance_boundary(samples)
        min_distance_equator = self.compute_min_distance_equator(samples, self.dim)

        return np.mean((min_distance_annulus <= r) * (min_distance_equator <= np.sqrt(self.dim) * r))


if __name__ == "__main__":
    cube = HyperCube(500)
    print(cube.estimate_annulus_volume(.05, 1000000))
    print(cube.estimate_meridian_volume(.05, 1000000))
    print(cube.estimate_intersection_meridian_annulus(.05,1000000))
