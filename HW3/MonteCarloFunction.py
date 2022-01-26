import numpy as np
import matplotlib.pyplot as plt


def sample(n: int, dim: int) -> np.array:
    """

    :param n: number of samples
    :param dim: number of dimensions
    :return: array of function values (n x 1)
    """
    samples = np.random.uniform(low=np.zeros((dim,)),
                                high=np.ones((dim,)),
                                size=(n, dim))
    return np.prod(samples * (1 - samples), axis=1) * (6 ** dim)


if __name__ == "__main__":
    samps = sample(1000000, 100)
    print(f"Mean: {samps.mean()}")
    print(f"Standard Deviation: {samps.std()}")
    plt.hist(samps,density = True)
    plt.show()

