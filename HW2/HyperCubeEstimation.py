import numpy as np


def sample(n: int, dim: int) -> np.array:
    """
    mean of n samples from a hypercube function of dim dimensions
    :param n:
    :param dim:
    :return: mean
    """
    realization = np.random.uniform(np.zeros(dim), np.ones(dim), size=(n, dim))
    return np.prod(realization + .5, axis=1)


if __name__ == "__main__":
    from multiprocessing.pool import ThreadPool
    import time

    num_samples = 10 ** 8
    n = [1] * num_samples
    dim_samples = [100] * num_samples
    pool = ThreadPool(20)
    start = time.time()
    results = pool.starmap(sample, zip(n, dim_samples))
    pool.close()
    pool.join()
    print(f"mean : {np.mean(results)}")
    print(f"CI: {2.58*np.std(results)/np.sqrt(num_samples)}")
    end = time.time()
    print(f"Time elapsed: {end - start} seconds")