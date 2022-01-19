import numpy as np


def sample(n: int, dim: int) -> np.array:
    """
    mean of n samples from a hypercube function of dim dimensions
    :param n:
    :param dim:
    :return: mean
    """
    realization = np.random.uniform(np.zeros(dim), np.ones(dim), size=(n, dim))
    return np.squeeze(np.prod(realization + .5, axis=1))


if __name__ == "__main__":
    from multiprocessing.pool import ThreadPool

    chunk_size = 10 ** 6
    num_samples = 10 ** 8
    samples_remaining = num_samples
    chunk_results = {}
    i = 0
    prev_sum = 0
    prev_ss = 0
    prev_n = 0
    while samples_remaining > 0:
        n = min(chunk_size, samples_remaining)
        dim_samples = [100] * n
        pool = ThreadPool(20)
        results = pool.starmap(sample, zip(n * [1], dim_samples))
        pool.close()
        pool.join()
        chunk_results[i] = {"n": n, "sum": np.sum(results)}
        chunk_sum = np.sum(results)
        chunk_ss = np.sum(np.power(np.array(results) - chunk_sum / n, 2))
        prev_ss += prev_ss + chunk_ss
        if i > 0:
            prev_ss += (prev_n / n) * (1 / (prev_n + n)) * ((n / prev_n) * prev_sum - chunk_sum) ** 2
        prev_sum += chunk_sum
        prev_n += n
        i += 1
        samples_remaining -= n

        print(f"mean : {prev_sum/prev_n}")
        print(f"CI: {2.58 * np.sqrt(prev_ss / (prev_n - 1)) / np.sqrt(prev_n)}")

