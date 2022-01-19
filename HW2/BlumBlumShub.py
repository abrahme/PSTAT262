import random

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def sieve_of_eratosthenes(n: int) -> List[int]:
    """

    :param n: upper bound of primes
    :return: list of all prime numbers less than n
    """
    possible_primes = {val: True for val in range(2, n + 1)}
    upper_bound = int((np.sqrt(n)))

    for num in range(2, upper_bound + 1):
        if possible_primes[num]:
            i = 0
            while num ** 2 + num * i <= n:
                possible_primes[num ** 2 + num * i] = False
                i += 1
    return [key for (key, val) in possible_primes.items() if val]


def find_blum_primes(primes: List[int]) -> List[int]:
    """

    :param primes: list of prime numbers
    :return: list of blum primes
    """
    return [prime for prime in primes if prime % 4 == 3]


def blum_blum_shub(n: int, seq_length: int) -> List[int]:
    """
    runs blum blum algorithm to return a list of random bits (1,0)
    :param seq_length: sequence of length to return
    :param n: how large of primes we want
    :return:  list of random bits (1,0)
    """

    primes = sieve_of_eratosthenes(n)
    blum_primes = find_blum_primes(primes)
    blum_n = blum_primes[-1] * blum_primes[-2]
    seed = random.choice(range(1, blum_n))
    x_i = (seed ** 2) % blum_n
    parities = [x_i % 2]

    i = 1
    while i < seq_length:
        x_i = x_i ** 2 % blum_n
        parities.append(x_i % 2)
        i += 1
    return parities


if __name__ == "__main__":
    print(len(blum_blum_shub(10000, 100)))
