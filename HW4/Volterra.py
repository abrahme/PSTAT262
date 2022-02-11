import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice, exponential
from typing import Tuple


class StochasticVolterra(object):
    """
    the stochastic volterra model
    """

    def __init__(self, alpha: float, beta: float, gamma: float, initial_state: np.array):
        """

        :param alpha: parameter
        :param beta: parameters
        :param gamma: parameter
        :param initial_state: parameter indicating start state of rabbits
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.state = initial_state  # 2 dimensional vector, with p as first element, q as second
        self.time = 0

    def generator(self, state: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param state: initial state
        :return: tuple of vector of 3 dimensions with generator for each state
        """
        p = state[0]
        q = state[1]
        a = np.zeros((3,))
        a[0] = self.alpha * p
        a[1] = self.beta * p * q
        a[2] = self.gamma * q
        transition_states = np.zeros((3, 2))
        transition_states[0, :] = np.array([p + 1, q])
        transition_states[1, :] = np.array([p - 1, q + 1])
        transition_states[2, :] = np.array([p, q - 1])
        return a, transition_states

    def update_state(self):
        """

        :return: time that has elapsed
        """
        a, transition_states = self.generator(self.state)
        t = exponential(scale=1 / np.sum(a))
        trans_probs = a / np.sum(a)
        new_state = choice(range(0, len(a)), p=trans_probs)
        self.time += t
        self.state = transition_states[new_state, :].copy()


if __name__ == "__main__":
    iterations = 10 ** 6
    g = 5
    a = 4
    b = .0102
    start_state = np.ones((2,)) * 1000
    rabbits = []
    wolves = []
    times = []
    volterra = StochasticVolterra(a, b, g, start_state)
    for j in range(iterations):
        if sum(volterra.state == 0) > 0:
            break
        volterra.update_state()
        times.append(volterra.time)
        cur_state = volterra.state
        rabbits.append(cur_state[0])
        wolves.append(cur_state[1])
    plt.plot(times, rabbits)
    plt.plot(times, wolves)
    plt.legend(["Rabbits", "Wolves"])
    plt.title(f"Stochastic Voletrra Model with alpha:{a}, beta:{b}, gamma: {g}")
    plt.xlabel("Time (years)")
    plt.ylabel("Population")
    plt.show()

    plt.plot(rabbits,wolves)
    plt.xlabel("Rabbits")
    plt.ylabel("Wolves")
    plt.title(f"Stochastic Voletrra Model with alpha:{a}, beta:{b}, gamma: {g}")
    plt.show()
