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


class ExplicitVolterra(object):

    def __init__(self, initial_state: np.ndarray, step_size: float):
        """

        :param initial_state: initial number of predator and prey (2,) array
        :param step_size: how much to increment in time
        """
        self.state = initial_state
        self.h = step_size

    def update_state(self):
        prev_q = self.state[1]
        prev_p = self.state[0]
        cur_q = prev_q + self.h * (prev_p - 2) * prev_q
        cur_p = prev_p + self.h * (2 - prev_q) * prev_p
        self.state = np.array([cur_p, cur_q])


class SymplecticVolterra(ExplicitVolterra):

    def __init__(self, initial_state: np.ndarray, step_size: float):
        super().__init__(initial_state, step_size)

    def update_state(self):
        prev_q = self.state[1]
        prev_p = self.state[0]
        cur_q = prev_q + self.h * (prev_p - 2) * prev_q
        cur_p = prev_p + self.h * (2 - cur_q) * prev_p
        self.state = np.array([cur_p, cur_q])


class ImplicitVolterra(ExplicitVolterra):

    def __init__(self, initial_state: np.ndarray, step_size: float):
        super().__init__(initial_state, step_size)

    def update_state(self):
        prev_q = self.state[1]
        prev_p = self.state[0]
        a = (2 * self.h ** 2 - self.h)
        b = (1 - 4 * self.h ** 2 + self.h * (prev_q + prev_p))
        c = -prev_p * (1 + 2 * self.h)

        cur_p = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        cur_q = prev_q / (1 - self.h * (cur_p - 2))
        self.state = np.array([cur_p, cur_q])


if __name__ == "__main__":
    iterations = 10 ** 4
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

    plt.plot(rabbits, wolves)
    plt.xlabel("Rabbits")
    plt.ylabel("Wolves")
    plt.title(f"Stochastic Voletrra Model with alpha:{a}, beta:{b}, gamma: {g}")
    plt.show()

    #### explicit euler
    step_size = .005
    start_state = np.ones((2,)) * 30
    explicit_volterra = ExplicitVolterra(start_state, step_size)
    symplectic_volterra = SymplecticVolterra(start_state, step_size)
    implicit_volterra = ImplicitVolterra(start_state, step_size)
    rabbits = []
    rabbits_symp = []
    rabbits_imp = []
    wolves = []
    wolves_symp = []
    wolves_imp = []
    for j in range(iterations):
        if sum(explicit_volterra.state == 0) > 0:
            break
        explicit_volterra.update_state()
        rabbits.append(explicit_volterra.state[0])
        wolves.append(explicit_volterra.state[1])
    for j in range(iterations):
        if sum(symplectic_volterra.state == 0) > 0:
            break
        symplectic_volterra.update_state()
        rabbits_symp.append(symplectic_volterra.state[0])
        wolves_symp.append(symplectic_volterra.state[1])
    for j in range(iterations):
        if sum(implicit_volterra.state == 0) > 0:
            break
        implicit_volterra.update_state()
        rabbits_imp.append(implicit_volterra.state[0])
        wolves_imp.append(implicit_volterra.state[1])
    plt.plot(rabbits, wolves)
    plt.plot(rabbits_symp, wolves_symp)
    plt.plot(rabbits_imp, wolves_imp)
    plt.legend(["Explicit","Symplectic","Implicit"])
    plt.xlabel("Rabbits")
    plt.ylabel("Wolves")
    plt.title("Volterra Model: Different ODE Methods")
    plt.show()
