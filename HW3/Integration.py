import numpy as np
import matplotlib.pyplot as plt


def trapezoid_rule(a: float, b: float, n: int, function) -> float:
    """

    :param a: lower bound of integration
    :param b: upper bound of integration
    :param n: number of points to evaluate
    :param function: a real valued function
    :return: estimate of integral
    """

    dx = (b - a) / n
    t = a + dx * np.array(range(1, n))
    return dx * (.5 * (function(a) + function(b)) + np.sum(function(t)))


def midpoint_rule(a: float, b: float, n: int, function) -> float:
    """

    :param a: lower bound of integration
    :param b: upper bound of integration
    :param n: number of points to evaluate
    :param function: a real valued function
    :return: estimate of integral
    """
    dx = (b - a) / n
    end_t = (a + dx * np.array(range(1, n + 1)))
    begin_t = (a + dx * np.array(range(0, n)))
    return dx * np.sum(function(.5 * (begin_t + end_t)))


if __name__ == "__main__":
    true_value = 1 / 3
    midpoint_value = []
    trapezoid_value = []
    for i in range(1, 5):
        n = 10 ** i
        midpoint_value.append(midpoint_rule(0, 1, n, lambda x: x ** 2))
        trapezoid_value.append(trapezoid_rule(0, 1, n, lambda x: x ** 2))

    plt.plot([10 ** i for i in range(1, 5)], trapezoid_value, label = "trapezoid")
    plt.plot([10 ** i for i in range(1, 5)], midpoint_value, label = "midpoint")
    plt.xlabel("Number of Design Points")
    plt.ylabel("Estimate of Integral")
    plt.title("Estimate of x^2 on [0,1]")
    plt.legend()
    plt.show()
