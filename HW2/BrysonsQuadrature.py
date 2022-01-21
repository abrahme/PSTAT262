import matplotlib.pyplot as plt
import numpy as np


def bryson_quadrature(n: int, radius: float) -> float:
    """
    method to estimate pi
    :param n: number of equally spaced polygons
    :param radius: radius of circle
    :return: estimate of pi
    """
    central_angle = np.pi * 2 / n
    apothem = np.cos(central_angle / 2) * radius
    base = 2 * np.sin(central_angle / 2) * radius
    return (n * apothem * base / 2) / (radius ** 2)


if __name__ == "__main__":
    r = 1
    estimates = []
    for num_points in range(3, 100):
        estimates.append(bryson_quadrature(num_points, r))
    plt.plot(estimates)
    plt.axhline(np.pi, color="r")
    plt.title("Bryson's Method for Estimating pi")
    plt.xlabel("Number of points")
    plt.ylabel("Estimate of pi")
    plt.show()
