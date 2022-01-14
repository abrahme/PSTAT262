import numpy as np
import matplotlib.pyplot as plt


class Box(object):
    """
    box in which we're dropping needles
    """

    def __init__(self, width: float, length: float):
        """
        originates at 0,0 and extends into positive direction both ways
        :param width: width of the box
        :param length: length of the box
        """

        assert width > 0
        assert length > 0
        self.d = width  ### just use one box
        self.width = width
        self.length = length

    def sample_box(self, n_samples) -> np.array:
        """

        :param n_samples: how many points to sample
        :return: n x 2 dim array for the x,y coordinates in the box
        """
        return np.random.uniform(low=[0, 0], high=[self.width, self.length], size=(n_samples, 2))

    def is_in_box(self, coordinates: np.array) -> np.array:
        """

        :param coordinates: coordinates of start and end of object (n x 4) -> (x1,y1,x2,y2)
        here x1 < x2
        :return: boolean vector length N to determine if object is in box
        """
        return (coordinates[:, 0] >= 0) & (coordinates[:, 2] <= self.d)


class Needle(object):
    def __init__(self, center: np.array, orientation: float, length: float):
        """

        :param center: center of needle (1 x 2 arrray)
        :param orientation: angle between 0 and 2*pi
        """
        assert length > 0
        self.center = center
        self.orientation = orientation
        self.length = length
        self.endpoints = np.expand_dims(self.calculate_endpoints(), 0)

    def calculate_endpoints(self) -> np.array:
        """

        :return:  1 x 4 numpy array of x1,y1,x2,y2 coordinates of endpoints of needle in plane
        x1 < x2
        """
        left_endpoint = np.array([-self.length / 2, 0])  ### centered at 0
        right_endpoint = np.array([self.length / 2, 0])

        rotation_matrix = np.zeros((2, 2))
        rotation_matrix[0, 0] = np.cos(self.orientation)
        rotation_matrix[1, 1] = rotation_matrix[0, 0]
        rotation_matrix[0, 1] = -np.sin(self.orientation)
        rotation_matrix[1, 0] = -rotation_matrix[0, 1]

        rotated_endpoint_1 = np.matmul(rotation_matrix, left_endpoint.T) + self.center
        rotated_endpoint_2 = np.matmul(rotation_matrix, right_endpoint.T) + self.center

        if rotated_endpoint_2[0] > rotated_endpoint_1[0]:
            return np.hstack((rotated_endpoint_1, rotated_endpoint_2))
        else:
            return np.hstack((rotated_endpoint_2, rotated_endpoint_1))


def conduct_experiment(needle_length: float, width: float, box_length: float, num_trials: int) -> float:
    """
    conducts buffons needle experiment
    :param needle_length: length of needle to use
    :param width: width of box (also distance between lines
    :param box_length: y direction max of the box
    :param num_trials: how many needles to throw
    :return: proportion of successes (needles hitting the line)
    """
    sample_orientations = np.random.uniform(low=0, high=2 * np.pi, size=num_trials)
    box = Box(width=width, length=box_length)
    sample_centers = box.sample_box(num_trials)
    needle_endpoints = np.concatenate(
        [Needle(sample_centers[i, :], sample_orientations[i], length=needle_length).endpoints for i in
         range(num_trials)], axis=0)
    return (1 - box.is_in_box(needle_endpoints)).mean()


if __name__ == "__main__":
    needle_length = 1
    width = 1
    box_length = 1
    num_trials = 1000000
    estimate_list = []
    for power in range(1,8):
        num_trials = 10 ** power
        estimate_pi = (2 * needle_length) / \
                      (width * conduct_experiment(needle_length, width, box_length, num_trials))
        estimate_list.append(estimate_pi)

    plt.scatter([j for j in range(1,8)],estimate_list)
    plt.axhline(y=np.pi, color='r', linestyle='-')
    plt.title("Estimate of pi vs Num Trials")
    plt.xlabel("log(Number of Trials)")
    plt.ylabel("Estimate")
    plt.show()

    #### repeated sampling of 100 dropped needles
    batch_average = []
    for batch_size in range(1,8):
        estimates = []
        for _ in range(10**batch_size):
            estimate_pi = (2 * needle_length) / \
                      (width * conduct_experiment(needle_length, width, box_length, 100))
            estimates.append(estimate_pi)
        batch_average.append(np.mean(estimates))

    plt.scatter([j for j in range(1, 8)], batch_average)
    plt.axhline(y=np.pi, color='r', linestyle='-')
    plt.title("Estimate of pi vs Number of Batches of size 100")
    plt.xlabel("log(Number of Batches of size 100)")
    plt.ylabel("Estimate")
    plt.show()


